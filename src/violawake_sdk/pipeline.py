"""Voice pipeline orchestration for wake, VAD, STT, and TTS."""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np

from violawake_sdk._constants import DEFAULT_THRESHOLD
from violawake_sdk._exceptions import PipelineError
from violawake_sdk.vad import VADEngine
from violawake_sdk.wake_detector import WakeDetector

logger = logging.getLogger(__name__)

CommandHandler: TypeAlias = Callable[[str], str | None]
WakeCallback: TypeAlias = Callable[[], None]
PipelineEventName: TypeAlias = Literal[
    "wake",
    "listen_start",
    "listen_end",
    "transcribe_start",
    "transcribe_end",
    "response",
]
PipelineEventCallback: TypeAlias = Callable[..., object]

_SUPPORTED_EVENTS: frozenset[PipelineEventName] = frozenset(
    {
        "wake",
        "listen_start",
        "listen_end",
        "transcribe_start",
        "transcribe_end",
        "response",
    }
)

_STATE_IDLE = "idle"
_STATE_LISTENING = "listening"
_STATE_TRANSCRIBING = "transcribing"
_STATE_RESPONDING = "responding"

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
MAX_COMMAND_DURATION_S = 10.0
SILENCE_FRAMES_STOP = 30
MAX_RECORDING_FRAMES = int(MAX_COMMAND_DURATION_S / (FRAME_SAMPLES / SAMPLE_RATE))


class TranscriptSegmentLike(Protocol):
    """Protocol for streaming STT segments."""

    text: str
    start: float
    end: float


class LazySTTEngine(Protocol):
    """Protocol for the lazily imported STT engine."""

    def prewarm(self) -> None: ...

    def transcribe(self, audio: np.ndarray) -> str: ...

    def transcribe_streaming(self, audio: np.ndarray) -> Iterator[TranscriptSegmentLike]: ...


class LazyTTSEngine(Protocol):
    """Protocol for the lazily imported TTS engine."""

    def synthesize(self, text: str) -> np.ndarray: ...

    def play(self, audio: np.ndarray, *, blocking: bool = True) -> None: ...


class VoicePipeline:
    """Wake -> listen -> transcribe -> respond voice pipeline."""

    def __init__(
        self,
        wake_word: str = "viola",
        stt_model: str = "base",
        tts_voice: str = "af_heart",
        threshold: float = DEFAULT_THRESHOLD,
        vad_backend: str = "auto",
        vad_threshold: float = 0.4,
        enable_tts: bool = True,
        device_index: int | None = None,
        on_wake: WakeCallback | None = None,
        streaming_stt: bool = False,
    ) -> None:
        self._wake_detector = WakeDetector(model=wake_word, threshold=threshold)
        self._vad = VADEngine(backend=vad_backend)
        self._vad_threshold = vad_threshold
        self._enable_tts = enable_tts
        self._device_index = device_index
        self._stt_model = stt_model
        self._tts_voice = tts_voice
        self._streaming_stt = streaming_stt

        self._state = _STATE_IDLE
        self._last_command: str | None = None
        self._last_score: float | None = None
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

        self._stt: LazySTTEngine | None = None
        self._tts: LazyTTSEngine | None = None
        self._command_handlers: list[CommandHandler] = []
        self._event_handlers: dict[PipelineEventName, list[PipelineEventCallback]] = {
            event: [] for event in _SUPPORTED_EVENTS
        }

        if on_wake is not None:
            self.on("wake", on_wake)

        logger.info(
            "VoicePipeline initialized: wake=%s, stt=%s, tts=%s, streaming_stt=%s",
            wake_word,
            stt_model,
            tts_voice,
            streaming_stt,
        )

    @property
    def state(self) -> str:
        """Return the current pipeline state."""
        with self._state_lock:
            return self._state

    @property
    def last_command(self) -> str | None:
        """Return the most recent transcription result."""
        with self._state_lock:
            return self._last_command

    @property
    def last_score(self) -> float | None:
        """Return the most recent wake score."""
        with self._state_lock:
            return self._last_score

    def on(
        self,
        event: PipelineEventName,
        callback: PipelineEventCallback | None = None,
    ) -> PipelineEventCallback | Callable[[PipelineEventCallback], PipelineEventCallback]:
        """Register a callback for a pipeline event."""
        self._validate_event(event)

        def decorator(fn: PipelineEventCallback) -> PipelineEventCallback:
            with self._event_lock:
                self._event_handlers[event].append(fn)
            return fn

        if callback is None:
            return decorator
        return decorator(callback)

    def on_command(self, handler: CommandHandler) -> CommandHandler:
        """Register a command handler."""
        self._command_handlers.append(handler)
        return handler

    def run(self) -> None:
        """Run the blocking microphone pipeline."""
        logger.info("VoicePipeline started. Say the wake word to begin.")
        self._stop_event.clear()

        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        except Exception as exc:
            raise PipelineError(f"Pipeline error: {exc}") from exc
        finally:
            self.stop()
            self._set_state(_STATE_IDLE)
            logger.info("VoicePipeline stopped.")

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the pipeline to stop and wait briefly for worker cleanup."""
        self._stop_event.set()
        worker = self._get_worker_thread()
        if worker is None or worker is threading.current_thread():
            return

        worker.join(timeout=timeout)
        if worker.is_alive():
            logger.warning("VoicePipeline worker thread did not exit within %.1f s", timeout)
        else:
            with self._worker_lock:
                if self._worker_thread is worker:
                    self._worker_thread = None

    def close(self) -> None:
        """Stop the pipeline and release resources."""
        self.stop()
        self._set_state(_STATE_IDLE)
        self._wake_detector.close()
        self._stt = None
        self._tts = None

    def __enter__(self) -> VoicePipeline:
        """Enter sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit sync context manager."""
        self.close()

    def speak(self, text: str) -> None:
        """Synthesize and play text via TTS."""
        if not self._enable_tts or self._stop_event.is_set():
            return

        tts = self._get_tts()
        if tts is None:
            logger.warning("TTS not available - install 'violawake[tts]'")
            return

        try:
            audio = tts.synthesize(text)
            tts.play(audio)
        except Exception as exc:
            logger.exception("TTS playback failed for text '%.50s': %s", text, exc)

    def _run_loop(self) -> None:
        """Main microphone capture and detection loop."""
        recording_buffer: list[bytes] = []
        silence_count = 0

        for frame in self._wake_detector.stream_mic(device_index=self._device_index):
            if self._stop_event.is_set():
                break

            state = self.state
            if state == _STATE_IDLE:
                # Re-read the live state here; once we've branched on an IDLE snapshot,
                # `state == _STATE_RESPONDING` is dead unless we check the current state again.
                if self._wake_detector.detect(frame, is_playing=self._is_playing()):
                    score = self._get_detector_score()
                    with self._state_lock:
                        self._last_score = score
                    logger.info("Wake word detected -> listening for command")
                    recording_buffer.clear()
                    silence_count = 0
                    self._emit("wake", score=score)
                    if not self._transition_state(_STATE_IDLE, _STATE_LISTENING):
                        continue
                    self._emit("listen_start", score=score)
                continue

            if state != _STATE_LISTENING:
                continue

            recording_buffer.append(frame)
            is_speech = self._vad.is_speech(frame, threshold=self._vad_threshold)
            silence_count = 0 if is_speech else silence_count + 1
            total_frames = len(recording_buffer)

            if silence_count < SILENCE_FRAMES_STOP and total_frames < MAX_RECORDING_FRAMES:
                continue

            duration_s = total_frames * FRAME_SAMPLES / SAMPLE_RATE
            audio_bytes = b"".join(recording_buffer)
            self._emit("listen_end", duration_s=duration_s, frame_count=total_frames)
            if not self._transition_state(_STATE_LISTENING, _STATE_TRANSCRIBING):
                continue
            self._emit("transcribe_start", duration_s=duration_s, frame_count=total_frames)
            self._start_worker(audio_bytes)

    def _transcribe_and_respond(self, audio_bytes: bytes) -> None:
        """Transcribe audio and dispatch command handlers."""
        stt = self._get_stt()
        if stt is None:
            logger.warning("STT not available - install 'violawake[stt]'")
            self._set_last_command(None)
            self._emit("transcribe_end", text="")
            self._set_state(_STATE_IDLE)
            return

        transcribe_end_emitted = False
        try:
            if len(audio_bytes) % 2 != 0:
                logger.warning(
                    "Audio buffer length %d is not a multiple of 2 bytes (int16); truncating",
                    len(audio_bytes),
                )
                audio_bytes = audio_bytes[: len(audio_bytes) & ~1]

            if len(audio_bytes) == 0:
                logger.warning("Empty audio buffer - skipping transcription")
                self._set_last_command(None)
                self._emit("transcribe_end", text="")
                transcribe_end_emitted = True
                return

            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if self._streaming_stt:
                segment_texts: list[str] = []
                for segment in stt.transcribe_streaming(pcm):
                    if self._stop_event.is_set():
                        logger.debug("Pipeline stopping; aborting streaming transcription")
                        return
                    logger.debug(
                        "Streaming segment [%.1f-%.1f]: '%s'",
                        segment.start,
                        segment.end,
                        segment.text,
                    )
                    segment_texts.append(segment.text)
                text = " ".join(segment_texts).strip()
            else:
                text = stt.transcribe(pcm)

            self._set_last_command(text.strip() or None)
            self._emit("transcribe_end", text=text)
            transcribe_end_emitted = True

            if self._stop_event.is_set():
                logger.debug("Pipeline stopping; dropping transcription result")
                return

            if text.strip():
                logger.info("Command: '%s'", text)
                self._dispatch_command(text)
            else:
                logger.debug("Empty transcription - returning to idle")
        except Exception:
            logger.exception("Transcription failed")
            if not transcribe_end_emitted:
                self._set_last_command(None)
                self._emit("transcribe_end", text="")
        finally:
            self._clear_worker_thread()
            self._set_state(_STATE_IDLE)

    def _dispatch_command(self, text: str) -> None:
        """Call registered command handlers."""
        if self._stop_event.is_set():
            return

        if not self._transition_state(
            (_STATE_IDLE, _STATE_TRANSCRIBING, _STATE_RESPONDING),
            _STATE_RESPONDING,
        ):
            return
        try:
            for handler in self._command_handlers:
                if self._stop_event.is_set():
                    break
                try:
                    response = handler(text)
                    if not response:
                        continue
                    self._emit(
                        "response",
                        command=text,
                        response=response,
                        handler=handler.__name__,
                    )
                    if self._enable_tts and not self._stop_event.is_set():
                        self.speak(response)
                except Exception:
                    logger.exception("Command handler '%s' failed", handler.__name__)
        finally:
            self._set_state(_STATE_IDLE)

    def _start_worker(self, audio_bytes: bytes) -> None:
        """Start the STT/TTS worker thread and retain it for shutdown."""
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.warning("Previous worker thread still alive - skipping new spawn")
                self._set_state(_STATE_IDLE)
                return
            worker = threading.Thread(
                target=self._transcribe_and_respond,
                args=(audio_bytes,),
                daemon=True,
            )
            self._worker_thread = worker
        worker.start()

    def _get_worker_thread(self) -> threading.Thread | None:
        """Return the active worker thread, if any."""
        with self._worker_lock:
            return self._worker_thread

    def _clear_worker_thread(self) -> None:
        """Clear the worker reference once the worker exits."""
        current = threading.current_thread()
        with self._worker_lock:
            if self._worker_thread is current:
                self._worker_thread = None

    def _validate_event(self, event: str) -> None:
        """Validate a pipeline event name."""
        if event not in _SUPPORTED_EVENTS:
            available = ", ".join(sorted(_SUPPORTED_EVENTS))
            raise ValueError(f"Unsupported pipeline event '{event}'. Available: {available}")

    def _emit(self, event: PipelineEventName, **payload: object) -> None:
        """Invoke callbacks registered for a pipeline event."""
        with self._event_lock:
            callbacks = list(self._event_handlers[event])

        event_payload = {"event": event, "pipeline": self, **payload}
        for callback in callbacks:
            try:
                self._invoke_callback(callback, event_payload)
            except Exception:
                logger.exception("Pipeline event callback failed for '%s'", event)

    def _invoke_callback(
        self,
        callback: PipelineEventCallback,
        payload: dict[str, object],
    ) -> None:
        """Call a callback with the subset of payload keys it accepts."""
        try:
            signature = inspect.signature(callback)
        except (TypeError, ValueError):
            callback()
            return

        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            callback(**payload)
            return

        accepted = {
            name: value for name, value in payload.items() if name in signature.parameters
        }
        callback(**accepted)

    def _set_state(self, state: str) -> None:
        """Update the pipeline state under lock."""
        with self._state_lock:
            self._state = state

    def _transition_state(self, expected_state: str | tuple[str, ...], new_state: str) -> bool:
        """Atomically update the pipeline state when it still matches the expected value."""
        expected_states = (
            (expected_state,) if isinstance(expected_state, str) else expected_state
        )
        with self._state_lock:
            # The old code split state reads and writes across different lock scopes,
            # which let stale snapshots overwrite newer RESPONDING/TRANSCRIBING states.
            if self._state not in expected_states:
                return False
            self._state = new_state
            return True

    def _is_playing(self) -> bool:
        """Return whether the pipeline is currently speaking a response."""
        with self._state_lock:
            return self._state == _STATE_RESPONDING

    def _set_last_command(self, command: str | None) -> None:
        """Update the last transcribed command under lock."""
        with self._state_lock:
            self._last_command = command

    def _get_detector_score(self) -> float | None:
        """Return the most recent wake detector score, if available."""
        scores = self._wake_detector.last_scores
        if not scores:
            return None
        return float(scores[-1])

    def _get_stt(self) -> LazySTTEngine | None:
        """Lazy-load the STT engine."""
        if self._stt is None:
            try:
                from violawake_sdk.stt import STTEngine

                self._stt = STTEngine(model=self._stt_model)
                self._stt.prewarm()
            except ImportError:
                return None
        return self._stt

    def _get_tts(self) -> LazyTTSEngine | None:
        """Lazy-load the TTS engine."""
        if self._tts is None and self._enable_tts:
            try:
                from violawake_sdk.tts import TTSEngine

                self._tts = TTSEngine(voice=self._tts_voice)
            except ImportError:
                return None
        return self._tts


class AsyncVoicePipeline:
    """Async wrapper around ``VoicePipeline``."""

    def __init__(self, pipeline: VoicePipeline | None = None, **kwargs: Any) -> None:
        if pipeline is not None and kwargs:
            raise ValueError("Provide either an existing pipeline or constructor kwargs, not both")
        self._pipeline = pipeline if pipeline is not None else VoicePipeline(**kwargs)

    async def __aenter__(self) -> AsyncVoicePipeline:
        """Enter async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context manager and release resources."""
        await self.close()

    @property
    def pipeline(self) -> VoicePipeline:
        """Return the wrapped sync pipeline."""
        return self._pipeline

    @property
    def state(self) -> str:
        """Return the current wrapped pipeline state."""
        return self._pipeline.state

    @property
    def last_command(self) -> str | None:
        """Return the most recent transcription result."""
        return self._pipeline.last_command

    @property
    def last_score(self) -> float | None:
        """Return the most recent wake score."""
        return self._pipeline.last_score

    def on(
        self,
        event: PipelineEventName,
        callback: PipelineEventCallback | None = None,
    ) -> PipelineEventCallback | Callable[[PipelineEventCallback], PipelineEventCallback]:
        """Register an event callback on the wrapped pipeline."""
        return self._pipeline.on(event, callback)

    def on_command(self, handler: CommandHandler) -> CommandHandler:
        """Register a command handler on the wrapped pipeline."""
        return self._pipeline.on_command(handler)

    async def run(self) -> None:
        """Run the wrapped pipeline in a background thread."""
        await asyncio.to_thread(self._pipeline.run)

    async def speak(self, text: str) -> None:
        """Speak text without blocking the event loop."""
        await asyncio.to_thread(self._pipeline.speak, text)

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the wrapped pipeline without blocking the event loop."""
        await asyncio.to_thread(self._pipeline.stop, timeout)

    async def close(self) -> None:
        """Close the wrapped pipeline without blocking the event loop."""
        await asyncio.to_thread(self._pipeline.close)

    async def aclose(self) -> None:
        """Alias for ``close()``."""
        await self.close()
