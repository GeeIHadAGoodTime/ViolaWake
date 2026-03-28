"""VoicePipeline — orchestrated Wake→VAD→STT→TTS pipeline.

Bundles all four engines into a single, easy-to-use pipeline with callback
registration, threaded execution, and clean shutdown.

Usage::

    pipeline = VoicePipeline(
        wake_word="viola",
        stt_model="base",
        tts_voice="af_heart",
    )

    @pipeline.on_command
    def handle(text: str) -> str | None:
        return f"You said: {text}"

    pipeline.run()  # blocks until Ctrl+C
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, TypeAlias

import numpy as np

from violawake_sdk._constants import DEFAULT_THRESHOLD
from violawake_sdk._exceptions import PipelineError
from violawake_sdk.vad import VADEngine
from violawake_sdk.wake_detector import WakeDetector

logger = logging.getLogger(__name__)

# Type alias for command handlers
CommandHandler: TypeAlias = Callable[[str], str | None]
WakeCallback: TypeAlias = Callable[[], None]

# Recording state machine states
_STATE_IDLE = "idle"
_STATE_LISTENING = "listening"
_STATE_TRANSCRIBING = "transcribing"
_STATE_RESPONDING = "responding"

# Audio constants
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20ms
MAX_COMMAND_DURATION_S = 10.0  # Max recording length before force-stop
SILENCE_FRAMES_STOP = 30  # Number of consecutive silence frames to stop recording (~600ms)
MAX_RECORDING_FRAMES = int(MAX_COMMAND_DURATION_S / (FRAME_SAMPLES / SAMPLE_RATE))  # Pre-computed


class VoicePipeline:
    """Orchestrated Wake→VAD→STT→TTS voice pipeline.

    State Machine
    =============
    Four states with the following transitions::

        IDLE ──(wake detected)──→ LISTENING
        LISTENING ──(silence/timeout)──→ TRANSCRIBING
        TRANSCRIBING ──(text ready)──→ RESPONDING
        RESPONDING ──(handlers done)──→ IDLE

    Thread ownership of transitions:
        - **Main thread** (``_run_loop``): IDLE→LISTENING, LISTENING→TRANSCRIBING.
          Owns the mic capture loop and wake/VAD detection.
        - **Worker thread** (``_transcribe_and_respond``): TRANSCRIBING→RESPONDING→IDLE.
          Spawned by ``_start_worker()`` as a daemon thread. Runs STT, dispatches
          command handlers, and triggers TTS playback. Only one worker thread is
          active at a time (guarded by ``_worker_lock``).

    Threading Model
    ===============
    - The main loop runs in the calling thread via ``run()`` (blocking).
    - When a command recording ends, ``_start_worker()`` spawns a single daemon
      thread for STT transcription + handler dispatch + TTS playback.
    - ``_state_lock`` guards all reads/writes of ``_state``.
    - ``_worker_lock`` guards ``_worker_thread`` reference management.

    Known Limitation
    ================
    There is a brief window between the worker thread completing (setting state
    back to IDLE) and the next wake detection where the pipeline is technically
    idle but the worker thread reference may not yet be cleared. During this
    window, a new wake detection will proceed normally since the state is IDLE.

    Usage::

        pipeline = VoicePipeline()

        @pipeline.on_command
        def handle(text: str) -> str | None:
            return f"You said: {text}"

        pipeline.run()
    """

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
    ) -> None:
        """Initialize the voice pipeline.

        Args:
            wake_word: Wake word model name. Default "viola".
            stt_model: Whisper model size. Default "base".
            tts_voice: TTS voice name. Default "af_heart".
            threshold: Wake word detection threshold. Default 0.80.
            vad_backend: VAD backend. Default "auto".
            vad_threshold: VAD speech detection threshold. Default 0.4.
            enable_tts: If True, speak responses via TTS. If False, skip TTS.
            device_index: Microphone device index. None = system default.
            on_wake: Optional callback fired after wake-word detection and
                     before command transcription begins.
        """
        self._wake_detector = WakeDetector(model=wake_word, threshold=threshold)
        self._vad = VADEngine(backend=vad_backend)
        self._vad_threshold = vad_threshold
        self._enable_tts = enable_tts
        self._device_index = device_index
        self._stt_model = stt_model
        self._tts_voice = tts_voice
        self._on_wake = on_wake

        # Typed as Any because STTEngine/TTSEngine are lazily imported to avoid
        # hard dependencies on optional extras (violawake[stt], violawake[tts]).
        self._stt: Any | None = None  # lazy — violawake_sdk.stt.STTEngine
        self._tts: Any | None = None  # lazy — violawake_sdk.tts.TTSEngine

        self._state = _STATE_IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

        self._command_handlers: list[CommandHandler] = []

        logger.info(
            "VoicePipeline initialized: wake=%s, stt=%s, tts=%s",
            wake_word,
            stt_model,
            tts_voice,
        )

    def on_command(self, handler: CommandHandler) -> CommandHandler:
        """Decorator to register a command handler.

        The handler receives the transcribed text and may return a string
        response (which is spoken via TTS) or None (no TTS response).

        Example::

            @pipeline.on_command
            def handle(text: str) -> str | None:
                if "weather" in text:
                    return "It's sunny!"
                return None
        """
        self._command_handlers.append(handler)
        return handler

    def run(self) -> None:
        """Run the pipeline. Blocks until ``stop()`` is called or Ctrl+C.

        Raises:
            PipelineError: If the pipeline encounters an unrecoverable error.
        """
        logger.info("VoicePipeline started. Say the wake word to begin.")
        self._stop_event.clear()

        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted by user.")
        except Exception as e:
            raise PipelineError(f"Pipeline error: {e}") from e
        finally:
            self.stop()
            with self._state_lock:
                self._state = _STATE_IDLE
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
            # Clear reference only after the worker has fully exited
            with self._worker_lock:
                if self._worker_thread is worker:
                    self._worker_thread = None

    def close(self) -> None:
        """Stop the pipeline and release all engine resources."""
        self.stop()
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
        """Exit sync context manager. Stops pipeline and releases resources."""
        self.close()

    def speak(self, text: str) -> None:
        """Synthesize and play text via TTS (called from within command handlers)."""
        if not self._enable_tts or self._stop_event.is_set():
            return

        tts = self._get_tts()
        if tts is None:
            logger.warning("TTS not available — install 'violawake[tts]'")
            return

        try:
            audio = tts.synthesize(text)  # type: ignore[attr-defined]
            tts.play(audio)  # type: ignore[attr-defined]
        except Exception as e:
            logger.exception("TTS playback failed for text '%.50s': %s", text, e)

    def _run_loop(self) -> None:
        """Main mic capture and detection loop."""
        recording_buffer: list[bytes] = []
        silence_count = 0

        for frame in self._wake_detector.stream_mic(device_index=self._device_index):
            if self._stop_event.is_set():
                break

            with self._state_lock:
                state = self._state

            if state == _STATE_IDLE:
                # Listening for wake word
                if self._wake_detector.detect(frame, is_playing=False):
                    logger.info("Wake word detected → listening for command")
                    recording_buffer.clear()
                    silence_count = 0
                    if self._on_wake is not None:
                        try:
                            self._on_wake()
                        except Exception:
                            logger.exception("on_wake callback failed")
                    with self._state_lock:
                        self._state = _STATE_LISTENING

            elif state == _STATE_LISTENING:
                # Recording command
                recording_buffer.append(frame)
                is_speech = self._vad.is_speech(frame, threshold=self._vad_threshold)

                if not is_speech:
                    silence_count += 1
                else:
                    silence_count = 0

                # Check stop conditions
                total_frames = len(recording_buffer)

                if silence_count >= SILENCE_FRAMES_STOP or total_frames >= MAX_RECORDING_FRAMES:
                    audio_bytes = b"".join(recording_buffer)
                    with self._state_lock:
                        self._state = _STATE_TRANSCRIBING

                    # Transcribe in a worker thread to not block the mic loop
                    self._start_worker(audio_bytes)

            # _STATE_TRANSCRIBING and _STATE_RESPONDING: keep looping (don't detect wake)

    def _transcribe_and_respond(self, audio_bytes: bytes) -> None:
        """Transcribe audio and dispatch to command handlers."""
        stt = self._get_stt()
        if stt is None:
            logger.warning("STT not available — install 'violawake[stt]'")
            with self._state_lock:
                self._state = _STATE_IDLE
            return

        try:
            if len(audio_bytes) % 2 != 0:
                logger.warning(
                    "Audio buffer length %d is not a multiple of 2 bytes (int16); "
                    "truncating to even boundary",
                    len(audio_bytes),
                )
                audio_bytes = audio_bytes[: len(audio_bytes) & ~1]
            if len(audio_bytes) == 0:
                logger.warning("Empty audio buffer — skipping transcription")
                return
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            text = stt.transcribe(pcm)  # type: ignore[attr-defined]

            if self._stop_event.is_set():
                logger.debug("Pipeline stopping; dropping transcription result")
                return

            if text.strip():
                logger.info("Command: '%s'", text)
                self._dispatch_command(text)
            else:
                logger.debug("Empty transcription — returning to idle")

        except Exception:
            logger.exception("Transcription failed")
        finally:
            self._clear_worker_thread()
            with self._state_lock:
                self._state = _STATE_IDLE

    def _dispatch_command(self, text: str) -> None:
        """Call all registered command handlers."""
        if self._stop_event.is_set():
            return

        with self._state_lock:
            self._state = _STATE_RESPONDING

        try:
            for handler in self._command_handlers:
                if self._stop_event.is_set():
                    break
                try:
                    response = handler(text)
                    if response and self._enable_tts and not self._stop_event.is_set():
                        self.speak(response)
                except Exception:
                    logger.exception("Command handler '%s' failed", handler.__name__)
        finally:
            with self._state_lock:
                self._state = _STATE_IDLE

    def _start_worker(self, audio_bytes: bytes) -> None:
        """Start the STT/TTS worker thread and retain it for shutdown.

        If a previous worker is still alive, skip spawning to prevent
        concurrent transcription.
        """
        with self._worker_lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                logger.warning("Previous worker thread still alive — skipping new spawn")
                with self._state_lock:
                    self._state = _STATE_IDLE
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

    def _get_stt(self) -> Any | None:
        """Lazy-load STT engine."""
        if self._stt is None:
            try:
                from violawake_sdk.stt import STTEngine

                self._stt = STTEngine(model=self._stt_model)
                self._stt.prewarm()  # type: ignore[attr-defined]
            except ImportError:
                return None
        return self._stt

    def _get_tts(self) -> Any | None:
        """Lazy-load TTS engine."""
        if self._tts is None and self._enable_tts:
            try:
                from violawake_sdk.tts import TTSEngine

                self._tts = TTSEngine(voice=self._tts_voice)
            except ImportError:
                return None
        return self._tts
