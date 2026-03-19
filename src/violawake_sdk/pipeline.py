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
from typing import TypeAlias

import numpy as np

from violawake_sdk._exceptions import PipelineError
from violawake_sdk.vad import VADEngine
from violawake_sdk.wake_detector import WakeDetector

logger = logging.getLogger(__name__)

# Type alias for command handlers
CommandHandler: TypeAlias = Callable[[str], str | None]

# Recording state machine states
_STATE_IDLE = "idle"
_STATE_LISTENING = "listening"
_STATE_TRANSCRIBING = "transcribing"
_STATE_RESPONDING = "responding"

# Audio constants
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320   # 20ms
MAX_COMMAND_DURATION_S = 10.0  # Max recording length before force-stop
SILENCE_FRAMES_STOP = 30  # Number of consecutive silence frames to stop recording (~600ms)


class VoicePipeline:
    """Orchestrated Wake→VAD→STT→TTS voice pipeline.

    State machine: idle → listening → transcribing → responding → idle

    Thread model:
        - Main thread: ``run()`` runs the mic capture loop
        - Worker thread: STT transcription runs off the mic thread to avoid blocking
        - Worker thread: TTS playback runs off the worker thread

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
        threshold: float = 0.80,
        vad_backend: str = "auto",
        enable_tts: bool = True,
        device_index: int | None = None,
    ) -> None:
        """Initialize the voice pipeline.

        Args:
            wake_word: Wake word model name. Default "viola".
            stt_model: Whisper model size. Default "base".
            tts_voice: TTS voice name. Default "af_heart".
            threshold: Wake word detection threshold. Default 0.80.
            vad_backend: VAD backend. Default "auto".
            enable_tts: If True, speak responses via TTS. If False, skip TTS.
            device_index: Microphone device index. None = system default.
        """
        self._wake_detector = WakeDetector(model=wake_word, threshold=threshold)
        self._vad = VADEngine(backend=vad_backend)
        self._enable_tts = enable_tts
        self._device_index = device_index
        self._stt_model = stt_model
        self._tts_voice = tts_voice

        self._stt: object | None = None  # lazy
        self._tts: object | None = None  # lazy

        self._state = _STATE_IDLE
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._command_handlers: list[CommandHandler] = []

        logger.info(
            "VoicePipeline initialized: wake=%s, stt=%s, tts=%s",
            wake_word, stt_model, tts_voice,
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
            self._state = _STATE_IDLE
            logger.info("VoicePipeline stopped.")

    def stop(self) -> None:
        """Signal the pipeline to stop after the current cycle."""
        self._stop_event.set()

    def speak(self, text: str) -> None:
        """Synthesize and play text via TTS (called from within command handlers)."""
        if not self._enable_tts:
            return

        tts = self._get_tts()
        if tts is None:
            logger.warning("TTS not available — install 'violawake[tts]'")
            return

        try:
            audio = tts.synthesize(text)  # type: ignore[attr-defined]
            tts.play(audio)  # type: ignore[attr-defined]
        except Exception:
            logger.exception("TTS playback failed")

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
                is_playing = state == _STATE_RESPONDING
                if self._wake_detector.detect(frame, is_playing=is_playing):
                    logger.info("Wake word detected → listening for command")
                    recording_buffer.clear()
                    silence_count = 0
                    with self._state_lock:
                        self._state = _STATE_LISTENING

            elif state == _STATE_LISTENING:
                # Recording command
                recording_buffer.append(frame)
                is_speech = self._vad.is_speech(frame, threshold=0.4)

                if not is_speech:
                    silence_count += 1
                else:
                    silence_count = 0

                # Check stop conditions
                total_frames = len(recording_buffer)
                max_frames = int(MAX_COMMAND_DURATION_S * 50)  # 50 frames/sec

                if silence_count >= SILENCE_FRAMES_STOP or total_frames >= max_frames:
                    audio_bytes = b"".join(recording_buffer)
                    with self._state_lock:
                        self._state = _STATE_TRANSCRIBING

                    # Transcribe in a worker thread to not block the mic loop
                    threading.Thread(
                        target=self._transcribe_and_respond,
                        args=(audio_bytes,),
                        daemon=True,
                    ).start()

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
            pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            text = stt.transcribe(pcm)  # type: ignore[attr-defined]

            if text.strip():
                logger.info("Command: '%s'", text)
                self._dispatch_command(text)
            else:
                logger.debug("Empty transcription — returning to idle")

        except Exception:
            logger.exception("Transcription failed")
        finally:
            with self._state_lock:
                self._state = _STATE_IDLE

    def _dispatch_command(self, text: str) -> None:
        """Call all registered command handlers."""
        with self._state_lock:
            self._state = _STATE_RESPONDING

        try:
            for handler in self._command_handlers:
                try:
                    response = handler(text)
                    if response and self._enable_tts:
                        self.speak(response)
                except Exception:
                    logger.exception("Command handler '%s' failed", handler.__name__)
        finally:
            with self._state_lock:
                self._state = _STATE_IDLE

    def _get_stt(self) -> object | None:
        """Lazy-load STT engine."""
        if self._stt is None:
            try:
                from violawake_sdk.stt import STTEngine
                self._stt = STTEngine(model=self._stt_model)
                self._stt.prewarm()  # type: ignore[attr-defined]
            except ImportError:
                return None
        return self._stt

    def _get_tts(self) -> object | None:
        """Lazy-load TTS engine."""
        if self._tts is None and self._enable_tts:
            try:
                from violawake_sdk.tts import TTSEngine
                self._tts = TTSEngine(voice=self._tts_voice)
            except ImportError:
                return None
        return self._tts
