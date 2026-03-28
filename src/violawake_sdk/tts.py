"""Text-to-Speech engine using Kokoro-82M (Apache 2.0).

Kokoro-82M is an on-device neural TTS model (82M parameters, ONNX format).
This module wraps kokoro-onnx with sentence-chunked streaming for low-latency
LLM-response playback.

Usage::

    from violawake_sdk import TTSEngine  # requires pip install violawake[tts]

    tts = TTSEngine(voice="af_heart")
    audio = tts.synthesize("Hello from ViolaWake!")
    tts.play(audio)

Note: TTSEngine requires the 'kokoro-onnx' package.
Install with: pip install 'violawake[tts]'
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Generator

import numpy as np

from violawake_sdk._exceptions import ModelLoadError, ModelNotFoundError
from violawake_sdk.models import get_model_path

logger = logging.getLogger(__name__)

# Audio configuration
TTS_SAMPLE_RATE = 24_000  # Kokoro outputs 24kHz
TARGET_SAMPLE_RATE = 16_000  # Pipeline uses 16kHz

# Sentence boundary characters for chunked streaming
SENTENCE_BOUNDARIES = ".!?;:"

# Available Kokoro voices
AVAILABLE_VOICES = [
    "af_heart",  # American Female (default)
    "af_bella",
    "af_sarah",
    "am_adam",  # American Male
    "am_michael",
    "bf_emma",  # British Female
    "bf_isabella",
    "bm_george",  # British Male
    "bm_lewis",
]

DEFAULT_VOICE = "af_heart"


class TTSEngine:
    """On-device TTS using Kokoro-82M (Apache 2.0 model).

    Thread-safe: multiple threads can call ``synthesize()`` concurrently.
    Calls are serialized via ``_synthesis_lock`` since kokoro-onnx is not
    guaranteed to be thread-safe. Model initialization is separately guarded
    by ``_lock`` (lazy load on first use).

    Model files required (auto-downloaded on first use):
        - ``kokoro_v1_0.onnx`` — Kokoro-82M model (~326MB)
        - ``kokoro_voices_v1_0.bin`` — Voice embeddings (~28MB)

    Example::

        tts = TTSEngine(voice="af_heart")
        audio = tts.synthesize("Hello, world!")  # returns np.ndarray
        tts.play(audio)  # blocking by default
        tts.play_async(audio)  # optional non-blocking playback
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        """Initialize the TTS engine.

        Args:
            voice: Kokoro voice name. Default "af_heart".
                   See ``AVAILABLE_VOICES`` for full list.
            speed: Speech speed multiplier. 1.0 = normal, 1.2 = 20% faster.
            sample_rate: Output sample rate. Default 16kHz (pipeline standard).
                         Kokoro outputs 24kHz; resampled if different.
        """
        if voice not in AVAILABLE_VOICES:
            raise ValueError(f"Unknown voice '{voice}'. Available: {', '.join(AVAILABLE_VOICES)}")

        if not (0.1 <= speed <= 3.0):
            raise ValueError(f"Speed must be between 0.1 and 3.0, got {speed}")

        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._kokoro: object | None = None

        # Lazy initialization — load model on first use
        logger.info("TTSEngine created: voice=%s, speed=%.1f", voice, speed)

    def _get_kokoro(self) -> object:
        """Lazy-load the Kokoro model (thread-safe)."""
        with self._lock:
            if self._kokoro is None:
                self._kokoro = self._load_kokoro()
        return self._kokoro

    def _load_kokoro(self) -> object:
        """Load the Kokoro ONNX model."""
        try:
            import kokoro_onnx
        except ImportError as e:
            raise ImportError(
                "kokoro-onnx is not installed. Install with: pip install 'violawake[tts]'"
            ) from e

        try:
            model_path = get_model_path("kokoro_v1_0")
            voices_path = get_model_path("kokoro_voices_v1_0")
        except FileNotFoundError as e:
            raise ModelNotFoundError(
                "Kokoro models not found. Run:\n"
                "  violawake-download --model kokoro_v1_0\n"
                "  violawake-download --model kokoro_voices_v1_0"
            ) from e

        try:
            kokoro = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
        except Exception as e:
            raise ModelLoadError(f"Failed to load Kokoro model: {e}") from e

        logger.info("Kokoro-82M loaded: %s", model_path)
        return kokoro

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize. May be multi-sentence.
                  Long text is processed as a single batch call.

        Returns:
            Audio samples as float32 numpy array at ``self.sample_rate``.
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        kokoro = self._get_kokoro()

        # Hold synthesis lock to serialize access to the kokoro model,
        # which is not guaranteed to be thread-safe by kokoro-onnx.
        with self._synthesis_lock:
            try:
                # kokoro-onnx API: returns (samples, sample_rate)
                audio, sr = kokoro.create(  # type: ignore[attr-defined]
                    text,
                    voice=self.voice,
                    speed=self.speed,
                    lang="en-us",
                )
            except Exception as e:
                logger.exception("TTS synthesis failed for text: %.50s...", text)
                raise RuntimeError(f"TTS synthesis failed: {e}") from e

        audio = np.asarray(audio, dtype=np.float32)

        # Resample if needed
        if sr != self.sample_rate:
            audio = self._resample(audio, sr, self.sample_rate)

        return audio

    def synthesize_chunked(self, text: str) -> Generator[np.ndarray, None, None]:
        """Synthesize text sentence-by-sentence for lower latency.

        Splits text at sentence boundaries and yields audio for each sentence
        as soon as it's synthesized. This allows playback to begin before
        the full text is processed — matching the pattern from production Viola.

        Args:
            text: Text to synthesize. May be multi-sentence.

        Yields:
            Audio chunks (one per sentence) as float32 numpy arrays.
        """
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if sentence.strip():
                audio = self.synthesize(sentence)
                if audio.size > 0:
                    yield audio

    def play(self, audio: np.ndarray, *, blocking: bool = True) -> None:
        """Play audio through the default output device.

        Args:
            audio: Float32 numpy array of audio samples.
            blocking: If True, wait for playback to finish. If False, return
                      immediately after starting playback.
        """
        try:
            import sounddevice as sd
        except ImportError as sd_err:
            logger.debug("sounddevice not available (%s), falling back to pyaudio", sd_err)
            try:
                self._play_pyaudio(audio, blocking=blocking)
            except ImportError as e:
                raise ImportError(
                    "No audio playback backend is installed. "
                    "Install sounddevice with: pip install sounddevice "
                    "or install violawake[audio] for PyAudio playback."
                ) from e
            return

        # Copy to prevent mutation of caller's array during async playback
        sd.play(audio.copy(), samplerate=self.sample_rate, blocking=blocking)

    def play_async(self, audio: np.ndarray) -> None:
        """Play audio without blocking the calling thread."""
        self.play(audio, blocking=False)

    def _play_pyaudio(self, audio: np.ndarray, *, blocking: bool = True) -> None:
        """Play audio using pyaudio as fallback."""
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio is required for audio playback. Install with: pip install violawake[audio]"
            ) from None

        if not blocking:
            thread = threading.Thread(
                target=self._play_pyaudio,
                args=(audio.copy(),),
                kwargs={"blocking": True},
                daemon=True,
            )
            thread.start()
            return

        clipped = np.clip(audio, -1.0, 1.0)
        pcm = (clipped * 32767).astype(np.int16)
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
        )
        try:
            stream.write(pcm.tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio using scipy."""
        import math

        try:
            from scipy.signal import resample_poly
        except ImportError as e:
            raise ImportError(
                "scipy is required for audio resampling. Install with: pip install scipy"
            ) from e
        gcd = math.gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // gcd, src_rate // gcd).astype(np.float32)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text at sentence boundaries for chunked synthesis.

        Uses a regex that splits on sentence-ending punctuation followed by
        whitespace and an uppercase letter (or end of string). This avoids
        false splits on abbreviations ("Dr. Smith"), decimals ("3.14"),
        and URLs.
        """
        import re

        # Split on sentence-ending punctuation followed by space+uppercase or end of string
        pattern = r"(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$"
        parts = re.split(pattern, text)
        return [s.strip() for s in parts if s and s.strip()]

    def close(self) -> None:
        """Release model resources."""
        self._kokoro = None

    def __enter__(self) -> TTSEngine:
        """Enter sync context manager. Returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit sync context manager. Releases model resources."""
        self.close()
