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
    "af_heart",   # American Female (default)
    "af_bella",
    "af_sarah",
    "am_adam",    # American Male
    "am_michael",
    "bf_emma",    # British Female
    "bf_isabella",
    "bm_george",  # British Male
    "bm_lewis",
]

DEFAULT_VOICE = "af_heart"


class TTSEngine:
    """On-device TTS using Kokoro-82M (Apache 2.0 model).

    Thread-safe: multiple threads can call ``synthesize()`` concurrently.
    Each call creates a temporary synthesis context — no shared mutable state.

    Model files required (auto-downloaded on first use):
        - ``kokoro_v1_0.onnx`` — Kokoro-82M model (~330MB)
        - ``kokoro_voices_v1_0.bin`` — Voice embeddings (~8MB)

    Example::

        tts = TTSEngine(voice="af_heart")
        audio = tts.synthesize("Hello, world!")  # returns np.ndarray
        tts.play(audio)  # plays via sounddevice or pyaudio
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
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {', '.join(AVAILABLE_VOICES)}"
            )

        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
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
            import kokoro_onnx  # type: ignore[import]
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

    def play(self, audio: np.ndarray) -> None:
        """Play audio through the default output device.

        Args:
            audio: Float32 numpy array of audio samples.
        """
        try:
            import sounddevice as sd  # type: ignore[import]
            sd.play(audio, samplerate=self.sample_rate, blocking=True)
        except ImportError:
            # Fall back to pyaudio
            self._play_pyaudio(audio)

    def _play_pyaudio(self, audio: np.ndarray) -> None:
        """Play audio using pyaudio as fallback."""
        import pyaudio
        pcm = (audio * 32767).astype(np.int16)
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

        from scipy.signal import resample_poly
        gcd = math.gcd(src_rate, dst_rate)
        return resample_poly(audio, dst_rate // gcd, src_rate // gcd).astype(np.float32)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text at sentence boundaries for chunked synthesis."""
        sentences: list[str] = []
        current: list[str] = []

        for char in text:
            current.append(char)
            if char in SENTENCE_BOUNDARIES:
                sentences.append("".join(current).strip())
                current = []

        if current:
            remaining = "".join(current).strip()
            if remaining:
                sentences.append(remaining)

        return [s for s in sentences if s]
