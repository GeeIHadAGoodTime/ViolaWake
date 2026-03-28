"""Speech-to-Text using faster-whisper (CTranslate2-optimized Whisper).

Batch-mode transcription: records audio buffer, then transcribes.
Not streaming/real-time (see ADR-003 rationale — streaming is Phase 2).

Usage::

    from violawake_sdk import STTEngine  # requires pip install violawake[stt]

    stt = STTEngine(model="base")
    text = stt.transcribe(audio_float32_16khz)

Note: STTEngine requires the 'faster-whisper' package.
Install with: pip install 'violawake[stt]'
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from faster_whisper import WhisperModel  # type: ignore[import]

logger = logging.getLogger(__name__)

# faster-whisper model sizes and their trade-offs
# WER measured on LibriSpeech test-clean (lower is better)
MODEL_PROFILES = {
    "tiny": {"vram_mb": 75, "wer": 14.0, "latency_ms": 120},
    "base": {"vram_mb": 145, "wer": 9.0, "latency_ms": 380},
    "small": {"vram_mb": 465, "wer": 7.0, "latency_ms": 850},
    "medium": {"vram_mb": 1500, "wer": 5.0, "latency_ms": 2100},
    "large-v3": {"vram_mb": 3000, "wer": 3.0, "latency_ms": 5000},
}

DEFAULT_MODEL = "base"
NO_SPEECH_THRESHOLD = 0.6  # Skip transcription if no_speech_prob > this value


@dataclass
class TranscriptSegment:
    """A single transcription segment with timing."""

    text: str
    start: float
    end: float
    no_speech_prob: float


@dataclass
class TranscriptResult:
    """Full transcription result."""

    text: str
    segments: list[TranscriptSegment]
    language: str
    language_prob: float
    duration_s: float
    no_speech_prob: float


class STTEngine:
    """Speech-to-text transcription via faster-whisper.

    Thread-safe: ``WhisperModel`` is thread-safe for concurrent ``transcribe()`` calls.

    Model is loaded once and reused. First call includes model load time
    (~1-3s). Subsequent calls are ~380ms (base model, CPU, 3s audio).

    Example::

        stt = STTEngine(model="base")
        text = stt.transcribe(audio_np_float32)
        print(text)  # "what's the weather today"
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        language_cache_ttl_s: float = 60.0,
    ) -> None:
        """Initialize the STT engine.

        Args:
            model: Whisper model size. One of: tiny, base, small, medium, large-v3.
                   Default "base" — good accuracy/speed balance (WER ~9%).
            device: "cpu" or "cuda". Default "cpu".
            compute_type: CTranslate2 compute type. "int8" (default), "float16", "float32".
                          "int8" is fastest on CPU with minimal accuracy loss.
            language: Force a specific language (e.g., "en"). None = auto-detect.
            language_cache_ttl_s: Cache detected language for N seconds to avoid
                                   per-call language detection overhead.
        """
        if model not in MODEL_PROFILES:
            available = ", ".join(MODEL_PROFILES.keys())
            raise ValueError(f"Unknown model '{model}'. Available: {available}")

        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.forced_language = language
        self._language_cache: tuple[str, float] | None = None  # (lang, cached_at)
        self._language_cache_ttl = language_cache_ttl_s
        self._model: WhisperModel | None = None
        self._model_lock = threading.Lock()

        profile = MODEL_PROFILES[model]
        logger.info(
            "STTEngine created: model=%s, device=%s (WER~%.0f%%, %dMB)",
            model,
            device,
            profile["wer"],
            profile["vram_mb"],
        )

    def _get_model(self) -> WhisperModel:
        """Lazy-load the Whisper model on first use (thread-safe)."""
        if self._model is not None:
            return self._model

        with self._model_lock:
            # Double-checked locking: another thread may have loaded
            # the model while we waited for the lock.
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel  # type: ignore[import]
            except ImportError as e:
                raise ImportError(
                    "faster-whisper is not installed. Install with: pip install 'violawake[stt]'"
                ) from e

            logger.info("Loading Whisper model '%s'...", self.model_name)
            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info("Whisper model loaded in %.0f ms", elapsed_ms)

        return self._model

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio to text.

        Note:
            This engine uses a progressive temperature fallback of
            ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`` during decoding, which can
            trigger up to 6 decoding passes and increase latency. For
            low-latency use cases, prefer a single-pass configuration such as
            ``temperature_fallback=[0.0]``.

        Args:
            audio: Float32 numpy array at 16kHz mono. Values should be in [-1.0, 1.0].

        Returns:
            Transcribed text as string. Empty string if no speech detected.
        """
        result = self.transcribe_full(audio)
        return result.text

    def transcribe_full(
        self,
        audio: np.ndarray,
        channels_first: bool | None = None,
    ) -> TranscriptResult:
        """Transcribe audio and return full result with segments, timing, and metadata.

        Args:
            audio: Float32 numpy array at 16kHz mono, or 2-D stereo.
            channels_first: Layout hint for 2-D stereo audio.
                ``True``  = (channels, samples)  e.g. shape (2, 48000).
                ``False`` = (samples, channels)  e.g. shape (48000, 2) — the
                standard layout.
                ``None`` (default) = fall back to a shape heuristic (smaller
                dimension is assumed to be channels).  Prefer passing an
                explicit value to avoid ambiguity with short audio clips.

        Returns:
            TranscriptResult with text, segments, language, and no_speech_prob.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            if channels_first is True:
                # Explicit: (channels, samples) — e.g. shape (2, 48000)
                audio = audio.mean(axis=0)
            elif channels_first is False:
                # Explicit: (samples, channels) — e.g. shape (48000, 2)
                audio = audio.mean(axis=1)
            else:
                # Legacy heuristic: channels axis is the smaller dimension.
                if audio.shape[0] < audio.shape[1]:
                    audio = audio.mean(axis=0)
                else:
                    audio = audio.mean(axis=1)

        # Determine language (use cache if available)
        language = self._get_language()

        model = self._get_model()
        t0 = time.perf_counter()

        segments_gen, info = model.transcribe(
            audio,
            language=language,
            vad_filter=True,  # Use Silero VAD for silence removal
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            beam_size=5,
            best_of=5,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Progressive fallback
        )

        # Consume the generator (transcription happens here)
        segments = list(segments_gen)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Update language cache (protected by _model_lock for thread safety)
        if language is None and info.language_probability > 0.5:
            with self._model_lock:
                self._language_cache = (info.language, time.monotonic())

        transcript_segments = [
            TranscriptSegment(
                text=s.text.strip(),
                start=s.start,
                end=s.end,
                no_speech_prob=s.no_speech_prob,
            )
            for s in segments
        ]

        full_text = " ".join(s.text for s in transcript_segments).strip()
        overall_no_speech = max((s.no_speech_prob for s in transcript_segments), default=0.0)

        if overall_no_speech > NO_SPEECH_THRESHOLD:
            logger.debug(
                "No speech detected (no_speech_prob=%.2f) — returning empty",
                overall_no_speech,
            )
            full_text = ""

        logger.debug(
            "Transcribed in %.0f ms: '%s'",
            elapsed_ms,
            full_text[:60] + "..." if len(full_text) > 60 else full_text,
        )

        return TranscriptResult(
            text=full_text,
            segments=transcript_segments,
            language=info.language,
            language_prob=info.language_probability,
            duration_s=info.duration,
            no_speech_prob=overall_no_speech,
        )

    def _get_language(self) -> str | None:
        """Return cached language or None for auto-detection.

        Thread-safe: reads ``_language_cache`` under ``_model_lock``.
        """
        if self.forced_language:
            return self.forced_language

        with self._model_lock:
            if self._language_cache is not None:
                lang, cached_at = self._language_cache
                if time.monotonic() - cached_at < self._language_cache_ttl:
                    return lang

        return None  # auto-detect

    def prewarm(self) -> None:
        """Load the model eagerly (avoids cold-start latency on first transcription)."""
        self._get_model()
        logger.info("STTEngine prewarmed: model '%s' loaded", self.model_name)

    def close(self) -> None:
        """Release model resources."""
        with self._model_lock:
            self._model = None

    def __enter__(self) -> STTEngine:
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
