"""Speech-to-Text using faster-whisper (CTranslate2-optimized Whisper).

Batch-mode transcription: records audio buffer, then transcribes.
Streaming transcription: yields TranscriptSegments one at a time via
``STTEngine.transcribe_streaming()``, or incrementally via
``StreamingSTTEngine`` which buffers incoming audio chunks.

Usage::

    from violawake_sdk import STTEngine  # requires pip install violawake[stt]

    stt = STTEngine(model="base")
    text = stt.transcribe(audio_float32_16khz)

    # Streaming (generator mode)
    for segment in stt.transcribe_streaming(audio_float32_16khz):
        print(segment.text, segment.start, segment.end)

    # Incremental streaming via StreamingSTTEngine
    from violawake_sdk.stt import StreamingSTTEngine
    streaming = StreamingSTTEngine(model="base", min_buffer_seconds=2.0)
    streaming.push_chunk(chunk1)
    streaming.push_chunk(chunk2)
    for segment in streaming.flush():
        print(segment.text)

Note: STTEngine and StreamingSTTEngine require the 'faster-whisper' package.
Install with: pip install 'violawake[stt]'
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
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

    def transcribe_streaming(
        self,
        audio: np.ndarray,
        channels_first: bool | None = None,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: list[float] | None = None,
    ) -> Iterator[TranscriptSegment]:
        """Stream transcription segments as they become available.

        Uses faster-whisper's generator mode: ``model.transcribe()`` returns a
        ``(segments_iterator, info)`` tuple.  This method yields each
        ``TranscriptSegment`` one at a time as faster-whisper decodes it,
        instead of collecting all segments first.

        This is useful when:
        - You want to display partial results before full transcription completes.
        - You need to pipe segments to a downstream consumer (TTS, logging, etc.)
          without waiting for the full buffer to finish.

        Note:
            Segments with ``no_speech_prob`` above ``NO_SPEECH_THRESHOLD`` are
            silently skipped (not yielded).

        Args:
            audio: Float32 numpy array at 16kHz mono, or 2-D stereo.
            channels_first: Layout hint for 2-D stereo audio (same semantics as
                ``transcribe_full``).
            beam_size: Beam search width. Default 5.
            best_of: Number of candidates when sampling. Default 5.
            temperature: Temperature schedule. Default ``[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]``.

        Yields:
            TranscriptSegment — one per decoded segment, in time order.

        Example::

            stt = STTEngine(model="base")
            for seg in stt.transcribe_streaming(audio_np):
                print(f"[{seg.start:.1f}s] {seg.text}")
        """
        if temperature is None:
            temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            if channels_first is True:
                audio = audio.mean(axis=0)
            elif channels_first is False:
                audio = audio.mean(axis=1)
            else:
                if audio.shape[0] < audio.shape[1]:
                    audio = audio.mean(axis=0)
                else:
                    audio = audio.mean(axis=1)

        language = self._get_language()
        model = self._get_model()

        logger.debug("transcribe_streaming: starting generator on %d samples", len(audio))

        segments_gen, info = model.transcribe(
            audio,
            language=language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
        )

        # Update language cache after model.transcribe() returns info — same
        # logic as transcribe_full, but we must do it before consuming the
        # generator so the cache is primed for subsequent calls.
        if language is None and info.language_probability > 0.5:
            with self._model_lock:
                self._language_cache = (info.language, time.monotonic())

        for seg in segments_gen:
            if seg.no_speech_prob > NO_SPEECH_THRESHOLD:
                logger.debug(
                    "Skipping silent segment [%.1f-%.1f] no_speech_prob=%.2f",
                    seg.start,
                    seg.end,
                    seg.no_speech_prob,
                )
                continue

            text = seg.text.strip()
            logger.debug("Streaming segment [%.1f-%.1f]: '%s'", seg.start, seg.end, text)
            yield TranscriptSegment(
                text=text,
                start=seg.start,
                end=seg.end,
                no_speech_prob=seg.no_speech_prob,
            )

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


# ---------------------------------------------------------------------------
# Incremental / chunk-based streaming
# ---------------------------------------------------------------------------

# Default sliding-window stride when the buffer is flushed automatically on
# each push.  A stride of 0.0 means "keep all samples" (no overlap removal).
_DEFAULT_STRIDE_S = 0.0


@dataclass
class StreamingSTTEngine:
    """Incremental streaming STT: accepts audio chunks, yields segments.

    Audio chunks are pushed one at a time via :meth:`push_chunk`.  When the
    accumulated buffer reaches ``min_buffer_seconds``, :meth:`push_chunk`
    transparently transcribes the buffer and yields any new segments.  You can
    also force a transcription at any time with :meth:`flush`.

    A sliding-window approach is supported via ``stride_seconds``: after each
    transcription pass the engine retains the last ``stride_seconds`` of audio
    so that words near the boundary are not lost on the next pass.  Set
    ``stride_seconds=0.0`` (default) to discard all audio after each pass.

    Thread safety: **not** thread-safe.  Call from a single thread or protect
    externally with a lock.

    Args:
        model: Whisper model size. One of ``tiny``, ``base``, ``small``,
               ``medium``, ``large-v3``. Default ``"base"``.
        device: ``"cpu"`` or ``"cuda"``. Default ``"cpu"``.
        compute_type: CTranslate2 compute type. Default ``"int8"``.
        language: Force a specific language code (e.g. ``"en"``). ``None``
                  for auto-detect.
        min_buffer_seconds: Minimum seconds of audio to accumulate before
                            attempting a transcription pass.  Shorter values
                            mean lower latency but more frequent (and
                            potentially noisier) passes.  Default ``2.0``.
        stride_seconds: Seconds of audio overlap to retain between passes
                        (sliding-window).  Default ``0.0`` (no overlap).
        sample_rate: Sample rate of incoming audio chunks. Default ``16000``.

    Example::

        streaming = StreamingSTTEngine(model="base", min_buffer_seconds=2.0)
        for chunk in mic_chunks:
            for segment in streaming.push_chunk(chunk):
                print(f"[{segment.start:.1f}s] {segment.text}")

        # Force final transcription when done
        for segment in streaming.flush():
            print(f"[{segment.start:.1f}s] {segment.text}")
    """

    model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    min_buffer_seconds: float = 2.0
    stride_seconds: float = _DEFAULT_STRIDE_S
    sample_rate: int = 16_000

    # Internal state — populated post-init; not part of the public constructor.
    _engine: STTEngine = field(init=False, repr=False)
    _buffer: list[np.ndarray] = field(init=False, repr=False, default_factory=list)
    _buffer_samples: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._engine = STTEngine(
            model=self.model,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
        )
        self._buffer = []
        self._buffer_samples = 0
        logger.info(
            "StreamingSTTEngine created: model=%s, min_buffer=%.1fs, stride=%.1fs",
            self.model,
            self.min_buffer_seconds,
            self.stride_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def buffer_duration_s(self) -> float:
        """Current accumulated audio duration in seconds."""
        return self._buffer_samples / self.sample_rate

    def push_chunk(self, chunk: np.ndarray | bytes) -> Iterator[TranscriptSegment]:
        """Push an audio chunk into the buffer.

        If the buffer has accumulated at least ``min_buffer_seconds`` of audio,
        a transcription pass is run and any yielded segments are returned.
        Otherwise, no segments are yielded and the chunk is silently buffered.

        Args:
            chunk: Float32 numpy array (16kHz mono) **or** raw ``int16`` PCM
                   bytes.  Bytes are automatically converted to float32.

        Yields:
            TranscriptSegment — segments decoded in this pass (may be empty).
        """
        arr = self._coerce_chunk(chunk)
        self._buffer.append(arr)
        self._buffer_samples += len(arr)

        min_samples = int(self.min_buffer_seconds * self.sample_rate)
        if self._buffer_samples >= min_samples:
            yield from self._run_pass()

    def flush(self) -> Iterator[TranscriptSegment]:
        """Transcribe whatever remains in the buffer and clear it.

        Call this when the audio stream ends to ensure trailing audio is
        transcribed.

        Yields:
            TranscriptSegment — segments from the remaining buffer.
        """
        if self._buffer_samples == 0:
            return

        logger.debug("StreamingSTTEngine.flush: %.2f s buffered", self.buffer_duration_s)
        yield from self._run_pass(force=True)

    def reset(self) -> None:
        """Discard the current buffer without transcribing."""
        self._buffer = []
        self._buffer_samples = 0
        logger.debug("StreamingSTTEngine buffer reset")

    def prewarm(self) -> None:
        """Eagerly load the underlying Whisper model."""
        self._engine.prewarm()

    def close(self) -> None:
        """Release model resources and discard the buffer."""
        self.reset()
        self._engine.close()

    def __enter__(self) -> StreamingSTTEngine:
        """Enter sync context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit sync context manager. Releases engine resources."""
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _coerce_chunk(self, chunk: np.ndarray | bytes) -> np.ndarray:
        """Convert raw int16 bytes or ensure float32 array."""
        if isinstance(chunk, (bytes, bytearray)):
            arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            return arr
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim > 1:
            # Best-effort stereo → mono using the shape heuristic from STTEngine
            if arr.shape[0] < arr.shape[1]:
                arr = arr.mean(axis=0)
            else:
                arr = arr.mean(axis=1)
        return arr

    def _run_pass(self, *, force: bool = False) -> Iterator[TranscriptSegment]:
        """Concatenate buffer, transcribe, apply sliding window, yield segments."""
        audio = np.concatenate(self._buffer)

        logger.debug(
            "StreamingSTTEngine pass: %.2f s (force=%s)",
            len(audio) / self.sample_rate,
            force,
        )

        yield from self._engine.transcribe_streaming(audio)

        # Sliding window: retain the last stride_seconds of audio so that
        # words near the boundary are not cut off on the next pass.
        stride_samples = int(self.stride_seconds * self.sample_rate)
        if stride_samples > 0 and len(audio) > stride_samples:
            retained = audio[-stride_samples:]
            self._buffer = [retained]
            self._buffer_samples = len(retained)
        else:
            self._buffer = []
            self._buffer_samples = 0
