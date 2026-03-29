"""Integration tests for streaming STT: transcribe_streaming and StreamingSTTEngine."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.stt import STTEngine, StreamingSTTEngine, TranscriptSegment

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_segment(
    text: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    no_speech_prob: float = 0.0,
) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.no_speech_prob = no_speech_prob
    return seg


def _make_info(
    *,
    language: str = "en",
    language_probability: float = 0.95,
    duration: float = 2.0,
) -> MagicMock:
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    info.duration = duration
    return info


def _silence_audio(seconds: float = 2.0, sample_rate: int = 16_000) -> np.ndarray:
    """Return a silent float32 array of the given duration."""
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)


# ---------------------------------------------------------------------------
# STTEngine.transcribe_streaming — returns an Iterator
# ---------------------------------------------------------------------------


class TestTranscribeStreamingReturnType:
    """Verify that transcribe_streaming() is an iterator."""

    def test_returns_iterator(self) -> None:
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([_make_segment(" hello ")]),
            _make_info(),
        )
        engine._model = model_mock

        result = engine.transcribe_streaming(_silence_audio())

        assert hasattr(result, "__iter__"), "transcribe_streaming must return an iterable"
        assert hasattr(result, "__next__"), "transcribe_streaming must return an iterator"

    def test_yields_transcript_segments(self) -> None:
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([
                _make_segment(" hello ", start=0.0, end=0.5),
                _make_segment(" world ", start=0.5, end=1.0),
            ]),
            _make_info(),
        )
        engine._model = model_mock

        segments = list(engine.transcribe_streaming(_silence_audio()))

        assert len(segments) == 2
        assert all(isinstance(s, TranscriptSegment) for s in segments)
        assert segments[0].text == "hello"
        assert segments[1].text == "world"

    def test_yields_correct_timing(self) -> None:
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([_make_segment("test", start=1.2, end=2.4)]),
            _make_info(),
        )
        engine._model = model_mock

        segments = list(engine.transcribe_streaming(_silence_audio()))

        assert segments[0].start == pytest.approx(1.2)
        assert segments[0].end == pytest.approx(2.4)

    def test_silent_segments_are_skipped(self) -> None:
        """Segments with high no_speech_prob must not be yielded."""
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([
                _make_segment("speech", no_speech_prob=0.1),
                _make_segment("noise", no_speech_prob=0.9),  # above threshold
            ]),
            _make_info(),
        )
        engine._model = model_mock

        segments = list(engine.transcribe_streaming(_silence_audio()))

        assert len(segments) == 1
        assert segments[0].text == "speech"

    def test_empty_audio_yields_no_segments(self) -> None:
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (iter([]), _make_info())
        engine._model = model_mock

        segments = list(engine.transcribe_streaming(_silence_audio()))

        assert segments == []

    def test_does_not_break_existing_transcribe(self) -> None:
        """Calling transcribe_streaming must not corrupt subsequent batch calls."""
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([_make_segment("hello")]),
            _make_info(),
        )
        engine._model = model_mock

        list(engine.transcribe_streaming(_silence_audio()))

        # Reset side_effect for next call
        model_mock.transcribe.return_value = (
            iter([_make_segment("world")]),
            _make_info(),
        )
        text = engine.transcribe(_silence_audio())
        assert text == "world"

    def test_2d_stereo_audio_is_handled(self) -> None:
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([_make_segment("stereo")]),
            _make_info(),
        )
        engine._model = model_mock

        stereo = np.zeros((2, 16_000), dtype=np.float32)  # channels_first
        segments = list(engine.transcribe_streaming(stereo, channels_first=True))

        passed_audio = model_mock.transcribe.call_args.args[0]
        assert passed_audio.ndim == 1, "Expected mono audio passed to model"
        assert segments[0].text == "stereo"

    def test_language_cache_updated_after_streaming(self) -> None:
        engine = STTEngine(model="base", language_cache_ttl_s=60.0)
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (
            iter([_make_segment("ciao")]),
            _make_info(language="it", language_probability=0.95),
        )
        engine._model = model_mock

        list(engine.transcribe_streaming(_silence_audio()))

        assert engine._language_cache is not None
        assert engine._language_cache[0] == "it"

    def test_custom_beam_size_and_temperature_forwarded(self) -> None:
        engine = STTEngine(model="base")
        model_mock = MagicMock()
        model_mock.transcribe.return_value = (iter([]), _make_info())
        engine._model = model_mock

        list(engine.transcribe_streaming(
            _silence_audio(),
            beam_size=3,
            best_of=3,
            temperature=[0.0],
        ))

        call_kwargs = model_mock.transcribe.call_args.kwargs
        assert call_kwargs["beam_size"] == 3
        assert call_kwargs["best_of"] == 3
        assert call_kwargs["temperature"] == [0.0]


# ---------------------------------------------------------------------------
# StreamingSTTEngine — incremental chunk-based streaming
# ---------------------------------------------------------------------------


class TestStreamingSTTEngine:
    """Tests for the StreamingSTTEngine chunk-accumulation interface."""

    def _make_engine(
        self,
        *,
        min_buffer_seconds: float = 1.0,
        stride_seconds: float = 0.0,
        model_segments: list[MagicMock] | None = None,
    ) -> tuple[StreamingSTTEngine, MagicMock]:
        """Return a StreamingSTTEngine with a mocked underlying WhisperModel."""
        if model_segments is None:
            model_segments = [_make_segment("chunk")]

        whisper_model_mock = MagicMock()
        whisper_model_mock.transcribe.return_value = (
            iter(model_segments),
            _make_info(),
        )

        engine = StreamingSTTEngine(
            model="base",
            min_buffer_seconds=min_buffer_seconds,
            stride_seconds=stride_seconds,
        )
        # Inject the mock at the STTEngine level
        engine._engine._model = whisper_model_mock
        return engine, whisper_model_mock

    # --- push_chunk ---

    def test_push_chunk_below_threshold_yields_nothing(self) -> None:
        engine, _ = self._make_engine(min_buffer_seconds=2.0)
        short_chunk = _silence_audio(0.5)  # 0.5 s < 2.0 s threshold

        results = list(engine.push_chunk(short_chunk))

        assert results == [], "Should not transcribe when buffer is below threshold"

    def test_push_chunk_above_threshold_yields_segments(self) -> None:
        segs = [_make_segment("hello")]
        engine, model_mock = self._make_engine(
            min_buffer_seconds=1.0,
            model_segments=segs,
        )
        chunk = _silence_audio(1.2)  # exceeds 1.0 s threshold

        model_mock.transcribe.return_value = (iter(segs), _make_info())
        results = list(engine.push_chunk(chunk))

        assert len(results) == 1
        assert results[0].text == "hello"

    def test_push_chunk_accepts_bytes(self) -> None:
        segs = [_make_segment("bytes chunk")]
        engine, model_mock = self._make_engine(min_buffer_seconds=0.5, model_segments=segs)
        model_mock.transcribe.return_value = (iter(segs), _make_info())

        # 1 second of silent int16 PCM
        pcm_bytes = (np.zeros(16_000, dtype=np.int16)).tobytes()
        results = list(engine.push_chunk(pcm_bytes))

        assert len(results) == 1
        assert results[0].text == "bytes chunk"

    def test_multiple_pushes_accumulate(self) -> None:
        engine, model_mock = self._make_engine(min_buffer_seconds=2.0)
        chunk = _silence_audio(0.9)  # each chunk alone is below threshold

        # First push — buffer = 0.9 s (no transcription)
        r1 = list(engine.push_chunk(chunk))
        assert r1 == []

        # Second push — buffer = 1.8 s (still below 2.0 s)
        r2 = list(engine.push_chunk(chunk))
        assert r2 == []

        # Third push — buffer = 2.7 s (exceeds threshold)
        segs = [_make_segment("accumulated")]
        model_mock.transcribe.return_value = (iter(segs), _make_info())
        r3 = list(engine.push_chunk(chunk))
        assert len(r3) == 1
        assert r3[0].text == "accumulated"

    def test_buffer_cleared_after_pass(self) -> None:
        segs = [_make_segment("cleared")]
        engine, model_mock = self._make_engine(min_buffer_seconds=0.5, model_segments=segs)
        model_mock.transcribe.return_value = (iter(segs), _make_info())

        list(engine.push_chunk(_silence_audio(0.6)))

        assert engine.buffer_duration_s == pytest.approx(0.0)
        assert engine._buffer_samples == 0

    # --- sliding window (stride_seconds) ---

    def test_stride_retains_tail_audio(self) -> None:
        segs = [_make_segment("strided")]
        engine, model_mock = self._make_engine(
            min_buffer_seconds=0.5,
            stride_seconds=0.25,
            model_segments=segs,
        )
        model_mock.transcribe.return_value = (iter(segs), _make_info())

        list(engine.push_chunk(_silence_audio(0.6)))

        # After the pass, buffer should retain ~0.25 s of audio
        assert engine.buffer_duration_s == pytest.approx(0.25, abs=0.01)

    # --- flush ---

    def test_flush_transcribes_remaining_buffer(self) -> None:
        engine, model_mock = self._make_engine(min_buffer_seconds=10.0)
        engine._buffer = [_silence_audio(0.5)]
        engine._buffer_samples = int(0.5 * 16_000)

        segs = [_make_segment("flushed")]
        model_mock.transcribe.return_value = (iter(segs), _make_info())
        results = list(engine.flush())

        assert len(results) == 1
        assert results[0].text == "flushed"

    def test_flush_on_empty_buffer_yields_nothing(self) -> None:
        engine, _ = self._make_engine()
        results = list(engine.flush())
        assert results == []

    def test_flush_clears_buffer(self) -> None:
        engine, model_mock = self._make_engine(min_buffer_seconds=10.0)
        engine._buffer = [_silence_audio(0.5)]
        engine._buffer_samples = int(0.5 * 16_000)
        model_mock.transcribe.return_value = (iter([]), _make_info())

        list(engine.flush())

        assert engine.buffer_duration_s == pytest.approx(0.0)

    # --- reset ---

    def test_reset_discards_buffer(self) -> None:
        engine, _ = self._make_engine(min_buffer_seconds=10.0)
        engine._buffer = [_silence_audio(1.0)]
        engine._buffer_samples = 16_000

        engine.reset()

        assert engine._buffer_samples == 0
        assert engine._buffer == []

    # --- context manager ---

    def test_context_manager_calls_close(self) -> None:
        engine, _ = self._make_engine()
        engine._buffer = [_silence_audio(0.5)]
        engine._buffer_samples = int(0.5 * 16_000)

        with engine:
            pass

        # After __exit__, buffer should be cleared
        assert engine._buffer_samples == 0

    # --- buffer_duration_s property ---

    def test_buffer_duration_reflects_pushed_audio(self) -> None:
        engine, _ = self._make_engine(min_buffer_seconds=10.0)
        chunk = _silence_audio(0.5)

        list(engine.push_chunk(chunk))
        list(engine.push_chunk(chunk))

        assert engine.buffer_duration_s == pytest.approx(1.0, abs=0.01)

    # --- prewarm ---

    def test_prewarm_delegates_to_inner_engine(self) -> None:
        engine, model_mock = self._make_engine()
        engine._engine._model = None  # reset so prewarm actually loads

        whisper_constructor = MagicMock(return_value=model_mock)
        mock_module = MagicMock()
        mock_module.WhisperModel = whisper_constructor

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            engine.prewarm()

        whisper_constructor.assert_called_once()
