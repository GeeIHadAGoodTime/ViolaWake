"""Unit tests for STTEngine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.stt import STTEngine, TranscriptResult
from violawake_sdk.stt_engine import STTFileEngine, transcribe_wav_file


def _make_segment(
    text: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    no_speech_prob: float = 0.0,
) -> MagicMock:
    segment = MagicMock()
    segment.text = text
    segment.start = start
    segment.end = end
    segment.no_speech_prob = no_speech_prob
    return segment


def _make_info(
    *,
    language: str = "en",
    language_probability: float = 0.95,
    duration: float = 1.5,
) -> MagicMock:
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    info.duration = duration
    return info


def _mock_faster_whisper(model: object | None = None) -> tuple[MagicMock, MagicMock]:
    constructor = MagicMock(return_value=model or MagicMock())
    module = MagicMock()
    module.WhisperModel = constructor
    return module, constructor


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        STTEngine(model="bad-model")


def test_valid_init() -> None:
    engine = STTEngine(model="base")

    assert engine.model_name == "base"
    assert engine.device == "cpu"
    assert engine.compute_type == "int8"


def test_transcribe_returns_str() -> None:
    engine = STTEngine(model="base")
    model = MagicMock()
    model.transcribe.return_value = (iter([_make_segment(" hello ")]), _make_info())
    mock_module, constructor = _mock_faster_whisper(model)

    with patch.dict("sys.modules", {"faster_whisper": mock_module}):
        result = engine.transcribe(np.array([0.1, -0.1], dtype=np.float32))

    assert result == "hello"
    constructor.assert_called_once_with("base", device="cpu", compute_type="int8")


def test_transcribe_uses_short_temperature_fallback() -> None:
    engine = STTEngine(model="base")
    engine._model = MagicMock()
    engine._model.transcribe.return_value = (iter([_make_segment("hello")]), _make_info())

    result = engine.transcribe(np.array([0.1, -0.1], dtype=np.float32))

    assert result == "hello"
    call_kwargs = engine._model.transcribe.call_args.kwargs
    assert call_kwargs["beam_size"] == 5
    assert call_kwargs["temperature"] == (0.0, 0.2, 0.4)


def test_no_speech_returns_empty() -> None:
    engine = STTEngine(model="base")
    engine._model = MagicMock()
    engine._model.transcribe.return_value = (
        iter([_make_segment("ignored", no_speech_prob=0.9)]),
        _make_info(),
    )

    result = engine.transcribe(np.array([0.1, -0.1], dtype=np.float32))

    assert result == ""


def test_transcribe_full() -> None:
    engine = STTEngine(model="base")
    model = MagicMock()
    model.transcribe.return_value = (
        iter(
            [
                _make_segment(" hello ", start=0.0, end=0.5),
                _make_segment(" world ", start=0.5, end=1.0),
            ]
        ),
        _make_info(language="en", language_probability=0.9, duration=1.0),
    )
    mock_module, _constructor = _mock_faster_whisper(model)

    with patch.dict("sys.modules", {"faster_whisper": mock_module}):
        result = engine.transcribe_full(np.array([0.1, -0.1], dtype=np.float32))

    assert isinstance(result, TranscriptResult)
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.language_prob == 0.9
    assert result.duration_s == 1.0
    assert len(result.segments) == 2


def test_2d_audio_converted_to_mono() -> None:
    engine = STTEngine(model="base")
    engine._model = MagicMock()
    engine._model.transcribe.return_value = (iter([_make_segment("stereo")]), _make_info())
    audio = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)

    result = engine.transcribe(audio)

    passed_audio = engine._model.transcribe.call_args.args[0]
    assert result == "stereo"
    assert passed_audio.ndim == 1
    assert np.allclose(passed_audio, np.array([2.0, 3.0], dtype=np.float32))


def test_forced_language_works() -> None:
    engine = STTEngine(model="base", language="fr")
    engine._model = MagicMock()
    engine._model.transcribe.return_value = (iter([_make_segment("bonjour")]), _make_info())

    result = engine.transcribe(np.array([0.1, -0.1], dtype=np.float32))

    assert result == "bonjour"
    assert engine._model.transcribe.call_args.kwargs["language"] == "fr"


def test_prewarm_loads_model() -> None:
    engine = STTEngine(model="base")
    model = MagicMock()
    mock_module, constructor = _mock_faster_whisper(model)

    with patch.dict("sys.modules", {"faster_whisper": mock_module}):
        engine.prewarm()

    assert engine._model is model
    constructor.assert_called_once_with("base", device="cpu", compute_type="int8")


def test_language_cache_respects_ttl() -> None:
    engine = STTEngine(model="base", language_cache_ttl_s=5.0)
    engine._language_cache = ("es", 100.0)

    with patch("violawake_sdk.stt.time.monotonic", return_value=104.0):
        assert engine._get_language() == "es"

    with patch("violawake_sdk.stt.time.monotonic", return_value=106.0):
        assert engine._get_language() is None


def test_transcribe_streaming_skips_no_speech_segments_and_primes_language_cache() -> None:
    engine = STTEngine(model="base")
    engine._model = MagicMock()
    engine._model.transcribe.return_value = (
        iter(
            [
                _make_segment(" hello ", start=0.0, end=0.3, no_speech_prob=0.2),
                _make_segment("ignored", start=0.3, end=0.6, no_speech_prob=0.95),
            ]
        ),
        _make_info(language="en", language_probability=0.9, duration=0.6),
    )

    with patch("violawake_sdk.stt.time.monotonic", return_value=123.0):
        segments = list(engine.transcribe_streaming(np.array([0.1, -0.1], dtype=np.float32)))

    assert [segment.text for segment in segments] == ["hello"]
    assert engine._language_cache == ("en", 123.0)


def test_close_releases_model() -> None:
    engine = STTEngine(model="base")
    engine._model = MagicMock()

    engine.close()

    assert engine._model is None


def test_context_manager_closes_model() -> None:
    engine = STTEngine(model="base")

    with patch.object(engine, "close") as close_mock:
        with engine as entered:
            assert entered is engine

    close_mock.assert_called_once()


class TestSTTFileEngine:
    """Tests for the WAV-file wrapper API in stt_engine.py."""

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with patch("violawake_sdk.stt_engine.STTEngine"):
            engine = STTFileEngine(model="base")

        with pytest.raises(FileNotFoundError, match="WAV file not found"):
            engine.transcribe_wav(tmp_path / "missing.wav")

    def test_wrong_sample_rate_raises_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(b"wav")

        with (
            patch("violawake_sdk.stt_engine.STTEngine"),
            patch("violawake_sdk.stt_engine.scipy.io.wavfile.read", return_value=(8000, np.zeros(8))),
        ):
            engine = STTFileEngine(model="base")
            with pytest.raises(ValueError, match="16000 Hz"):
                engine.transcribe_wav(path)

    def test_int16_audio_is_normalized_before_transcribe(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(b"wav")
        stt = MagicMock()
        stt.transcribe.return_value = "hello"

        with (
            patch("violawake_sdk.stt_engine.STTEngine", return_value=stt),
            patch(
                "violawake_sdk.stt_engine.scipy.io.wavfile.read",
                return_value=(16_000, np.array([-32768, 0, 32767], dtype=np.int16)),
            ),
        ):
            engine = STTFileEngine(model="base")
            result = engine.transcribe_wav(path)

        assert result == "hello"
        audio = stt.transcribe.call_args.args[0]
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, np.array([-1.0, 0.0, 32767 / 32768], dtype=np.float32))

    def test_float32_audio_passes_through(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(b"wav")
        stt = MagicMock()
        stt.transcribe.return_value = "ok"
        audio_input = np.array([0.25, -0.5], dtype=np.float32)

        with (
            patch("violawake_sdk.stt_engine.STTEngine", return_value=stt),
            patch("violawake_sdk.stt_engine.scipy.io.wavfile.read", return_value=(16_000, audio_input)),
        ):
            engine = STTFileEngine(model="base")
            engine.transcribe_wav(path)

        np.testing.assert_array_equal(stt.transcribe.call_args.args[0], audio_input)

    def test_transcribe_wav_file_uses_wrapper_engine(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"

        with patch("violawake_sdk.stt_engine.STTFileEngine") as engine_cls:
            engine_cls.return_value.transcribe_wav.return_value = "spoken"

            result = transcribe_wav_file(path, model="small")

        assert result == "spoken"
        engine_cls.assert_called_once_with(model="small")
        engine_cls.return_value.transcribe_wav.assert_called_once_with(path)
