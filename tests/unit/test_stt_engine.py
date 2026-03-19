"""Unit tests for STTEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.stt import STTEngine, TranscriptResult


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
