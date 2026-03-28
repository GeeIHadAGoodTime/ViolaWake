"""Unit tests for STTFileEngine (WAV-file wrapper around STTEngine).

All tests mock faster_whisper so the package is never required to be installed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.io.wavfile  # type: ignore[import]

from violawake_sdk.stt_engine import STTFileEngine, transcribe_wav_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(text: str, *, no_speech_prob: float = 0.0) -> MagicMock:
    seg = MagicMock()
    seg.text = text
    seg.start = 0.0
    seg.end = 1.0
    seg.no_speech_prob = no_speech_prob
    return seg


def _make_info() -> MagicMock:
    info = MagicMock()
    info.language = "en"
    info.language_probability = 0.95
    info.duration = 1.0
    return info


def _mock_whisper_model(return_text: str = "hello") -> MagicMock:
    """Return a fake WhisperModel that produces *return_text* on every call."""
    model = MagicMock()
    # Use side_effect so each call gets a fresh iterator (return_value reuses the
    # same exhausted generator after the first call).
    def _side_effect(*args: object, **kwargs: object) -> tuple[object, object]:
        del args, kwargs  # intentionally unused: mock side_effect for model.transcribe
        return (iter([_make_segment(f" {return_text} ")]), _make_info())

    model.transcribe.side_effect = _side_effect
    return model


def _mock_faster_whisper(model: MagicMock) -> MagicMock:
    """Return a fake faster_whisper module containing *model* as WhisperModel."""
    module = MagicMock()
    module.WhisperModel = MagicMock(return_value=model)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def silence_wav(tmp_path: Path) -> Path:
    """Write a 1-second silence WAV at 16 kHz to a temp file."""
    sample_rate = 16_000
    duration_samples = sample_rate  # 1 second
    audio = np.zeros(duration_samples, dtype=np.int16)
    wav_path = tmp_path / "silence.wav"
    scipy.io.wavfile.write(str(wav_path), sample_rate, audio)
    return wav_path


@pytest.fixture()
def wrong_rate_wav(tmp_path: Path) -> Path:
    """Write a 1-second WAV at 44 100 Hz (wrong rate)."""
    sample_rate = 44_100
    audio = np.zeros(sample_rate, dtype=np.int16)
    wav_path = tmp_path / "wrong_rate.wav"
    scipy.io.wavfile.write(str(wav_path), sample_rate, audio)
    return wav_path


# ---------------------------------------------------------------------------
# STTFileEngine tests
# ---------------------------------------------------------------------------

def test_transcribe_wav_returns_string(silence_wav: Path) -> None:
    """transcribe_wav() should call through to STTEngine and return a string."""
    engine = STTFileEngine(model="base")
    whisper_model = _mock_whisper_model("hello world")
    fake_module = _mock_faster_whisper(whisper_model)

    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        result = engine.transcribe_wav(silence_wav)

    assert isinstance(result, str)
    assert result == "hello world"


def test_transcribe_wav_file_not_found(tmp_path: Path) -> None:
    """transcribe_wav() raises FileNotFoundError for a missing file."""
    engine = STTFileEngine(model="base")
    missing = tmp_path / "does_not_exist.wav"

    with pytest.raises(FileNotFoundError, match="does_not_exist.wav"):
        engine.transcribe_wav(missing)


def test_transcribe_wav_wrong_sample_rate(wrong_rate_wav: Path) -> None:
    """transcribe_wav() raises ValueError when the WAV is not at 16 kHz."""
    engine = STTFileEngine(model="base")

    with pytest.raises(ValueError, match="16000"):
        engine.transcribe_wav(wrong_rate_wav)


def test_transcribe_wav_accepts_path_object(silence_wav: Path) -> None:
    """transcribe_wav() accepts both str and Path arguments."""
    engine = STTFileEngine(model="base")
    whisper_model = _mock_whisper_model("ok")
    fake_module = _mock_faster_whisper(whisper_model)

    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        result_path = engine.transcribe_wav(silence_wav)
        result_str = engine.transcribe_wav(str(silence_wav))

    assert result_path == result_str == "ok"


def test_transcribe_wav_int16_conversion(silence_wav: Path) -> None:
    """Int16 WAV data must be divided by 32768 before being passed to STTEngine."""
    engine = STTFileEngine(model="base")
    whisper_model = _mock_whisper_model("check")
    fake_module = _mock_faster_whisper(whisper_model)

    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        engine.transcribe_wav(silence_wav)

    # The audio passed to model.transcribe should be float32
    call_args = whisper_model.transcribe.call_args
    audio_arg = call_args.args[0]
    assert audio_arg.dtype == np.float32
    # Silence (zeros) divided by 32768 should still be zeros
    assert np.all(audio_arg == 0.0)


def test_transcribe_wav_int32_conversion(tmp_path: Path) -> None:
    """Int32 WAV data must be normalized via np.iinfo(dtype) without str() wrapping."""
    sample_rate = 16_000
    # Write a 16-bit WAV, then we'll patch scipy.io.wavfile.read to return int32
    audio_int32 = np.array([0, 1000, -1000, 2147483647], dtype=np.int32)
    engine = STTFileEngine(model="base")
    whisper_model = _mock_whisper_model("int32test")
    fake_module = _mock_faster_whisper(whisper_model)

    wav_path = tmp_path / "int32.wav"
    # Write a dummy WAV so the file exists
    scipy.io.wavfile.write(str(wav_path), sample_rate, np.zeros(4, dtype=np.int16))

    # Patch scipy.io.wavfile.read to return int32 data
    with (
        patch.dict("sys.modules", {"faster_whisper": fake_module}),
        patch("scipy.io.wavfile.read", return_value=(sample_rate, audio_int32)),
    ):
        result = engine.transcribe_wav(wav_path)

    assert result == "int32test"
    # Verify the audio was converted to float32
    call_args = whisper_model.transcribe.call_args
    audio_arg = call_args.args[0]
    assert audio_arg.dtype == np.float32
    # Max int32 should map to ~1.0
    assert abs(audio_arg[3] - 1.0) < 0.01


def test_transcribe_wav_empty_speech_returns_empty(silence_wav: Path) -> None:
    """If faster-whisper returns no-speech, the result is an empty string."""
    engine = STTFileEngine(model="base")
    fake_model = MagicMock()
    fake_model.transcribe.return_value = (
        iter([_make_segment("ignored", no_speech_prob=0.9)]),
        _make_info(),
    )
    fake_module = _mock_faster_whisper(fake_model)

    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        result = engine.transcribe_wav(silence_wav)

    assert result == ""


# ---------------------------------------------------------------------------
# transcribe_wav_file convenience function
# ---------------------------------------------------------------------------

def test_convenience_function_returns_string(silence_wav: Path) -> None:
    """transcribe_wav_file() is a thin wrapper that must return a string."""
    whisper_model = _mock_whisper_model("viola")
    fake_module = _mock_faster_whisper(whisper_model)

    with patch.dict("sys.modules", {"faster_whisper": fake_module}):
        result = transcribe_wav_file(silence_wav, model="base")

    assert result == "viola"


def test_convenience_function_file_not_found(tmp_path: Path) -> None:
    """transcribe_wav_file() propagates FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        transcribe_wav_file(tmp_path / "ghost.wav")


def test_convenience_function_wrong_rate(wrong_rate_wav: Path) -> None:
    """transcribe_wav_file() propagates ValueError for wrong sample rate."""
    with pytest.raises(ValueError, match="16000"):
        transcribe_wav_file(wrong_rate_wav)
