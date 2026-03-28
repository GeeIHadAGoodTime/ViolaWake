"""Unit tests for violawake_sdk.audio module.

Tests pad_or_trim, load_audio, and error handling.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from violawake_sdk.audio import load_audio, pad_or_trim


# ---------------------------------------------------------------------------
# pad_or_trim tests
# ---------------------------------------------------------------------------


class TestPadOrTrim:
    """Tests for pad_or_trim()."""

    def test_exact_length_unchanged(self) -> None:
        """Audio exactly at target length should be returned as-is."""
        audio = np.ones(100, dtype=np.float32)
        result = pad_or_trim(audio, target_length=100)
        assert len(result) == 100
        np.testing.assert_array_equal(result, audio)

    def test_short_audio_padded_with_zeros(self) -> None:
        """Audio shorter than target should be zero-padded at the end."""
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = pad_or_trim(audio, target_length=6)
        assert len(result) == 6
        np.testing.assert_array_equal(result[:3], audio)
        np.testing.assert_array_equal(result[3:], np.zeros(3))

    def test_long_audio_trimmed_deterministically(self) -> None:
        """Audio longer than target should be trimmed from the start (deterministic)."""
        audio = np.arange(10, dtype=np.float32)
        result = pad_or_trim(audio, target_length=5)
        assert len(result) == 5
        # Should take from the beginning
        np.testing.assert_array_equal(result, np.arange(5, dtype=np.float32))

    def test_trim_is_deterministic(self) -> None:
        """Multiple calls with the same input should produce identical output."""
        audio = np.arange(20, dtype=np.float32)
        result1 = pad_or_trim(audio, target_length=10)
        result2 = pad_or_trim(audio, target_length=10)
        np.testing.assert_array_equal(result1, result2)

    def test_empty_audio_padded(self) -> None:
        """Empty audio should be padded to target length with zeros."""
        audio = np.array([], dtype=np.float32)
        result = pad_or_trim(audio, target_length=5)
        assert len(result) == 5
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_single_sample(self) -> None:
        """Single-sample audio should be padded correctly."""
        audio = np.array([0.5], dtype=np.float32)
        result = pad_or_trim(audio, target_length=3)
        assert len(result) == 3
        assert result[0] == 0.5
        assert result[1] == 0.0
        assert result[2] == 0.0


# ---------------------------------------------------------------------------
# load_audio tests
# ---------------------------------------------------------------------------


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 16000) -> None:
    """Write a mono 16-bit WAV file."""
    int16_data = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16_data.tobytes())


class TestLoadAudio:
    """Tests for load_audio()."""

    def test_load_valid_wav(self, tmp_path: Path) -> None:
        """load_audio should successfully load a valid WAV file."""
        wav_path = tmp_path / "test.wav"
        original = np.sin(np.linspace(0, 2 * np.pi, 16000)).astype(np.float32)
        _write_wav(wav_path, original, sample_rate=16000)

        result = load_audio(wav_path, target_sr=16000)
        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == 16000

    def test_load_wav_returns_float(self, tmp_path: Path) -> None:
        """Loaded audio should be float32."""
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, np.zeros(1000, dtype=np.float32))

        result = load_audio(wav_path)
        assert result is not None
        assert result.dtype == np.float32

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Loading a nonexistent file should return None (not raise)."""
        result = load_audio(tmp_path / "nonexistent.wav")
        assert result is None

    def test_load_corrupt_file_returns_none(self, tmp_path: Path) -> None:
        """Loading a corrupt file should return None and log the error."""
        bad_path = tmp_path / "corrupt.wav"
        bad_path.write_bytes(b"not a wav file")

        result = load_audio(bad_path)
        assert result is None

    def test_load_audio_error_is_logged(self, tmp_path: Path) -> None:
        """Errors during load should be logged, not silently swallowed."""
        bad_path = tmp_path / "corrupt.wav"
        bad_path.write_bytes(b"not a wav file")

        with patch("violawake_sdk.audio.logger") as mock_logger:
            result = load_audio(bad_path)

        assert result is None
        mock_logger.error.assert_called_once()
