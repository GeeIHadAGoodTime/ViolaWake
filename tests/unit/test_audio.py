"""Unit tests for violawake_sdk.audio module.

Tests pad_or_trim, load_audio, and error handling.
"""

from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import violawake_sdk.audio as audio_module
from violawake_sdk.audio import (
    center_crop,
    compute_features,
    compute_mel_spectrogram,
    compute_mel_spectrogram_v2,
    compute_rms,
    is_silent,
    load_audio,
    normalize_audio,
    normalize_audio_rms,
    pad_or_trim,
)


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
        mock_logger.exception.assert_called_once()


class TestCenterCrop:
    """Tests for center_crop()."""

    def test_short_audio_is_padded_symmetrically(self) -> None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = center_crop(audio, target_length=7)

        np.testing.assert_array_equal(
            result,
            np.array([0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0], dtype=np.float32),
        )

    def test_long_audio_is_center_cropped(self) -> None:
        audio = np.arange(9, dtype=np.float32)
        result = center_crop(audio, target_length=5)

        np.testing.assert_array_equal(result, np.array([2, 3, 4, 5, 6], dtype=np.float32))


class TestFeatureExtraction:
    """Tests for mel features, PCEN, and dispatch."""

    def test_compute_mel_spectrogram_returns_finite_features(self) -> None:
        audio = np.sin(np.linspace(0, 8 * np.pi, 4096, dtype=np.float32))
        fake_signal = SimpleNamespace(
            spectrogram=MagicMock(
                return_value=(
                    np.arange(audio_module.N_MELS + 4, dtype=np.float32),
                    np.arange(3, dtype=np.float32),
                    np.ones((audio_module.N_MELS + 4, 3), dtype=np.float32),
                )
            )
        )
        fake_scipy = SimpleNamespace(signal=fake_signal)

        with patch.dict(sys.modules, {"scipy": fake_scipy, "scipy.signal": fake_signal}):
            features = compute_mel_spectrogram(audio)

        assert features.shape == (audio_module.N_MELS, 3)
        assert np.isfinite(features).all()

    def test_compute_mel_spectrogram_v2_uses_librosa_pcen_when_available(self) -> None:
        mel = np.ones((audio_module.N_MELS_MEL, 4), dtype=np.float32)
        expected = np.full_like(mel, 7.0)
        fake_librosa = SimpleNamespace(
            feature=SimpleNamespace(melspectrogram=MagicMock(return_value=mel)),
            pcen=MagicMock(return_value=expected),
        )

        with patch.dict(sys.modules, {"librosa": fake_librosa}):
            features = compute_mel_spectrogram_v2(np.ones(320, dtype=np.float32))

        np.testing.assert_array_equal(features, expected)
        fake_librosa.pcen.assert_called_once()

    def test_compute_mel_spectrogram_v2_falls_back_to_manual_pcen(self) -> None:
        mel = np.ones((audio_module.N_MELS_MEL, 3), dtype=np.float32)
        expected = np.full_like(mel, 0.25)
        fake_librosa = SimpleNamespace(
            feature=SimpleNamespace(melspectrogram=MagicMock(return_value=mel)),
            pcen=MagicMock(side_effect=AttributeError("pcen missing")),
        )

        with (
            patch.dict(sys.modules, {"librosa": fake_librosa}),
            patch("violawake_sdk.audio._apply_pcen_manual", return_value=expected) as pcen_manual,
        ):
            features = compute_mel_spectrogram_v2(np.ones(320, dtype=np.float32))

        np.testing.assert_array_equal(features, expected)
        pcen_manual.assert_called_once_with(mel)

    def test_compute_mel_spectrogram_v2_uses_log_features_when_pcen_disabled(self) -> None:
        mel = np.full((audio_module.N_MELS_MEL, 2), 2.0, dtype=np.float32)
        fake_librosa = SimpleNamespace(
            feature=SimpleNamespace(melspectrogram=MagicMock(return_value=mel)),
            pcen=MagicMock(),
        )

        with patch.dict(sys.modules, {"librosa": fake_librosa}), patch.object(
            audio_module, "USE_PCEN", False
        ):
            features = compute_mel_spectrogram_v2(np.ones(320, dtype=np.float32))

        np.testing.assert_allclose(features, np.log(mel + 1e-9))
        fake_librosa.pcen.assert_not_called()

    def test_compute_features_dispatches_to_linear_features(self) -> None:
        expected = np.ones((audio_module.N_MELS, 2), dtype=np.float32)
        with (
            patch.object(audio_module, "FEATURE_TYPE", "linear"),
            patch("violawake_sdk.audio.compute_mel_spectrogram", return_value=expected) as linear,
        ):
            result = compute_features(np.ones(320, dtype=np.float32))

        assert result is expected
        linear.assert_called_once()

    def test_compute_features_dispatches_to_mel_features(self) -> None:
        expected = np.ones((audio_module.N_MELS_MEL, 2), dtype=np.float32)
        with (
            patch.object(audio_module, "FEATURE_TYPE", "mel_pcen"),
            patch("violawake_sdk.audio.compute_mel_spectrogram_v2", return_value=expected) as mel,
        ):
            result = compute_features(np.ones(320, dtype=np.float32))

        assert result is expected
        mel.assert_called_once()

    def test_compute_features_rejects_unknown_feature_type(self) -> None:
        with patch.object(audio_module, "FEATURE_TYPE", "bogus"):
            with pytest.raises(ValueError, match="Unknown FEATURE_TYPE"):
                compute_features(np.ones(320, dtype=np.float32))


class TestNormalization:
    """Tests for normalization and silence helpers."""

    def test_normalize_audio_caps_gain(self) -> None:
        audio = np.array([0.1, -0.2], dtype=np.float32)

        result = normalize_audio(audio, target_peak=0.95, max_gain=2.0)

        np.testing.assert_allclose(result, audio * 2.0)

    def test_normalize_audio_returns_silence_unchanged(self) -> None:
        audio = np.zeros(4, dtype=np.float32)

        result = normalize_audio(audio)

        np.testing.assert_array_equal(result, audio)

    def test_normalize_audio_rms_targets_requested_level(self) -> None:
        audio = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)

        result = normalize_audio_rms(audio, target_rms=0.25)

        assert compute_rms(result) == pytest.approx(0.25, rel=1e-3)

    def test_normalize_audio_rms_clips_to_unit_range(self) -> None:
        audio = np.array([0.9, -0.9], dtype=np.float32)

        result = normalize_audio_rms(audio, target_rms=2.0)

        assert np.max(np.abs(result)) <= 1.0

    def test_compute_rms_and_is_silent(self) -> None:
        audio = np.array([0.3, -0.3, 0.3, -0.3], dtype=np.float32)

        assert compute_rms(audio) == pytest.approx(0.3, rel=1e-3)
        assert is_silent(audio, threshold=0.31) is True
        assert is_silent(audio, threshold=0.29) is False
