"""Tests for Room Impulse Response augmentation (J2).

Verifies that RIR augmentation:
  - Generates valid synthetic RIRs (correct shape, dtype, decay)
  - Convolves audio correctly (preserves length, changes signal)
  - Loads RIR files from directories
  - Falls back to synthetic RIR when no file provided
  - Handles edge cases (very short audio, missing scipy)
  - Integrates with AugmentationPipeline
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from violawake_sdk.training.augment import (
    AugmentConfig,
    AugmentationPipeline,
    apply_rir,
    generate_synthetic_rir,
    load_rir_dataset,
    rir_augment,
)


@pytest.fixture
def speech_signal() -> np.ndarray:
    """1.5s synthetic speech-like signal at 16kHz."""
    rng = np.random.default_rng(42)
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    signal = (
        np.sin(2 * np.pi * 200 * t) * 0.3
        + np.sin(2 * np.pi * 400 * t) * 0.2
        + rng.normal(0, 0.05, len(t))
    )
    return signal.astype(np.float32)


@pytest.fixture
def rir_wav_dir(tmp_path: Path) -> Path:
    """Create a directory with synthetic RIR WAV files."""
    rir_dir = tmp_path / "rirs"
    rir_dir.mkdir()

    rng = np.random.default_rng(42)
    for i in range(3):
        wav_path = rir_dir / f"rir_{i:02d}.wav"
        # Simple exponential decay RIR
        n = 8000  # 0.5 seconds at 16kHz
        rir = rng.standard_normal(n).astype(np.float32)
        decay = np.exp(-np.arange(n, dtype=np.float32) * 6.0 / n)
        rir = rir * decay
        rir[0] = 1.0
        # Normalize
        rir = rir / np.abs(rir).max()

        pcm = (np.clip(rir, -1, 1) * 32767).astype(np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(pcm.tobytes())

    return rir_dir


class TestGenerateSyntheticRIR:
    """Test synthetic RIR generation."""

    def test_basic_generation(self) -> None:
        """Should produce a valid float32 array."""
        rir = generate_synthetic_rir()
        assert isinstance(rir, np.ndarray)
        assert rir.dtype == np.float32
        assert len(rir) > 0

    def test_length_matches_rt60(self) -> None:
        """RIR length should approximate rt60 * sample_rate."""
        rir = generate_synthetic_rir(sample_rate=16000, rt60=0.5)
        expected_len = int(16000 * 0.5)
        assert len(rir) == expected_len

    def test_direct_path_peak(self) -> None:
        """After normalization, max abs value should be 1.0."""
        rir = generate_synthetic_rir(sample_rate=16000, rt60=0.5, rng=np.random.default_rng(42))
        # After normalization, peak absolute value should be 1.0
        assert np.abs(rir).max() == pytest.approx(1.0, abs=1e-6)
        # rir[0] is set to 1.0 before normalization, but may not remain exactly 1.0
        # after dividing by the peak (which could be elsewhere in the noise)
        assert rir[0] > 0, "Direct path (index 0) should be positive"

    def test_exponential_decay(self) -> None:
        """Energy should decay over time."""
        rir = generate_synthetic_rir(sample_rate=16000, rt60=0.5, rng=np.random.default_rng(42))
        n = len(rir)
        # Compare energy of first quarter vs last quarter
        first_quarter_energy = np.mean(rir[: n // 4] ** 2)
        last_quarter_energy = np.mean(rir[3 * n // 4 :] ** 2)
        assert first_quarter_energy > last_quarter_energy

    def test_reproducible_with_rng(self) -> None:
        """Same rng should produce identical RIRs."""
        rir1 = generate_synthetic_rir(rng=np.random.default_rng(42))
        rir2 = generate_synthetic_rir(rng=np.random.default_rng(42))
        np.testing.assert_array_equal(rir1, rir2)

    def test_different_rng_differ(self) -> None:
        """Different rng seeds should produce different RIRs."""
        rir1 = generate_synthetic_rir(rng=np.random.default_rng(42))
        rir2 = generate_synthetic_rir(rng=np.random.default_rng(99))
        assert not np.array_equal(rir1, rir2)

    def test_random_rt60_range(self) -> None:
        """When rt60=None, should sample from [0.1, 0.8]."""
        rng = np.random.default_rng(42)
        rir = generate_synthetic_rir(sample_rate=16000, rt60=None, rng=rng)
        # Length should be between 0.1*16000=1600 and 0.8*16000=12800
        assert 1600 <= len(rir) <= 12800

    def test_very_short_rt60(self) -> None:
        """Very short rt60 should produce at least 2 samples."""
        rir = generate_synthetic_rir(sample_rate=16000, rt60=0.0001)
        assert len(rir) >= 2


class TestApplyRIR:
    """Test the apply_rir convolution function."""

    def test_output_length_preserved(self, speech_signal: np.ndarray) -> None:
        """Output should have same length as input."""
        rir = generate_synthetic_rir(rt60=0.3, rng=np.random.default_rng(42))
        result = apply_rir(speech_signal, rir)
        assert len(result) == len(speech_signal)

    def test_output_dtype_preserved(self, speech_signal: np.ndarray) -> None:
        """Output dtype should match input."""
        rir = generate_synthetic_rir(rt60=0.3, rng=np.random.default_rng(42))
        result = apply_rir(speech_signal, rir)
        assert result.dtype == speech_signal.dtype

    def test_signal_modified(self, speech_signal: np.ndarray) -> None:
        """Convolution with a non-trivial RIR should change the signal."""
        rir = generate_synthetic_rir(rt60=0.5, rng=np.random.default_rng(42))
        result = apply_rir(speech_signal, rir)
        assert not np.allclose(result, speech_signal, atol=1e-4)

    def test_peak_normalization(self, speech_signal: np.ndarray) -> None:
        """Output peak should match input peak (prevents clipping)."""
        rir = generate_synthetic_rir(rt60=0.5, rng=np.random.default_rng(42))
        result = apply_rir(speech_signal, rir)
        original_peak = np.abs(speech_signal).max()
        result_peak = np.abs(result).max()
        # Peak should be approximately preserved
        assert result_peak <= original_peak + 0.01


class TestRIRAugment:
    """Test the rir_augment convenience function."""

    def test_synthetic_fallback(self, speech_signal: np.ndarray) -> None:
        """Without rir or rir_path, should generate synthetic RIR."""
        rng = np.random.default_rng(42)
        result = rir_augment(speech_signal, rng=rng)
        assert len(result) == len(speech_signal)
        assert result.dtype == speech_signal.dtype
        # Should differ from original (synthetic RIR applied)
        assert not np.allclose(result, speech_signal, atol=1e-4)

    def test_with_preloaded_rir(self, speech_signal: np.ndarray) -> None:
        """Should use the provided RIR array."""
        rir = generate_synthetic_rir(rt60=0.3, rng=np.random.default_rng(42))
        result = rir_augment(speech_signal, rir=rir)
        # Should be same as calling apply_rir directly
        expected = apply_rir(speech_signal, rir)
        np.testing.assert_array_equal(result, expected)

    def test_with_rir_file(self, speech_signal: np.ndarray, rir_wav_dir: Path) -> None:
        """Should load and apply RIR from a WAV file."""
        rir_files = sorted(rir_wav_dir.glob("*.wav"))
        assert len(rir_files) > 0
        result = rir_augment(speech_signal, rir_path=rir_files[0])
        assert len(result) == len(speech_signal)

    def test_rir_takes_precedence_over_path(self, speech_signal: np.ndarray) -> None:
        """Provided rir array should take precedence over rir_path."""
        rir = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # Identity-like
        result = rir_augment(
            speech_signal, rir=rir, rir_path="/nonexistent/path.wav"
        )
        # Should not crash (rir_path not loaded since rir is provided)
        assert len(result) == len(speech_signal)


class TestLoadRIRDataset:
    """Test loading RIR datasets from directories."""

    def test_load_valid_directory(self, rir_wav_dir: Path) -> None:
        """Should load all WAV files from directory."""
        rirs = load_rir_dataset(rir_wav_dir)
        assert len(rirs) == 3
        for rir in rirs:
            assert isinstance(rir, np.ndarray)
            assert rir.dtype == np.float32
            assert len(rir) > 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory should return empty list."""
        empty_dir = tmp_path / "empty_rirs"
        empty_dir.mkdir()
        rirs = load_rir_dataset(empty_dir)
        assert rirs == []

    def test_nonexistent_directory(self) -> None:
        """Nonexistent directory should return empty list."""
        rirs = load_rir_dataset("/nonexistent/rir/dir")
        assert rirs == []


class TestPipelineRIRIntegration:
    """Test RIR integration with AugmentationPipeline."""

    def test_rir_with_files(self, speech_signal: np.ndarray, rir_wav_dir: Path) -> None:
        """Pipeline should use loaded RIR files when configured."""
        rir_files = [str(f) for f in sorted(rir_wav_dir.glob("*.wav"))]
        cfg = AugmentConfig(p_rir=1.0, rir_files=rir_files)
        pipeline = AugmentationPipeline(config=cfg, seed=42)
        assert len(pipeline._rir_cache) == 3
        variants = pipeline.augment_clip(speech_signal, factor=3)
        assert len(variants) == 3

    def test_rir_synthetic_fallback(self, speech_signal: np.ndarray) -> None:
        """Pipeline with p_rir>0 but no files should use synthetic RIRs."""
        cfg = AugmentConfig(p_rir=1.0, rir_files=[])
        pipeline = AugmentationPipeline(config=cfg, seed=42)
        assert len(pipeline._rir_cache) == 0
        variants = pipeline.augment_clip(speech_signal, factor=3)
        assert len(variants) == 3
        # At least one variant should differ from original
        # (synthetic RIR convolution changes the signal)
        any_different = False
        for v in variants:
            if not np.allclose(v[:len(speech_signal)], speech_signal[:len(v)], atol=1e-4):
                any_different = True
                break
        assert any_different

    def test_rir_disabled(self, speech_signal: np.ndarray) -> None:
        """Pipeline with p_rir=0 should not apply RIR."""
        cfg = AugmentConfig(
            p_rir=0.0,
            p_gain=0.0,
            p_time_stretch=0.0,
            p_pitch_shift=0.0,
            p_noise=0.0,
            p_time_shift=0.0,
        )
        pipeline = AugmentationPipeline(config=cfg, seed=42)
        variants = pipeline.augment_clip(speech_signal, factor=3)
        # With all augmentations disabled, variants should be copies
        for v in variants:
            np.testing.assert_array_equal(v, speech_signal)
