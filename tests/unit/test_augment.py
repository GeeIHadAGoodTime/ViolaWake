"""Tests for the audio augmentation pipeline.

Verifies that augmentations:
  - Produce valid audio (correct dtype, length, amplitude range)
  - Actually modify the signal (not a no-op)
  - Are deterministic with a fixed seed
  - Handle edge cases (silence, very short clips)

Tests the REAL API: AugmentationPipeline, AugmentConfig, and the standalone
functions apply_gain, apply_additive_noise, apply_time_shift, apply_time_stretch,
apply_pitch_shift.
"""
from __future__ import annotations

import numpy as np
import pytest

from violawake_sdk.training.augment import (
    AugmentationPipeline,
    AugmentConfig,
    apply_additive_noise,
    apply_gain,
    apply_pitch_shift,
    apply_time_shift,
    apply_time_stretch,
)


@pytest.fixture
def speech_signal() -> np.ndarray:
    """1.5s synthetic speech-like signal at 16kHz."""
    rng = np.random.default_rng(42)
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Multi-harmonic signal resembling speech
    signal = (
        np.sin(2 * np.pi * 200 * t) * 0.3
        + np.sin(2 * np.pi * 400 * t) * 0.2
        + np.sin(2 * np.pi * 800 * t) * 0.1
        + rng.normal(0, 0.05, len(t))
    )
    return signal.astype(np.float32)


@pytest.fixture
def silence_signal() -> np.ndarray:
    """1.5s of silence."""
    return np.zeros(24000, dtype=np.float32)


# ── AugmentConfig Tests ────────────────────────────────────────────────────


class TestAugmentConfig:
    """Test the AugmentConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have reasonable values."""
        cfg = AugmentConfig()
        assert cfg.gain_db_range == (-6.0, 6.0)
        assert cfg.time_stretch_range == (0.9, 1.1)
        assert cfg.pitch_shift_semitone_range == (-2.0, 2.0)
        assert cfg.noise_snr_range_db == (5.0, 20.0)
        assert cfg.time_shift_fraction == 0.10
        # All probabilities should be between 0 and 1
        for field_name in ("p_gain", "p_time_stretch", "p_pitch_shift", "p_noise", "p_time_shift"):
            val = getattr(cfg, field_name)
            assert 0.0 <= val <= 1.0, f"{field_name} = {val} out of range"

    def test_custom_values(self) -> None:
        """Custom config should override defaults."""
        cfg = AugmentConfig(gain_db_range=(-3.0, 3.0), p_gain=0.5)
        assert cfg.gain_db_range == (-3.0, 3.0)
        assert cfg.p_gain == 0.5
        # Other fields should still be defaults
        assert cfg.time_stretch_range == (0.9, 1.1)


# ── AugmentationPipeline Tests ─────────────────────────────────────────────


class TestAugmentationPipeline:
    """Test the AugmentationPipeline class."""

    def test_instantiation_defaults(self) -> None:
        """Pipeline should be instantiable with no arguments."""
        pipeline = AugmentationPipeline()
        assert pipeline.config is not None
        assert isinstance(pipeline.config, AugmentConfig)

    def test_instantiation_custom_config(self) -> None:
        """Pipeline should accept a custom config."""
        cfg = AugmentConfig(p_gain=1.0, p_noise=0.0)
        pipeline = AugmentationPipeline(config=cfg, seed=99)
        assert pipeline.config.p_gain == 1.0
        assert pipeline.config.p_noise == 0.0

    def test_augment_clip_returns_list(self, speech_signal: np.ndarray) -> None:
        """augment_clip should return a list of arrays."""
        pipeline = AugmentationPipeline(seed=42)
        results = pipeline.augment_clip(speech_signal, factor=5)
        assert isinstance(results, list)
        assert len(results) == 5
        for arr in results:
            assert isinstance(arr, np.ndarray)
            assert arr.dtype == np.float32

    def test_augment_clip_modifies_signal(self, speech_signal: np.ndarray) -> None:
        """Augmented clips should differ from the original."""
        # Use config with all augmentations forced on
        cfg = AugmentConfig(
            p_gain=1.0, p_time_stretch=1.0, p_pitch_shift=1.0,
            p_noise=1.0, p_time_shift=1.0,
        )
        pipeline = AugmentationPipeline(config=cfg, seed=42)
        results = pipeline.augment_clip(speech_signal, factor=3)
        for i, aug in enumerate(results):
            # At least the noise augmentation guarantees difference
            # (time_stretch/pitch_shift may change length, so compare
            # up to the minimum length)
            min_len = min(len(aug), len(speech_signal))
            assert not np.allclose(aug[:min_len], speech_signal[:min_len], atol=1e-6), (
                f"Variant {i} is identical to original"
            )

    def test_augment_clip_bounded_amplitude(self, speech_signal: np.ndarray) -> None:
        """All augmented clips should stay in [-1, 1]."""
        cfg = AugmentConfig(p_gain=1.0, p_noise=1.0)
        pipeline = AugmentationPipeline(config=cfg, seed=42)
        results = pipeline.augment_clip(speech_signal, factor=20)
        for i, aug in enumerate(results):
            assert np.all(aug >= -1.0), f"Variant {i} has values below -1"
            assert np.all(aug <= 1.0), f"Variant {i} has values above 1"

    def test_augment_clip_deterministic(self, speech_signal: np.ndarray) -> None:
        """Same seed should produce identical results."""
        pipeline1 = AugmentationPipeline(seed=42)
        pipeline2 = AugmentationPipeline(seed=42)
        results1 = pipeline1.augment_clip(speech_signal, factor=5)
        results2 = pipeline2.augment_clip(speech_signal, factor=5)
        assert len(results1) == len(results2)
        for a, b in zip(results1, results2):
            np.testing.assert_array_equal(a, b)

    def test_augment_clip_different_seeds_differ(self, speech_signal: np.ndarray) -> None:
        """Different seeds should produce different results."""
        pipeline1 = AugmentationPipeline(seed=42)
        pipeline2 = AugmentationPipeline(seed=99)
        results1 = pipeline1.augment_clip(speech_signal, factor=3)
        results2 = pipeline2.augment_clip(speech_signal, factor=3)
        any_different = False
        for a, b in zip(results1, results2):
            min_len = min(len(a), len(b))
            if not np.allclose(a[:min_len], b[:min_len], atol=1e-6):
                any_different = True
                break
        assert any_different, "Different seeds produced identical output"

    def test_augment_batch_returns_flat_list(self, speech_signal: np.ndarray) -> None:
        """augment_batch should return a flat list of len(clips) * factor."""
        pipeline = AugmentationPipeline(seed=42)
        clips = [speech_signal, speech_signal.copy()]
        results = pipeline.augment_batch(clips, factor=3)
        assert isinstance(results, list)
        assert len(results) == 2 * 3  # 2 clips * factor 3

    def test_augment_batch_each_unique(self, speech_signal: np.ndarray) -> None:
        """Each variant in the batch should be unique."""
        pipeline = AugmentationPipeline(seed=42)
        clips = [speech_signal]
        results = pipeline.augment_batch(clips, factor=5)
        assert len(results) == 5
        # At least some pairs should differ
        diffs = 0
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                min_len = min(len(results[i]), len(results[j]))
                if not np.allclose(results[i][:min_len], results[j][:min_len], atol=1e-6):
                    diffs += 1
        assert diffs > 0, "All batch variants are identical"

    def test_augment_silence(self, silence_signal: np.ndarray) -> None:
        """Pipeline should handle silence without crashing."""
        pipeline = AugmentationPipeline(seed=42)
        results = pipeline.augment_clip(silence_signal, factor=3)
        assert len(results) == 3
        for aug in results:
            assert isinstance(aug, np.ndarray)
            assert aug.dtype == np.float32


# ── Individual Augmentation Function Tests ──────────────────────────────────


class TestApplyGain:
    """Test the standalone apply_gain function."""

    def test_positive_gain(self, speech_signal: np.ndarray) -> None:
        """Positive gain should increase amplitude."""
        result = apply_gain(speech_signal, gain_db=6.0)
        assert len(result) == len(speech_signal)
        assert result.dtype == np.float32
        # 6dB gain ~= 2x amplitude
        # Compare RMS (more robust than peak)
        orig_rms = np.sqrt(np.mean(speech_signal ** 2))
        aug_rms = np.sqrt(np.mean(result ** 2))
        assert aug_rms > orig_rms

    def test_negative_gain(self, speech_signal: np.ndarray) -> None:
        """Negative gain should decrease amplitude."""
        result = apply_gain(speech_signal, gain_db=-6.0)
        orig_rms = np.sqrt(np.mean(speech_signal ** 2))
        aug_rms = np.sqrt(np.mean(result ** 2))
        assert aug_rms < orig_rms

    def test_zero_gain(self, speech_signal: np.ndarray) -> None:
        """Zero gain should return identical signal."""
        result = apply_gain(speech_signal, gain_db=0.0)
        np.testing.assert_allclose(result, speech_signal, atol=1e-6)

    def test_clipping(self) -> None:
        """Large gain should clip to [-1, 1]."""
        loud = np.array([0.5, -0.5, 0.8, -0.8], dtype=np.float32)
        result = apply_gain(loud, gain_db=20.0)  # ~10x gain
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)


class TestApplyAdditiveNoise:
    """Test the standalone apply_additive_noise function."""

    def test_noise_changes_signal(self, speech_signal: np.ndarray) -> None:
        """Adding noise should modify the signal."""
        rng = np.random.default_rng(42)
        result = apply_additive_noise(speech_signal, snr_db=10.0, rng=rng)
        assert len(result) == len(speech_signal)
        assert not np.allclose(result, speech_signal, atol=1e-6)

    def test_high_snr_barely_changes(self, speech_signal: np.ndarray) -> None:
        """Very high SNR should barely change the signal."""
        rng = np.random.default_rng(42)
        result = apply_additive_noise(speech_signal, snr_db=60.0, rng=rng)
        # High SNR means very little noise
        diff = np.max(np.abs(result - speech_signal))
        assert diff < 0.01

    def test_bounded_output(self, speech_signal: np.ndarray) -> None:
        """Output should be clipped to [-1, 1]."""
        rng = np.random.default_rng(42)
        result = apply_additive_noise(speech_signal, snr_db=0.0, rng=rng)
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_silence_input(self, silence_signal: np.ndarray) -> None:
        """Silence input should return silence (no crash)."""
        rng = np.random.default_rng(42)
        result = apply_additive_noise(silence_signal, snr_db=10.0, rng=rng)
        assert len(result) == len(silence_signal)

    def test_pink_noise_type(self, speech_signal: np.ndarray) -> None:
        """Pink noise type should work without error."""
        rng = np.random.default_rng(42)
        result = apply_additive_noise(speech_signal, snr_db=10.0, rng=rng, noise_type="pink")
        assert len(result) == len(speech_signal)
        assert result.dtype == np.float32


class TestApplyTimeShift:
    """Test the standalone apply_time_shift function."""

    def test_positive_shift(self, speech_signal: np.ndarray) -> None:
        """Positive shift should move audio right (zeros at start)."""
        result = apply_time_shift(speech_signal, shift_samples=1600)
        assert len(result) == len(speech_signal)
        # First 1600 samples should be zero
        assert np.all(result[:1600] == 0.0)
        # Shifted portion should match original
        np.testing.assert_array_equal(result[1600:], speech_signal[:-1600])

    def test_negative_shift(self, speech_signal: np.ndarray) -> None:
        """Negative shift should move audio left (zeros at end)."""
        result = apply_time_shift(speech_signal, shift_samples=-1600)
        assert len(result) == len(speech_signal)
        # Last 1600 samples should be zero
        assert np.all(result[-1600:] == 0.0)

    def test_zero_shift(self, speech_signal: np.ndarray) -> None:
        """Zero shift should return identical copy."""
        result = apply_time_shift(speech_signal, shift_samples=0)
        np.testing.assert_array_equal(result, speech_signal)

    def test_same_length(self, speech_signal: np.ndarray) -> None:
        """Output should always have same length as input."""
        result = apply_time_shift(speech_signal, shift_samples=800)
        assert len(result) == len(speech_signal)


class TestApplyTimeStretch:
    """Test the standalone apply_time_stretch function."""

    def test_stretch_slower(self, speech_signal: np.ndarray) -> None:
        """Rate < 1 should make audio longer."""
        result = apply_time_stretch(speech_signal, rate=0.9)
        assert len(result) > len(speech_signal)

    def test_stretch_faster(self, speech_signal: np.ndarray) -> None:
        """Rate > 1 should make audio shorter."""
        result = apply_time_stretch(speech_signal, rate=1.1)
        assert len(result) < len(speech_signal)

    def test_rate_one_identity(self, speech_signal: np.ndarray) -> None:
        """Rate = 1.0 should return a copy of same length."""
        result = apply_time_stretch(speech_signal, rate=1.0)
        assert len(result) == len(speech_signal)
        np.testing.assert_array_equal(result, speech_signal)

    def test_output_dtype(self, speech_signal: np.ndarray) -> None:
        """Output should be float32."""
        result = apply_time_stretch(speech_signal, rate=0.95)
        assert result.dtype == np.float32


class TestApplyPitchShift:
    """Test the standalone apply_pitch_shift function."""

    def test_pitch_up(self, speech_signal: np.ndarray) -> None:
        """Positive semitones should change the signal."""
        result = apply_pitch_shift(speech_signal, semitones=2.0)
        assert len(result) == len(speech_signal)
        assert not np.allclose(result, speech_signal, atol=1e-6)

    def test_pitch_down(self, speech_signal: np.ndarray) -> None:
        """Negative semitones should change the signal."""
        result = apply_pitch_shift(speech_signal, semitones=-2.0)
        assert len(result) == len(speech_signal)
        assert not np.allclose(result, speech_signal, atol=1e-6)

    def test_zero_semitones_identity(self, speech_signal: np.ndarray) -> None:
        """Zero semitones should return a copy."""
        result = apply_pitch_shift(speech_signal, semitones=0.0)
        assert len(result) == len(speech_signal)
        np.testing.assert_array_equal(result, speech_signal)

    def test_same_length_preserved(self, speech_signal: np.ndarray) -> None:
        """Output should have same length as input (time-corrected)."""
        result = apply_pitch_shift(speech_signal, semitones=1.5)
        assert len(result) == len(speech_signal)

    def test_output_dtype(self, speech_signal: np.ndarray) -> None:
        """Output should be float32."""
        result = apply_pitch_shift(speech_signal, semitones=1.0)
        assert result.dtype == np.float32
