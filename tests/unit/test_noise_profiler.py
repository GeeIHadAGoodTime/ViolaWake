"""Unit tests for K4: Noise robustness metrics and adaptive threshold.

Tests the NoiseProfiler class and NoiseProfile dataclass without requiring
model files or hardware.
"""

from __future__ import annotations

import numpy as np
import pytest

from violawake_sdk.noise_profiler import (
    DEFAULT_MAX_THRESHOLD,
    DEFAULT_MIN_THRESHOLD,
    DEFAULT_NOISE_WINDOW_S,
    NoiseProfile,
    NoiseProfiler,
)


class TestNoiseProfilerBasics:
    """Test basic NoiseProfiler operations."""

    def test_initial_state(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80)
        assert profiler.base_threshold == 0.80
        assert profiler.noise_floor == 0.0

    def test_update_returns_threshold(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80)
        frame = np.random.randn(320).astype(np.float32) * 100
        result = profiler.update(frame)
        assert isinstance(result, float)
        assert DEFAULT_MIN_THRESHOLD <= result <= DEFAULT_MAX_THRESHOLD

    def test_update_with_silence(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80)
        # Feed many silence frames
        silence = np.zeros(320, dtype=np.float32)
        for _ in range(20):
            profiler.update(silence)
        assert profiler.noise_floor < 0.001

    def test_noise_floor_estimation(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80, noise_window_s=1.0)
        rng = np.random.default_rng(42)
        # Feed low-level noise frames
        for _ in range(100):
            noise = (rng.standard_normal(320) * 10).astype(np.float32)
            profiler.update(noise)
        # Noise floor should be non-zero but small
        assert profiler.noise_floor > 0.0
        assert profiler.noise_floor < 100.0


class TestAdaptiveThreshold:
    """Test threshold adaptation under different SNR conditions."""

    def test_returns_base_with_insufficient_data(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80)
        frame = np.random.randn(320).astype(np.float32) * 100
        # Fewer than 10 frames -> should return base
        for _ in range(5):
            result = profiler.update(frame)
        assert result == 0.80

    def test_high_snr_lowers_threshold(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80, snr_boost_db=6.0)
        rng = np.random.default_rng(42)
        # Build noise history with low-energy frames
        for _ in range(50):
            quiet = (rng.standard_normal(320) * 1.0).astype(np.float32)
            profiler.update(quiet)
        # Now send a loud frame (high SNR)
        loud = (rng.standard_normal(320) * 1000.0).astype(np.float32)
        adapted = profiler.update(loud)
        # Should lower threshold
        assert adapted < 0.80

    def test_low_snr_raises_threshold(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80, snr_penalty_db=3.0)
        rng = np.random.default_rng(42)
        # Build noise history with moderate-energy frames
        for _ in range(50):
            moderate = (rng.standard_normal(320) * 100.0).astype(np.float32)
            profiler.update(moderate)
        # Now send a frame close to noise level (low SNR)
        similar = (rng.standard_normal(320) * 100.0).astype(np.float32)
        adapted = profiler.update(similar)
        # Should be at or above base (noise ~ signal)
        assert adapted >= 0.80 - 0.01  # allow tiny floating point tolerance

    def test_threshold_clamped_to_min(self) -> None:
        profiler = NoiseProfiler(
            base_threshold=0.60, min_threshold=0.60, max_threshold=0.95,
        )
        rng = np.random.default_rng(42)
        for _ in range(50):
            quiet = (rng.standard_normal(320) * 0.001).astype(np.float32)
            profiler.update(quiet)
        loud = (rng.standard_normal(320) * 100000).astype(np.float32)
        adapted = profiler.update(loud)
        assert adapted >= 0.60

    def test_threshold_clamped_to_max(self) -> None:
        profiler = NoiseProfiler(
            base_threshold=0.95, min_threshold=0.60, max_threshold=0.95,
        )
        rng = np.random.default_rng(42)
        for _ in range(100):
            noisy = (rng.standard_normal(320) * 100).astype(np.float32)
            adapted = profiler.update(noisy)
        assert adapted <= 0.95


class TestNoiseProfile:
    """Test NoiseProfile snapshot."""

    def test_get_profile(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80)
        rng = np.random.default_rng(42)
        for _ in range(20):
            frame = (rng.standard_normal(320) * 50).astype(np.float32)
            profiler.update(frame)
        profile = profiler.get_profile()
        assert isinstance(profile, NoiseProfile)
        assert profile.base_threshold == 0.80
        assert profile.noise_rms >= 0.0
        assert profile.signal_rms >= 0.0
        assert isinstance(profile.snr_db, float)
        assert DEFAULT_MIN_THRESHOLD <= profile.adjusted_threshold <= DEFAULT_MAX_THRESHOLD

    def test_frozen_dataclass(self) -> None:
        profile = NoiseProfile(
            noise_rms=10.0, signal_rms=50.0, snr_db=14.0,
            adjusted_threshold=0.75, base_threshold=0.80,
        )
        with pytest.raises(AttributeError):
            profile.noise_rms = 20.0  # type: ignore[misc]


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self) -> None:
        profiler = NoiseProfiler(base_threshold=0.80)
        rng = np.random.default_rng(42)
        for _ in range(50):
            frame = (rng.standard_normal(320) * 50).astype(np.float32)
            profiler.update(frame)
        assert profiler.noise_floor > 0.0
        profiler.reset()
        assert profiler.noise_floor == 0.0
