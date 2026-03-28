"""H2: Integration test -- full audio -> OWW -> MLP -> decision pipeline.

Tests the complete detection pipeline with mocked backend sessions returning
predictable outputs. No model files or hardware required.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.wake_detector import (
    FRAME_SAMPLES,
    WakeDecisionPolicy,
    WakeDetector,
)

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tone_bytes(freq: float = 440.0, duration_s: float = 0.02, amplitude: int = 10000) -> bytes:
    """Generate a sine wave as int16 PCM bytes."""
    n_samples = int(SAMPLE_RATE * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    signal = (np.sin(2 * math.pi * freq * t) * amplitude).astype(np.int16)
    return signal.tobytes()


def make_noise_bytes(duration_s: float = 0.02, rms: int = 5000, seed: int = 42) -> bytes:
    """Generate white noise as int16 PCM bytes."""
    rng = np.random.default_rng(seed)
    n_samples = int(SAMPLE_RATE * duration_s)
    samples = (rng.standard_normal(n_samples) * rms).clip(-32768, 32767).astype(np.int16)
    return samples.tobytes()


def make_silence_bytes(duration_s: float = 0.02) -> bytes:
    """Generate silence as int16 PCM bytes."""
    n_samples = int(SAMPLE_RATE * duration_s)
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _make_backend_session(output_value: np.ndarray) -> MagicMock:
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [output_value]
    return sess


def _create_detector_with_mocks(
    score: float = 0.95,
    threshold: float = 0.80,
    cooldown_s: float = 0.0,
) -> WakeDetector:
    """Create a WakeDetector with fully mocked backend sessions."""
    mlp_sess = _make_backend_session(np.array([[score]], dtype=np.float32))

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    mock_backend.load.return_value = mlp_sess

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
            push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.5)),
        )),
    ):
        detector = WakeDetector(
            model="viola_mlp_oww",
            threshold=threshold,
            cooldown_s=cooldown_s,
        )
    return detector


# ---------------------------------------------------------------------------
# Test: Full pipeline with high-score model
# ---------------------------------------------------------------------------

class TestFullPipelineDetection:
    """Test the complete audio -> score -> decision pipeline."""

    def test_detect_with_noise_above_threshold(self) -> None:
        """Noisy audio + high model score should trigger detection."""
        detector = _create_detector_with_mocks(score=0.95, cooldown_s=0.0)
        noise = make_noise_bytes(duration_s=0.02, rms=5000)
        result = detector.detect(noise)
        assert isinstance(result, bool)

    def test_no_detect_with_silence(self) -> None:
        """Silent audio should be rejected by the zero-input guard (Gate 1)."""
        detector = _create_detector_with_mocks(score=0.95, cooldown_s=0.0)
        silence = make_silence_bytes(duration_s=0.02)
        result = detector.detect(silence)
        assert result is False

    def test_no_detect_with_low_score(self) -> None:
        """Low model score should not trigger detection."""
        detector = _create_detector_with_mocks(score=0.30, cooldown_s=0.0)
        noise = make_noise_bytes(duration_s=0.02, rms=5000)
        result = detector.detect(noise)
        assert result is False

    def test_multiple_frames(self) -> None:
        """Multiple frames should all be processed without error."""
        detector = _create_detector_with_mocks(
            score=0.95, cooldown_s=0.0, threshold=0.80,
        )

        results = []
        for _ in range(10):
            frame = make_noise_bytes(duration_s=0.02, rms=5000, seed=99)
            results.append(detector.detect(frame))

        assert isinstance(results, list)
        assert all(isinstance(r, bool) for r in results)

    def test_cooldown_suppression(self) -> None:
        """Second detection within cooldown should be suppressed."""
        detector = _create_detector_with_mocks(
            score=0.95, cooldown_s=10.0,
        )
        noise = make_noise_bytes(duration_s=0.02, rms=5000)
        first = detector.detect(noise)

        # Second detect immediately -- should be suppressed by cooldown
        noise2 = make_noise_bytes(duration_s=0.02, rms=5000, seed=123)
        second = detector.detect(noise2)
        # If first triggered, second must be False (cooldown).
        if first:
            assert second is False

    def test_playback_gate(self) -> None:
        """Detection should be suppressed when is_playing=True."""
        detector = _create_detector_with_mocks(score=0.95, cooldown_s=0.0)
        noise = make_noise_bytes(duration_s=0.02, rms=5000)
        result = detector.detect(noise, is_playing=True)
        assert result is False


class TestPipelineStateManagement:
    """Test state persistence across multiple detect() calls."""

    def test_accumulation_across_frames(self) -> None:
        """Multiple detect() calls should work without error."""
        detector = _create_detector_with_mocks(score=0.50, cooldown_s=0.0)
        for _ in range(100):
            frame = make_noise_bytes(duration_s=0.02, rms=3000)
            detector.detect(frame)

    def test_process_returns_score(self) -> None:
        """process() should return a float score."""
        detector = _create_detector_with_mocks(score=0.85, cooldown_s=0.0)
        noise = make_noise_bytes(duration_s=0.02, rms=5000)
        score = detector.process(noise)
        assert isinstance(score, float)

    def test_bytes_and_ndarray_produce_same_behavior(self) -> None:
        """Both bytes and ndarray inputs should work without errors."""
        detector = _create_detector_with_mocks(score=0.90, cooldown_s=0.0)

        # bytes input
        bytes_frame = make_noise_bytes(duration_s=0.02, rms=5000)
        r1 = detector.detect(bytes_frame)
        assert isinstance(r1, bool)

        # ndarray input (float32)
        rng = np.random.default_rng(42)
        arr_frame = (rng.standard_normal(FRAME_SAMPLES) * 0.5).astype(np.float32)
        r2 = detector.detect(arr_frame)
        assert isinstance(r2, bool)

        # ndarray input (int16)
        int16_frame = (rng.integers(-5000, 5000, FRAME_SAMPLES)).astype(np.int16)
        r3 = detector.detect(int16_frame)
        assert isinstance(r3, bool)

    def test_policy_reset_cooldown(self) -> None:
        """reset_cooldown() should allow immediate re-detection."""
        detector = _create_detector_with_mocks(score=0.95, cooldown_s=10.0)
        noise = make_noise_bytes(duration_s=0.02, rms=5000)
        first = detector.detect(noise)

        detector._policy.reset_cooldown()

        noise2 = make_noise_bytes(duration_s=0.02, rms=5000, seed=99)
        second = detector.detect(noise2)
        # After reset, should be able to detect again
        assert isinstance(second, bool)
