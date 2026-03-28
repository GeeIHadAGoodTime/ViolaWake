"""Unit tests for WakeDetector.process() and .detect() with mocked ONNX.

Tests the core detection pipeline without requiring real model files or
ONNX Runtime -- all sessions (OWW backbone, MLP) are mocked via the backend.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.wake_detector import FRAME_SAMPLES, WakeDetector


# ---------------------------------------------------------------------------
# Helper to build a WakeDetector with fully mocked backend
# ---------------------------------------------------------------------------

def _make_backend_session(output_value: np.ndarray) -> MagicMock:
    """Return a mock BackendSession that always returns *output_value*."""
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    inp.shape = [1, 96]
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [output_value]
    return sess


def _make_fake_backbone() -> MagicMock:
    backbone = MagicMock()
    backbone.push_audio.return_value = (True, np.ones(96, dtype=np.float32) * 0.5)
    backbone.last_embedding = np.ones(96, dtype=np.float32) * 0.5
    return backbone


def _build_detector(
    mlp_score: float = 0.95,
    threshold: float = 0.80,
    cooldown_s: float = 0.0,
) -> WakeDetector:
    """Build a WakeDetector with mocked backend sessions.

    The backbone returns a 96-dim embedding and the MLP returns *mlp_score*.
    """
    mlp_sess = _make_backend_session(np.array([[mlp_score]], dtype=np.float32))

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    mock_backend.load.return_value = mlp_sess

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=_make_fake_backbone()),
    ):
        det = WakeDetector(
            threshold=threshold,
            cooldown_s=cooldown_s,
        )

    return det


# ---------------------------------------------------------------------------
# Score computation from MLP
# ---------------------------------------------------------------------------

class TestScoreComputation:
    """Verify MLP score computation."""

    def test_returns_mlp_score(self) -> None:
        """process() returns the MLP output score."""
        det = _build_detector(mlp_score=0.73)
        frame = np.random.default_rng(42).integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16)
        score = det.process(frame.tobytes())
        assert abs(score - 0.73) < 0.01

    def test_high_score(self) -> None:
        """High MLP score is returned correctly."""
        det = _build_detector(mlp_score=0.99)
        frame = np.random.default_rng(42).integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16)
        score = det.process(frame.tobytes())
        assert abs(score - 0.99) < 0.01

    def test_low_score(self) -> None:
        """Low MLP score is returned correctly."""
        det = _build_detector(mlp_score=0.02)
        frame = np.random.default_rng(42).integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16)
        score = det.process(frame.tobytes())
        assert abs(score - 0.02) < 0.01


# ---------------------------------------------------------------------------
# detect() applies decision policy on top of process()
# ---------------------------------------------------------------------------

class TestDetectAppliesPolicy:
    """detect() wraps process() with the 4-gate decision policy."""

    def test_detect_true_on_high_score(self, loud_noise_frame: bytes) -> None:
        """detect() returns True when score > threshold and RMS is high."""
        det = _build_detector(mlp_score=0.95, cooldown_s=0.0)
        result = det.detect(loud_noise_frame)
        assert isinstance(result, bool)

    def test_detect_false_on_low_score(self, loud_noise_frame: bytes) -> None:
        """detect() returns False when score < threshold."""
        det = _build_detector(mlp_score=0.10, cooldown_s=0.0)
        result = det.detect(loud_noise_frame)
        assert result is False

    def test_detect_false_on_silent_frame(self, silent_frame: bytes) -> None:
        """detect() returns False on silence (RMS gate)."""
        det = _build_detector(mlp_score=0.95, cooldown_s=0.0)
        result = det.detect(silent_frame)
        assert result is False

    def test_detect_false_when_playing(self, loud_noise_frame: bytes) -> None:
        """detect() returns False when is_playing=True (playback gate)."""
        det = _build_detector(mlp_score=0.95, cooldown_s=0.0)
        result = det.detect(loud_noise_frame, is_playing=True)
        assert result is False

    def test_detect_float32_not_rejected_by_rms_gate(self) -> None:
        """Float32 audio in [-1, 1] must NOT be silently rejected by Gate 1.

        Regression test for critical bug: RMS of float32 input in [-1, 1]
        is ~0.0-0.7, always below rms_floor=1.0 (calibrated for int16).
        The fix scales float32 RMS to int16 range before comparison.
        """
        det = _build_detector(mlp_score=0.95, cooldown_s=0.0)
        # Typical speech-level float32 audio (RMS ~0.3 in float32 scale)
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(FRAME_SAMPLES) * 0.3).astype(np.float32)
        result = det.detect(audio)
        # With score=0.95 (above threshold) and non-silent audio, detect
        # must return True.  Before the fix, this always returned False.
        assert result is True

    def test_detect_int16_bytes_and_float32_agree(self) -> None:
        """Same audio as bytes (int16) and float32 ndarray should both detect."""
        det_bytes = _build_detector(mlp_score=0.95, cooldown_s=0.0)
        det_float = _build_detector(mlp_score=0.95, cooldown_s=0.0)

        # Create int16 audio with speech-level amplitude
        pcm_int16 = (np.sin(np.linspace(0, 2 * np.pi * 440, FRAME_SAMPLES)) * 10000).astype(
            np.int16
        )
        pcm_float32 = (pcm_int16 / 32768.0).astype(np.float32)

        result_bytes = det_bytes.detect(pcm_int16.tobytes())
        result_float = det_float.detect(pcm_float32)
        assert result_bytes == result_float


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    """Verify that different input formats are handled."""

    def test_float32_input(self) -> None:
        """Float32 input should be accepted."""
        det = _build_detector()
        audio = np.array([0.5, -0.5, 1.0, -1.0] * 80, dtype=np.float32)
        score = det.process(audio)
        assert isinstance(score, float)

    def test_int16_input(self) -> None:
        """Int16 input should be accepted."""
        det = _build_detector()
        audio = np.array([1000, -1000, 32000, -32000] * 80, dtype=np.int16)
        score = det.process(audio)
        assert isinstance(score, float)

    def test_bytes_input(self) -> None:
        """Bytes input is converted from int16 PCM to float32."""
        det = _build_detector()
        pcm_int16 = np.array([500, -500, 1000, -1000] * 80, dtype=np.int16)
        score = det.process(pcm_int16.tobytes())
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Repeated process calls
# ---------------------------------------------------------------------------

class TestRepeatedProcessCalls:
    """Verify stability over many sequential calls."""

    def test_hundred_frames_no_crash(self) -> None:
        """100 consecutive process() calls should not crash."""
        det = _build_detector()
        frame = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
        for _ in range(100):
            score = det.process(frame)
            assert isinstance(score, float)


class TestReset:
    """Verify reset() clears detector state."""

    def test_reset_clears_score_history(self, loud_noise_frame: bytes) -> None:
        det = _build_detector(mlp_score=0.95, cooldown_s=60.0)

        det.detect(loud_noise_frame)
        assert det.last_scores
        assert det.get_confidence().raw_score > 0.0

        det.reset()

        assert det.last_scores == ()
        confidence = det.get_confidence()
        assert confidence.raw_score == 0.0
        assert confidence.confirm_count == 0
