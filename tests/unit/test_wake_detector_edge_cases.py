"""Edge case and corrupted input tests for WakeDetector.

Tests that process() and detect() handle degenerate inputs gracefully:
NaN values, empty arrays, wrong dtypes, very short/long audio, zero-length bytes.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.wake_detector import FRAME_SAMPLES, WakeDetector


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _make_backend_session(output_value: np.ndarray) -> MagicMock:
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
) -> WakeDetector:
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
        det = WakeDetector(threshold=0.80, cooldown_s=0.0)

    return det


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNInput:
    """process() with NaN values in audio should not crash."""

    def test_nan_float32_array(self) -> None:
        det = _build_detector()
        audio = np.full(FRAME_SAMPLES, np.nan, dtype=np.float32)
        # Must not raise -- score may be garbage but no crash
        score = det.process(audio)
        assert isinstance(score, float)

    def test_partial_nan(self) -> None:
        det = _build_detector()
        audio = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        audio[0] = np.nan
        audio[100] = np.nan
        score = det.process(audio)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """process() with empty arrays."""

    def test_empty_float32(self) -> None:
        det = _build_detector()
        audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            det.process(audio)

    def test_empty_int16(self) -> None:
        det = _build_detector()
        audio = np.array([], dtype=np.int16)
        with pytest.raises(ValueError, match="empty"):
            det.process(audio)

    def test_empty_bytes(self) -> None:
        det = _build_detector()
        with pytest.raises(ValueError, match="empty"):
            det.process(b"")


# ---------------------------------------------------------------------------
# Various dtypes
# ---------------------------------------------------------------------------

class TestVariousDtypes:
    """process() with various dtypes."""

    def test_float64(self) -> None:
        det = _build_detector()
        audio = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        score = det.process(audio)
        assert isinstance(score, float)

    def test_int16(self) -> None:
        det = _build_detector()
        audio = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        score = det.process(audio)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Extreme lengths
# ---------------------------------------------------------------------------

class TestExtremeLengths:
    """process() with very short or very long audio."""

    def test_single_sample(self) -> None:
        det = _build_detector()
        audio = np.array([1000], dtype=np.int16)
        score = det.process(audio)
        assert isinstance(score, float)

    def test_ten_frames_is_still_allowed(self) -> None:
        det = _build_detector()
        audio = np.zeros(FRAME_SAMPLES * 10, dtype=np.int16)
        score = det.process(audio)
        assert isinstance(score, float)

    def test_more_than_ten_frames_raises(self) -> None:
        """Frames larger than 10x the expected size should be rejected."""
        det = _build_detector()
        audio = np.zeros(FRAME_SAMPLES * 10 + 1, dtype=np.int16)
        with pytest.raises(ValueError, match="expected 320 samples"):
            det.process(audio)

    def test_exactly_one_frame(self) -> None:
        """Exactly one 20ms frame."""
        det = _build_detector()
        audio = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        score = det.process(audio.tobytes())
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# detect() edge cases
# ---------------------------------------------------------------------------

class TestDetectEdgeCases:
    """detect() with degenerate inputs."""

    def test_zero_length_bytes_raises(self) -> None:
        """detect() with empty bytes should raise ValueError."""
        det = _build_detector()
        with pytest.raises(ValueError, match="empty"):
            det.detect(b"")

    def test_detect_with_nan_audio(self) -> None:
        """detect() with NaN audio should not crash."""
        det = _build_detector()
        audio = np.full(FRAME_SAMPLES, np.nan, dtype=np.float32)
        result = det.detect(audio)
        assert isinstance(result, bool)

    def test_detect_single_byte_pair(self) -> None:
        """detect() with 2 bytes (1 int16 sample) and below-threshold score."""
        det = _build_detector(mlp_score=0.10)
        result = det.detect(b"\x00\x01")
        assert result is False


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
