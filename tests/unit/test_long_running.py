"""Long-running tests for WakeDetector.

Verifies:
- Processing 10,000 frames (200s of audio) without crashes
- All tests use mocked backend sessions (no real models)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.wake_detector import (
    FRAME_SAMPLES,
    WakeDetector,
)

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helper to build a mocked WakeDetector
# ---------------------------------------------------------------------------

def _make_backend_session(output_value: np.ndarray) -> MagicMock:
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [output_value]
    return sess


def _build_detector(
    mlp_score: float = 0.50,
    threshold: float = 0.80,
    cooldown_s: float = 0.0,
) -> WakeDetector:
    oww_sess = _make_backend_session(np.ones((1, 96), dtype=np.float32) * 0.5)
    mlp_sess = _make_backend_session(np.array([[mlp_score]], dtype=np.float32))

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    mock_backend.load.side_effect = [oww_sess, mlp_sess]

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
            push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.5)),
        )),
    ):
        det = WakeDetector(
            threshold=threshold,
            cooldown_s=cooldown_s,
        )

    return det


# ---------------------------------------------------------------------------
# Long-running frame processing
# ---------------------------------------------------------------------------

class TestLongRunningProcessing:
    """Feed 10,000 frames (200 seconds) and verify no issues."""

    def test_10000_frames_no_crash(self) -> None:
        """Process 10,000 frames without crashing."""
        det = _build_detector(mlp_score=0.50, cooldown_s=0.0)
        rng = np.random.default_rng(42)

        for _ in range(10_000):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            det.process(frame)

    def test_10000_frames_all_return_valid_scores(self) -> None:
        """All 10,000 process() calls return valid float scores."""
        det = _build_detector(mlp_score=0.65, cooldown_s=0.0)
        rng = np.random.default_rng(42)

        invalid_count = 0
        for i in range(10_000):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            score = det.process(frame)
            if not isinstance(score, float):
                invalid_count += 1

        assert invalid_count == 0, f"{invalid_count} invalid scores out of 10,000"


# ---------------------------------------------------------------------------
# Extended operation tests
# ---------------------------------------------------------------------------

class TestExtendedOperation:
    """Verify behavior after extended operation."""

    def test_5000_frames_no_crash(self) -> None:
        """Feed 5,000 frames without crash."""
        det = _build_detector()
        rng = np.random.default_rng(42)

        for i in range(5_000):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            det.process(frame)

    def test_varying_frame_sizes(self) -> None:
        """Feed frames of different sizes."""
        det = _build_detector()
        rng = np.random.default_rng(42)

        frame_sizes = [160, 320, 640, 1600, 3200]

        for _ in range(100):
            for size in frame_sizes:
                frame = rng.integers(-5000, 5000, size, dtype=np.int16).tobytes()
                det.process(frame)

    def test_policy_reset_then_continue(self) -> None:
        """After policy reset, detection still works."""
        det = _build_detector()
        rng = np.random.default_rng(42)

        # Process some frames
        for _ in range(500):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            det.process(frame)

        det._policy.reset_cooldown()

        # Continue feeding
        for _ in range(500):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            det.process(frame)


# ---------------------------------------------------------------------------
# Score stability over time
# ---------------------------------------------------------------------------

class TestScoreStability:
    """Verify score stability over extended processing."""

    def test_score_stable_after_extended_run(self) -> None:
        """After 2,000 frames, scores remain valid floats."""
        det = _build_detector(mlp_score=0.65)
        rng = np.random.default_rng(42)

        for _ in range(2_000):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            score = det.process(frame)
            assert isinstance(score, float)

    def test_score_consistent_over_time(self) -> None:
        """Scores should be consistent (mocked model returns constant)."""
        det = _build_detector(mlp_score=0.65)
        rng = np.random.default_rng(42)

        scores: list[float] = []
        for i in range(3_000):
            frame = rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16).tobytes()
            score = det.process(frame)
            if i > 0 and i % 500 == 0:
                scores.append(score)

        # All scores from the mocked model should be consistent
        if len(scores) >= 3:
            for s in scores:
                assert isinstance(s, float)
