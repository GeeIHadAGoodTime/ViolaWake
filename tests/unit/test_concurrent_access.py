"""Concurrent access tests for WakeDetector.

Verifies thread safety:
- Multiple threads calling detect() on the same WakeDetector
- Multiple threads calling process() simultaneously
- No crashes, no data corruption
"""

from __future__ import annotations

import threading
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
# Concurrent detect() calls
# ---------------------------------------------------------------------------

class TestConcurrentDetect:
    """Multiple threads calling detect() on the same WakeDetector."""

    def test_concurrent_detect_no_crash(self) -> None:
        """4 threads calling detect() 100+ times each must not crash."""
        det = _build_detector(mlp_score=0.50, cooldown_s=0.0)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                local_rng = np.random.default_rng(thread_id)
                for _ in range(100):
                    frame = local_rng.integers(-10000, 10000, FRAME_SAMPLES, dtype=np.int16)
                    result = det.detect(frame.tobytes())
                    assert isinstance(result, bool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Errors in threads: {errors}"

    def test_concurrent_detect_results_are_bools(self) -> None:
        """All detect() results must be bool, even under contention."""
        det = _build_detector(mlp_score=0.90, cooldown_s=0.0)
        results: list[bool] = []
        lock = threading.Lock()

        def worker(thread_id: int) -> None:
            local_rng = np.random.default_rng(thread_id + 100)
            for _ in range(50):
                frame = local_rng.integers(-10000, 10000, FRAME_SAMPLES, dtype=np.int16)
                r = det.detect(frame.tobytes())
                with lock:
                    results.append(r)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(results) == 200
        assert all(isinstance(r, bool) for r in results)


# ---------------------------------------------------------------------------
# Concurrent process() calls
# ---------------------------------------------------------------------------

class TestConcurrentProcess:
    """Multiple threads calling process() simultaneously."""

    def test_concurrent_process_no_crash(self) -> None:
        """4 threads calling process() 100+ times each must not crash."""
        det = _build_detector(mlp_score=0.50)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            try:
                local_rng = np.random.default_rng(thread_id + 200)
                for _ in range(100):
                    frame = local_rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16)
                    score = det.process(frame.tobytes())
                    assert isinstance(score, float)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Errors in threads: {errors}"

    def test_concurrent_process_scores_valid(self) -> None:
        """All process() return values must be valid floats."""
        det = _build_detector(mlp_score=0.73)
        scores: list[float] = []
        lock = threading.Lock()

        def worker(thread_id: int) -> None:
            local_rng = np.random.default_rng(thread_id + 300)
            for _ in range(50):
                frame = local_rng.integers(-5000, 5000, FRAME_SAMPLES, dtype=np.int16)
                s = det.process(frame.tobytes())
                with lock:
                    scores.append(s)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(scores) == 200
        for s in scores:
            assert isinstance(s, float)
