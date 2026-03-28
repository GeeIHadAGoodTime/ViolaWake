"""H3: Stress tests for WakeDetector.

Tests:
- Rapid sequential detect() calls (1000 frames)
- Memory leak detection (process 10K frames, check RSS)
- Thread safety (concurrent detect() from multiple threads)
- Large audio chunks, tiny audio chunks, empty chunks
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.wake_detector import (
    FRAME_SAMPLES,
    WakeDecisionPolicy,
    WakeDetector,
    validate_audio_chunk,
)

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers (mocked backend pattern)
# ---------------------------------------------------------------------------

def _make_backend_session(output_value: np.ndarray) -> MagicMock:
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [output_value]
    return sess


def _create_detector(score: float = 0.50) -> WakeDetector:
    oww_sess = _make_backend_session(np.ones((1, 96), dtype=np.float32) * 0.1)
    mlp_sess = _make_backend_session(np.array([[score]], dtype=np.float32))

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    mock_backend.load.side_effect = [oww_sess, mlp_sess]

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
            push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.1)),
        )),
    ):
        return WakeDetector(
            model="viola_mlp_oww",
            threshold=0.80,
            cooldown_s=0.0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRapidSequentialCalls:
    """Stress test: rapid sequential detect() calls."""

    def test_1000_frames_no_crash(self) -> None:
        """Process 1000 frames rapidly without crashing."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)

        for i in range(1000):
            frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()
            result = detector.detect(frame)
            assert isinstance(result, bool)

    def test_1000_frames_timing(self) -> None:
        """1000 frames should complete in reasonable time (< 30s total)."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)
        frames = [
            (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()
            for _ in range(1000)
        ]

        t0 = time.perf_counter()
        for frame in frames:
            detector.detect(frame)
        elapsed = time.perf_counter() - t0

        # 1000 frames at 20ms each = 20s real time. Processing should be much faster.
        assert elapsed < 30.0, f"1000 frames took {elapsed:.2f}s (expected < 30s)"

    def test_alternating_detect_and_policy_reset(self) -> None:
        """Alternating detect+policy_reset should not corrupt state."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)

        for _ in range(500):
            frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()
            detector.detect(frame)
            detector._policy.reset_cooldown()


class TestMemoryLeak:
    """Stress test: memory stability over many frames."""

    def test_10k_frames_memory_stable(self) -> None:
        """Process 10K frames and verify RSS does not grow unboundedly."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed -- skipping memory test")

        import os
        process = psutil.Process(os.getpid())

        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)

        # Warmup
        for _ in range(100):
            frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()
            detector.detect(frame)

        baseline_rss = process.memory_info().rss

        # Process 10K frames
        for _ in range(10_000):
            frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()
            detector.detect(frame)

        final_rss = process.memory_info().rss
        growth_mb = (final_rss - baseline_rss) / (1024 * 1024)

        assert growth_mb < 50.0, (
            f"Memory grew by {growth_mb:.1f} MB after 10K frames (expected < 50MB)"
        )


class TestThreadSafety:
    """Stress test: concurrent detect() from multiple threads."""

    def test_concurrent_detect(self) -> None:
        """Multiple threads calling detect() should not crash or corrupt state."""
        detector = _create_detector(score=0.50)
        errors: list[Exception] = []

        def worker(thread_id: int) -> None:
            local_rng = np.random.default_rng(thread_id)
            try:
                for _ in range(200):
                    frame = (
                        local_rng.standard_normal(FRAME_SAMPLES) * 3000
                    ).astype(np.int16).tobytes()
                    result = detector.detect(frame)
                    assert isinstance(result, bool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_detect_and_policy_reset(self) -> None:
        """One thread resetting policy while others detect should not crash."""
        detector = _create_detector(score=0.50)
        errors: list[Exception] = []
        stop_event = threading.Event()

        def detect_worker() -> None:
            rng = np.random.default_rng(99)
            try:
                while not stop_event.is_set():
                    frame = (
                        rng.standard_normal(FRAME_SAMPLES) * 3000
                    ).astype(np.int16).tobytes()
                    detector.detect(frame)
            except Exception as e:
                errors.append(e)

        def reset_worker() -> None:
            try:
                for _ in range(50):
                    time.sleep(0.01)
                    detector._policy.reset_cooldown()
            except Exception as e:
                errors.append(e)

        detect_threads = [threading.Thread(target=detect_worker) for _ in range(3)]
        reset_thread = threading.Thread(target=reset_worker)

        for t in detect_threads:
            t.start()
        reset_thread.start()

        reset_thread.join(timeout=10)
        stop_event.set()
        for t in detect_threads:
            t.join(timeout=5)

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestChunkSizes:
    """Stress test: various chunk sizes."""

    def test_large_chunk(self) -> None:
        """A chunk at the max allowed size (10x frame) should work."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)
        large = (rng.standard_normal(FRAME_SAMPLES * 10) * 3000).astype(np.int16).tobytes()
        result = detector.detect(large)
        assert isinstance(result, bool)

    def test_tiny_chunk(self) -> None:
        """A single-sample chunk should not crash."""
        detector = _create_detector(score=0.50)
        tiny = np.array([1000], dtype=np.int16).tobytes()
        result = detector.detect(tiny)
        assert isinstance(result, bool)

    def test_single_sample_ndarray(self) -> None:
        """A 1-element ndarray should not crash."""
        detector = _create_detector(score=0.50)
        tiny = np.array([0.5], dtype=np.float32)
        result = detector.detect(tiny)
        assert isinstance(result, bool)

    def test_empty_chunk_raises(self) -> None:
        """An empty chunk should raise ValueError."""
        detector = _create_detector(score=0.50)
        with pytest.raises(ValueError, match="empty"):
            detector.detect(b"")

    def test_empty_ndarray_raises(self) -> None:
        """An empty ndarray should raise ValueError."""
        detector = _create_detector(score=0.50)
        with pytest.raises(ValueError, match="empty"):
            detector.detect(np.array([], dtype=np.float32))

    def test_varied_chunk_sizes(self) -> None:
        """Process frames of varying sizes without crash."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)

        sizes = [1, 10, 100, 320, 640, 1600, 3200]
        for size in sizes:
            frame = (rng.standard_normal(size) * 3000).astype(np.int16).tobytes()
            result = detector.detect(frame)
            assert isinstance(result, bool), f"Failed for chunk size {size}"


class TestDecisionPolicyStress:
    """Stress test the WakeDecisionPolicy independently."""

    def test_100k_evaluations(self) -> None:
        """100K rapid evaluations should not leak or crash."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        rng = np.random.default_rng(42)

        for _ in range(100_000):
            score = float(rng.random())
            rms = float(rng.random() * 1000)
            result = policy.evaluate(score=score, rms=rms)
            assert isinstance(result, bool)

    def test_rapid_cooldown_transitions(self) -> None:
        """Rapid transitions in/out of cooldown should not corrupt state."""
        policy = WakeDecisionPolicy(threshold=0.50, cooldown_s=0.001)

        for _ in range(10_000):
            result = policy.evaluate(score=0.90, rms=500.0)
            assert isinstance(result, bool)
            if result:
                # Very short cooldown -- next call may or may not be in cooldown
                pass
