"""H7: Performance regression tests for WakeDetector.

Tests:
- detect() latency < 5ms per frame (with mocked backend)
- Memory usage < 50MB RSS
- Uses pytest-benchmark if available, falls back to time.perf_counter
"""

from __future__ import annotations

import time
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
# Latency tests
# ---------------------------------------------------------------------------

class TestDetectLatency:
    """Verify detect() latency stays under 5ms per frame (mocked inference)."""

    def _measure_detect_latency(self, n_frames: int = 500) -> list[float]:
        """Return per-frame latencies in ms."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)
        frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()

        # Warmup
        for _ in range(50):
            detector.detect(frame)

        latencies = []
        for _ in range(n_frames):
            t0 = time.perf_counter()
            detector.detect(frame)
            latencies.append((time.perf_counter() - t0) * 1000)

        return latencies

    def test_detect_p50_under_5ms(self) -> None:
        """Median detect() latency should be < 5ms per frame."""
        latencies = self._measure_detect_latency(500)
        p50 = float(np.percentile(latencies, 50))
        assert p50 < 5.0, f"p50 latency = {p50:.2f}ms (expected < 5ms)"

    def test_detect_p95_under_10ms(self) -> None:
        """95th percentile detect() latency should be < 10ms."""
        latencies = self._measure_detect_latency(500)
        p95 = float(np.percentile(latencies, 95))
        assert p95 < 10.0, f"p95 latency = {p95:.2f}ms (expected < 10ms)"

    def test_detect_mean_under_5ms(self) -> None:
        """Mean detect() latency should be < 5ms."""
        latencies = self._measure_detect_latency(500)
        mean = float(np.mean(latencies))
        assert mean < 5.0, f"mean latency = {mean:.2f}ms (expected < 5ms)"


class TestDecisionPolicyLatency:
    """Verify WakeDecisionPolicy.evaluate() is fast."""

    def test_evaluate_under_1us(self) -> None:
        """evaluate() should take < 0.1ms (100us) per call."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        n = 10_000

        t0 = time.perf_counter()
        for _ in range(n):
            policy.evaluate(score=0.50, rms=500.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        per_call_ms = elapsed_ms / n
        assert per_call_ms < 0.1, f"evaluate() = {per_call_ms:.4f}ms/call (expected < 0.1ms)"


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemoryUsage:
    """Verify memory usage stays bounded."""

    def test_detector_rss_under_50mb(self) -> None:
        """WakeDetector + 1K frames should use < 50MB RSS above baseline."""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed -- skipping RSS test")

        import os
        process = psutil.Process(os.getpid())
        baseline_rss = process.memory_info().rss

        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)

        for _ in range(1000):
            frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()
            detector.detect(frame)

        final_rss = process.memory_info().rss
        growth_mb = (final_rss - baseline_rss) / (1024 * 1024)

        assert growth_mb < 50.0, f"RSS grew by {growth_mb:.1f} MB (expected < 50MB)"


# ---------------------------------------------------------------------------
# pytest-benchmark integration (if available)
# ---------------------------------------------------------------------------

class TestBenchmarkIntegration:
    """Use pytest-benchmark if available for more precise measurements."""

    def test_detect_benchmark(self, benchmark) -> None:
        """Benchmark detect() with pytest-benchmark."""
        try:
            # Check if benchmark fixture is real (not a no-op)
            if not hasattr(benchmark, "pedantic"):
                pytest.skip("pytest-benchmark not available")
        except Exception:
            pytest.skip("pytest-benchmark not available")

        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(42)
        frame = (rng.standard_normal(FRAME_SAMPLES) * 3000).astype(np.int16).tobytes()

        # Warmup
        for _ in range(50):
            detector.detect(frame)

        result = benchmark(detector.detect, frame)
        assert isinstance(result, bool)

    def test_evaluate_benchmark(self, benchmark) -> None:
        """Benchmark WakeDecisionPolicy.evaluate() with pytest-benchmark."""
        try:
            if not hasattr(benchmark, "pedantic"):
                pytest.skip("pytest-benchmark not available")
        except Exception:
            pytest.skip("pytest-benchmark not available")

        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        result = benchmark(policy.evaluate, score=0.50, rms=500.0)
        assert isinstance(result, bool)
