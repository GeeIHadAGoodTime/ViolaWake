"""Unit tests for K8: Benchmark suite.

Tests the benchmark utility functions (memory measurement, report generation)
without requiring ONNX models. Model-dependent benchmarks are tested only
for their structure, not execution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make the tools directory importable
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "tools"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Import after path setup
import benchmark  # noqa: E402


class TestMemoryMeasurement:
    """Test memory measurement utilities."""

    def test_get_process_memory_returns_float(self) -> None:
        result = benchmark.get_process_memory_mb()
        assert isinstance(result, float)
        # Should be positive on any running process (or -1 if undetectable)
        assert result > 0 or result == -1.0

    def test_get_peak_memory_returns_float(self) -> None:
        result = benchmark.get_peak_memory_mb()
        assert isinstance(result, float)
        # Either a valid positive number or -1.0
        assert result > 0 or result == -1.0


class TestCPUTimeMeasurement:
    """Test CPU time measurement utility."""

    def test_measure_cpu_time(self) -> None:
        def dummy_work():
            total = 0
            for i in range(10000):
                total += i
            return total

        result, wall_s, cpu_s = benchmark.measure_cpu_time(dummy_work)
        assert result == sum(range(10000))
        assert wall_s >= 0
        assert cpu_s >= 0

    def test_measure_cpu_time_with_args(self) -> None:
        def add(a, b):
            return a + b

        result, wall_s, cpu_s = benchmark.measure_cpu_time(add, 3, 4)
        assert result == 7


class TestReportGeneration:
    """Test markdown report generation."""

    def test_generate_report_structure(self) -> None:
        latency = {
            "n_frames": 1000,
            "frame_ms": 20,
            "p50_ms": 0.5,
            "p95_ms": 1.0,
            "p99_ms": 2.0,
            "mean_ms": 0.6,
            "std_ms": 0.2,
            "max_ms": 5.0,
            "min_ms": 0.1,
            "throughput_fps": 2000.0,
            "realtime_factor": 0.025,
        }
        memory = {
            "rss_before_mb": 50.0,
            "rss_after_load_mb": 65.0,
            "rss_after_process_mb": 67.0,
            "model_overhead_mb": 15.0,
            "runtime_overhead_mb": 2.0,
            "peak_rss_mb": 70.0,
        }
        cpu = {
            "n_frames": 1000,
            "wall_time_s": 0.6,
            "cpu_time_s": 0.5,
            "cpu_utilization_pct": 83.3,
            "cpu_per_frame_ms": 0.5,
        }
        throughput = {
            "duration_s": 5.0,
            "frames_processed": 10000,
            "throughput_fps": 2000.0,
            "realtime_multiple": 40.0,
        }

        report = benchmark.generate_report(latency, memory, cpu, throughput, "test_model")

        # Check structure
        assert "# ViolaWake Benchmark Report" in report
        assert "## Inference Latency" in report
        assert "## Memory Usage" in report
        assert "## CPU Utilization" in report
        assert "## Sustained Throughput" in report
        assert "## Interpretation" in report
        assert "test_model" in report

    def test_report_pass_for_low_latency(self) -> None:
        latency = {
            "n_frames": 100, "frame_ms": 20,
            "p50_ms": 0.5, "p95_ms": 1.0, "p99_ms": 2.0,
            "mean_ms": 0.6, "std_ms": 0.2, "max_ms": 5.0, "min_ms": 0.1,
            "throughput_fps": 2000.0, "realtime_factor": 0.025,
        }
        memory = {
            "rss_before_mb": 50.0, "rss_after_load_mb": 65.0,
            "rss_after_process_mb": 67.0, "model_overhead_mb": 15.0,
            "runtime_overhead_mb": 2.0, "peak_rss_mb": 70.0,
        }
        cpu = {
            "n_frames": 100, "wall_time_s": 0.6, "cpu_time_s": 0.5,
            "cpu_utilization_pct": 83.3, "cpu_per_frame_ms": 0.5,
        }
        throughput = {
            "duration_s": 5.0, "frames_processed": 10000,
            "throughput_fps": 2000.0, "realtime_multiple": 40.0,
        }

        report = benchmark.generate_report(latency, memory, cpu, throughput, "viola")
        assert "**PASS**" in report

    def test_report_fail_for_high_latency(self) -> None:
        latency = {
            "n_frames": 100, "frame_ms": 20,
            "p50_ms": 25.0, "p95_ms": 30.0, "p99_ms": 35.0,
            "mean_ms": 25.0, "std_ms": 5.0, "max_ms": 40.0, "min_ms": 20.0,
            "throughput_fps": 40.0, "realtime_factor": 1.25,
        }
        memory = {
            "rss_before_mb": 50.0, "rss_after_load_mb": 65.0,
            "rss_after_process_mb": 67.0, "model_overhead_mb": 15.0,
            "runtime_overhead_mb": 2.0, "peak_rss_mb": 70.0,
        }
        cpu = {
            "n_frames": 100, "wall_time_s": 2.5, "cpu_time_s": 2.0,
            "cpu_utilization_pct": 80.0, "cpu_per_frame_ms": 20.0,
        }
        throughput = {
            "duration_s": 5.0, "frames_processed": 200,
            "throughput_fps": 40.0, "realtime_multiple": 0.8,
        }

        report = benchmark.generate_report(latency, memory, cpu, throughput, "viola")
        assert "**FAIL**" in report

    def test_report_p99_warning(self) -> None:
        latency = {
            "n_frames": 100, "frame_ms": 20,
            "p50_ms": 5.0, "p95_ms": 15.0, "p99_ms": 25.0,
            "mean_ms": 6.0, "std_ms": 3.0, "max_ms": 30.0, "min_ms": 2.0,
            "throughput_fps": 160.0, "realtime_factor": 0.25,
        }
        memory = {
            "rss_before_mb": 50.0, "rss_after_load_mb": 65.0,
            "rss_after_process_mb": 67.0, "model_overhead_mb": 15.0,
            "runtime_overhead_mb": 2.0, "peak_rss_mb": 70.0,
        }
        cpu = {
            "n_frames": 100, "wall_time_s": 0.6, "cpu_time_s": 0.5,
            "cpu_utilization_pct": 83.3, "cpu_per_frame_ms": 5.0,
        }
        throughput = {
            "duration_s": 5.0, "frames_processed": 800,
            "throughput_fps": 160.0, "realtime_multiple": 3.2,
        }

        report = benchmark.generate_report(latency, memory, cpu, throughput, "viola")
        assert "**WARNING**" in report
        assert "p99" in report.lower()
