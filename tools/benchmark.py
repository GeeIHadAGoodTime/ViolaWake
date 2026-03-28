#!/usr/bin/env python3
"""K8: Comprehensive benchmark suite for ViolaWake inference.

Measures inference latency (p50/p95/p99), memory usage (RSS/peak),
CPU utilization, throughput (frames/sec), and generates a markdown report.

Usage:
    python tools/benchmark.py                       # Run all benchmarks
    python tools/benchmark.py --frames 5000         # More frames for stable stats
    python tools/benchmark.py --output report.md    # Custom output path
    python tools/benchmark.py --model viola_cnn_v4  # Specific model
    python tools/benchmark.py --no-report           # Print results, no file
"""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Ensure the SDK is importable from the repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def get_process_memory_mb() -> float:
    """Get current process RSS in megabytes."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    # Fallback for Linux
    if platform.system() == "Linux":
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except Exception:
            pass

    # Windows fallback via ctypes
    if platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / (1024 * 1024)
        except Exception:
            pass

    return -1.0


def get_peak_memory_mb() -> float:
    """Get peak RSS (high watermark) in megabytes."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().peak_wss / (1024 * 1024)  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pass

    if platform.system() == "Linux":
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass

    if platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / (1024 * 1024)
        except Exception:
            pass

    return -1.0


def measure_cpu_time(func, *args, **kwargs):
    """Measure wall time and CPU time for a callable."""
    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()
    result = func(*args, **kwargs)
    cpu_s = time.process_time() - t0_cpu
    wall_s = time.perf_counter() - t0_wall
    return result, wall_s, cpu_s


def run_latency_benchmark(
    n_frames: int = 2000,
    frame_ms: int = 20,
    model: str = "viola",
) -> dict:
    """Measure per-frame inference latency.

    Returns dict with p50/p95/p99/mean/max in ms, plus throughput.
    """
    from violawake_sdk._constants import SAMPLE_RATE
    from violawake_sdk.wake_detector import WakeDetector

    frame_samples = int(SAMPLE_RATE * frame_ms / 1000)
    rng = np.random.default_rng(42)

    # Create detector
    detector = WakeDetector(model=model, threshold=0.80)

    # Warmup: fill buffer and stabilize
    warmup_noise = (rng.standard_normal(frame_samples) * 100).astype(np.float32)
    for _ in range(50):
        detector.process(warmup_noise)

    # Benchmark
    latencies = np.zeros(n_frames, dtype=np.float64)
    for i in range(n_frames):
        noise = (rng.standard_normal(frame_samples) * 100).astype(np.float32)
        t0 = time.perf_counter()
        detector.process(noise)
        latencies[i] = (time.perf_counter() - t0) * 1000.0

    frames_per_sec = n_frames / (latencies.sum() / 1000.0)

    return {
        "n_frames": n_frames,
        "frame_ms": frame_ms,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "mean_ms": float(latencies.mean()),
        "std_ms": float(latencies.std()),
        "max_ms": float(latencies.max()),
        "min_ms": float(latencies.min()),
        "throughput_fps": frames_per_sec,
        "realtime_factor": float(latencies.mean() / frame_ms),
    }


def run_memory_benchmark(model: str = "viola") -> dict:
    """Measure memory usage before and after model loading."""
    from violawake_sdk._constants import SAMPLE_RATE
    from violawake_sdk.wake_detector import WakeDetector

    gc.collect()
    mem_before = get_process_memory_mb()

    detector = WakeDetector(model=model, threshold=0.80)

    mem_after_load = get_process_memory_mb()

    # Process some frames to allocate runtime buffers
    frame_samples = int(SAMPLE_RATE * 20 / 1000)
    rng = np.random.default_rng(42)
    for _ in range(100):
        noise = (rng.standard_normal(frame_samples) * 100).astype(np.float32)
        detector.process(noise)

    mem_after_process = get_process_memory_mb()
    peak = get_peak_memory_mb()

    return {
        "rss_before_mb": mem_before,
        "rss_after_load_mb": mem_after_load,
        "rss_after_process_mb": mem_after_process,
        "model_overhead_mb": mem_after_load - mem_before if mem_before > 0 else -1,
        "runtime_overhead_mb": mem_after_process - mem_after_load if mem_after_load > 0 else -1,
        "peak_rss_mb": peak,
    }


def run_cpu_benchmark(
    n_frames: int = 1000,
    model: str = "viola",
) -> dict:
    """Measure CPU utilization during inference."""
    from violawake_sdk._constants import SAMPLE_RATE
    from violawake_sdk.wake_detector import WakeDetector

    frame_samples = int(SAMPLE_RATE * 20 / 1000)
    rng = np.random.default_rng(42)

    detector = WakeDetector(model=model, threshold=0.80)

    # Warmup
    warmup_noise = (rng.standard_normal(frame_samples) * 100).astype(np.float32)
    for _ in range(50):
        detector.process(warmup_noise)

    # Timed run
    frames_data = [
        (rng.standard_normal(frame_samples) * 100).astype(np.float32)
        for _ in range(n_frames)
    ]

    t0_wall = time.perf_counter()
    t0_cpu = time.process_time()

    for frame in frames_data:
        detector.process(frame)

    wall_s = time.perf_counter() - t0_wall
    cpu_s = time.process_time() - t0_cpu

    return {
        "n_frames": n_frames,
        "wall_time_s": wall_s,
        "cpu_time_s": cpu_s,
        "cpu_utilization_pct": (cpu_s / wall_s * 100) if wall_s > 0 else 0,
        "cpu_per_frame_ms": (cpu_s / n_frames) * 1000,
    }


def run_throughput_benchmark(
    duration_s: float = 5.0,
    model: str = "viola",
) -> dict:
    """Measure maximum throughput (frames per second)."""
    from violawake_sdk._constants import SAMPLE_RATE
    from violawake_sdk.wake_detector import WakeDetector

    frame_samples = int(SAMPLE_RATE * 20 / 1000)
    rng = np.random.default_rng(42)
    noise = (rng.standard_normal(frame_samples) * 100).astype(np.float32)

    detector = WakeDetector(model=model, threshold=0.80)

    # Warmup
    for _ in range(50):
        detector.process(noise)

    # Sustained throughput test
    frames_done = 0
    t_start = time.perf_counter()
    t_end = t_start + duration_s

    while time.perf_counter() < t_end:
        detector.process(noise)
        frames_done += 1

    elapsed = time.perf_counter() - t_start
    fps = frames_done / elapsed

    # 50 fps = real-time (20ms frames)
    return {
        "duration_s": elapsed,
        "frames_processed": frames_done,
        "throughput_fps": fps,
        "realtime_multiple": fps / 50.0,
    }


def generate_report(
    latency: dict,
    memory: dict,
    cpu: dict,
    throughput: dict,
    model: str,
) -> str:
    """Generate a markdown benchmark report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# ViolaWake Benchmark Report",
        "",
        f"**Generated**: {now}",
        f"**Model**: `{model}`",
        f"**Platform**: {platform.platform()}",
        f"**Python**: {platform.python_version()}",
        f"**CPU**: {platform.processor() or 'unknown'}",
        "",
        "## Inference Latency",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Frames tested | {latency['n_frames']:,} |",
        f"| Frame duration | {latency['frame_ms']}ms |",
        f"| **p50 latency** | **{latency['p50_ms']:.3f}ms** |",
        f"| p95 latency | {latency['p95_ms']:.3f}ms |",
        f"| p99 latency | {latency['p99_ms']:.3f}ms |",
        f"| Mean latency | {latency['mean_ms']:.3f}ms |",
        f"| Std deviation | {latency['std_ms']:.3f}ms |",
        f"| Max latency | {latency['max_ms']:.3f}ms |",
        f"| Min latency | {latency['min_ms']:.3f}ms |",
        f"| Throughput | {latency['throughput_fps']:.1f} fps |",
        f"| Real-time factor | {latency['realtime_factor']:.4f}x |",
        "",
        "## Memory Usage",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| RSS before load | {memory['rss_before_mb']:.1f} MB |",
        f"| RSS after load | {memory['rss_after_load_mb']:.1f} MB |",
        f"| RSS after processing | {memory['rss_after_process_mb']:.1f} MB |",
        f"| Model overhead | {memory['model_overhead_mb']:.1f} MB |",
        f"| Runtime overhead | {memory['runtime_overhead_mb']:.1f} MB |",
        f"| Peak RSS | {memory['peak_rss_mb']:.1f} MB |",
        "",
        "## CPU Utilization",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Frames | {cpu['n_frames']:,} |",
        f"| Wall time | {cpu['wall_time_s']:.3f}s |",
        f"| CPU time | {cpu['cpu_time_s']:.3f}s |",
        f"| **CPU utilization** | **{cpu['cpu_utilization_pct']:.1f}%** |",
        f"| CPU per frame | {cpu['cpu_per_frame_ms']:.3f}ms |",
        "",
        "## Sustained Throughput",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Test duration | {throughput['duration_s']:.1f}s |",
        f"| Frames processed | {throughput['frames_processed']:,} |",
        f"| **Throughput** | **{throughput['throughput_fps']:.1f} fps** |",
        f"| Real-time multiple | {throughput['realtime_multiple']:.1f}x |",
        "",
        "## Interpretation",
        "",
        f"- Real-time requirement: 20ms per frame (50 fps)",
    ]

    rtf = latency["realtime_factor"]
    if rtf < 0.5:
        lines.append(f"- **PASS**: p50 latency is {rtf:.2f}x real-time (well within budget)")
    elif rtf < 1.0:
        lines.append(f"- **PASS**: p50 latency is {rtf:.2f}x real-time (within budget)")
    else:
        lines.append(f"- **FAIL**: p50 latency is {rtf:.2f}x real-time (exceeds budget!)")

    p99 = latency["p99_ms"]
    if p99 < 20.0:
        lines.append(f"- p99 latency ({p99:.1f}ms) fits within a single 20ms frame")
    else:
        lines.append(f"- **WARNING**: p99 latency ({p99:.1f}ms) exceeds 20ms frame budget")

    rtm = throughput["realtime_multiple"]
    lines.append(f"- Sustained throughput: {rtm:.1f}x real-time")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="ViolaWake Benchmark Suite")
    parser.add_argument("--frames", type=int, default=2000, help="Number of frames for latency test")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration for throughput test (s)")
    parser.add_argument("--model", type=str, default="viola", help="Model name or path")
    parser.add_argument("--output", type=str, default=None, help="Output markdown file path")
    parser.add_argument("--no-report", action="store_true", help="Print results without saving")
    args = parser.parse_args()

    output_path = args.output or str(_REPO_ROOT / "benchmark_report.md")

    print(f"ViolaWake Benchmark Suite")
    print(f"Model: {args.model}")
    print(f"Frames: {args.frames}")
    print()

    print("[1/4] Latency benchmark...", flush=True)
    latency = run_latency_benchmark(n_frames=args.frames, model=args.model)
    print(f"  p50={latency['p50_ms']:.3f}ms  p95={latency['p95_ms']:.3f}ms  p99={latency['p99_ms']:.3f}ms")

    print("[2/4] Memory benchmark...", flush=True)
    memory = run_memory_benchmark(model=args.model)
    print(f"  RSS={memory['rss_after_process_mb']:.1f}MB  peak={memory['peak_rss_mb']:.1f}MB")

    print("[3/4] CPU benchmark...", flush=True)
    cpu = run_cpu_benchmark(n_frames=min(args.frames, 1000), model=args.model)
    print(f"  CPU={cpu['cpu_utilization_pct']:.1f}%  per_frame={cpu['cpu_per_frame_ms']:.3f}ms")

    print("[4/4] Throughput benchmark...", flush=True)
    throughput = run_throughput_benchmark(duration_s=args.duration, model=args.model)
    print(f"  {throughput['throughput_fps']:.1f} fps  ({throughput['realtime_multiple']:.1f}x real-time)")

    print()
    report = generate_report(latency, memory, cpu, throughput, args.model)

    if not args.no_report:
        Path(output_path).write_text(report, encoding="utf-8")
        print(f"Report saved to: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
