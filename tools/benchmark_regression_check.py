"""Benchmark regression check tool.

Used in CI to detect performance regressions between nightly benchmark runs.

Usage:
    python tools/benchmark_regression_check.py --threshold 0.10

Exits with code 0 if no regressions, code 1 if regressions detected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Regressions beyond this percentage trigger a CI failure
DEFAULT_THRESHOLD = 0.10  # 10%


def load_benchmark_results(path: Path) -> dict:
    """Load a pytest-benchmark JSON result file."""
    with open(path) as f:
        return json.load(f)


def find_baseline(results_dir: Path, current_sha: str) -> Path | None:
    """Find the most recent non-current benchmark result file."""
    result_files = sorted(results_dir.glob("latency-*.json"), key=lambda p: p.stat().st_mtime)
    for f in reversed(result_files):
        if current_sha not in f.name:
            return f
    return None


def check_regressions(current: dict, baseline: dict, threshold: float) -> list[str]:
    """Compare benchmark results and return list of regression descriptions."""
    regressions = []

    current_benchmarks = {b["name"]: b for b in current.get("benchmarks", [])}
    baseline_benchmarks = {b["name"]: b for b in baseline.get("benchmarks", [])}

    for name, current_bench in current_benchmarks.items():
        if name not in baseline_benchmarks:
            print(f"  NEW: {name} (no baseline to compare)")
            continue

        baseline_bench = baseline_benchmarks[name]
        current_mean = current_bench["stats"]["mean"]
        baseline_mean = baseline_bench["stats"]["mean"]

        if baseline_mean == 0:
            continue

        change = (current_mean - baseline_mean) / baseline_mean

        if change > threshold:
            regressions.append(
                f"REGRESSION: {name}\n"
                f"  Baseline: {baseline_mean * 1000:.2f}ms\n"
                f"  Current:  {current_mean * 1000:.2f}ms\n"
                f"  Change:   +{change * 100:.1f}% (threshold: {threshold * 100:.0f}%)"
            )
            print(f"  ❌ {name}: +{change * 100:.1f}% (REGRESSED)")
        elif change < -0.05:  # >5% improvement
            print(f"  ✅ {name}: {change * 100:.1f}% (improved)")
        else:
            print(f"  ✓  {name}: {change * 100:.1f}% (within threshold)")

    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(description="Check benchmark regressions")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmark-results"),
        help="Directory containing benchmark JSON files",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Regression threshold (default: {DEFAULT_THRESHOLD} = {DEFAULT_THRESHOLD * 100:.0f}%%)",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help="Path to current benchmark JSON (default: most recent file)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Path to baseline benchmark JSON (default: second most recent file)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"No benchmark results directory found at {results_dir}")
        print("Run benchmarks first: pytest tests/benchmarks/ --benchmark-json=benchmark-results/latency.json")
        return 0  # Not an error — might be first run

    result_files = sorted(results_dir.glob("latency-*.json"), key=lambda p: p.stat().st_mtime)

    if len(result_files) < 2 and args.current is None:
        print(f"Need at least 2 benchmark result files for comparison. Found: {len(result_files)}")
        return 0

    current_file = args.current or result_files[-1]
    baseline_file = args.baseline or (result_files[-2] if len(result_files) >= 2 else None)

    if baseline_file is None:
        print("No baseline available — skipping regression check (first run)")
        return 0

    print(f"Comparing:")
    print(f"  Current:  {current_file}")
    print(f"  Baseline: {baseline_file}")
    print(f"  Threshold: {args.threshold * 100:.0f}%")
    print()

    current_data = load_benchmark_results(current_file)
    baseline_data = load_benchmark_results(baseline_file)

    regressions = check_regressions(current_data, baseline_data, args.threshold)

    if regressions:
        print()
        print("=" * 60)
        print("BENCHMARK REGRESSIONS DETECTED:")
        print("=" * 60)
        for r in regressions:
            print(r)
        print()
        print(f"Total regressions: {len(regressions)}")
        return 1

    print()
    print("No regressions detected. ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
