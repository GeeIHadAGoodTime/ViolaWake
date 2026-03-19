"""
violawake-eval CLI — Evaluate a wake word model using d-prime metric.

Entry point: ``violawake-eval`` (declared in pyproject.toml).

Usage::

    violawake-eval --model models/viola_mlp_oww.onnx \\
                   --test-dir data/test/ \\
                   --threshold 0.50 \\
                   --report

The test directory must contain:
    positives/  — WAV/FLAC files containing the wake word
    negatives/  — WAV/FLAC files of background audio (no wake word)

Output::

    d-prime: 15.10
    False Accept Rate: 0.28/hr (at threshold=0.50)
    False Reject Rate: 1.8% (at threshold=0.50)
    ROC AUC: 0.998
    Positives: 150 | Negatives: 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-eval",
        description="Evaluate a ViolaWake ONNX model using d-prime metric.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to the ONNX model file (e.g., viola_mlp_oww.onnx)",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        metavar="DIR",
        help="Test directory with positives/ and negatives/ subdirectories",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        metavar="FLOAT",
        help="Classification threshold for FAR/FRR calculation (default: 0.50)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a detailed text report to stdout",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Save full results as JSON to this file",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    test_dir = Path(args.test_dir)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    if not test_dir.exists():
        print(f"ERROR: Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating model: {model_path.name}")
    print(f"Test set:         {test_dir}")
    print(f"Threshold:        {args.threshold}")
    print()

    try:
        from violawake_sdk.training.evaluate import evaluate_onnx_model
    except ImportError as e:
        print(f"ERROR: Missing dependencies: {e}", file=sys.stderr)
        print("Install with: pip install 'violawake[training]'", file=sys.stderr)
        sys.exit(1)

    try:
        results = evaluate_onnx_model(
            model_path=model_path,
            test_dir=test_dir,
            threshold=args.threshold,
        )
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    d = results["d_prime"]
    far = results["far_per_hour"]
    frr = results["frr"] * 100
    auc = results["roc_auc"]
    n_pos = results["n_positives"]
    n_neg = results["n_negatives"]

    print(f"d-prime:              {d:.2f}")
    print(f"False Accept Rate:    {far:.2f}/hr (at threshold={args.threshold})")
    print(f"False Reject Rate:    {frr:.1f}% (at threshold={args.threshold})")
    print(f"ROC AUC:              {auc:.3f}")
    print(f"Positives scored:     {n_pos}")
    print(f"Negatives scored:     {n_neg}")
    print()

    # Grade the model
    if d >= 15.0:
        grade = "EXCELLENT (production-grade, Viola standard)"
    elif d >= 10.0:
        grade = "GOOD (acceptable for deployment)"
    elif d >= 5.0:
        grade = "FAIR (may need more training data or augmentation)"
    else:
        grade = "POOR (d-prime < 5.0 — collect more positive samples)"

    print(f"Grade: {grade}")

    if args.report:
        print()
        print("─" * 50)
        print("SCORE DISTRIBUTION")
        print("─" * 50)
        import numpy as np
        pos_arr = np.array(results["tp_scores"])
        neg_arr = np.array(results["fp_scores"])
        print(f"Positive mean:  {pos_arr.mean():.3f}  std: {pos_arr.std():.3f}  min: {pos_arr.min():.3f}  max: {pos_arr.max():.3f}")
        print(f"Negative mean:  {neg_arr.mean():.3f}  std: {neg_arr.std():.3f}  min: {neg_arr.min():.3f}  max: {neg_arr.max():.3f}")

    if args.json:
        import json
        # Remove list fields for JSON output (they can be large)
        json_results = {k: v for k, v in results.items() if not isinstance(v, list)}
        with open(args.json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {args.json}")

    # Exit with non-zero if d-prime is below acceptable threshold
    if d < 5.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
