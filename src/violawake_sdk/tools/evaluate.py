"""
violawake-eval CLI.

Evaluates an ONNX wake-word model on a test set with:
  - architecture auto-detection from ONNX input shape
  - optional threshold sweep
  - confusion-matrix reporting
  - optional per-file CSV output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def evaluate_onnx_model(
    model_path: str | Path,
    test_dir: str | Path,
    threshold: float = 0.50,
    dump_scores_csv: str | Path | None = None,
    sweep: bool = True,
) -> dict:
    """Proxy the public evaluation helper for callers importing from tools."""
    from violawake_sdk.training.evaluate import evaluate_onnx_model as _evaluate_onnx_model

    return _evaluate_onnx_model(
        model_path=model_path,
        test_dir=test_dir,
        threshold=threshold,
        dump_scores_csv=dump_scores_csv,
        sweep=sweep,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-eval",
        description="Evaluate a ViolaWake ONNX model using Cohen's d plus FAR/FRR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to the ONNX model file (for example: viola_mlp_oww.onnx)",
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
        "--sweep",
        action="store_true",
        help="Sweep thresholds from 0.00 to 1.00 in 0.01 steps to find the best threshold",
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
    parser.add_argument(
        "--output-csv",
        "--dump-scores",
        dest="output_csv",
        metavar="FILE",
        help="Write per-file scores to a CSV (columns: file, label, score, correct)",
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
    print(f"Threshold:        {args.threshold:.2f}")
    print(f"Threshold sweep:  {'enabled' if args.sweep else 'disabled'}")
    print()

    try:
        results = evaluate_onnx_model(
            model_path=model_path,
            test_dir=test_dir,
            threshold=args.threshold,
            dump_scores_csv=args.output_csv,
            sweep=args.sweep,
        )
    except ImportError as e:
        print(f"ERROR: Missing dependencies: {e}", file=sys.stderr)
        print("Install with: pip install 'violawake[training]'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}", file=sys.stderr)
        sys.exit(1)

    arch = results["architecture"]
    d_prime = results["d_prime"]
    far_per_hour = results["far_per_hour"]
    frr = results["frr"] * 100
    roc_auc = results["roc_auc"]
    n_pos = results["n_positives"]
    n_neg = results["n_negatives"]

    print(f"Architecture:         {arch} (auto-detected)")
    print(f"Cohen's d:            {d_prime:.2f} (synthetic negatives may inflate this metric)")
    print(f"False Accept Rate:    {far_per_hour:.2f}/hr (at threshold={args.threshold:.2f})")
    print(f"False Reject Rate:    {frr:.1f}% (at threshold={args.threshold:.2f})")
    print(f"ROC AUC:              {roc_auc:.3f}")

    if args.sweep:
        opt_thresh = results["optimal_threshold"]
        opt_far = results["optimal_far"] * 100
        opt_frr = results["optimal_frr"] * 100
        eer = results["eer_approx"] * 100
        print(
            f"Optimal threshold:    {opt_thresh:.2f} "
            f"(FPR={opt_far:.1f}%, FNR={opt_frr:.1f}%, EER~{eer:.1f}%)"
        )
    else:
        print("Optimal threshold:    not computed (enable --sweep)")

    print(f"Positives scored:     {n_pos}")
    print(f"Negatives scored:     {n_neg}")
    print()

    if args.sweep:
        cm = results["optimal_confusion_matrix"]
        cm_threshold = results["optimal_threshold"]
        print(f"Confusion matrix (optimal threshold={cm_threshold:.2f}):")
    else:
        cm = results["confusion_matrix"]
        cm_threshold = args.threshold
        print(f"Confusion matrix (threshold={cm_threshold:.2f}):")
    print(f"  TP={cm['tp']:>5}  FP={cm['fp']:>5}")
    print(f"  FN={cm['fn']:>5}  TN={cm['tn']:>5}")
    print(f"  Precision={cm['precision']:.3f}  Recall={cm['recall']:.3f}  F1={cm['f1']:.3f}")
    print()

    if d_prime >= 15.0:
        grade = "EXCELLENT on this benchmark (validate with speech negatives before shipping)"
    elif d_prime >= 10.0:
        grade = "GOOD on this benchmark (still validate with real speech/background negatives)"
    elif d_prime >= 5.0:
        grade = "FAIR (may need more training data, augmentation, or better negatives)"
    else:
        grade = "POOR (Cohen's d < 5.0; collect more positive samples)"

    print(f"Grade: {grade}")
    print("Metric note: this score is Cohen's d over the supplied negatives.")
    print("Synthetic-only negatives can materially inflate it versus speech/background corpora.")

    if args.report:
        print()
        print("-" * 50)
        print("SCORE DISTRIBUTION")
        print("-" * 50)
        import numpy as np

        pos_arr = np.array(results["tp_scores"])
        neg_arr = np.array(results["fp_scores"])
        print(
            f"Positive mean:  {pos_arr.mean():.3f}  std: {pos_arr.std():.3f}  "
            f"min: {pos_arr.min():.3f}  max: {pos_arr.max():.3f}"
        )
        print(
            f"Negative mean:  {neg_arr.mean():.3f}  std: {neg_arr.std():.3f}  "
            f"min: {neg_arr.min():.3f}  max: {neg_arr.max():.3f}"
        )

    if args.output_csv:
        print(f"\nPer-file scores written to: {args.output_csv}")

    if args.json:
        import json

        json_results = {k: v for k, v in results.items() if not isinstance(v, list)}
        with open(args.json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {args.json}")

    if d_prime < 5.0:
        sys.exit(1)


if __name__ == "__main__":
    main()
