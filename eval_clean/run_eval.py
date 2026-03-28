#!/usr/bin/env python3
"""
Run ViolaWake evaluation on the clean eval set.
Uses Viola's venv for all dependencies.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

# Setup
sys.path.insert(0, "J:/CLAUDE/PROJECTS/Wakeword/src")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from violawake_sdk.training.evaluate import evaluate_onnx_model

EVAL_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/eval_clean")
MODEL_DIR = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA/violawake_data/trained_models")

def run_eval(model_name: str, model_path: str, csv_name: str) -> dict:
    """Run evaluation for a single model and return results dict."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'='*70}")

    results = evaluate_onnx_model(
        model_path=model_path,
        test_dir=str(EVAL_DIR),
        threshold=0.50,
        dump_scores_csv=str(EVAL_DIR / csv_name),
        device="cpu",
    )

    print(f"\n=== {model_name} Results ===")
    print(f"Architecture: {results['architecture']}")
    print(f"D-prime (Cohen's d): {results['d_prime']:.4f}")
    print(f"AUC: {results['roc_auc']:.4f}")
    print(f"FRR at 0.50: {results['frr']:.4f} ({results['frr']*100:.1f}%)")
    print(f"FAR/hr at 0.50: {results['far_per_hour']:.2f}")
    print(f"Optimal threshold: {results['optimal_threshold']:.2f}")
    print(f"Optimal FAR: {results['optimal_far']:.4f}")
    print(f"Optimal FRR: {results['optimal_frr']:.4f}")
    print(f"EER approx: {results['eer_approx']:.4f}")
    print(f"Pos mean score: {results['tp_mean']:.6f}")
    print(f"Neg mean score: {results['fp_mean']:.6f}")
    print(f"N positives: {results['n_positives']}")
    print(f"N negatives: {results['n_negatives']}")
    cm = results['confusion_matrix']
    print(f"Confusion at 0.50: TP={cm['tp']} FP={cm['fp']} TN={cm['tn']} FN={cm['fn']}")
    print(f"Precision: {cm['precision']:.4f}")
    print(f"Recall: {cm['recall']:.4f}")
    print(f"F1: {cm['f1']:.4f}")

    return results


def main():
    # Model 1: Mean-pool (production)
    results_meanpool = run_eval(
        "Mean-Pool Model (Production)",
        str(MODEL_DIR / "viola_mlp_oww.onnx"),
        "scores_meanpool.csv",
    )

    # Model 2: Maxpool
    results_maxpool = run_eval(
        "Maxpool Model (Previous Production)",
        str(MODEL_DIR / "viola_mlp_oww_maxpool.onnx"),
        "scores_maxpool.csv",
    )

    # Save raw results as JSON for analysis
    for name, results in [("meanpool", results_meanpool), ("maxpool", results_maxpool)]:
        # Convert scores to serializable format
        json_results = {
            k: v for k, v in results.items()
            if k not in ('tp_scores', 'fp_scores')
        }
        json_results['tp_scores_summary'] = {
            'count': len(results['tp_scores']),
            'mean': float(sum(results['tp_scores']) / len(results['tp_scores'])),
            'min': float(min(results['tp_scores'])),
            'max': float(max(results['tp_scores'])),
        }
        json_results['fp_scores_summary'] = {
            'count': len(results['fp_scores']),
            'mean': float(sum(results['fp_scores']) / len(results['fp_scores'])),
            'min': float(min(results['fp_scores'])),
            'max': float(max(results['fp_scores'])),
        }

        with open(EVAL_DIR / f"results_{name}.json", "w") as f:
            json.dump(json_results, f, indent=2)

    # Comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Mean-Pool':<20} {'Maxpool':<20}")
    print(f"{'-'*70}")
    print(f"{'D-prime':<30} {results_meanpool['d_prime']:<20.4f} {results_maxpool['d_prime']:<20.4f}")
    print(f"{'AUC':<30} {results_meanpool['roc_auc']:<20.4f} {results_maxpool['roc_auc']:<20.4f}")
    print(f"{'FRR @ 0.50':<30} {results_meanpool['frr']:<20.4f} {results_maxpool['frr']:<20.4f}")
    print(f"{'FAR/hr @ 0.50':<30} {results_meanpool['far_per_hour']:<20.2f} {results_maxpool['far_per_hour']:<20.2f}")
    print(f"{'Optimal threshold':<30} {results_meanpool['optimal_threshold']:<20.2f} {results_maxpool['optimal_threshold']:<20.2f}")
    print(f"{'EER approx':<30} {results_meanpool['eer_approx']:<20.4f} {results_maxpool['eer_approx']:<20.4f}")
    print(f"{'N positives':<30} {results_meanpool['n_positives']:<20} {results_maxpool['n_positives']:<20}")
    print(f"{'N negatives':<30} {results_meanpool['n_negatives']:<20} {results_maxpool['n_negatives']:<20}")


if __name__ == "__main__":
    main()
