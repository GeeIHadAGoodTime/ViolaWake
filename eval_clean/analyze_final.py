#!/usr/bin/env python3
"""
Final analysis with precise phrase filtering.

Split into:
  - Trained phrases: "viola" (standalone), "hey viola", "ok viola"
  - Untrained phrases: "viola wake up", "viola please"
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = float(row["score"])
            rows.append(row)
    return rows


def extract_phrase(filename: str) -> str:
    name = Path(filename).stem
    for suffix in ["_noisy", "_reverb"]:
        name = name.replace(suffix, "")
    m = re.match(r"en-\w+-\w+Neural_(.+)$", name)
    if m:
        return m.group(1).replace("_", " ")
    parts = name.split("_")
    if len(parts) >= 3:
        phrase_parts = parts[2:]
        while phrase_parts and phrase_parts[-1].isdigit():
            phrase_parts.pop()
        return " ".join(phrase_parts)
    return name


def is_trained_phrase(phrase: str) -> bool:
    """Check if this is a phrase the model was trained on."""
    p = phrase.strip().lower()
    # Exact match or equivalent
    if p in ("viola", "hey viola", "ok viola"):
        return True
    # pyttsx3 voices produce names like "David Desktop - English (United States) viola"
    if p.endswith("viola") and "wake" not in p and "please" not in p:
        return True
    if p.endswith("hey viola"):
        return True
    if p.endswith("ok viola"):
        return True
    return False


def compute_dprime(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pooled_std = np.sqrt(0.5 * (pos.var() + neg.var()))
    if pooled_std < 1e-10:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled_std)


def bootstrap_ci(pos, neg, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    pos = np.asarray(pos)
    neg = np.asarray(neg)
    ds = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        ps = np.sqrt(0.5 * (p.var() + n.var()))
        ds.append(float((p.mean() - n.mean()) / ps) if ps > 1e-10 else 0.0)
    ds = np.array(ds)
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5), ds.std()


def analyze(csv_path: str, model_name: str):
    rows = load_csv(csv_path)
    positives = [r for r in rows if r["label"] == "positive"]
    negatives = [r for r in rows if r["label"] == "negative"]

    # Classify positives
    trained = []
    untrained = []
    for r in positives:
        phrase = extract_phrase(r["file"])
        if is_trained_phrase(phrase):
            trained.append(r)
        else:
            untrained.append(r)

    neg_scores = np.array([r["score"] for r in negatives])
    all_pos_scores = np.array([r["score"] for r in positives])
    trained_scores = np.array([r["score"] for r in trained])
    untrained_scores = np.array([r["score"] for r in untrained]) if untrained else np.array([])

    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")

    # Full eval
    d_full = compute_dprime(all_pos_scores, neg_scores)
    ci_low, ci_high, se = bootstrap_ci(all_pos_scores, neg_scores)
    frr_full = float(np.sum(all_pos_scores < 0.50) / len(all_pos_scores))
    fp_count = int(np.sum(neg_scores >= 0.50))

    print(f"\n--- ALL PHRASES ---")
    print(f"N positives: {len(positives)} (trained: {len(trained)}, untrained: {len(untrained)})")
    print(f"N negatives: {len(negatives)}")
    print(f"D-prime: {d_full:.4f}  95% CI: [{ci_low:.4f}, {ci_high:.4f}]  SE: {se:.4f}")
    print(f"FRR @ 0.50: {frr_full*100:.1f}%  ({int(np.sum(all_pos_scores < 0.50))}/{len(all_pos_scores)})")
    print(f"FP @ 0.50: {fp_count}/{len(negatives)}")
    print(f"Pos mean: {all_pos_scores.mean():.4f}  Neg mean: {neg_scores.mean():.6f}")

    # Trained phrases only
    d_trained = compute_dprime(trained_scores, neg_scores)
    ci_t_lo, ci_t_hi, se_t = bootstrap_ci(trained_scores, neg_scores)
    frr_trained = float(np.sum(trained_scores < 0.50) / len(trained_scores))

    print(f"\n--- TRAINED PHRASES ONLY (\"viola\", \"hey viola\", \"ok viola\") ---")
    print(f"N positives: {len(trained)}")
    print(f"D-prime: {d_trained:.4f}  95% CI: [{ci_t_lo:.4f}, {ci_t_hi:.4f}]  SE: {se_t:.4f}")
    print(f"FRR @ 0.50: {frr_trained*100:.1f}%  ({int(np.sum(trained_scores < 0.50))}/{len(trained_scores)})")
    print(f"Pos mean: {trained_scores.mean():.4f}")

    # Per-phrase detail
    phrase_data = defaultdict(list)
    for r in positives:
        p = extract_phrase(r["file"])
        phrase_data[p].append(r["score"])

    print(f"\n  Phrase breakdown:")
    for phrase, scores in sorted(phrase_data.items(), key=lambda x: np.mean(x[1])):
        arr = np.array(scores)
        n_pass = sum(1 for s in scores if s >= 0.50)
        tag = "[TRAINED]" if is_trained_phrase(phrase) else "[NOT TRAINED]"
        print(f"    {phrase:<50} n={len(scores):>3}  mean={arr.mean():.4f}  pass={n_pass}/{len(scores):>3}  {tag}")

    # Untrained phrases
    if len(untrained_scores) > 0:
        d_unt = compute_dprime(untrained_scores, neg_scores)
        print(f"\n--- UNTRAINED PHRASES (\"viola wake up\", \"viola please\") ---")
        print(f"N: {len(untrained)}, D-prime: {d_unt:.4f}, mean: {untrained_scores.mean():.4f}, FRR: {float(np.sum(untrained_scores < 0.50) / len(untrained_scores))*100:.1f}%")

    # Threshold sweep for trained phrases
    print(f"\n--- THRESHOLD SWEEP (trained phrases) ---")
    print(f"{'Threshold':>10} {'FRR':>8} {'FP':>5} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    for thresh in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        tp = int(np.sum(trained_scores >= thresh))
        fn = int(np.sum(trained_scores < thresh))
        fp = int(np.sum(neg_scores >= thresh))
        tn = int(np.sum(neg_scores < thresh))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0
        print(f"{thresh:>10.2f} {frr*100:>7.1f}% {fp:>5} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

    return {
        "d_full": d_full, "ci_full": (ci_low, ci_high), "se_full": se,
        "d_trained": d_trained, "ci_trained": (ci_t_lo, ci_t_hi), "se_trained": se_t,
        "frr_full": frr_full, "frr_trained": frr_trained,
        "fp_count": fp_count,
        "n_pos_all": len(positives), "n_pos_trained": len(trained),
        "n_neg": len(negatives),
    }


def main():
    mp = analyze(
        "J:/CLAUDE/PROJECTS/Wakeword/eval_clean/scores_meanpool.csv",
        "MEAN-POOL MODEL (Production: viola_mlp_oww.onnx)",
    )

    mx = analyze(
        "J:/CLAUDE/PROJECTS/Wakeword/eval_clean/scores_maxpool.csv",
        "MAXPOOL MODEL (viola_mlp_oww_maxpool.onnx)",
    )

    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"")
    print(f"{'Metric':<40} {'MeanPool':>12} {'Maxpool':>12} {'Prior':>12}")
    print(f"{'-'*76}")
    print(f"{'d-prime (all phrases)':<40} {mp['d_full']:>12.4f} {mx['d_full']:>12.4f} {'11.56':>12}")
    print(f"{'d-prime (trained phrases)':<40} {mp['d_trained']:>12.4f} {mx['d_trained']:>12.4f} {'11.56':>12}")
    print(f"{'d-prime 95% CI (trained)':<40} [{mp['ci_trained'][0]:.2f},{mp['ci_trained'][1]:.2f}]  [{mx['ci_trained'][0]:.2f},{mx['ci_trained'][1]:.2f}]")
    print(f"{'FRR @ 0.50 (trained)':<40} {mp['frr_trained']*100:>11.1f}% {mx['frr_trained']*100:>11.1f}%")
    print(f"{'FP @ 0.50':<40} {mp['fp_count']:>12} {mx['fp_count']:>12}")
    print(f"{'N pos (all / trained)':<40} {mp['n_pos_all']}/{mp['n_pos_trained']:>11} {mx['n_pos_all']}/{mx['n_pos_trained']:>11}")
    print(f"{'N neg':<40} {mp['n_neg']:>12} {mx['n_neg']:>12}")
    print(f"")
    print(f"CONCLUSION: The prior d-prime of 11.56 was measured on a contaminated eval set.")
    print(f"On a clean eval set with zero training overlap, the production model achieves")
    print(f"d-prime {mp['d_trained']:.2f} (95% CI [{mp['ci_trained'][0]:.2f}, {mp['ci_trained'][1]:.2f}]).")
    print(f"This is approximately {11.56 / mp['d_trained']:.1f}x lower than claimed.")


if __name__ == "__main__":
    main()
