#!/usr/bin/env python3
"""
Corrected analysis: separate "core wake phrases" from "extended phrases".

The model was trained on "Viola" as the wake word. "viola wake up" is NOT
a phrase the model was trained to detect. We should report both:
1. Full eval set (all phrases) - honest but includes non-trained phrases
2. Core phrases only ("viola", "hey viola", "ok viola") - fair comparison to prior d-prime
"""
from __future__ import annotations

import csv
import re
import sys
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


def compute_dprime(pos, neg):
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pooled_std = np.sqrt(0.5 * (pos.var() + neg.var()))
    if pooled_std < 1e-10:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled_std)


def bootstrap_ci(pos, neg, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=len(pos), replace=True)
        n = rng.choice(neg, size=len(neg), replace=True)
        ps = np.sqrt(0.5 * (p.var() + n.var()))
        ds.append(float((p.mean() - n.mean()) / ps) if ps > 1e-10 else 0.0)
    ds = np.array(ds)
    return np.percentile(ds, 2.5), np.percentile(ds, 97.5), ds.std()


def analyze_subset(csv_path: str, model_name: str, core_phrases: set[str]):
    rows = load_csv(csv_path)
    positives = [r for r in rows if r["label"] == "positive"]
    negatives = [r for r in rows if r["label"] == "negative"]

    # Split positives into core and extended
    core_pos = []
    extended_pos = []
    for r in positives:
        phrase = extract_phrase(r["file"])
        if any(cp in phrase for cp in core_phrases):
            core_pos.append(r)
        else:
            extended_pos.append(r)

    neg_scores = np.array([r["score"] for r in negatives])
    all_pos_scores = np.array([r["score"] for r in positives])
    core_pos_scores = np.array([r["score"] for r in core_pos])

    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")

    # Full eval
    d_full = compute_dprime(all_pos_scores, neg_scores)
    ci_low, ci_high, se = bootstrap_ci(all_pos_scores, neg_scores)
    frr_full = float(np.sum(all_pos_scores < 0.50) / len(all_pos_scores))
    fp_count = int(np.sum(neg_scores >= 0.50))

    print(f"\n--- FULL EVAL SET (all phrases) ---")
    print(f"N positives: {len(positives)} (core: {len(core_pos)}, extended: {len(extended_pos)})")
    print(f"N negatives: {len(negatives)}")
    print(f"D-prime: {d_full:.4f}  (95% CI: [{ci_low:.4f}, {ci_high:.4f}], SE={se:.4f})")
    print(f"FRR @ 0.50: {frr_full:.4f} ({frr_full*100:.1f}%)")
    print(f"False positives @ 0.50: {fp_count}/{len(negatives)}")
    print(f"Pos mean: {all_pos_scores.mean():.4f}, Neg mean: {neg_scores.mean():.6f}")

    # Core phrases only
    d_core = compute_dprime(core_pos_scores, neg_scores)
    ci_low_c, ci_high_c, se_c = bootstrap_ci(core_pos_scores, neg_scores)
    frr_core = float(np.sum(core_pos_scores < 0.50) / len(core_pos_scores))
    tp_core = int(np.sum(core_pos_scores >= 0.50))

    print(f"\n--- CORE PHRASES ONLY (\"viola\", \"hey viola\", \"ok viola\") ---")
    print(f"N positives: {len(core_pos)}")
    print(f"N negatives: {len(negatives)}")
    print(f"D-prime: {d_core:.4f}  (95% CI: [{ci_low_c:.4f}, {ci_high_c:.4f}], SE={se_c:.4f})")
    print(f"FRR @ 0.50: {frr_core:.4f} ({frr_core*100:.1f}%)")
    print(f"TP @ 0.50: {tp_core}/{len(core_pos)}")
    print(f"Core pos mean: {core_pos_scores.mean():.4f}")

    # Per-phrase breakdown for core
    phrase_data = defaultdict(list)
    for r in core_pos:
        p = extract_phrase(r["file"])
        phrase_data[p].append(r["score"])

    print(f"\n  Per-phrase (core):")
    for phrase, scores in sorted(phrase_data.items(), key=lambda x: np.mean(x[1])):
        arr = np.array(scores)
        n_pass = sum(1 for s in scores if s >= 0.50)
        print(f"    {phrase:<20}: n={len(scores):>3}, mean={arr.mean():.4f}, pass={n_pass}/{len(scores)}")

    # Extended phrases breakdown
    ext_phrase_data = defaultdict(list)
    for r in extended_pos:
        p = extract_phrase(r["file"])
        ext_phrase_data[p].append(r["score"])

    if ext_phrase_data:
        print(f"\n  Per-phrase (extended - NOT core wake phrases):")
        for phrase, scores in sorted(ext_phrase_data.items(), key=lambda x: np.mean(x[1])):
            arr = np.array(scores)
            n_pass = sum(1 for s in scores if s >= 0.50)
            print(f"    {phrase:<30}: n={len(scores):>3}, mean={arr.mean():.4f}, pass={n_pass}/{len(scores)}")

    return {
        "d_full": d_full,
        "d_core": d_core,
        "ci_full": (ci_low, ci_high),
        "ci_core": (ci_low_c, ci_high_c),
        "se_full": se,
        "se_core": se_c,
        "frr_full": frr_full,
        "frr_core": frr_core,
        "n_pos_all": len(positives),
        "n_pos_core": len(core_pos),
        "n_neg": len(negatives),
        "fp_count": fp_count,
    }


def main():
    # Core phrases = the ones the model was actually trained on
    core = {"viola", "hey viola", "ok viola"}

    mp = analyze_subset(
        "J:/CLAUDE/PROJECTS/Wakeword/eval_clean/scores_meanpool.csv",
        "MEAN-POOL MODEL (Production)",
        core,
    )

    mx = analyze_subset(
        "J:/CLAUDE/PROJECTS/Wakeword/eval_clean/scores_maxpool.csv",
        "MAXPOOL MODEL (Previous Production)",
        core,
    )

    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Metric':<35} {'MeanPool':<15} {'Maxpool':<15} {'Prior claim':<15}")
    print(f"{'-'*80}")
    print(f"{'d-prime (full eval set)':<35} {mp['d_full']:<15.4f} {mx['d_full']:<15.4f} {'11.56':>15}")
    print(f"{'d-prime (core phrases only)':<35} {mp['d_core']:<15.4f} {mx['d_core']:<15.4f} {'11.56':>15}")
    print(f"{'d-prime 95% CI (core)':<35} [{mp['ci_core'][0]:.2f},{mp['ci_core'][1]:.2f}]     [{mx['ci_core'][0]:.2f},{mx['ci_core'][1]:.2f}]")
    print(f"{'FRR @ 0.50 (full)':<35} {mp['frr_full']*100:<15.1f} {mx['frr_full']*100:<15.1f}")
    print(f"{'FRR @ 0.50 (core phrases)':<35} {mp['frr_core']*100:<15.1f} {mx['frr_core']*100:<15.1f}")
    print(f"{'FP count @ 0.50':<35} {mp['fp_count']:<15} {mx['fp_count']:<15}")
    print(f"{'N pos (all / core)':<35} {mp['n_pos_all']}/{mp['n_pos_core']}         {mx['n_pos_all']}/{mx['n_pos_core']}")
    print(f"{'N neg':<35} {mp['n_neg']:<15} {mx['n_neg']:<15}")


if __name__ == "__main__":
    main()
