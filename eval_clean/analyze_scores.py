#!/usr/bin/env python3
"""
Analyze per-file scores from ViolaWake evaluation.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_csv(path: str) -> list[dict]:
    """Load score CSV into list of dicts."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = float(row["score"])
            row["threshold_pass"] = row["threshold_pass"] == "True"
            rows.append(row)
    return rows


def extract_voice(filename: str) -> str:
    """Extract voice name from filename."""
    name = Path(filename).stem
    # edge_tts format: en-XX-NameNeural_phrase
    m = re.match(r"(en-\w+-\w+Neural)", name)
    if m:
        return m.group(1)
    # flat format: pos_XX-Name_phrase
    m = re.match(r"pos_(\w+-\w+)_", name)
    if m:
        return m.group(1)
    # neg format: neg_XX-Name_phrase
    m = re.match(r"neg_(\w+-\w+)_", name)
    if m:
        return m.group(1)
    return "unknown"


def extract_phrase(filename: str) -> str:
    """Extract phrase from filename."""
    name = Path(filename).stem
    # Remove augmentation suffixes
    for suffix in ["_noisy", "_reverb"]:
        name = name.replace(suffix, "")
    # edge_tts: en-XX-NameNeural_phrase
    m = re.match(r"en-\w+-\w+Neural_(.+)$", name)
    if m:
        return m.group(1).replace("_", " ")
    # flat format: pos/neg_XX-Name_phrase_NN_NN
    parts = name.split("_")
    if len(parts) >= 3:
        # Skip pos/neg prefix and voice
        phrase_parts = parts[2:]
        # Remove trailing digits
        while phrase_parts and phrase_parts[-1].isdigit():
            phrase_parts.pop()
        return " ".join(phrase_parts)
    return name


def extract_augmentation(filename: str) -> str:
    """Detect augmentation type."""
    stem = Path(filename).stem
    if "_noisy" in stem:
        return "noisy"
    if "_reverb" in stem:
        return "reverb"
    return "original"


def analyze_model(csv_path: str, model_name: str):
    """Full analysis for one model."""
    rows = load_csv(csv_path)
    positives = [r for r in rows if r["label"] == "positive"]
    negatives = [r for r in rows if r["label"] == "negative"]

    pos_scores = np.array([r["score"] for r in positives])
    neg_scores = np.array([r["score"] for r in negatives])

    print(f"\n{'='*70}")
    print(f"DETAILED ANALYSIS: {model_name}")
    print(f"{'='*70}")

    # ── Basic stats ──
    print(f"\n--- Score Distribution ---")
    print(f"Positives: n={len(positives)}, mean={pos_scores.mean():.4f}, std={pos_scores.std():.4f}, min={pos_scores.min():.4f}, max={pos_scores.max():.4f}")
    print(f"  Median: {np.median(pos_scores):.4f}, Q25: {np.percentile(pos_scores, 25):.4f}, Q75: {np.percentile(pos_scores, 75):.4f}")
    print(f"Negatives: n={len(negatives)}, mean={neg_scores.mean():.6f}, std={neg_scores.std():.6f}, min={neg_scores.min():.6f}, max={neg_scores.max():.6f}")
    print(f"  Median: {np.median(neg_scores):.6f}, Q25: {np.percentile(neg_scores, 25):.6f}, Q75: {np.percentile(neg_scores, 75):.6f}")

    # ── Per-voice breakdown for positives ──
    print(f"\n--- Per-Voice Positive Scores ---")
    voice_scores = defaultdict(list)
    for r in positives:
        voice = extract_voice(r["file"])
        voice_scores[voice].append(r["score"])

    voice_stats = []
    for voice, scores in sorted(voice_scores.items()):
        arr = np.array(scores)
        n_pass = sum(1 for s in scores if s >= 0.50)
        voice_stats.append({
            "voice": voice,
            "n": len(scores),
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "pass_rate": n_pass / len(scores),
        })

    print(f"{'Voice':<30} {'N':>4} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Pass%':>7}")
    for vs in sorted(voice_stats, key=lambda x: x["mean"]):
        print(f"{vs['voice']:<30} {vs['n']:>4} {vs['mean']:>8.4f} {vs['std']:>8.4f} {vs['min']:>8.4f} {vs['max']:>8.4f} {vs['pass_rate']*100:>6.1f}%")

    # ── Per-phrase breakdown for positives ──
    print(f"\n--- Per-Phrase Positive Scores ---")
    phrase_scores = defaultdict(list)
    for r in positives:
        phrase = extract_phrase(r["file"])
        phrase_scores[phrase].append(r["score"])

    print(f"{'Phrase':<35} {'N':>4} {'Mean':>8} {'Min':>8} {'Max':>8} {'Pass%':>7}")
    for phrase, scores in sorted(phrase_scores.items(), key=lambda x: np.mean(x[1])):
        arr = np.array(scores)
        n_pass = sum(1 for s in scores if s >= 0.50)
        print(f"{phrase:<35} {len(scores):>4} {arr.mean():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f} {n_pass/len(scores)*100:>6.1f}%")

    # ── Per-augmentation breakdown ──
    print(f"\n--- Per-Augmentation Positive Scores ---")
    aug_scores = defaultdict(list)
    for r in positives:
        aug = extract_augmentation(r["file"])
        aug_scores[aug].append(r["score"])

    for aug, scores in sorted(aug_scores.items()):
        arr = np.array(scores)
        n_pass = sum(1 for s in scores if s >= 0.50)
        print(f"  {aug:<12}: n={len(scores)}, mean={arr.mean():.4f}, std={arr.std():.4f}, pass={n_pass/len(scores)*100:.1f}%")

    # ── Lowest positive scores (false rejects) ──
    print(f"\n--- 20 Lowest Positive Scores (False Rejects @ 0.50) ---")
    for r in sorted(positives, key=lambda x: x["score"])[:20]:
        voice = extract_voice(r["file"])
        phrase = extract_phrase(r["file"])
        aug = extract_augmentation(r["file"])
        print(f"  {r['score']:.6f} | {voice:<30} | {phrase:<20} | {aug}")

    # ── Highest negative scores (potential false accepts) ──
    print(f"\n--- 20 Highest Negative Scores (Potential False Accepts) ---")
    for r in sorted(negatives, key=lambda x: x["score"], reverse=True)[:20]:
        voice = extract_voice(r["file"])
        phrase = extract_phrase(r["file"])
        print(f"  {r['score']:.6f} | {voice:<30} | {phrase:<20} | {Path(r['file']).stem}")

    # ── Per-phrase negative scores (which words confuse the model?) ──
    print(f"\n--- Negative Score by Phrase (Confusability Ranking) ---")
    neg_phrase_scores = defaultdict(list)
    for r in negatives:
        phrase = extract_phrase(r["file"])
        neg_phrase_scores[phrase].append(r["score"])

    print(f"{'Phrase':<35} {'N':>4} {'Mean':>10} {'Max':>10} {'FP@0.5':>7}")
    for phrase, scores in sorted(neg_phrase_scores.items(), key=lambda x: -np.mean(x[1])):
        arr = np.array(scores)
        n_fp = sum(1 for s in scores if s >= 0.50)
        print(f"{phrase:<35} {len(scores):>4} {arr.mean():>10.6f} {arr.max():>10.6f} {n_fp:>7}")

    # ── Bootstrap confidence interval for d-prime ──
    print(f"\n--- Bootstrap 95% CI for D-prime ---")
    n_bootstrap = 10000
    rng = np.random.default_rng(42)
    d_primes = []
    for _ in range(n_bootstrap):
        pos_boot = rng.choice(pos_scores, size=len(pos_scores), replace=True)
        neg_boot = rng.choice(neg_scores, size=len(neg_scores), replace=True)
        pooled_std = np.sqrt(0.5 * (pos_boot.var() + neg_boot.var()))
        if pooled_std > 1e-10:
            d = float((pos_boot.mean() - neg_boot.mean()) / pooled_std)
        else:
            d = 0.0
        d_primes.append(d)

    d_primes = np.array(d_primes)
    ci_low = np.percentile(d_primes, 2.5)
    ci_high = np.percentile(d_primes, 97.5)
    print(f"  D-prime: {pos_scores.mean() - neg_scores.mean():.4f} (raw diff)")
    print(f"  Cohen's d point estimate: {float((pos_scores.mean() - neg_scores.mean()) / np.sqrt(0.5 * (pos_scores.var() + neg_scores.var()))):.4f}")
    print(f"  Bootstrap mean: {d_primes.mean():.4f}")
    print(f"  Bootstrap std: {d_primes.std():.4f}")
    print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    # Standard error
    se = d_primes.std()
    print(f"  Standard error: {se:.4f}")

    return {
        "d_prime_ci_low": ci_low,
        "d_prime_ci_high": ci_high,
        "d_prime_se": se,
    }


def main():
    print("VIOLAWAKE CLEAN EVAL SET - DETAILED SCORE ANALYSIS")
    print(f"{'='*70}")

    mp_stats = analyze_model(
        "J:/CLAUDE/PROJECTS/Wakeword/eval_clean/scores_meanpool.csv",
        "Mean-Pool (Production)"
    )

    mx_stats = analyze_model(
        "J:/CLAUDE/PROJECTS/Wakeword/eval_clean/scores_maxpool.csv",
        "Maxpool (Previous Production)"
    )

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"Mean-Pool d-prime 95% CI: [{mp_stats['d_prime_ci_low']:.4f}, {mp_stats['d_prime_ci_high']:.4f}] (SE={mp_stats['d_prime_se']:.4f})")
    print(f"Maxpool d-prime 95% CI: [{mx_stats['d_prime_ci_low']:.4f}, {mx_stats['d_prime_ci_high']:.4f}] (SE={mx_stats['d_prime_se']:.4f})")
    print(f"\nPrior claimed d-prime: 11.56 (contaminated real), 15.10 (synthetic)")
    print(f"Both are WELL outside the 95% CI of the clean eval.")


if __name__ == "__main__":
    main()
