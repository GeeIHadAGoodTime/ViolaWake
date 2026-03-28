#!/usr/bin/env python3
"""
Head-to-head benchmark: ViolaWake temporal_cnn vs OpenWakeWord
Each system evaluated on its OWN best wake word using identical methodology.

ViolaWake: "viola" (mean-pool production model)
OpenWakeWord: "alexa" (pre-trained model)

Same TTS voices, same augmentation, same negative corpus, same metrics.
"""
from __future__ import annotations

import csv
import json
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── ViolaWake imports ──
sys.path.insert(0, "J:/CLAUDE/PROJECTS/Wakeword/src")
from violawake_sdk.training.evaluate import evaluate_onnx_model

# ── OpenWakeWord imports ──
from openwakeword.model import Model as OWWModel

# ── Paths ──
EVAL_CLEAN = Path("J:/CLAUDE/PROJECTS/Wakeword/eval_clean")
OWW_POS_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_oww/oww_positives")
VIOLAWAKE_MODEL = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA/violawake_data/trained_models/viola_mlp_oww.onnx")
OUTPUT_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_oww")

THRESHOLDS = [0.30, 0.50, 0.70, 0.80, 0.90]


@dataclass
class BenchResult:
    name: str
    wake_word: str
    pos_scores: list[float] = field(default_factory=list)
    neg_scores: list[float] = field(default_factory=list)
    pos_files: list[str] = field(default_factory=list)
    neg_files: list[str] = field(default_factory=list)

    @property
    def pos_mean(self) -> float:
        return float(np.mean(self.pos_scores)) if self.pos_scores else 0.0

    @property
    def pos_std(self) -> float:
        return float(np.std(self.pos_scores)) if self.pos_scores else 0.0

    @property
    def neg_mean(self) -> float:
        return float(np.mean(self.neg_scores)) if self.neg_scores else 0.0

    @property
    def neg_std(self) -> float:
        return float(np.std(self.neg_scores)) if self.neg_scores else 0.0

    def cohens_d(self) -> float:
        if not self.pos_scores or not self.neg_scores:
            return 0.0
        pos = np.array(self.pos_scores)
        neg = np.array(self.neg_scores)
        pooled_std = np.sqrt((pos.std()**2 + neg.std()**2) / 2)
        if pooled_std == 0:
            return float('inf') if pos.mean() != neg.mean() else 0.0
        return float((pos.mean() - neg.mean()) / pooled_std)

    def metrics_at_threshold(self, threshold: float) -> dict:
        pos = np.array(self.pos_scores)
        neg = np.array(self.neg_scores)
        tp = int(np.sum(pos >= threshold))
        fn = int(np.sum(pos < threshold))
        fp = int(np.sum(neg >= threshold))
        tn = int(np.sum(neg < threshold))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        far = fp / len(neg) if len(neg) > 0 else 0.0
        frr = fn / len(pos) if len(pos) > 0 else 0.0
        return {
            "threshold": threshold,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "far": far, "frr": frr,
        }

    def roc_auc(self) -> float:
        """Compute ROC AUC using trapezoidal rule."""
        pos = np.array(self.pos_scores)
        neg = np.array(self.neg_scores)
        thresholds = np.linspace(0, 1, 1001)
        tpr_list = []
        fpr_list = []
        for t in thresholds:
            tpr = np.mean(pos >= t)
            fpr = np.mean(neg >= t)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        # Sort by FPR for proper AUC calculation
        pairs = sorted(zip(fpr_list, tpr_list))
        fpr_sorted = [p[0] for p in pairs]
        tpr_sorted = [p[1] for p in pairs]
        return float(np.trapz(tpr_sorted, fpr_sorted))

    def eer(self) -> float:
        """Approximate Equal Error Rate."""
        thresholds = np.linspace(0, 1, 1001)
        pos = np.array(self.pos_scores)
        neg = np.array(self.neg_scores)
        best_diff = float('inf')
        best_eer = 0.5
        for t in thresholds:
            frr = np.mean(pos < t)
            far = np.mean(neg >= t)
            diff = abs(frr - far)
            if diff < best_diff:
                best_diff = diff
                best_eer = (frr + far) / 2
        return float(best_eer)

    def bootstrap_ci(self, n_boot: int = 10000, seed: int = 42) -> tuple[float, float, float]:
        """Bootstrap 95% CI for Cohen's d."""
        rng = np.random.default_rng(seed)
        pos = np.array(self.pos_scores)
        neg = np.array(self.neg_scores)
        ds = []
        for _ in range(n_boot):
            bp = rng.choice(pos, size=len(pos), replace=True)
            bn = rng.choice(neg, size=len(neg), replace=True)
            pooled = np.sqrt((bp.std()**2 + bn.std()**2) / 2)
            if pooled > 0:
                ds.append(float((bp.mean() - bn.mean()) / pooled))
        ds = sorted(ds)
        lo = ds[int(0.025 * len(ds))]
        hi = ds[int(0.975 * len(ds))]
        return float(np.mean(ds)), lo, hi


# ── OWW Evaluation ──

def evaluate_oww(model_name: str = "alexa") -> BenchResult:
    """Evaluate OWW on its own wake word using same negatives as ViolaWake."""
    print(f"\n{'='*70}")
    print(f"Evaluating OpenWakeWord: {model_name}")
    print(f"{'='*70}")

    oww = OWWModel()
    result = BenchResult(name="OpenWakeWord", wake_word=model_name)

    # Score positives
    pos_files = sorted(OWW_POS_DIR.glob("*.wav"))
    print(f"Scoring {len(pos_files)} positive samples...")
    for i, f in enumerate(pos_files):
        score = score_oww_clip(oww, str(f), model_name)
        result.pos_scores.append(score)
        result.pos_files.append(str(f))
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(pos_files)} done...")

    # Score negatives (same corpus as ViolaWake)
    neg_dirs = [
        EVAL_CLEAN / "negatives" / "adversarial_tts",
        EVAL_CLEAN / "negatives" / "noise",
        EVAL_CLEAN / "negatives" / "speech",
    ]
    neg_files = []
    for d in neg_dirs:
        neg_files.extend(sorted(d.glob("*.wav")))

    print(f"Scoring {len(neg_files)} negative samples...")
    for i, f in enumerate(neg_files):
        score = score_oww_clip(oww, str(f), model_name)
        result.neg_scores.append(score)
        result.neg_files.append(str(f))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(neg_files)} done...")

    print(f"OWW done: {len(result.pos_scores)} pos, {len(result.neg_scores)} neg")
    return result


def score_oww_clip(model: OWWModel, wav_path: str, model_name: str) -> float:
    """Score a single WAV file through OWW, return max prediction for model_name."""
    model.reset()
    try:
        predictions = model.predict_clip(wav_path)
    except Exception as e:
        print(f"  ERROR scoring {wav_path}: {e}")
        return 0.0

    if not predictions:
        return 0.0

    max_score = max(p.get(model_name, 0.0) for p in predictions)
    return float(max_score)


# ── ViolaWake Evaluation ──

def evaluate_violawake() -> BenchResult:
    """Evaluate ViolaWake mean-pool on its trained phrases from eval_clean."""
    print(f"\n{'='*70}")
    print("Evaluating ViolaWake (mean-pool) on 'viola'")
    print(f"{'='*70}")

    # Use the existing evaluate_onnx_model but we need raw scores
    # Let's read from the existing scores CSV which was already generated
    scores_csv = EVAL_CLEAN / "scores_meanpool.csv"

    result = BenchResult(name="ViolaWake", wake_word="viola")

    # Filter to trained phrases only (viola, hey viola, ok viola)
    trained_phrases = {"viola", "hey_viola", "ok_viola"}

    with open(scores_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filepath = row["file"]
            score = float(row["score"])
            label = row["label"]

            if label == "positive":
                # Check if trained phrase
                fname = Path(filepath).stem.lower()
                # Extract phrase from filename: voice_phrase[_variant].wav
                # e.g. en-AU-NatashaNeural_hey_viola_noisy
                is_trained = False
                for phrase in trained_phrases:
                    if f"_{phrase}" in fname or fname.endswith(phrase):
                        is_trained = True
                        break
                # Also exclude untrained phrases
                if "viola_wake_up" in fname or "viola_please" in fname:
                    is_trained = False

                if is_trained:
                    result.pos_scores.append(score)
                    result.pos_files.append(filepath)
            else:
                result.neg_scores.append(score)
                result.neg_files.append(filepath)

    print(f"ViolaWake done: {len(result.pos_scores)} pos (trained only), {len(result.neg_scores)} neg")
    return result


# ── Report Generation ──

def generate_report(vw: BenchResult, oww: BenchResult) -> str:
    """Generate the head-to-head comparison report."""
    lines = []
    lines.append("# Wake Word Detection Benchmark: ViolaWake vs OpenWakeWord")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **ViolaWake**: `viola_mlp_oww.onnx` (mean-pool MLP on OWW embeddings) detecting **\"viola\"**")
    lines.append(f"  - Positives: {len(vw.pos_scores)} TTS-generated samples (trained phrases: viola, hey viola, ok viola)")
    lines.append(f"  - Negatives: {len(vw.neg_scores)} samples (adversarial confusables + common speech + noise)")
    lines.append(f"- **OpenWakeWord**: pre-trained `alexa` model detecting **\"alexa\"**")
    lines.append(f"  - Positives: {len(oww.pos_scores)} TTS-generated samples (alexa, hey alexa, ok alexa)")
    lines.append(f"  - Negatives: {len(oww.neg_scores)} samples (same corpus as ViolaWake)")
    lines.append("")
    lines.append("**Methodology**: Same 20 Edge TTS voices (en-US, en-GB, en-AU, en-IN, en-ZA, en-IE, en-CA),")
    lines.append("same augmentations (clean + noisy + reverb), same negative corpus, same metrics.")
    lines.append("Each system evaluated on its own best wake word -- this is NOT about detecting 'viola' with OWW.")
    lines.append("")

    # Summary at threshold 0.50
    vw_m = vw.metrics_at_threshold(0.50)
    oww_m = oww.metrics_at_threshold(0.50)

    lines.append("## Results at Threshold 0.50")
    lines.append("")
    lines.append("| Metric | ViolaWake (viola) | OWW (alexa) |")
    lines.append("|--------|-------------------|-------------|")
    lines.append(f"| **Cohen's d** | **{vw.cohens_d():.2f}** | **{oww.cohens_d():.2f}** |")
    lines.append(f"| ROC AUC | {vw.roc_auc():.4f} | {oww.roc_auc():.4f} |")
    lines.append(f"| EER | {vw.eer():.4f} | {oww.eer():.4f} |")
    lines.append(f"| FAR | {vw_m['far']*100:.1f}% ({vw_m['fp']}/{len(vw.neg_scores)}) | {oww_m['far']*100:.1f}% ({oww_m['fp']}/{len(oww.neg_scores)}) |")
    lines.append(f"| FRR | {vw_m['frr']*100:.1f}% ({vw_m['fn']}/{len(vw.pos_scores)}) | {oww_m['frr']*100:.1f}% ({oww_m['fn']}/{len(oww.pos_scores)}) |")
    lines.append(f"| Precision | {vw_m['precision']:.4f} | {oww_m['precision']:.4f} |")
    lines.append(f"| Recall | {vw_m['recall']:.4f} | {oww_m['recall']:.4f} |")
    lines.append(f"| F1 | {vw_m['f1']:.4f} | {oww_m['f1']:.4f} |")
    lines.append("")

    # Multi-threshold
    lines.append("## Multi-Threshold Results")
    lines.append("")
    lines.append("| Threshold | VW FAR | VW FRR | VW F1 | OWW FAR | OWW FRR | OWW F1 |")
    lines.append("|-----------|--------|--------|-------|---------|---------|--------|")
    for t in THRESHOLDS:
        vm = vw.metrics_at_threshold(t)
        om = oww.metrics_at_threshold(t)
        lines.append(f"| {t:.2f} | {vm['far']*100:.1f}% | {vm['frr']*100:.1f}% | {vm['f1']:.3f} | {om['far']*100:.1f}% | {om['frr']*100:.1f}% | {om['f1']:.3f} |")
    lines.append("")

    # Score distributions
    lines.append("## Score Distributions")
    lines.append("")
    lines.append("| Statistic | ViolaWake (viola) | OWW (alexa) |")
    lines.append("|-----------|-------------------|-------------|")
    lines.append(f"| Pos mean | {vw.pos_mean:.4f} | {oww.pos_mean:.4f} |")
    lines.append(f"| Pos std | {vw.pos_std:.4f} | {oww.pos_std:.4f} |")
    lines.append(f"| Pos min | {min(vw.pos_scores):.4f} | {min(oww.pos_scores):.4f} |")
    lines.append(f"| Pos max | {max(vw.pos_scores):.4f} | {max(oww.pos_scores):.4f} |")
    lines.append(f"| Neg mean | {vw.neg_mean:.4f} | {oww.neg_mean:.4f} |")
    lines.append(f"| Neg std | {vw.neg_std:.4f} | {oww.neg_std:.4f} |")
    lines.append(f"| Neg min | {min(vw.neg_scores):.4f} | {min(oww.neg_scores):.4f} |")
    lines.append(f"| Neg max | {max(vw.neg_scores):.4f} | {max(oww.neg_scores):.4f} |")
    lines.append("")

    # Bootstrap CIs
    vw_boot_mean, vw_lo, vw_hi = vw.bootstrap_ci()
    oww_boot_mean, oww_lo, oww_hi = oww.bootstrap_ci()
    lines.append("## Bootstrap Confidence Intervals (10,000 resamples)")
    lines.append("")
    lines.append("| System | Cohen's d | 95% CI |")
    lines.append("|--------|-----------|--------|")
    lines.append(f"| ViolaWake (viola) | {vw.cohens_d():.2f} | [{vw_lo:.2f}, {vw_hi:.2f}] |")
    lines.append(f"| OWW (alexa) | {oww.cohens_d():.2f} | [{oww_lo:.2f}, {oww_hi:.2f}] |")
    lines.append("")

    # Per-phrase breakdown for OWW
    lines.append("## Per-Phrase Breakdown: OWW (alexa)")
    lines.append("")
    phrase_scores: dict[str, list[float]] = {}
    for f, s in zip(oww.pos_files, oww.pos_scores):
        fname = Path(f).stem.lower()
        if "ok_alexa" in fname:
            phrase = "ok alexa"
        elif "hey_alexa" in fname:
            phrase = "hey alexa"
        else:
            phrase = "alexa"
        phrase_scores.setdefault(phrase, []).append(s)

    lines.append("| Phrase | N | Mean Score | Pass @ 0.50 |")
    lines.append("|--------|---|-----------|-------------|")
    for phrase in ["alexa", "hey alexa", "ok alexa"]:
        scores = phrase_scores.get(phrase, [])
        n = len(scores)
        mean = np.mean(scores) if scores else 0
        passing = sum(1 for s in scores if s >= 0.50)
        lines.append(f"| {phrase} | {n} | {mean:.4f} | {passing}/{n} ({passing/n*100:.1f}%) |")
    lines.append("")

    # Per-phrase for ViolaWake
    lines.append("## Per-Phrase Breakdown: ViolaWake (viola)")
    lines.append("")
    vw_phrase_scores: dict[str, list[float]] = {}
    for f, s in zip(vw.pos_files, vw.pos_scores):
        fname = Path(f).stem.lower()
        if "ok_viola" in fname:
            phrase = "ok viola"
        elif "hey_viola" in fname:
            phrase = "hey viola"
        else:
            phrase = "viola"
        vw_phrase_scores.setdefault(phrase, []).append(s)

    lines.append("| Phrase | N | Mean Score | Pass @ 0.50 |")
    lines.append("|--------|---|-----------|-------------|")
    for phrase in ["viola", "hey viola", "ok viola"]:
        scores = vw_phrase_scores.get(phrase, [])
        n = len(scores)
        mean = np.mean(scores) if scores else 0
        passing = sum(1 for s in scores if s >= 0.50)
        pct = passing/n*100 if n > 0 else 0
        lines.append(f"| {phrase} | {n} | {mean:.4f} | {passing}/{n} ({pct:.1f}%) |")
    lines.append("")

    # Analysis
    lines.append("## Analysis")
    lines.append("")
    vw_d = vw.cohens_d()
    oww_d = oww.cohens_d()
    vw_auc = vw.roc_auc()
    oww_auc = oww.roc_auc()

    if oww_d > vw_d:
        ratio = oww_d / vw_d if vw_d > 0 else float('inf')
        lines.append(f"**OpenWakeWord wins on separability.** OWW's Cohen's d ({oww_d:.2f}) is {ratio:.1f}x higher than ViolaWake's ({vw_d:.2f}).")
    elif vw_d > oww_d:
        ratio = vw_d / oww_d if oww_d > 0 else float('inf')
        lines.append(f"**ViolaWake wins on separability.** ViolaWake's Cohen's d ({vw_d:.2f}) is {ratio:.1f}x higher than OWW's ({oww_d:.2f}).")
    else:
        lines.append(f"**Tie on separability.** Both systems achieve Cohen's d of {vw_d:.2f}.")

    lines.append("")
    lines.append(f"ROC AUC: ViolaWake {vw_auc:.4f} vs OWW {oww_auc:.4f}.")
    lines.append(f"EER: ViolaWake {vw.eer():.4f} vs OWW {oww.eer():.4f}.")
    lines.append("")

    lines.append("### Context")
    lines.append("")
    lines.append("- OWW's \"alexa\" model was trained by Amazon/David Scripka on a large corpus of real speech.")
    lines.append("- ViolaWake's \"viola\" model is a custom MLP trained on OWW's embedding features with TTS-generated data.")
    lines.append("- Both are evaluated here on TTS-generated audio only (no real recordings).")
    lines.append("- The negative corpus contains phonetically adversarial words for ViolaWake (vanilla, villa, violet, etc.)")
    lines.append("  but NOT adversarial words for OWW (e.g., no 'alexis', 'election', 'electric'). This gives OWW a slight")
    lines.append("  advantage on FAR since the negatives weren't designed to trick it.")
    lines.append("")

    return "\n".join(lines)


def save_scores_csv(result: BenchResult, path: Path):
    """Save raw scores to CSV for further analysis."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "score", "threshold_pass_050"])
        for fp, s in zip(result.pos_files, result.pos_scores):
            writer.writerow([fp, "positive", f"{s:.6f}", s >= 0.50])
        for fp, s in zip(result.neg_files, result.neg_scores):
            writer.writerow([fp, "negative", f"{s:.6f}", s >= 0.50])


def main():
    # Run ViolaWake evaluation (from existing scores)
    vw = evaluate_violawake()

    # Run OWW evaluation
    oww = evaluate_oww("alexa")

    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'ViolaWake (viola)':<25} {'OWW (alexa)':<25}")
    print(f"{'-'*75}")
    label = "Cohen's d"
    print(f"{label:<25} {vw.cohens_d():<25.4f} {oww.cohens_d():<25.4f}")
    print(f"{'ROC AUC':<25} {vw.roc_auc():<25.4f} {oww.roc_auc():<25.4f}")
    print(f"{'EER':<25} {vw.eer():<25.4f} {oww.eer():<25.4f}")
    print(f"{'Pos mean':<25} {vw.pos_mean:<25.4f} {oww.pos_mean:<25.4f}")
    print(f"{'Pos std':<25} {vw.pos_std:<25.4f} {oww.pos_std:<25.4f}")
    print(f"{'Neg mean':<25} {vw.neg_mean:<25.4f} {oww.neg_mean:<25.4f}")
    print(f"{'Neg std':<25} {vw.neg_std:<25.4f} {oww.neg_std:<25.4f}")

    for t in THRESHOLDS:
        vm = vw.metrics_at_threshold(t)
        om = oww.metrics_at_threshold(t)
        print(f"\nAt threshold {t:.2f}:")
        print(f"  VW: FAR={vm['far']*100:.1f}% FRR={vm['frr']*100:.1f}% F1={vm['f1']:.3f}")
        print(f"  OWW: FAR={om['far']*100:.1f}% FRR={om['frr']*100:.1f}% F1={om['f1']:.3f}")

    # Save outputs
    report = generate_report(vw, oww)
    report_path = OUTPUT_DIR / "BENCHMARK_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    save_scores_csv(oww, OUTPUT_DIR / "oww_alexa_scores.csv")
    save_scores_csv(vw, OUTPUT_DIR / "violawake_viola_scores.csv")
    print(f"Scores saved to: {OUTPUT_DIR}")

    # Save raw results as JSON
    raw = {
        "violawake": {
            "cohens_d": vw.cohens_d(),
            "roc_auc": vw.roc_auc(),
            "eer": vw.eer(),
            "pos_mean": vw.pos_mean,
            "pos_std": vw.pos_std,
            "neg_mean": vw.neg_mean,
            "neg_std": vw.neg_std,
            "n_pos": len(vw.pos_scores),
            "n_neg": len(vw.neg_scores),
        },
        "oww": {
            "cohens_d": oww.cohens_d(),
            "roc_auc": oww.roc_auc(),
            "eer": oww.eer(),
            "pos_mean": oww.pos_mean,
            "pos_std": oww.pos_std,
            "neg_mean": oww.neg_mean,
            "neg_std": oww.neg_std,
            "n_pos": len(oww.pos_scores),
            "n_neg": len(oww.neg_scores),
        },
    }
    with open(OUTPUT_DIR / "benchmark_results.json", 'w') as f:
        json.dump(raw, f, indent=2)


if __name__ == "__main__":
    main()
