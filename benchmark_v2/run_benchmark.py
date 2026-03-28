#!/usr/bin/env python3
"""
ViolaWake vs OpenWakeWord — Corrected Benchmark v2

Unified streaming inference pipeline for both systems:
1. Load audio as 16kHz int16
2. Feed in 1280-sample chunks (80ms, matching OWW's native chunk size)
3. Collect per-chunk scores
4. File score = max of all chunk scores
5. Record all scores in CSV

Fixes every flaw from v1:
- Same negative corpus for both
- No wake word contamination in negatives
- Adversarial negatives for BOTH wake words
- Same inference pipeline (streaming, chunk-by-chunk)
- EER and FAR-at-fixed-FRR as primary metrics
- Same positive sample count (180 each)
"""
from __future__ import annotations

import csv
import json
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ── ViolaWake imports ──
sys.path.insert(0, "J:/CLAUDE/PROJECTS/Wakeword/src")
from violawake_sdk.wake_detector import WakeDetector

# ── OpenWakeWord imports ──
from openwakeword.model import Model as OWWModel

# ── Paths ──
CORPUS_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_v2/corpus")
OUTPUT_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_v2")

# Streaming config — matches OWW backbone's native chunk size
CHUNK_SAMPLES = 1280  # 80ms at 16kHz
SAMPLE_RATE = 16_000


# ── Audio loading ──

def load_wav_int16(path: Path) -> np.ndarray:
    """Load a WAV file as mono int16 at 16kHz."""
    try:
        with wave.open(str(path), "rb") as wf:
            assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
            assert wf.getframerate() == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {wf.getframerate()}Hz"
            assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth()*8}-bit"
            frames = wf.readframes(wf.getnframes())
            return np.frombuffer(frames, dtype=np.int16)
    except Exception:
        # Fallback: use soundfile for non-standard WAVs
        import soundfile as sf_mod
        audio, sr = sf_mod.read(str(path), dtype="int16")
        if sr != SAMPLE_RATE:
            # Simple resample via linear interpolation
            ratio = SAMPLE_RATE / sr
            indices = np.arange(0, len(audio), 1 / ratio)[:int(len(audio) * ratio)]
            audio = np.interp(indices, np.arange(len(audio)), audio.astype(np.float64)).astype(np.int16)
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio


# ── Streaming scoring ──

def score_violawake_streaming(detector: WakeDetector, audio: np.ndarray) -> tuple[float, list[float]]:
    """Score audio through ViolaWake in 1280-sample streaming chunks.

    Returns (max_score, all_chunk_scores).
    """
    detector.reset()
    chunk_scores = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            # Pad last chunk with zeros
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode="constant")
        # Feed as int16 bytes (matching production pipeline)
        score = detector.process(chunk.astype(np.int16).tobytes())
        chunk_scores.append(score)

    max_score = max(chunk_scores) if chunk_scores else 0.0
    return max_score, chunk_scores


def score_oww_streaming(model: OWWModel, audio: np.ndarray, model_name: str = "alexa") -> tuple[float, list[float]]:
    """Score audio through OWW in 1280-sample streaming chunks.

    Returns (max_score, all_chunk_scores).
    """
    model.reset()
    chunk_scores = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start:start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)), mode="constant")
        # OWW expects int16 numpy array
        prediction = model.predict(chunk.astype(np.int16))
        score = prediction.get(model_name, 0.0)
        chunk_scores.append(score)

    max_score = max(chunk_scores) if chunk_scores else 0.0
    return max_score, chunk_scores


# ── Metric computation ──

@dataclass
class SystemResults:
    name: str
    wake_word: str
    pos_scores: list[float] = field(default_factory=list)
    neg_scores: list[float] = field(default_factory=list)
    pos_files: list[str] = field(default_factory=list)
    neg_files: list[str] = field(default_factory=list)

    def _arr(self, which: str) -> np.ndarray:
        return np.array(self.pos_scores if which == "pos" else self.neg_scores)

    def roc_curve(self, n_thresholds: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute ROC curve (FPR, TPR, thresholds)."""
        pos = self._arr("pos")
        neg = self._arr("neg")
        thresholds = np.linspace(0, 1, n_thresholds)
        tpr = np.array([np.mean(pos >= t) for t in thresholds])
        fpr = np.array([np.mean(neg >= t) for t in thresholds])
        return fpr, tpr, thresholds

    def roc_auc(self) -> float:
        """Compute ROC AUC using trapezoidal rule."""
        fpr, tpr, _ = self.roc_curve()
        # Sort by FPR ascending
        order = np.argsort(fpr)
        return float(np.trapz(tpr[order], fpr[order]))

    def eer(self) -> float:
        """Compute Equal Error Rate (where FAR == FRR)."""
        pos = self._arr("pos")
        neg = self._arr("neg")
        thresholds = np.linspace(0, 1, 10001)
        best_diff = float("inf")
        best_eer = 0.5
        for t in thresholds:
            frr = np.mean(pos < t)
            far = np.mean(neg >= t)
            diff = abs(frr - far)
            if diff < best_diff:
                best_diff = diff
                best_eer = (frr + far) / 2
        return float(best_eer)

    def far_at_frr(self, target_frr: float) -> float:
        """Find FAR when FRR is closest to target."""
        pos = self._arr("pos")
        neg = self._arr("neg")
        thresholds = np.linspace(0, 1, 10001)
        best_diff = float("inf")
        best_far = 1.0
        for t in thresholds:
            frr = np.mean(pos < t)
            far = np.mean(neg >= t)
            diff = abs(frr - target_frr)
            if diff < best_diff:
                best_diff = diff
                best_far = far
        return float(best_far)

    def frr_at_far(self, target_far: float) -> float:
        """Find FRR when FAR is closest to target."""
        pos = self._arr("pos")
        neg = self._arr("neg")
        thresholds = np.linspace(0, 1, 10001)
        best_diff = float("inf")
        best_frr = 1.0
        for t in thresholds:
            frr = np.mean(pos < t)
            far = np.mean(neg >= t)
            diff = abs(far - target_far)
            if diff < best_diff:
                best_diff = diff
                best_frr = frr
        return float(best_frr)

    def score_stats(self, which: str) -> dict:
        """Compute score distribution statistics."""
        arr = self._arr(which)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "q25": float(np.percentile(arr, 25)),
            "q75": float(np.percentile(arr, 75)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    def per_phrase_breakdown(self, phrases: dict[str, str]) -> dict[str, dict]:
        """Break down positive scores by phrase."""
        results = {}
        for phrase_key, phrase_text in phrases.items():
            scores = []
            for f, s in zip(self.pos_files, self.pos_scores):
                fname = Path(f).stem.lower()
                if f"_{phrase_key}" in fname or fname.startswith(phrase_key):
                    # Exclude cross-phrase matches
                    # e.g., for "viola", exclude "hey_viola" and "ok_viola"
                    if phrase_key == self.wake_word:
                        if f"hey_{phrase_key}" in fname or f"ok_{phrase_key}" in fname:
                            continue
                    scores.append(s)
            results[phrase_text] = {
                "n": len(scores),
                "mean": float(np.mean(scores)) if scores else 0.0,
                "std": float(np.std(scores)) if scores else 0.0,
                "min": float(np.min(scores)) if scores else 0.0,
                "max": float(np.max(scores)) if scores else 0.0,
            }
        return results


# ── File collection ──

def collect_negatives() -> list[Path]:
    """Collect all negative WAV files from the shared corpus."""
    neg_dirs = [
        CORPUS_DIR / "negatives" / "adversarial_viola",
        CORPUS_DIR / "negatives" / "adversarial_alexa",
        CORPUS_DIR / "negatives" / "speech",
        CORPUS_DIR / "negatives" / "speech_existing",
        CORPUS_DIR / "negatives" / "noise",
    ]
    files = []
    for d in neg_dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.wav")))
    return files


def collect_positives(wake_word: str) -> list[Path]:
    """Collect positive WAV files for a wake word."""
    pos_dir = CORPUS_DIR / "positives" / wake_word
    if not pos_dir.exists():
        raise FileNotFoundError(f"Positive directory not found: {pos_dir}")
    return sorted(pos_dir.glob("*.wav"))


# ── Evaluation runners ──

def evaluate_violawake(neg_files: list[Path], pos_files: list[Path]) -> SystemResults:
    """Evaluate ViolaWake temporal_cnn on the corpus using streaming inference."""
    print(f"\n{'='*70}")
    print("Evaluating ViolaWake (temporal_cnn) — streaming 80ms chunks")
    print(f"{'='*70}")

    # Initialize detector with very low threshold to get raw scores
    # Use direct path to avoid auto-download attempt
    model_path = "J:/CLAUDE/PROJECTS/Wakeword/experiments/models/j5_temporal/temporal_cnn.onnx"
    detector = WakeDetector(model=model_path, threshold=0.01, cooldown_s=0.0)
    result = SystemResults(name="ViolaWake", wake_word="viola")

    # Score positives
    print(f"Scoring {len(pos_files)} positive samples...")
    t0 = time.time()
    for i, f in enumerate(pos_files):
        try:
            audio = load_wav_int16(f)
            max_score, _ = score_violawake_streaming(detector, audio)
            result.pos_scores.append(max_score)
            result.pos_files.append(str(f))
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}")
            result.pos_scores.append(0.0)
            result.pos_files.append(str(f))
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(pos_files)} done ({elapsed:.1f}s)")

    # Score negatives
    print(f"Scoring {len(neg_files)} negative samples...")
    t0 = time.time()
    for i, f in enumerate(neg_files):
        try:
            audio = load_wav_int16(f)
            max_score, _ = score_violawake_streaming(detector, audio)
            result.neg_scores.append(max_score)
            result.neg_files.append(str(f))
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}")
            result.neg_scores.append(0.0)
            result.neg_files.append(str(f))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(neg_files)} done ({elapsed:.1f}s)")

    print(f"ViolaWake done: {len(result.pos_scores)} pos, {len(result.neg_scores)} neg")
    return result


def evaluate_oww(neg_files: list[Path], pos_files: list[Path]) -> SystemResults:
    """Evaluate OpenWakeWord (alexa) on the corpus using streaming inference."""
    print(f"\n{'='*70}")
    print("Evaluating OpenWakeWord (alexa) — streaming 80ms chunks")
    print(f"{'='*70}")

    oww = OWWModel()
    result = SystemResults(name="OpenWakeWord", wake_word="alexa")

    # Score positives
    print(f"Scoring {len(pos_files)} positive samples...")
    t0 = time.time()
    for i, f in enumerate(pos_files):
        try:
            audio = load_wav_int16(f)
            max_score, _ = score_oww_streaming(oww, audio, "alexa")
            result.pos_scores.append(max_score)
            result.pos_files.append(str(f))
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}")
            result.pos_scores.append(0.0)
            result.pos_files.append(str(f))
        if (i + 1) % 30 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(pos_files)} done ({elapsed:.1f}s)")

    # Score negatives
    print(f"Scoring {len(neg_files)} negative samples...")
    t0 = time.time()
    for i, f in enumerate(neg_files):
        try:
            audio = load_wav_int16(f)
            max_score, _ = score_oww_streaming(oww, audio, "alexa")
            result.neg_scores.append(max_score)
            result.neg_files.append(str(f))
        except Exception as e:
            print(f"  ERROR on {f.name}: {e}")
            result.neg_scores.append(0.0)
            result.neg_files.append(str(f))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(neg_files)} done ({elapsed:.1f}s)")

    print(f"OWW done: {len(result.pos_scores)} pos, {len(result.neg_scores)} neg")
    return result


# ── CSV output ──

def save_scores_csv(result: SystemResults, path: Path) -> None:
    """Save all scores to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "score", "category"])
        for fp, s in zip(result.pos_files, result.pos_scores):
            writer.writerow([fp, "positive", f"{s:.6f}", _categorize_file(fp)])
        for fp, s in zip(result.neg_files, result.neg_scores):
            writer.writerow([fp, "negative", f"{s:.6f}", _categorize_file(fp)])


def _categorize_file(filepath: str) -> str:
    """Categorize a file by its directory."""
    p = Path(filepath)
    parts = p.parts
    for part in parts:
        if part.startswith("adversarial_"):
            return part
        if part in ("speech", "speech_existing", "noise"):
            return part
        if part in ("viola", "alexa"):
            return f"positive_{part}"
    return "unknown"


# ── Report generation ──

def generate_report(vw: SystemResults, oww: SystemResults, neg_count_by_category: dict) -> str:
    """Generate the definitive benchmark report."""
    lines = []
    lines.append("## ViolaWake vs OpenWakeWord -- Corrected Benchmark v2")
    lines.append("")

    # Methodology
    total_neg = sum(neg_count_by_category.values())
    lines.append("### Methodology")
    lines.append(f"- Shared negative corpus: {total_neg} files")
    for cat, count in sorted(neg_count_by_category.items()):
        lines.append(f"  - {cat}: {count} files")
    lines.append(f"- Matched positives: {len(vw.pos_scores)} viola, {len(oww.pos_scores)} alexa")
    lines.append("- Same 20 Edge TTS voices, same 3 augmentations (clean, noisy, reverb)")
    lines.append(f"- Streaming inference: {CHUNK_SAMPLES}-sample chunks (80ms at 16kHz), max-score per file")
    lines.append("- Primary metrics: EER, FAR@FRR")
    lines.append("")

    # Results table
    vw_eer = vw.eer()
    oww_eer = oww.eer()
    vw_auc = vw.roc_auc()
    oww_auc = oww.roc_auc()
    vw_pos = vw.score_stats("pos")
    oww_pos = oww.score_stats("pos")
    vw_neg = vw.score_stats("neg")
    oww_neg = oww.score_stats("neg")

    lines.append("### Results")
    lines.append("")
    lines.append("| Metric | ViolaWake (viola) | OWW (alexa) |")
    lines.append("|--------|-------------------|-------------|")
    lines.append(f"| EER | {vw_eer*100:.2f}% | {oww_eer*100:.2f}% |")
    lines.append(f"| ROC AUC | {vw_auc:.4f} | {oww_auc:.4f} |")

    # FAR at fixed FRR
    for target_frr in [0.01, 0.03, 0.05, 0.10]:
        vw_far = vw.far_at_frr(target_frr)
        oww_far = oww.far_at_frr(target_frr)
        lines.append(f"| FAR @ FRR={target_frr*100:.0f}% | {vw_far*100:.2f}% | {oww_far*100:.2f}% |")

    # FRR at fixed FAR
    for target_far in [0.001, 0.005, 0.01, 0.05]:
        vw_frr = vw.frr_at_far(target_far)
        oww_frr = oww.frr_at_far(target_far)
        lines.append(f"| FRR @ FAR={target_far*100:.1f}% | {vw_frr*100:.2f}% | {oww_frr*100:.2f}% |")

    lines.append("")

    # Score distributions
    lines.append("### Score Distributions")
    lines.append("")
    lines.append("| Statistic | ViolaWake (viola) | OWW (alexa) |")
    lines.append("|-----------|-------------------|-------------|")
    lines.append(f"| Pos mean +/- std | {vw_pos['mean']:.4f} +/- {vw_pos['std']:.4f} | {oww_pos['mean']:.4f} +/- {oww_pos['std']:.4f} |")
    lines.append(f"| Pos median [IQR] | {vw_pos['median']:.4f} [{vw_pos['q25']:.4f}-{vw_pos['q75']:.4f}] | {oww_pos['median']:.4f} [{oww_pos['q25']:.4f}-{oww_pos['q75']:.4f}] |")
    lines.append(f"| Pos range | [{vw_pos['min']:.4f}, {vw_pos['max']:.4f}] | [{oww_pos['min']:.4f}, {oww_pos['max']:.4f}] |")
    lines.append(f"| Neg mean +/- std | {vw_neg['mean']:.4f} +/- {vw_neg['std']:.4f} | {oww_neg['mean']:.4f} +/- {oww_neg['std']:.4f} |")
    lines.append(f"| Neg median [IQR] | {vw_neg['median']:.4f} [{vw_neg['q25']:.4f}-{vw_neg['q75']:.4f}] | {oww_neg['median']:.4f} [{oww_neg['q25']:.4f}-{oww_neg['q75']:.4f}] |")
    lines.append(f"| Neg range | [{vw_neg['min']:.4f}, {vw_neg['max']:.4f}] | [{oww_neg['min']:.4f}, {oww_neg['max']:.4f}] |")
    lines.append("")

    # Per-phrase breakdown
    lines.append("### Per-Phrase Breakdown")
    lines.append("")
    lines.append("| Phrase | VW Score (mean +/- std) | OWW Score (mean +/- std) |")
    lines.append("|--------|------------------------|--------------------------|")

    vw_phrases = {
        "viola": "Viola",
        "hey_viola": "Hey Viola",
        "ok_viola": "OK Viola",
    }
    oww_phrases = {
        "alexa": "Alexa",
        "hey_alexa": "Hey Alexa",
        "ok_alexa": "OK Alexa",
    }

    vw_breakdown = vw.per_phrase_breakdown(vw_phrases)
    oww_breakdown = oww.per_phrase_breakdown(oww_phrases)

    # Align by position: standalone, hey, ok
    phrase_pairs = [
        ("Viola", "Alexa", "Standalone word"),
        ("Hey Viola", "Hey Alexa", '"hey [word]"'),
        ("OK Viola", "OK Alexa", '"ok [word]"'),
    ]
    for vw_phrase, oww_phrase, label in phrase_pairs:
        vwd = vw_breakdown.get(vw_phrase, {"mean": 0, "std": 0, "n": 0})
        owwd = oww_breakdown.get(oww_phrase, {"mean": 0, "std": 0, "n": 0})
        lines.append(
            f"| {label} | {vwd['mean']:.4f} +/- {vwd['std']:.4f} (n={vwd['n']}) | "
            f"{owwd['mean']:.4f} +/- {owwd['std']:.4f} (n={owwd['n']}) |"
        )
    lines.append("")

    # Adversarial breakdown
    lines.append("### Adversarial Resistance")
    lines.append("")
    lines.append("How each system scores on the OTHER system's adversarial words:")
    lines.append("")

    # VW scores on alexa-adversarial
    vw_alexa_adv_scores = [
        s for f, s in zip(vw.neg_files, vw.neg_scores)
        if "adversarial_alexa" in f
    ]
    vw_viola_adv_scores = [
        s for f, s in zip(vw.neg_files, vw.neg_scores)
        if "adversarial_viola" in f
    ]
    oww_alexa_adv_scores = [
        s for f, s in zip(oww.neg_files, oww.neg_scores)
        if "adversarial_alexa" in f
    ]
    oww_viola_adv_scores = [
        s for f, s in zip(oww.neg_files, oww.neg_scores)
        if "adversarial_viola" in f
    ]

    lines.append("| Adversarial Set | VW mean score | OWW mean score |")
    lines.append("|-----------------|---------------|----------------|")
    if vw_viola_adv_scores:
        lines.append(
            f"| Viola-confusables (n={len(vw_viola_adv_scores)}) | "
            f"{np.mean(vw_viola_adv_scores):.4f} +/- {np.std(vw_viola_adv_scores):.4f} | "
            f"{np.mean(oww_viola_adv_scores):.4f} +/- {np.std(oww_viola_adv_scores):.4f} |"
        )
    if vw_alexa_adv_scores:
        lines.append(
            f"| Alexa-confusables (n={len(vw_alexa_adv_scores)}) | "
            f"{np.mean(vw_alexa_adv_scores):.4f} +/- {np.std(vw_alexa_adv_scores):.4f} | "
            f"{np.mean(oww_alexa_adv_scores):.4f} +/- {np.std(oww_alexa_adv_scores):.4f} |"
        )
    lines.append("")

    # Analysis
    lines.append("### Analysis")
    lines.append("")
    if vw_eer < oww_eer:
        lines.append(f"**ViolaWake has lower EER** ({vw_eer*100:.2f}% vs {oww_eer*100:.2f}%), indicating better overall discrimination.")
    elif oww_eer < vw_eer:
        lines.append(f"**OWW has lower EER** ({oww_eer*100:.2f}% vs {vw_eer*100:.2f}%), indicating better overall discrimination.")
    else:
        lines.append(f"**Tied on EER** ({vw_eer*100:.2f}%).")

    lines.append("")
    if vw_auc > oww_auc:
        lines.append(f"ViolaWake has higher AUC ({vw_auc:.4f} vs {oww_auc:.4f}).")
    elif oww_auc > vw_auc:
        lines.append(f"OWW has higher AUC ({oww_auc:.4f} vs {vw_auc:.4f}).")
    lines.append("")

    lines.append("### Context")
    lines.append("")
    lines.append("- OWW's 'alexa' model: pre-trained by David Scripka on large real-speech corpus")
    lines.append("- ViolaWake's 'viola' model: temporal CNN on OWW embeddings, TTS-trained")
    lines.append("- Both evaluated on TTS audio only (no real recordings in this benchmark)")
    lines.append("- Adversarial negatives included for BOTH systems (v1 only had viola adversarials)")
    lines.append("- Negatives do NOT contain either actual wake word")
    lines.append("")

    return "\n".join(lines)


# ── Main ──

def main():
    print("=" * 70)
    print("ViolaWake vs OpenWakeWord -- Corrected Benchmark v2")
    print("=" * 70)

    # Verify corpus exists
    if not CORPUS_DIR.exists():
        print(f"ERROR: Corpus not found at {CORPUS_DIR}")
        print("Run build_corpus.py first!")
        sys.exit(1)

    # Collect files
    neg_files = collect_negatives()
    viola_pos = collect_positives("viola")
    alexa_pos = collect_positives("alexa")

    # Count negatives by category
    neg_count_by_category: dict[str, int] = {}
    for f in neg_files:
        cat = _categorize_file(str(f))
        neg_count_by_category[cat] = neg_count_by_category.get(cat, 0) + 1

    print(f"\nCorpus summary:")
    print(f"  Viola positives: {len(viola_pos)}")
    print(f"  Alexa positives: {len(alexa_pos)}")
    print(f"  Shared negatives: {len(neg_files)}")
    for cat, count in sorted(neg_count_by_category.items()):
        print(f"    {cat}: {count}")

    # Run evaluations
    t_start = time.time()

    vw = evaluate_violawake(neg_files, viola_pos)
    oww = evaluate_oww(neg_files, alexa_pos)

    t_total = time.time() - t_start

    # Print summary
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"Total time: {t_total:.1f}s")
    print()
    print(f"{'Metric':<25} {'ViolaWake (viola)':<25} {'OWW (alexa)':<25}")
    print(f"{'-'*75}")
    print(f"{'EER':<25} {vw.eer()*100:<25.2f}% {oww.eer()*100:<25.2f}%")
    print(f"{'ROC AUC':<25} {vw.roc_auc():<25.4f} {oww.roc_auc():<25.4f}")
    print(f"{'Pos mean':<25} {np.mean(vw.pos_scores):<25.4f} {np.mean(oww.pos_scores):<25.4f}")
    print(f"{'Neg mean':<25} {np.mean(vw.neg_scores):<25.4f} {np.mean(oww.neg_scores):<25.4f}")

    for target_frr in [0.01, 0.05, 0.10]:
        label = f"FAR @ FRR={target_frr*100:.0f}%"
        print(f"{label:<25} {vw.far_at_frr(target_frr)*100:<25.2f}% {oww.far_at_frr(target_frr)*100:<25.2f}%")

    # Save outputs
    report = generate_report(vw, oww, neg_count_by_category)
    report_path = OUTPUT_DIR / "BENCHMARK_REPORT_v2.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport: {report_path}")

    save_scores_csv(vw, OUTPUT_DIR / "violawake_scores_v2.csv")
    save_scores_csv(oww, OUTPUT_DIR / "oww_scores_v2.csv")
    print(f"Scores: {OUTPUT_DIR}")

    # Save JSON results
    raw = {
        "metadata": {
            "version": "2",
            "chunk_samples": CHUNK_SAMPLES,
            "sample_rate": SAMPLE_RATE,
            "n_voices": 20,
            "n_viola_pos": len(vw.pos_scores),
            "n_alexa_pos": len(oww.pos_scores),
            "n_negatives": len(neg_files),
            "neg_categories": neg_count_by_category,
            "total_time_s": t_total,
        },
        "violawake": {
            "eer": vw.eer(),
            "roc_auc": vw.roc_auc(),
            "pos_stats": vw.score_stats("pos"),
            "neg_stats": vw.score_stats("neg"),
            "far_at_frr": {
                f"frr_{int(frr*100)}pct": vw.far_at_frr(frr)
                for frr in [0.01, 0.03, 0.05, 0.10]
            },
            "frr_at_far": {
                f"far_{f:.1f}pct".replace(".", "p"): vw.frr_at_far(f / 100)
                for f in [0.1, 0.5, 1.0, 5.0]
            },
        },
        "oww": {
            "eer": oww.eer(),
            "roc_auc": oww.roc_auc(),
            "pos_stats": oww.score_stats("pos"),
            "neg_stats": oww.score_stats("neg"),
            "far_at_frr": {
                f"frr_{int(frr*100)}pct": oww.far_at_frr(frr)
                for frr in [0.01, 0.03, 0.05, 0.10]
            },
            "frr_at_far": {
                f"far_{f:.1f}pct".replace(".", "p"): oww.frr_at_far(f / 100)
                for f in [0.1, 0.5, 1.0, 5.0]
            },
        },
    }
    with open(OUTPUT_DIR / "benchmark_results_v2.json", "w") as f:
        json.dump(raw, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
