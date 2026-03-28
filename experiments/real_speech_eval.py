"""
Comprehensive real-speech evaluation and cross-speaker generalization test for ViolaWake.

Scores ALL real recordings through the best model and computes:
- Per-file scores
- Detection rates at multiple thresholds
- FA rates on adversarial/confusable negatives
- EER
- Cross-speaker breakdown (Jihad vs Sierra)
- Cross-condition breakdown (normal, music, whisper)
- Duplicate analysis between training and eval sets
"""

from __future__ import annotations

import hashlib
import json
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from openwakeword.model import Model as OWWModel

# Add src to path for violawake_sdk
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from violawake_sdk.audio import load_audio, center_crop
from violawake_sdk._constants import CLIP_SAMPLES

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA/violawake_data")
MODEL_PATH = Path(__file__).parent / "models" / "D_combined_bce_s42.onnx"
OUTPUT_JSON = Path(__file__).parent / "real_speech_eval.json"
OUTPUT_MD = Path(__file__).parent / "REAL_SPEECH_EVAL.md"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# ── Data sources ────────────────────────────────────────────────────────
# Training positives (real speech)
TRAIN_POS_DIRS = {
    "jihad_music":   DATA_ROOT / "positives" / "real" / "jihad_Music",
    "jihad_normal":  DATA_ROOT / "positives" / "real" / "jihad_normal",
    "jihad_whisper": DATA_ROOT / "positives" / "real" / "jihad_whisper",
    "sierra_normal": DATA_ROOT / "positives" / "real" / "sierra_Normal",
    "sierra_music":  DATA_ROOT / "positives" / "real" / "sierra_music",
    "sierra_whisper":DATA_ROOT / "positives" / "real" / "sierra_whisper",
    "jihad_legacy":  DATA_ROOT / "positives" / "real" / "speaker_jihad",
}
LEGACY_CUSTOM_DIR = DATA_ROOT / "positives" / "legacy_custom"

# Eval positives (real speech)
EVAL_POS_DIRS = {
    "jihad_music":   DATA_ROOT / "eval_real" / "positives" / "jihad_music",
    "jihad_normal":  DATA_ROOT / "eval_real" / "positives" / "jihad_normal",
    "jihad_whisper": DATA_ROOT / "eval_real" / "positives" / "jihad_whisper",
    "sierra_music":  DATA_ROOT / "eval_real" / "positives" / "sierra_music",
    "sierra_normal": DATA_ROOT / "eval_real" / "positives" / "sierra_normal",
    "sierra_whisper":DATA_ROOT / "eval_real" / "positives" / "sierra_whisper",
}

# Eval negatives (real speech)
EVAL_NEG_DIRS = {
    "adversarial":    DATA_ROOT / "eval_real" / "negatives" / "adversarial",
    "legacy_hard":    DATA_ROOT / "eval_real" / "negatives" / "legacy_hard",
    "music":          DATA_ROOT / "eval_real" / "negatives" / "music",
    "music_hard":     DATA_ROOT / "eval_real" / "negatives" / "music_hard",
    "real_confusable": DATA_ROOT / "eval_real" / "negatives" / "real_confusable",
    "real_fp_captures":DATA_ROOT / "eval_real" / "negatives" / "real_fp_captures",
}


def load_model():
    """Load OWW preprocessor and ONNX model."""
    print(f"Loading model: {MODEL_PATH.name}")
    oww = OWWModel()
    preprocessor = oww.preprocessor
    sess = ort.InferenceSession(str(MODEL_PATH))
    input_name = sess.get_inputs()[0].name
    return preprocessor, sess, input_name


def score_file(path: str, preprocessor, sess, input_name) -> float:
    """Score a single audio file. Returns probability [0, 1]."""
    audio = load_audio(path)
    audio = center_crop(audio, CLIP_SAMPLES)
    audio_int16 = np.clip(audio, -1, 1)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    if len(audio_int16) < CLIP_SAMPLES:
        audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
    embs = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    emb = embs.mean(axis=1).astype(np.float32)
    logit = sess.run(None, {input_name: emb})[0][0][0]
    # Apply sigmoid if model outputs logits
    if logit < -10 or logit > 10:
        return float(1.0 / (1.0 + np.exp(-logit)))
    # If output is already in ~[0,1] range, use as-is
    if 0 <= logit <= 1:
        return float(logit)
    # Otherwise sigmoid
    return float(1.0 / (1.0 + np.exp(-logit)))


def score_directory(dir_path: Path, preprocessor, sess, input_name, label: str) -> list[dict]:
    """Score all .wav files in a directory."""
    files = sorted(glob.glob(str(dir_path / "*.wav")))
    results = []
    for f in files:
        try:
            score = score_file(f, preprocessor, sess, input_name)
            results.append({
                "file": os.path.basename(f),
                "path": f,
                "score": score,
                "label": label,
            })
        except Exception as e:
            results.append({
                "file": os.path.basename(f),
                "path": f,
                "score": None,
                "label": label,
                "error": str(e),
            })
    return results


def compute_eer(pos_scores: list[float], neg_scores: list[float]) -> tuple[float, float]:
    """Compute Equal Error Rate and its threshold."""
    all_scores = sorted(set(pos_scores + neg_scores))
    best_eer = 1.0
    best_thresh = 0.5
    for thresh in np.linspace(0, 1, 1001):
        fnr = sum(1 for s in pos_scores if s < thresh) / max(len(pos_scores), 1)
        fpr = sum(1 for s in neg_scores if s >= thresh) / max(len(neg_scores), 1)
        eer_candidate = (fnr + fpr) / 2
        if abs(fnr - fpr) < abs(best_eer * 2 - (sum(1 for s in pos_scores if s < best_thresh) / max(len(pos_scores), 1) + sum(1 for s in neg_scores if s >= best_thresh) / max(len(neg_scores), 1))):
            best_eer = (fnr + fpr) / 2
            best_thresh = thresh
            if abs(fnr - fpr) < 0.005:
                break
    # More precise: find crossing point
    fnrs = []
    fprs = []
    threshs = np.linspace(0, 1, 10001)
    for thresh in threshs:
        fnr = sum(1 for s in pos_scores if s < thresh) / max(len(pos_scores), 1)
        fpr = sum(1 for s in neg_scores if s >= thresh) / max(len(neg_scores), 1)
        fnrs.append(fnr)
        fprs.append(fpr)
    fnrs = np.array(fnrs)
    fprs = np.array(fprs)
    # Find where FNR crosses FPR
    diff = fnrs - fprs
    idx = np.argmin(np.abs(diff))
    eer = (fnrs[idx] + fprs[idx]) / 2
    return float(eer), float(threshs[idx])


def find_duplicates(all_results: list[dict]) -> dict:
    """Find files that exist in both training and eval sets."""
    hash_map = defaultdict(list)
    for r in all_results:
        if r.get("score") is None:
            continue
        try:
            h = hashlib.md5(open(r["path"], "rb").read()).hexdigest()
            hash_map[h].append(r["path"])
        except Exception:
            pass
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def main():
    preprocessor, sess, input_name = load_model()

    all_results = {}
    timing = {}

    # ── Score training positives ──────────────────────────────────────
    print("\n═══ TRAINING POSITIVES (real speech) ═══")
    train_pos_results = {}
    for name, dir_path in TRAIN_POS_DIRS.items():
        if not dir_path.exists():
            print(f"  SKIP {name}: {dir_path} not found")
            continue
        t0 = time.time()
        results = score_directory(dir_path, preprocessor, sess, input_name, f"train_pos_{name}")
        dt = time.time() - t0
        scores = [r["score"] for r in results if r["score"] is not None]
        if scores:
            print(f"  {name}: {len(results)} files, mean={np.mean(scores):.4f}, "
                  f"min={np.min(scores):.4f}, max={np.max(scores):.4f} ({dt:.1f}s)")
        else:
            print(f"  {name}: {len(results)} files, NO VALID SCORES")
        train_pos_results[name] = results

    # Legacy custom
    print("\n═══ LEGACY CUSTOM POSITIVES ═══")
    t0 = time.time()
    legacy_results = score_directory(LEGACY_CUSTOM_DIR, preprocessor, sess, input_name, "legacy_custom")
    dt = time.time() - t0
    legacy_scores = [r["score"] for r in legacy_results if r["score"] is not None]
    print(f"  legacy_custom: {len(legacy_results)} files, mean={np.mean(legacy_scores):.4f}, "
          f"min={np.min(legacy_scores):.4f}, max={np.max(legacy_scores):.4f} ({dt:.1f}s)")

    # ── Score eval positives ──────────────────────────────────────────
    print("\n═══ EVAL POSITIVES (real speech) ═══")
    eval_pos_results = {}
    for name, dir_path in EVAL_POS_DIRS.items():
        if not dir_path.exists():
            print(f"  SKIP {name}: {dir_path} not found")
            continue
        t0 = time.time()
        results = score_directory(dir_path, preprocessor, sess, input_name, f"eval_pos_{name}")
        dt = time.time() - t0
        scores = [r["score"] for r in results if r["score"] is not None]
        if scores:
            print(f"  {name}: {len(results)} files, mean={np.mean(scores):.4f}, "
                  f"min={np.min(scores):.4f}, max={np.max(scores):.4f} ({dt:.1f}s)")
        else:
            print(f"  {name}: {len(results)} files, NO VALID SCORES")
        eval_pos_results[name] = results

    # ── Score eval negatives ──────────────────────────────────────────
    print("\n═══ EVAL NEGATIVES (real speech) ═══")
    eval_neg_results = {}
    for name, dir_path in EVAL_NEG_DIRS.items():
        if not dir_path.exists():
            print(f"  SKIP {name}: {dir_path} not found")
            continue
        t0 = time.time()
        results = score_directory(dir_path, preprocessor, sess, input_name, f"eval_neg_{name}")
        dt = time.time() - t0
        scores = [r["score"] for r in results if r["score"] is not None]
        if scores:
            print(f"  {name}: {len(results)} files, mean={np.mean(scores):.4f}, "
                  f"min={np.min(scores):.4f}, max={np.max(scores):.4f} ({dt:.1f}s)")
        else:
            print(f"  {name}: {len(results)} files, NO VALID SCORES")
        eval_neg_results[name] = results

    # ── Aggregate metrics ─────────────────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("          AGGREGATE METRICS")
    print("═══════════════════════════════════════════")

    # Collect all eval positive scores
    all_eval_pos = []
    for name, results in eval_pos_results.items():
        all_eval_pos.extend([r["score"] for r in results if r["score"] is not None])

    # Collect all eval negative scores
    all_eval_neg = []
    for name, results in eval_neg_results.items():
        all_eval_neg.extend([r["score"] for r in results if r["score"] is not None])

    # Collect all training positive scores
    all_train_pos = []
    for name, results in train_pos_results.items():
        all_train_pos.extend([r["score"] for r in results if r["score"] is not None])

    print(f"\nEval positives: {len(all_eval_pos)} files")
    print(f"Eval negatives: {len(all_eval_neg)} files")
    print(f"Train positives: {len(all_train_pos)} files")
    print(f"Legacy custom: {len(legacy_scores)} files")

    # Detection rates at thresholds
    print("\n── Detection Rate (eval positives) ──")
    det_rates = {}
    for t in THRESHOLDS:
        rate = sum(1 for s in all_eval_pos if s >= t) / max(len(all_eval_pos), 1)
        det_rates[str(t)] = rate
        print(f"  threshold={t:.2f}: {rate:.4f} ({sum(1 for s in all_eval_pos if s >= t)}/{len(all_eval_pos)})")

    # FA rates at thresholds
    print("\n── False Accept Rate (eval negatives) ──")
    fa_rates = {}
    for t in THRESHOLDS:
        rate = sum(1 for s in all_eval_neg if s >= t) / max(len(all_eval_neg), 1)
        fa_rates[str(t)] = rate
        print(f"  threshold={t:.2f}: {rate:.4f} ({sum(1 for s in all_eval_neg if s >= t)}/{len(all_eval_neg)})")

    # EER
    eer, eer_thresh = compute_eer(all_eval_pos, all_eval_neg)
    print(f"\nEER: {eer:.4f} at threshold {eer_thresh:.4f}")

    # ── Per-speaker breakdown ─────────────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("       CROSS-SPEAKER BREAKDOWN")
    print("═══════════════════════════════════════════")

    speaker_metrics = {}
    for speaker in ["jihad", "sierra"]:
        # Eval positives for this speaker
        speaker_scores = []
        for name, results in eval_pos_results.items():
            if name.startswith(speaker):
                speaker_scores.extend([r["score"] for r in results if r["score"] is not None])
        if not speaker_scores:
            continue

        # Train positives for this speaker
        train_speaker_scores = []
        for name, results in train_pos_results.items():
            if name.startswith(speaker):
                train_speaker_scores.extend([r["score"] for r in results if r["score"] is not None])

        sm = {
            "eval_count": len(speaker_scores),
            "eval_mean": float(np.mean(speaker_scores)),
            "eval_std": float(np.std(speaker_scores)),
            "eval_min": float(np.min(speaker_scores)),
            "eval_max": float(np.max(speaker_scores)),
            "train_count": len(train_speaker_scores),
            "train_mean": float(np.mean(train_speaker_scores)) if train_speaker_scores else None,
        }
        # Detection rates per speaker
        for t in THRESHOLDS:
            sm[f"det_rate_{t}"] = sum(1 for s in speaker_scores if s >= t) / len(speaker_scores)
        speaker_metrics[speaker] = sm

        print(f"\n  {speaker.upper()}:")
        print(f"    Eval: {sm['eval_count']} files, mean={sm['eval_mean']:.4f}, "
              f"std={sm['eval_std']:.4f}, range=[{sm['eval_min']:.4f}, {sm['eval_max']:.4f}]")
        if train_speaker_scores:
            print(f"    Train: {sm['train_count']} files, mean={sm['train_mean']:.4f}")
        for t in [0.5, 0.7, 0.8, 0.9]:
            print(f"    Det@{t}: {sm[f'det_rate_{t}']:.4f}")

    # ── Per-condition breakdown ───────────────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("       PER-CONDITION BREAKDOWN")
    print("═══════════════════════════════════════════")

    condition_metrics = {}
    for condition in ["music", "normal", "whisper"]:
        cond_scores = []
        for name, results in eval_pos_results.items():
            if condition in name:
                cond_scores.extend([r["score"] for r in results if r["score"] is not None])
        if not cond_scores:
            continue
        cm = {
            "count": len(cond_scores),
            "mean": float(np.mean(cond_scores)),
            "std": float(np.std(cond_scores)),
            "min": float(np.min(cond_scores)),
            "max": float(np.max(cond_scores)),
        }
        for t in THRESHOLDS:
            cm[f"det_rate_{t}"] = sum(1 for s in cond_scores if s >= t) / len(cond_scores)
        condition_metrics[condition] = cm

        print(f"\n  {condition.upper()}:")
        print(f"    {cm['count']} files, mean={cm['mean']:.4f}, "
              f"std={cm['std']:.4f}, range=[{cm['min']:.4f}, {cm['max']:.4f}]")
        for t in [0.5, 0.7, 0.8, 0.9]:
            print(f"    Det@{t}: {cm[f'det_rate_{t}']:.4f}")

    # ── Per-negative-category breakdown ───────────────────────────────
    print("\n═══════════════════════════════════════════")
    print("       PER-NEGATIVE-CATEGORY BREAKDOWN")
    print("═══════════════════════════════════════════")

    neg_cat_metrics = {}
    for name, results in eval_neg_results.items():
        scores = [r["score"] for r in results if r["score"] is not None]
        if not scores:
            continue
        nm = {
            "count": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
        }
        for t in THRESHOLDS:
            nm[f"fa_rate_{t}"] = sum(1 for s in scores if s >= t) / len(scores)
        neg_cat_metrics[name] = nm

        print(f"\n  {name}:")
        print(f"    {nm['count']} files, mean={nm['mean']:.4f}, "
              f"std={nm['std']:.4f}, range=[{nm['min']:.4f}, {nm['max']:.4f}]")
        for t in [0.5, 0.7, 0.8, 0.9]:
            print(f"    FA@{t}: {nm[f'fa_rate_{t}']:.4f}")

    # ── Duplicate analysis ────────────────────────────────────────────
    all_scored = []
    for results in train_pos_results.values():
        all_scored.extend(results)
    all_scored.extend(legacy_results)
    for results in eval_pos_results.values():
        all_scored.extend(results)
    for results in eval_neg_results.values():
        all_scored.extend(results)

    dupes = find_duplicates(all_scored)
    print(f"\n═══ DUPLICATE ANALYSIS ═══")
    print(f"Duplicate file groups (same content in train+eval): {len(dupes)}")

    # ── Score distribution histograms (for plotting) ──────────────────
    def histogram_data(scores, bins=50):
        counts, edges = np.histogram(scores, bins=bins, range=(0, 1))
        return {"counts": counts.tolist(), "edges": edges.tolist()}

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "model": MODEL_PATH.name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "eval_pos_count": len(all_eval_pos),
            "eval_neg_count": len(all_eval_neg),
            "train_pos_count": len(all_train_pos),
            "legacy_custom_count": len(legacy_scores),
            "eer": eer,
            "eer_threshold": eer_thresh,
            "detection_rates": det_rates,
            "fa_rates": fa_rates,
            "duplicate_groups": len(dupes),
        },
        "speaker_metrics": speaker_metrics,
        "condition_metrics": condition_metrics,
        "negative_category_metrics": neg_cat_metrics,
        "distributions": {
            "eval_positives": histogram_data(all_eval_pos),
            "eval_negatives": histogram_data(all_eval_neg),
            "train_positives": histogram_data(all_train_pos),
            "legacy_custom": histogram_data(legacy_scores),
        },
        "per_file_scores": {
            "train_positives": {
                name: [{"file": r["file"], "score": r["score"]} for r in results]
                for name, results in train_pos_results.items()
            },
            "legacy_custom": [{"file": r["file"], "score": r["score"]} for r in legacy_results],
            "eval_positives": {
                name: [{"file": r["file"], "score": r["score"]} for r in results]
                for name, results in eval_pos_results.items()
            },
            "eval_negatives": {
                name: [{"file": r["file"], "score": r["score"]} for r in results]
                for name, results in eval_neg_results.items()
            },
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # ── Generate markdown report ──────────────────────────────────────
    generate_report(output)
    print(f"Report saved to {OUTPUT_MD}")


def generate_report(data: dict):
    """Generate a markdown report from the evaluation results."""
    s = data["summary"]
    lines = [
        "# ViolaWake Real Speech Evaluation",
        "",
        f"**Model**: `{data['model']}`",
        f"**Date**: {data['timestamp']}",
        "",
        "## Dataset Overview",
        "",
        "| Set | Count | Description |",
        "|-----|-------|-------------|",
        f"| Eval Positives | {s['eval_pos_count']} | Real \"Viola\" recordings (held-out eval) |",
        f"| Eval Negatives | {s['eval_neg_count']} | Real adversarial/confusable/music negatives |",
        f"| Train Positives | {s['train_pos_count']} | Real \"Viola\" recordings (in training set) |",
        f"| Legacy Custom | {s['legacy_custom_count']} | Legacy custom recordings |",
        f"| Duplicate Groups | {s['duplicate_groups']} | Files appearing in both train and eval |",
        "",
        "## Overall Performance",
        "",
        f"**EER: {s['eer']:.4f} ({s['eer']*100:.2f}%) at threshold {s['eer_threshold']:.4f}**",
        "",
        "### Detection Rate (Eval Positives)",
        "",
        "| Threshold | Detection Rate | Detected/Total |",
        "|-----------|---------------|----------------|",
    ]

    for t_str, rate in s["detection_rates"].items():
        t = float(t_str)
        detected = int(rate * s["eval_pos_count"])
        lines.append(f"| {t:.2f} | {rate:.4f} ({rate*100:.1f}%) | {detected}/{s['eval_pos_count']} |")

    lines.extend([
        "",
        "### False Accept Rate (Eval Negatives)",
        "",
        "| Threshold | FA Rate | False Accepts/Total |",
        "|-----------|---------|---------------------|",
    ])

    for t_str, rate in s["fa_rates"].items():
        t = float(t_str)
        fa = int(rate * s["eval_neg_count"])
        lines.append(f"| {t:.2f} | {rate:.4f} ({rate*100:.1f}%) | {fa}/{s['eval_neg_count']} |")

    # Cross-speaker
    lines.extend([
        "",
        "## Cross-Speaker Analysis",
        "",
        "| Speaker | Eval Count | Mean Score | Std | Det@0.5 | Det@0.7 | Det@0.8 | Det@0.9 |",
        "|---------|-----------|------------|-----|---------|---------|---------|---------|",
    ])
    for speaker, m in data["speaker_metrics"].items():
        lines.append(
            f"| {speaker.title()} | {m['eval_count']} | {m['eval_mean']:.4f} | "
            f"{m['eval_std']:.4f} | {m.get('det_rate_0.5', 0):.4f} | {m.get('det_rate_0.7', 0):.4f} | "
            f"{m.get('det_rate_0.8', 0):.4f} | {m.get('det_rate_0.9', 0):.4f} |"
        )

    # Per-condition
    lines.extend([
        "",
        "## Per-Condition Analysis",
        "",
        "| Condition | Count | Mean Score | Std | Det@0.5 | Det@0.7 | Det@0.8 | Det@0.9 |",
        "|-----------|-------|------------|-----|---------|---------|---------|---------|",
    ])
    for cond, m in data["condition_metrics"].items():
        lines.append(
            f"| {cond.title()} | {m['count']} | {m['mean']:.4f} | "
            f"{m['std']:.4f} | {m.get('det_rate_0.5', 0):.4f} | {m.get('det_rate_0.7', 0):.4f} | "
            f"{m.get('det_rate_0.8', 0):.4f} | {m.get('det_rate_0.9', 0):.4f} |"
        )

    # Per-negative-category
    lines.extend([
        "",
        "## Negative Category Breakdown",
        "",
        "| Category | Count | Mean Score | Max Score | FA@0.5 | FA@0.7 | FA@0.8 | FA@0.9 |",
        "|----------|-------|------------|-----------|--------|--------|--------|--------|",
    ])
    for cat, m in data["negative_category_metrics"].items():
        lines.append(
            f"| {cat} | {m['count']} | {m['mean']:.4f} | "
            f"{m['max']:.4f} | {m.get('fa_rate_0.5', 0):.4f} | {m.get('fa_rate_0.7', 0):.4f} | "
            f"{m.get('fa_rate_0.8', 0):.4f} | {m.get('fa_rate_0.9', 0):.4f} |"
        )

    # Score distribution summary
    lines.extend([
        "",
        "## Score Distributions",
        "",
        "### Eval Positives",
        "```",
    ])
    hist = data["distributions"]["eval_positives"]
    edges = hist["edges"]
    counts = hist["counts"]
    max_count = max(counts) if counts else 1
    for i, c in enumerate(counts):
        bar = "#" * int(40 * c / max_count) if max_count > 0 else ""
        if c > 0:
            lines.append(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {bar} ({c})")
    lines.append("```")

    lines.extend([
        "",
        "### Eval Negatives",
        "```",
    ])
    hist = data["distributions"]["eval_negatives"]
    edges = hist["edges"]
    counts = hist["counts"]
    max_count = max(counts) if counts else 1
    for i, c in enumerate(counts):
        bar = "#" * int(40 * c / max_count) if max_count > 0 else ""
        if c > 0:
            lines.append(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {bar} ({c})")
    lines.append("```")

    lines.append("")

    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
