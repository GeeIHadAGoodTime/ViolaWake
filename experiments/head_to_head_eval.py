"""Head-to-head evaluation: ViolaWake temporal_cnn vs MLP (r3_10x_s42) vs raw OWW baseline.

Scores each test clip through the full WakeDetector pipeline (OWW backbone -> model)
to get proper scores, then computes FAR, FRR, Cohen's d, and grades.

Usage:
    python experiments/head_to_head_eval.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAKEWORD / "src"))

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20ms
CLIP_SAMPLES = 24000  # 1.5s

TEST_DIR = WAKEWORD / "eval_clean"
POS_DIR = TEST_DIR / "positives"
NEG_DIR = TEST_DIR / "negatives"

THRESHOLD = 0.50  # Evaluation threshold


def load_audio_clip(path: Path) -> np.ndarray | None:
    """Load audio file and return int16 array at 16kHz."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        # Pad or truncate to CLIP_SAMPLES
        if len(audio) < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
        else:
            audio = audio[:CLIP_SAMPLES]
        return (audio * 32767).clip(-32768, 32767).astype(np.int16)
    except Exception as e:
        print(f"  WARN: Failed to load {path.name}: {e}")
        return None


def score_clip_streaming(detector, audio_int16: np.ndarray) -> float:
    """Feed audio through detector frame-by-frame and return max score."""
    detector.reset()
    max_score = 0.0
    for start in range(0, len(audio_int16) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = audio_int16[start:start + FRAME_SAMPLES].tobytes()
        score = detector.process(frame)
        max_score = max(max_score, score)
    return max_score


def compute_metrics(pos_scores: np.ndarray, neg_scores: np.ndarray, threshold: float) -> dict:
    """Compute FAR, FRR, Cohen's d, ROC AUC, confusion matrix."""
    from sklearn.metrics import auc, roc_curve

    # Cohen's d
    pooled_std = np.sqrt(0.5 * (pos_scores.var() + neg_scores.var()))
    d_prime = float((pos_scores.mean() - neg_scores.mean()) / pooled_std) if pooled_std > 1e-10 else 0.0

    # ROC
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    # FAR/FRR at threshold
    tp = int(np.sum(pos_scores >= threshold))
    fn = int(np.sum(pos_scores < threshold))
    fp = int(np.sum(neg_scores >= threshold))
    tn = int(np.sum(neg_scores < threshold))

    frr = fn / (tp + fn) if (tp + fn) > 0 else 1.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 1.0

    # FAR per hour (assume each negative is ~1.5s clip)
    neg_hours = len(neg_scores) * 1.5 / 3600
    far_per_hour = fp / neg_hours if neg_hours > 0 else float("inf")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Optimal threshold sweep
    best_thresh, best_cost = 0.5, float("inf")
    best_far_opt, best_frr_opt = 1.0, 1.0
    for t in np.arange(0.0, 1.01, 0.01):
        f = float(np.sum(neg_scores >= t) / len(neg_scores))
        r = float(np.sum(pos_scores < t) / len(pos_scores))
        if f + r < best_cost:
            best_cost = f + r
            best_thresh = float(t)
            best_far_opt = f
            best_frr_opt = r

    # Grade
    if d_prime >= 15.0:
        grade = "A (Excellent)"
    elif d_prime >= 10.0:
        grade = "B (Good)"
    elif d_prime >= 5.0:
        grade = "C (Fair)"
    elif d_prime >= 2.0:
        grade = "D (Poor)"
    else:
        grade = "F (Fail)"

    return {
        "d_prime": d_prime,
        "far": far,
        "frr": frr,
        "far_per_hour": far_per_hour,
        "roc_auc": roc_auc,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "grade": grade,
        "optimal_threshold": best_thresh,
        "optimal_far": best_far_opt,
        "optimal_frr": best_frr_opt,
        "pos_mean": float(pos_scores.mean()),
        "pos_std": float(pos_scores.std()),
        "neg_mean": float(neg_scores.mean()),
        "neg_std": float(neg_scores.std()),
    }


def evaluate_model(model_name: str, model_path: str, pos_files: list[Path], neg_files: list[Path], threshold: float) -> dict:
    """Evaluate a model on the test set."""
    from violawake_sdk.wake_detector import WakeDetector

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"  Model: {model_path}")
    print(f"  Threshold: {threshold}")
    print(f"{'='*60}")

    detector = WakeDetector(model=model_path, threshold=0.01, cooldown_s=0.0)

    # Score positives
    print(f"  Scoring {len(pos_files)} positive clips...")
    t0 = time.time()
    pos_scores = []
    for i, f in enumerate(pos_files):
        audio = load_audio_clip(f)
        if audio is None:
            continue
        score = score_clip_streaming(detector, audio)
        pos_scores.append(score)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(pos_files)} done...")

    # Score negatives
    print(f"  Scoring {len(neg_files)} negative clips...")
    neg_scores = []
    for i, f in enumerate(neg_files):
        audio = load_audio_clip(f)
        if audio is None:
            continue
        score = score_clip_streaming(detector, audio)
        neg_scores.append(score)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(neg_files)} done...")

    elapsed = time.time() - t0
    print(f"  Scored {len(pos_scores)} pos + {len(neg_scores)} neg in {elapsed:.1f}s")

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)

    metrics = compute_metrics(pos_arr, neg_arr, threshold)
    metrics["n_pos"] = len(pos_scores)
    metrics["n_neg"] = len(neg_scores)
    metrics["scoring_time_s"] = elapsed

    return metrics


def evaluate_raw_oww_baseline(pos_files: list[Path], neg_files: list[Path], threshold: float) -> dict:
    """Score clips using only OWW's internal embedding magnitude (no classifier).

    This gives a baseline: how well can OWW embeddings alone distinguish "viola"
    from non-wake-word audio? Uses embedding L2 norm as a proxy score.
    """
    from violawake_sdk.oww_backbone import OpenWakeWordBackbone
    from violawake_sdk.backends import get_backend

    print(f"\n{'='*60}")
    print(f"Evaluating: Raw OWW Backbone (embedding norm baseline)")
    print(f"{'='*60}")

    backend = get_backend("auto", providers=["CPUExecutionProvider"])
    backbone = OpenWakeWordBackbone(backend)

    def score_clip_oww(audio_int16: np.ndarray) -> float:
        backbone.reset()
        max_norm = 0.0
        for start in range(0, len(audio_int16) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            frame = audio_int16[start:start + FRAME_SAMPLES].tobytes()
            produced, embedding = backbone.push_audio(frame)
            if produced and embedding is not None:
                norm = float(np.linalg.norm(embedding.flatten()))
                max_norm = max(max_norm, norm)
        return max_norm

    # Score positives
    print(f"  Scoring {len(pos_files)} positive clips...")
    t0 = time.time()
    pos_scores = []
    for i, f in enumerate(pos_files):
        audio = load_audio_clip(f)
        if audio is None:
            continue
        score = score_clip_oww(audio)
        pos_scores.append(score)

    # Score negatives
    print(f"  Scoring {len(neg_files)} negative clips...")
    neg_scores = []
    for i, f in enumerate(neg_files):
        audio = load_audio_clip(f)
        if audio is None:
            continue
        score = score_clip_oww(audio)
        neg_scores.append(score)

    elapsed = time.time() - t0
    print(f"  Scored {len(pos_scores)} pos + {len(neg_scores)} neg in {elapsed:.1f}s")

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)

    # Raw norms aren't 0-1, so we normalize for comparison
    all_scores = np.concatenate([pos_arr, neg_arr])
    score_min, score_max = all_scores.min(), all_scores.max()
    if score_max > score_min:
        pos_norm = (pos_arr - score_min) / (score_max - score_min)
        neg_norm = (neg_arr - score_min) / (score_max - score_min)
    else:
        pos_norm = pos_arr
        neg_norm = neg_arr

    metrics = compute_metrics(pos_norm, neg_norm, threshold)
    metrics["n_pos"] = len(pos_scores)
    metrics["n_neg"] = len(neg_scores)
    metrics["scoring_time_s"] = elapsed
    metrics["raw_pos_mean"] = float(pos_arr.mean())
    metrics["raw_neg_mean"] = float(neg_arr.mean())

    return metrics


def main():
    print("ViolaWake Head-to-Head Evaluation")
    print(f"Test directory: {TEST_DIR}")
    print(f"Evaluation threshold: {THRESHOLD}")

    # Collect test files
    pos_files = sorted(list(POS_DIR.rglob("*.wav")) + list(POS_DIR.rglob("*.flac")))
    neg_files = sorted(list(NEG_DIR.rglob("*.wav")) + list(NEG_DIR.rglob("*.flac")))
    print(f"Positive samples: {len(pos_files)}")
    print(f"Negative samples: {len(neg_files)}")

    if not pos_files or not neg_files:
        print("ERROR: No test files found!")
        sys.exit(1)

    results = {}

    # 1. temporal_cnn (production default)
    temporal_cnn_path = str(WAKEWORD / "experiments" / "models" / "j5_temporal" / "temporal_cnn.onnx")
    results["temporal_cnn"] = evaluate_model(
        "temporal_cnn", temporal_cnn_path, pos_files, neg_files, THRESHOLD,
    )

    # 2. r3_10x_s42 (MLP on OWW -- best previous MLP)
    r3_path = str(WAKEWORD / "experiments" / "models" / "r3_10x_s42.onnx")
    results["r3_10x_s42"] = evaluate_model(
        "r3_10x_s42 (MLP)", r3_path, pos_files, neg_files, THRESHOLD,
    )

    # 3. Raw OWW backbone baseline
    results["oww_baseline"] = evaluate_raw_oww_baseline(pos_files, neg_files, THRESHOLD)

    # ================================================================
    # REPORT
    # ================================================================
    print("\n" + "=" * 80)
    print("## ViolaWake temporal_cnn vs MLP (r3_10x_s42) vs Raw OWW -- Head-to-Head")
    print("=" * 80)

    print(f"\n### Test Setup")
    print(f"- Positive samples: {len(pos_files)} files (TTS-generated 'viola' utterances)")
    print(f"- Negative samples: {len(neg_files)} files (adversarial TTS + speech + noise)")
    print(f"- Evaluation threshold: {THRESHOLD}")
    print(f"- Scoring method: streaming (20ms frames through full pipeline)")

    print(f"\n### Results\n")
    print(f"| Metric              | temporal_cnn  | r3_10x_s42 (MLP) | OWW Baseline  | Winner         |")
    print(f"|---------------------|---------------|-------------------|---------------|----------------|")

    for metric_name, key, fmt, lower_better in [
        ("FAR (rate)",          "far",          ".4f",  True),
        ("FRR (rate)",          "frr",          ".4f",  True),
        ("FAR/hr",              "far_per_hour", ".1f",  True),
        ("Cohen's d",           "d_prime",      ".2f",  False),
        ("ROC AUC",             "roc_auc",      ".4f",  False),
        ("Precision",           "precision",    ".3f",  False),
        ("Recall",              "recall",       ".3f",  False),
        ("F1",                  "f1",           ".3f",  False),
        ("Grade",               "grade",        "s",    None),
        ("Optimal threshold",   "optimal_threshold", ".2f", None),
    ]:
        vals = {}
        for name in ["temporal_cnn", "r3_10x_s42", "oww_baseline"]:
            v = results[name][key]
            if fmt == "s":
                vals[name] = str(v)
            else:
                vals[name] = f"{v:{fmt}}"

        # Determine winner
        if lower_better is not None and fmt != "s":
            numeric = {k: results[k][key] for k in ["temporal_cnn", "r3_10x_s42", "oww_baseline"]}
            if lower_better:
                winner_key = min(numeric, key=numeric.get)
            else:
                winner_key = max(numeric, key=numeric.get)
            winner_name = {"temporal_cnn": "temporal_cnn", "r3_10x_s42": "r3_10x_s42 (MLP)", "oww_baseline": "OWW Baseline"}[winner_key]
        else:
            winner_name = "-"

        print(f"| {metric_name:<19} | {vals['temporal_cnn']:>13} | {vals['r3_10x_s42']:>17} | {vals['oww_baseline']:>13} | {winner_name:<14} |")

    print(f"\n### Score Distributions\n")
    for name, label in [("temporal_cnn", "temporal_cnn"), ("r3_10x_s42", "r3_10x_s42 (MLP)"), ("oww_baseline", "OWW Baseline")]:
        r = results[name]
        print(f"**{label}**:")
        print(f"  Positive scores: mean={r['pos_mean']:.4f}, std={r['pos_std']:.4f}")
        print(f"  Negative scores: mean={r['neg_mean']:.4f}, std={r['neg_std']:.4f}")
        print(f"  Separation:      {r['pos_mean'] - r['neg_mean']:.4f}")
        print(f"  Confusion: TP={r['tp']}, FP={r['fp']}, TN={r['tn']}, FN={r['fn']}")
        print()

    print("### Analysis\n")
    # Auto-generate analysis
    tc = results["temporal_cnn"]
    mlp = results["r3_10x_s42"]
    oww = results["oww_baseline"]

    if tc["d_prime"] > mlp["d_prime"]:
        print(f"temporal_cnn achieves a Cohen's d of {tc['d_prime']:.2f} vs {mlp['d_prime']:.2f} for the MLP,")
        print(f"demonstrating {((tc['d_prime'] / mlp['d_prime']) - 1) * 100:.0f}% better score separation.")
    else:
        print(f"r3_10x_s42 (MLP) achieves a Cohen's d of {mlp['d_prime']:.2f} vs {tc['d_prime']:.2f} for temporal_cnn.")

    print(f"\nThe raw OWW backbone (embedding norm only, no trained classifier) achieves d'={oww['d_prime']:.2f},")
    print(f"showing that the trained classifiers add {'significant' if tc['d_prime'] > oww['d_prime'] * 2 else 'some'} value over raw embeddings.")

    if tc["far"] <= mlp["far"] and tc["frr"] <= mlp["frr"]:
        print(f"\ntemporal_cnn dominates on both FAR ({tc['far']:.4f} vs {mlp['far']:.4f}) and FRR ({tc['frr']:.4f} vs {mlp['frr']:.4f}).")
    elif tc["f1"] > mlp["f1"]:
        print(f"\ntemporal_cnn has better overall F1 ({tc['f1']:.3f} vs {mlp['f1']:.3f}).")

    print(f"\nScoring time: temporal_cnn={tc['scoring_time_s']:.1f}s, MLP={mlp['scoring_time_s']:.1f}s, OWW={oww['scoring_time_s']:.1f}s")


if __name__ == "__main__":
    main()
