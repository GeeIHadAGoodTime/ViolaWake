#!/usr/bin/env python3
"""
Experiment F: Decision Policies on D_combined Model
=====================================================

Runs all per-frame decision policies from Experiment A on the D_combined
model (our current champion retrained model) to determine whether temporal
smoothing improves the already-retrained model.

Answers empirical question Q4:
  "Does temporal smoothing reduce EER on an already-better model?"

Baseline comparison: D_combined raw per-clip EER = 13.2%

Policies tested:
  - mean:           mean of all frame scores (baseline equivalent)
  - max:            max of all frame scores
  - median:         median of all frame scores
  - top3_mean:      mean of top-3 frame scores
  - consecutive_2:  1.0 if any 2 consecutive frames >= threshold, else max*0.5
  - consecutive_3:  1.0 if any 3 consecutive frames >= threshold, else max*0.5
  - moving_avg_3:   max of 3-frame moving average

Usage:
    python experiments/exp_f_policy_on_d.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# SDK on path
sys.path.insert(0, "J:/CLAUDE/PROJECTS/Wakeword/src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# -- Paths -----------------------------------------------------------------------

MODEL_PATH = Path("J:/CLAUDE/PROJECTS/Wakeword/experiments/models/D_combined.onnx")
EVAL_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/eval_clean")
POS_DIR = EVAL_DIR / "positives"
NEG_DIR = EVAL_DIR / "negatives"
RESULTS_PATH = Path("J:/CLAUDE/PROJECTS/Wakeword/experiments/exp_f_policy_on_d.json")

# D_combined raw per-clip EER baseline (from prior eval)
D_COMBINED_RAW_EER = 0.132


# -- Decision policies -----------------------------------------------------------

def policy_mean(scores: list[float], _threshold: float = 0.5) -> float:
    """Mean of all frame scores (current baseline equivalent)."""
    return float(np.mean(scores))


def policy_max(scores: list[float], _threshold: float = 0.5) -> float:
    """Maximum frame score."""
    return float(np.max(scores))


def policy_median(scores: list[float], _threshold: float = 0.5) -> float:
    """Median frame score."""
    return float(np.median(scores))


def policy_top3_mean(scores: list[float], _threshold: float = 0.5) -> float:
    """Mean of top-3 frame scores."""
    s = sorted(scores, reverse=True)
    top = s[: min(3, len(s))]
    return float(np.mean(top))


def policy_consecutive_2(scores: list[float], threshold: float = 0.5) -> float:
    """1.0 if any 2 consecutive frames >= threshold, else max_score * 0.5."""
    for i in range(len(scores) - 1):
        if scores[i] >= threshold and scores[i + 1] >= threshold:
            return 1.0
    return float(max(scores) * 0.5)


def policy_consecutive_3(scores: list[float], threshold: float = 0.5) -> float:
    """1.0 if any 3 consecutive frames >= threshold, else max_score * 0.5."""
    for i in range(len(scores) - 2):
        if scores[i] >= threshold and scores[i + 1] >= threshold and scores[i + 2] >= threshold:
            return 1.0
    return float(max(scores) * 0.5)


def policy_moving_avg_3(scores: list[float], _threshold: float = 0.5) -> float:
    """Max of 3-frame moving average."""
    if len(scores) < 3:
        return float(np.mean(scores))
    avgs = [
        np.mean(scores[i : i + 3]) for i in range(len(scores) - 2)
    ]
    return float(np.max(avgs))


POLICIES: dict[str, callable] = {
    "mean": policy_mean,
    "max": policy_max,
    "median": policy_median,
    "top3_mean": policy_top3_mean,
    "consecutive_2": policy_consecutive_2,
    "consecutive_3": policy_consecutive_3,
    "moving_avg_3": policy_moving_avg_3,
}


# -- Metrics --------------------------------------------------------------------

def compute_dprime(pos: np.ndarray, neg: np.ndarray) -> float:
    """Cohen's d separability metric."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    pooled_std = np.sqrt(0.5 * (pos.var() + neg.var()))
    if pooled_std < 1e-10:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled_std)


def compute_eer_and_auc(pos: np.ndarray, neg: np.ndarray) -> dict:
    """Compute EER and AUC from positive/negative score arrays."""
    from sklearn.metrics import auc, roc_curve

    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.concatenate([pos, neg])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = float(auc(fpr, tpr))

    # EER: where FPR == FNR
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    return {"eer": eer, "eer_threshold": eer_threshold, "auc": roc_auc}


def compute_operating_points(
    pos: np.ndarray,
    neg: np.ndarray,
    target_fa_per_hour: list[float],
    neg_duration_sec: float = 1.5,
) -> list[dict]:
    """Compute FRR at specific FA/hr operating points."""
    total_neg_hours = len(neg) * neg_duration_sec / 3600.0
    n_pos = len(pos)

    thresholds = np.arange(0.0, 1.001, 0.001)
    results = []

    for target_fa in target_fa_per_hour:
        best_thresh = 1.0
        best_frr = 1.0
        best_fa_hr = 0.0

        for t in thresholds:
            n_fa = float(np.sum(neg >= t))
            fa_hr = n_fa / total_neg_hours if total_neg_hours > 0 else float("inf")
            if fa_hr <= target_fa:
                frr = float(np.sum(pos < t) / n_pos) if n_pos > 0 else 1.0
                if frr < best_frr or (frr == best_frr and fa_hr > best_fa_hr):
                    best_frr = frr
                    best_thresh = float(t)
                    best_fa_hr = fa_hr

        results.append({
            "target_fa_per_hour": target_fa,
            "threshold": best_thresh,
            "frr": best_frr,
            "actual_fa_per_hour": best_fa_hr,
        })

    return results


def compute_confidence_interval(eer: float, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """
    Approximate 95% confidence interval for EER using normal approximation.
    EER ~ a proportion, so SE = sqrt(p*(1-p)/n).
    """
    se = np.sqrt(eer * (1 - eer) / n_total)
    return (max(0.0, eer - z * se), min(1.0, eer + z * se))


# -- Embedding extraction & per-frame scoring -----------------------------------

def build_perframe_scorer():
    """
    Build infrastructure for per-frame scoring.

    Returns (preprocessor, onnx_session, input_name) for reuse across all files.
    """
    import onnxruntime as ort
    from openwakeword.model import Model as OWWModel

    logger.info("Loading OWW preprocessor...")
    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    logger.info("Loading D_combined ONNX model from %s", MODEL_PATH)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    logger.info("Model input: name=%s, shape=%s", input_name, input_shape)

    return preprocessor, session, input_name


def extract_perframe_scores(
    wav_path: Path,
    preprocessor,
    session,
    input_name: str,
) -> list[float] | None:
    """
    Extract OWW embeddings for a WAV file and score each frame individually.

    Returns a list of T float scores, or None on failure.
    """
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    audio = load_audio(wav_path)
    if audio is None:
        return None

    audio = center_crop(audio, CLIP_SAMPLES)
    # Convert to int16 -- matches training pipeline exactly
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    if len(audio_int16) < CLIP_SAMPLES:
        audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
    else:
        audio_int16 = audio_int16[:CLIP_SAMPLES]

    try:
        # Get (1, T, 96) embeddings -- DO NOT mean-pool
        embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
        n_frames = embeddings.shape[1]

        frame_scores = []
        for t in range(n_frames):
            frame_emb = embeddings[0, t, :].astype(np.float32).reshape(1, -1)
            score = session.run(None, {input_name: frame_emb})[0]
            frame_scores.append(float(np.asarray(score).flatten()[0]))

        return frame_scores
    except Exception as e:
        logger.warning("Failed to score %s: %s", wav_path, e)
        return None


# -- Main -----------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 80)
    print("EXPERIMENT F: Decision Policies on D_combined Model")
    print("=" * 80)
    print(f"Model:              {MODEL_PATH.name}")
    print(f"Eval dir:           {EVAL_DIR}")
    print(f"D_combined raw EER: {D_COMBINED_RAW_EER:.1%} (baseline to beat)")
    print()

    # Build scorer
    preprocessor, session, input_name = build_perframe_scorer()

    # Collect WAV files
    pos_files = sorted(list(POS_DIR.rglob("*.wav")) + list(POS_DIR.rglob("*.flac")))
    neg_files = sorted(list(NEG_DIR.rglob("*.wav")) + list(NEG_DIR.rglob("*.flac")))
    print(f"Positive files: {len(pos_files)}")
    print(f"Negative files: {len(neg_files)}")
    print()

    # Extract per-frame scores for all files
    print("Extracting per-frame scores for positives...")
    pos_frame_scores: list[list[float]] = []
    pos_valid_files: list[Path] = []
    for i, f in enumerate(pos_files):
        scores = extract_perframe_scores(f, preprocessor, session, input_name)
        if scores is not None:
            pos_frame_scores.append(scores)
            pos_valid_files.append(f)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(pos_files)}")
    print(f"  Scored {len(pos_frame_scores)}/{len(pos_files)} positives")

    pos_frame_counts = [len(s) for s in pos_frame_scores]
    print(f"  Frames per clip: min={min(pos_frame_counts)}, max={max(pos_frame_counts)}, "
          f"mean={np.mean(pos_frame_counts):.1f}")

    print("\nExtracting per-frame scores for negatives...")
    neg_frame_scores: list[list[float]] = []
    neg_valid_files: list[Path] = []
    for i, f in enumerate(neg_files):
        scores = extract_perframe_scores(f, preprocessor, session, input_name)
        if scores is not None:
            neg_frame_scores.append(scores)
            neg_valid_files.append(f)
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(neg_files)}")
    print(f"  Scored {len(neg_frame_scores)}/{len(neg_files)} negatives")

    neg_frame_counts = [len(s) for s in neg_frame_scores]
    print(f"  Frames per clip: min={min(neg_frame_counts)}, max={max(neg_frame_counts)}, "
          f"mean={np.mean(neg_frame_counts):.1f}")

    extraction_time = time.time() - t0
    print(f"\nExtraction took {extraction_time:.1f}s")

    # -- Apply policies and compute metrics ----------------------------------------

    print("\n" + "=" * 80)
    print("POLICY COMPARISON (D_combined model)")
    print("=" * 80)

    all_results = {}
    target_fa_rates = [1.0, 5.0, 10.0]
    n_total = len(pos_frame_scores) + len(neg_frame_scores)

    for policy_name, policy_fn in POLICIES.items():
        # Aggregate per-frame scores into one score per clip
        pos_agg = np.array([policy_fn(s) for s in pos_frame_scores])
        neg_agg = np.array([policy_fn(s) for s in neg_frame_scores])

        # Core metrics
        dprime = compute_dprime(pos_agg, neg_agg)
        metrics = compute_eer_and_auc(pos_agg, neg_agg)
        eer = metrics["eer"]
        auc_val = metrics["auc"]

        # Confidence interval for EER
        ci_lo, ci_hi = compute_confidence_interval(eer, n_total)

        # Operating points
        ops = compute_operating_points(pos_agg, neg_agg, target_fa_rates)

        # Score distribution stats
        pos_mean = float(pos_agg.mean())
        pos_std = float(pos_agg.std())
        neg_mean = float(neg_agg.mean())
        neg_std = float(neg_agg.std())

        # Improvement vs D_combined raw EER
        eer_delta = eer - D_COMBINED_RAW_EER
        eer_relative_change_pct = (eer_delta / D_COMBINED_RAW_EER) * 100.0

        all_results[policy_name] = {
            "eer": eer,
            "eer_threshold": metrics["eer_threshold"],
            "auc": auc_val,
            "dprime": dprime,
            "pos_mean": pos_mean,
            "pos_std": pos_std,
            "neg_mean": neg_mean,
            "neg_std": neg_std,
            "eer_95ci_lo": ci_lo,
            "eer_95ci_hi": ci_hi,
            "vs_raw_eer_delta": eer_delta,
            "vs_raw_eer_relative_pct": eer_relative_change_pct,
            "beats_raw_eer": eer < D_COMBINED_RAW_EER,
            "operating_points": ops,
            "n_pos": len(pos_agg),
            "n_neg": len(neg_agg),
        }

    # -- Print comparison table ----------------------------------------------------

    print(f"\n{'Policy':<18} {'EER':>8} {'AUC':>8} {'d-prime':>8} "
          f"{'pos_mu':>8} {'pos_sd':>8} {'neg_mu':>8} {'neg_sd':>8} "
          f"{'vs 13.2%':>10}")
    print("-" * 108)

    for name, r in all_results.items():
        delta_str = f"{r['vs_raw_eer_delta']:+.4f}"
        print(f"{name:<18} {r['eer']:>8.4f} {r['auc']:>8.4f} {r['dprime']:>8.2f} "
              f"{r['pos_mean']:>8.4f} {r['pos_std']:>8.4f} {r['neg_mean']:>8.4f} {r['neg_std']:>8.4f} "
              f"{delta_str:>10}")

    # D_combined raw baseline row
    print("-" * 108)
    print(f"{'D_combined (raw)':<18} {D_COMBINED_RAW_EER:>8.4f} {'--':>8} {'--':>8} "
          f"{'--':>8} {'--':>8} {'--':>8} {'--':>8} {'baseline':>10}")

    # Operating points table
    print(f"\n{'':=<90}")
    print("OPERATING POINTS: FRR at target FA/hr")
    print(f"{'':=<90}")

    header = f"{'Policy':<18}"
    for fa in target_fa_rates:
        header += f" {'FRR@' + str(int(fa)) + 'FA/hr':>14} {'(thresh)':>10}"
    print(header)
    print("-" * 90)

    for name, r in all_results.items():
        row = f"{name:<18}"
        for op in r["operating_points"]:
            frr_pct = op["frr"] * 100
            row += f" {frr_pct:>13.1f}% {op['threshold']:>10.3f}"
        print(row)

    # EER with confidence intervals
    print(f"\n{'':=<70}")
    print("EER WITH 95% CONFIDENCE INTERVALS")
    print(f"{'':=<70}")
    print(f"{'Policy':<18} {'EER':>8} {'95% CI':>20} {'Beats 13.2%?':>14}")
    print("-" * 62)
    for name, r in all_results.items():
        ci_str = f"[{r['eer_95ci_lo']:.4f}, {r['eer_95ci_hi']:.4f}]"
        beats = "YES" if r["beats_raw_eer"] else "no"
        print(f"{name:<18} {r['eer']:>8.4f} {ci_str:>20} {beats:>14}")

    # -- Ranking -------------------------------------------------------------------

    print(f"\n{'':=<60}")
    print("RANKING (by EER, lower is better)")
    print(f"{'':=<60}")
    ranked = sorted(all_results.items(), key=lambda x: x[1]["eer"])
    for i, (name, r) in enumerate(ranked):
        marker = " <-- BEST" if i == 0 else ""
        delta = f"({r['vs_raw_eer_delta']:+.4f} vs raw)" if r["beats_raw_eer"] else f"({r['vs_raw_eer_delta']:+.4f} vs raw)"
        print(f"  {i+1}. {name:<18} EER={r['eer']:.4f}  AUC={r['auc']:.4f}  d'={r['dprime']:.2f}  {delta}{marker}")

    print(f"\n{'':=<60}")
    print("RANKING (by AUC, higher is better)")
    print(f"{'':=<60}")
    ranked_auc = sorted(all_results.items(), key=lambda x: x[1]["auc"], reverse=True)
    for i, (name, r) in enumerate(ranked_auc):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1}. {name:<18} AUC={r['auc']:.4f}  EER={r['eer']:.4f}  d'={r['dprime']:.2f}{marker}")

    # -- Summary verdict -----------------------------------------------------------

    best_name, best_r = ranked[0]
    any_beats_raw = any(r["beats_raw_eer"] for r in all_results.values())
    beating_policies = [(n, r) for n, r in all_results.items() if r["beats_raw_eer"]]

    print(f"\n{'':=<80}")
    print("VERDICT: Does temporal smoothing improve D_combined?")
    print(f"{'':=<80}")
    print(f"D_combined raw per-clip EER:  {D_COMBINED_RAW_EER:.1%}")
    print(f"Best per-frame policy EER:    {best_r['eer']:.4f} ({best_name})")
    print(f"Best policy AUC:              {best_r['auc']:.4f}")
    print(f"Best policy d-prime:          {best_r['dprime']:.2f}")
    print()

    if any_beats_raw:
        print(f"ANSWER: YES -- {len(beating_policies)} policies beat the raw 13.2% EER:")
        for n, r in sorted(beating_policies, key=lambda x: x[1]["eer"]):
            print(f"  - {n}: EER={r['eer']:.4f} ({r['vs_raw_eer_relative_pct']:+.1f}% relative change)")
        improvement = D_COMBINED_RAW_EER - best_r["eer"]
        print(f"\nBest achievable EER: {best_r['eer']:.4f} ({best_name})")
        print(f"Absolute improvement: {improvement:.4f} ({improvement/D_COMBINED_RAW_EER*100:.1f}% relative)")
        # Statistical significance check
        ci_lo, ci_hi = best_r["eer_95ci_lo"], best_r["eer_95ci_hi"]
        if ci_hi < D_COMBINED_RAW_EER:
            print(f"Statistical significance: YES (95% CI upper bound {ci_hi:.4f} < {D_COMBINED_RAW_EER:.4f})")
        else:
            print(f"Statistical significance: NO (95% CI [{ci_lo:.4f}, {ci_hi:.4f}] overlaps {D_COMBINED_RAW_EER:.4f})")
    else:
        print("ANSWER: NO -- No policy beats the raw 13.2% EER.")
        print(f"Closest policy: {best_name} at EER={best_r['eer']:.4f}")
        gap = best_r["eer"] - D_COMBINED_RAW_EER
        print(f"Gap from raw baseline: +{gap:.4f} ({gap/D_COMBINED_RAW_EER*100:.1f}% worse)")
        print("\nConclusion: Temporal smoothing does NOT help D_combined.")
        print("The retrained model already captures temporal patterns in its embeddings.")

    # -- Save results ---------------------------------------------------------------

    total_time = time.time() - t0

    output = {
        "experiment": "F",
        "description": "Decision policies on D_combined model (temporal smoothing on retrained model)",
        "question": "Q4: Does temporal smoothing reduce EER on an already-better model?",
        "model": str(MODEL_PATH),
        "eval_dir": str(EVAL_DIR),
        "d_combined_raw_eer": D_COMBINED_RAW_EER,
        "n_positives": len(pos_frame_scores),
        "n_negatives": len(neg_frame_scores),
        "frame_stats": {
            "pos_frames_per_clip": {
                "min": int(min(pos_frame_counts)),
                "max": int(max(pos_frame_counts)),
                "mean": float(np.mean(pos_frame_counts)),
            },
            "neg_frames_per_clip": {
                "min": int(min(neg_frame_counts)),
                "max": int(max(neg_frame_counts)),
                "mean": float(np.mean(neg_frame_counts)),
            },
        },
        "policies": {
            name: {k: v for k, v in r.items() if k != "operating_points"}
            | {"operating_points": r["operating_points"]}
            for name, r in all_results.items()
        },
        "rankings": {
            "by_eer": [name for name, _ in ranked],
            "by_auc": [name for name, _ in ranked_auc],
        },
        "verdict": {
            "any_policy_beats_raw": any_beats_raw,
            "best_policy": best_name,
            "best_eer": best_r["eer"],
            "best_auc": best_r["auc"],
            "best_dprime": best_r["dprime"],
            "n_policies_beating_raw": len(beating_policies),
            "beating_policies": [n for n, _ in beating_policies],
        },
        "elapsed_seconds": total_time,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
