"""
ViolaWake Evaluation
=====================

D-prime evaluation of wake word models on real-world test sets.

Copied and adapted from Viola's violawake/training/evaluate_real.py
and violawake/training/trainer.py::evaluate_real_samples.

Usage::

    from violawake_sdk.training.evaluate import evaluate_model, compute_dprime

    results = evaluate_model(
        model_path="models/viola_mlp_oww.onnx",
        test_dir="data/test/",  # must contain positives/ and negatives/
    )
    print(f"d-prime: {results['d_prime']:.2f}")
    print(f"FAR: {results['far_per_hour']:.2f}/hr")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_dprime(pos_scores: list[float] | np.ndarray, neg_scores: list[float] | np.ndarray) -> float:
    """
    Compute d-prime discriminability index.

    d' = (mean_pos - mean_neg) / sqrt(0.5 * (var_pos + var_neg))

    Higher is better. d' >= 15.0 is production-grade (Viola standard).
    d' >= 10.0 is the minimum acceptable for a custom-trained model.

    Args:
        pos_scores: Model scores on positive (wake word) samples.
        neg_scores: Model scores on negative (background) samples.

    Returns:
        d-prime value. Returns 0.0 if either list is empty.
    """
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)

    if len(pos) == 0 or len(neg) == 0:
        return 0.0

    pooled_std = np.sqrt(0.5 * (pos.var() + neg.var()))
    if pooled_std < 1e-10:
        return 0.0

    return float((pos.mean() - neg.mean()) / pooled_std)


def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, int]:
    """
    Compute Equal Error Rate (EER) from ROC curve.

    Args:
        fpr: False positive rates (from sklearn roc_curve).
        tpr: True positive rates (from sklearn roc_curve).

    Returns:
        (eer, threshold_index) tuple.
    """
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, idx


def evaluate_onnx_model(
    model_path: str | Path,
    test_dir: str | Path,
    threshold: float = 0.50,
) -> dict:
    """
    Evaluate an ONNX wake word model (MLP-on-OWW architecture) on a test set.

    The test_dir must contain:
        positives/  — WAV files containing the wake word
        negatives/  — WAV files of background audio (no wake word)

    Args:
        model_path: Path to the .onnx model file.
        test_dir: Directory containing positives/ and negatives/ subdirs.
        threshold: Classification threshold for FAR/FRR computation.

    Returns:
        dict with keys: d_prime, far_per_hour, frr, roc_auc, tp_scores,
                        fp_scores, tp_mean, fp_mean, n_positives, n_negatives.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime required. pip install onnxruntime") from e

    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as e:
        raise ImportError("scikit-learn required. pip install 'violawake[training]'") from e

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, compute_features, load_audio

    model_path = Path(model_path)
    test_dir = Path(test_dir)
    pos_dir = test_dir / "positives"
    neg_dir = test_dir / "negatives"

    if not pos_dir.exists():
        raise FileNotFoundError(f"positives/ directory not found in {test_dir}")
    if not neg_dir.exists():
        raise FileNotFoundError(f"negatives/ directory not found in {test_dir}")

    # Load ONNX session
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def _score_file(wav_path: Path) -> float | None:
        audio = load_audio(wav_path)
        if audio is None:
            return None
        audio = center_crop(audio, CLIP_SAMPLES)
        features = compute_features(audio)
        feat_input = features[np.newaxis, :, :].astype(np.float32)
        outputs = session.run(None, {input_name: feat_input})
        return float(np.asarray(outputs[0]).flatten()[0])

    logger.info("Scoring positive samples in %s", pos_dir)
    pos_files = sorted(list(pos_dir.rglob("*.wav")) + list(pos_dir.rglob("*.flac")))
    pos_scores = []
    for f in pos_files:
        s = _score_file(f)
        if s is not None:
            pos_scores.append(s)

    logger.info("Scoring negative samples in %s", neg_dir)
    neg_files = sorted(list(neg_dir.rglob("*.wav")) + list(neg_dir.rglob("*.flac")))
    neg_scores = []
    for f in neg_files:
        s = _score_file(f)
        if s is not None:
            neg_scores.append(s)

    if not pos_scores:
        raise RuntimeError(f"No valid positive audio files found in {pos_dir}")
    if not neg_scores:
        raise RuntimeError(f"No valid negative audio files found in {neg_dir}")

    pos_arr = np.array(pos_scores)
    neg_arr = np.array(neg_scores)

    d_prime = compute_dprime(pos_arr, neg_arr)

    # ROC curve and AUC
    all_scores = np.concatenate([pos_arr, neg_arr])
    all_labels = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    # FAR and FRR at the given threshold
    tp_at_thresh = np.sum(pos_arr >= threshold)
    fp_at_thresh = np.sum(neg_arr >= threshold)
    frr = float(1.0 - tp_at_thresh / len(pos_arr)) if len(pos_arr) > 0 else 1.0

    # FAR in events/hour — assumes each negative file is approximately 1 second
    # (standard test set convention). Adjust if your test set files are longer.
    neg_total_hours = len(neg_arr) / 3600.0
    far_per_hour = float(fp_at_thresh / neg_total_hours) if neg_total_hours > 0 else float("inf")

    results = {
        "d_prime": d_prime,
        "far_per_hour": far_per_hour,
        "frr": frr,
        "roc_auc": roc_auc,
        "tp_scores": pos_scores,
        "fp_scores": neg_scores,
        "tp_mean": float(pos_arr.mean()),
        "fp_mean": float(neg_arr.mean()),
        "n_positives": len(pos_scores),
        "n_negatives": len(neg_scores),
        "threshold_used": threshold,
    }

    logger.info(
        "Evaluation complete: d'=%.2f, FAR=%.2f/hr, FRR=%.1f%%, AUC=%.3f",
        d_prime, far_per_hour, frr * 100, roc_auc,
    )

    return results
