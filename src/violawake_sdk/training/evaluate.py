"""
ViolaWake Evaluation
=====================

Wake word evaluation using a Cohen's d-style separability score plus FAR/FRR.

Supports two model architectures:
  - **MLP-on-OWW** (primary): MLP classifier on OpenWakeWord embeddings (96-dim).
    Auto-detected via .config.json or ONNX input shape.
  - **CNN** (legacy): CNN on mel spectrograms. Used only for old viola_v2/v3 models.

Copied and adapted from Viola's violawake/training/evaluate_real.py
and violawake/training/trainer.py::evaluate_real_samples.

Usage::

    from violawake_sdk.training.evaluate import evaluate_onnx_model, compute_dprime

    results = evaluate_onnx_model(
        model_path="models/viola_mlp_oww.onnx",
        test_dir="data/test/",  # must contain positives/ and negatives/
    )
    print(f"Cohen's d: {results['d_prime']:.2f}")
    print(f"FAR: {results['far_per_hour']:.2f}/hr")
    print(f"Optimal threshold: {results['optimal_threshold']:.2f}")
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Architecture detection ────────────────────────────────────────────────────

# Input dimensions that indicate MLP-on-OWW architecture.
# OWW embeddings are typically 96-dim, but allow some variation.
_OWW_EMBEDDING_DIM_RANGE = (32, 256)


def detect_architecture(model_path: Path, session) -> str:
    """
    Auto-detect model architecture from config file or ONNX input shape.

    Detection order:
      1. Check for .config.json alongside the .onnx file
      2. Fall back to ONNX input shape heuristic

    Args:
        model_path: Path to the .onnx model file.
        session: An onnxruntime InferenceSession for the model.

    Returns:
        "mlp_on_oww", "temporal_oww", or "cnn"
    """
    # 1. Check for .config.json
    config_path = model_path.with_suffix(".config.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            arch = config.get("architecture", "")
            if arch in ("mlp_on_oww", "temporal_cnn", "temporal_oww"):
                # Normalize temporal_cnn → temporal_oww for scorer routing
                result = "temporal_oww" if arch == "temporal_cnn" else arch
                logger.info("Architecture detected from config: %s", result)
                return result
            elif arch == "cnn":
                logger.info("Architecture detected from config: cnn")
                return "cnn"
            # If architecture key is missing or unrecognized, fall through
        except (json.JSONDecodeError, OSError):
            pass  # Fall through to shape heuristic

    # 2. ONNX input shape heuristic
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape  # e.g. [1, 96] for MLP or [1, 40, N] for CNN
    input_name = input_info.name

    # Filter out dynamic/string dims, keep only integer dims
    numeric_dims = [d for d in input_shape if isinstance(d, int)]

    if len(numeric_dims) >= 1:
        if len(input_shape) == 2:
            # 2D input: (batch, embedding_dim) — MLP-on-OWW
            last_dim = numeric_dims[-1] if numeric_dims else None
            if last_dim and _OWW_EMBEDDING_DIM_RANGE[0] <= last_dim <= _OWW_EMBEDDING_DIM_RANGE[1]:
                logger.info(
                    "Architecture detected from input shape %s: mlp_on_oww (dim=%d)",
                    input_shape, last_dim,
                )
                return "mlp_on_oww"
        elif len(input_shape) >= 3:
            # 3D input: could be temporal OWW (batch, seq, 96) or CNN (batch, mels, time)
            # Temporal OWW: (batch, seq_len=5-15, embedding_dim=96), input named "embeddings"
            # CNN: (batch, n_mels=32-40, time_steps=50-200+)
            last_dim = numeric_dims[-1] if numeric_dims else None
            seq_dim = numeric_dims[0] if len(numeric_dims) >= 2 else None
            is_temporal_oww = (
                last_dim == 96
                and seq_dim is not None
                and seq_dim <= 20
            ) or input_name == "embeddings"
            if is_temporal_oww:
                logger.info(
                    "Architecture detected from input shape %s: temporal_oww (dim=%s, seq=%s)",
                    input_shape, last_dim, seq_dim,
                )
                return "temporal_oww"
            logger.info(
                "Architecture detected from input shape %s: cnn (3D+ input)",
                input_shape,
            )
            return "cnn"

    # Default: assume CNN (legacy behavior)
    logger.warning(
        "Could not determine architecture from shape %s; defaulting to cnn",
        input_shape,
    )
    return "cnn"


# ── Core metrics ──────────────────────────────────────────────────────────────


def compute_dprime(pos_scores: list[float] | np.ndarray, neg_scores: list[float] | np.ndarray) -> float:
    """
    Compute the repo's historical "d-prime" metric.

    This is Cohen's d:

        (mean_pos - mean_neg) / sqrt(0.5 * (var_pos + var_neg))

    It is not the standard signal-detection d-prime metric often reported in
    wake word benchmarks. The score can be materially inflated when negatives
    are synthetic-only noise/silence instead of real speech/background audio.

    Higher is better. In this repo, values around 10-15 have historically been
    used as internal synthetic-benchmark targets, not as real-world accuracy
    guarantees.

    Args:
        pos_scores: Model scores on positive (wake word) samples.
        neg_scores: Model scores on negative (background) samples.

    Returns:
        Cohen's d value. Returns 0.0 if either list is empty.
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


def find_optimal_threshold(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    step: float = 0.01,
) -> dict:
    """
    Find the optimal classification threshold by sweeping from 0.0 to 1.0.

    For each candidate threshold, computes:
      - FAR (False Accept Rate): fraction of negatives scoring >= threshold
      - FRR (False Reject Rate): fraction of positives scoring < threshold

    Returns the threshold that minimizes (FAR + FRR), which approximates
    the Equal Error Rate (EER) operating point.

    Args:
        pos_scores: Model scores on positive samples.
        neg_scores: Model scores on negative samples.
        step: Threshold increment (default 0.01 for 101 steps).

    Returns:
        dict with keys:
          - optimal_threshold: float
          - optimal_far: float (FAR at optimal threshold)
          - optimal_frr: float (FRR at optimal threshold)
          - eer_approx: float (approximate EER = (FAR + FRR) / 2 at optimal)
    """
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_thresh = 0.5
    best_cost = float("inf")
    best_far = 1.0
    best_frr = 1.0

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return {
            "optimal_threshold": 0.5,
            "optimal_far": 1.0,
            "optimal_frr": 1.0,
            "eer_approx": 1.0,
        }

    for t in thresholds:
        far = float(np.sum(neg_scores >= t) / n_neg)
        frr = float(np.sum(pos_scores < t) / n_pos)
        cost = far + frr
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(t)
            best_far = far
            best_frr = frr

    return {
        "optimal_threshold": best_thresh,
        "optimal_far": best_far,
        "optimal_frr": best_frr,
        "eer_approx": (best_far + best_frr) / 2.0,
    }


def compute_confusion_matrix(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute confusion matrix metrics at a given threshold.

    Args:
        pos_scores: Model scores on positive (wake word) samples.
        neg_scores: Model scores on negative (background) samples.
        threshold: Classification threshold (score >= threshold = positive).

    Returns:
        dict with keys: tp, fp, tn, fn, precision, recall, f1
    """
    tp = int(np.sum(pos_scores >= threshold))
    fn = int(np.sum(pos_scores < threshold))
    fp = int(np.sum(neg_scores >= threshold))
    tn = int(np.sum(neg_scores < threshold))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


# ── Per-file score dumping ────────────────────────────────────────────────────


def _dump_scores_csv(
    pos_files: list[Path],
    pos_scores: list[float],
    neg_files: list[Path],
    neg_scores: list[float],
    threshold: float,
    csv_path: Path,
) -> None:
    """
    Write per-file scores to a CSV for debugging false rejects/accepts.

    Columns: file, label, score, threshold_pass

    Args:
        pos_files: List of positive file paths (aligned with pos_scores).
        pos_scores: Scores for positive files.
        neg_files: List of negative file paths (aligned with neg_scores).
        neg_scores: Scores for negative files.
        threshold: Classification threshold.
        csv_path: Output CSV path.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if len(pos_files) != len(pos_scores):
        raise ValueError(f"Positive files/scores mismatch: {len(pos_files)} vs {len(pos_scores)}")
    if len(neg_files) != len(neg_scores):
        raise ValueError(f"Negative files/scores mismatch: {len(neg_files)} vs {len(neg_scores)}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "score", "threshold_pass"])
        for fpath, score in zip(pos_files, pos_scores, strict=True):
            writer.writerow([str(fpath), "positive", f"{score:.6f}", score >= threshold])
        for fpath, score in zip(neg_files, neg_scores, strict=True):
            writer.writerow([str(fpath), "negative", f"{score:.6f}", score >= threshold])

    logger.info("Score dump written to %s (%d entries)", csv_path, len(pos_scores) + len(neg_scores))


# ── Scoring backends ──────────────────────────────────────────────────────────


def _build_oww_scorer(session, input_name: str):
    """
    Build a scoring function for MLP-on-OWW models.

    Loads the OpenWakeWord preprocessor once and returns a closure that
    extracts OWW embeddings and runs the MLP classifier.

    This MUST match the embedding extraction in train.py::_extract_embedding exactly.

    Args:
        session: ONNX InferenceSession for the MLP model.
        input_name: Name of the model's input tensor.

    Returns:
        Callable that takes a wav Path and returns a float score (or None on failure).
    """
    try:
        from openwakeword.model import Model as OWWModel  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "openwakeword required for MLP-on-OWW evaluation. "
            "pip install openwakeword"
        ) from e

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    def _score_file_oww(wav_path: Path) -> float | None:
        audio = load_audio(wav_path)
        if audio is None:
            return None
        audio = center_crop(audio, CLIP_SAMPLES)
        # Convert to int16 — matches train.py::_extract_embedding exactly
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        if len(audio_int16) < CLIP_SAMPLES:
            audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:CLIP_SAMPLES]
        try:
            embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            # Mean-pool across time axis (production Viola approach used for the
            # synthetic-benchmark Cohen's d reference model).
            embedding = embeddings.mean(axis=1)[0].astype(np.float32)
            score = session.run(None, {input_name: embedding.reshape(1, -1)})[0]
            return float(np.asarray(score).flatten()[0])
        except Exception:
            logger.warning("Failed to score file (OWW path): %s", wav_path, exc_info=True)
            return None

    return _score_file_oww


def _build_cnn_scorer(session, input_name: str):
    """
    Build a scoring function for legacy CNN models (mel spectrogram input).

    Args:
        session: ONNX InferenceSession for the CNN model.
        input_name: Name of the model's input tensor.

    Returns:
        Callable that takes a wav Path and returns a float score (or None on failure).
    """
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, compute_features, load_audio

    def _score_file_cnn(wav_path: Path) -> float | None:
        audio = load_audio(wav_path)
        if audio is None:
            return None
        audio = center_crop(audio, CLIP_SAMPLES)
        features = compute_features(audio)
        feat_input = features[np.newaxis, :, :].astype(np.float32)
        try:
            outputs = session.run(None, {input_name: feat_input})
            return float(np.asarray(outputs[0]).flatten()[0])
        except Exception:
            logger.warning("Failed to score file (CNN path): %s", wav_path, exc_info=True)
            return None

    return _score_file_cnn


def _build_temporal_oww_scorer(session, input_name: str):
    """
    Build a scoring function for temporal OWW models (e.g. TemporalCNN).

    These models expect input shape (batch, seq_len, 96) — a sliding window
    of OWW embeddings.  For each file, extract all embeddings, build all
    possible windows, score each, and return the max score.

    Args:
        session: ONNX InferenceSession for the temporal model.
        input_name: Name of the model's input tensor.

    Returns:
        Callable that takes a wav Path and returns a float score (or None on failure).
    """
    try:
        from openwakeword.model import Model as OWWModel  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "openwakeword required for temporal OWW evaluation. "
            "pip install openwakeword"
        ) from e

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    # Read seq_len from model input shape
    model_shape = session.get_inputs()[0].shape
    numeric_dims = [d for d in model_shape if isinstance(d, int)]
    seq_len = numeric_dims[0] if len(numeric_dims) >= 2 else 9  # default 9

    def _score_file_temporal(wav_path: Path) -> float | None:
        audio = load_audio(wav_path)
        if audio is None:
            return None
        audio = center_crop(audio, CLIP_SAMPLES)
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        if len(audio_int16) < CLIP_SAMPLES:
            audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:CLIP_SAMPLES]
        try:
            embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            # embeddings shape: (1, n_frames, 96) or (n_frames, 96)
            emb = np.squeeze(embeddings)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            n_frames = emb.shape[0]

            if n_frames < seq_len:
                # Pad with zeros if not enough frames
                padded = np.zeros((seq_len, emb.shape[1]), dtype=np.float32)
                padded[:n_frames] = emb
                window = padded[np.newaxis, :, :].astype(np.float32)
                score = session.run(None, {input_name: window})[0]
                return float(np.asarray(score).flatten()[0])

            # Slide window and take max score
            max_score = -1.0
            for i in range(n_frames - seq_len + 1):
                window = emb[i:i + seq_len][np.newaxis, :, :].astype(np.float32)
                score = session.run(None, {input_name: window})[0]
                s = float(np.asarray(score).flatten()[0])
                if s > max_score:
                    max_score = s
            return max_score
        except Exception:
            logger.warning("Failed to score file (temporal OWW path): %s", wav_path, exc_info=True)
            return None

    return _score_file_temporal


def build_model_scorer(
    model_path: str | Path,
) -> tuple[str, Callable[[Path], float | None]]:
    """
    Build a clip-level scoring function for an ONNX wake word model.

    Args:
        model_path: Path to the .onnx model file.

    Returns:
        (architecture, scorer) where scorer accepts a WAV/FLAC path and
        returns a float score or None if scoring fails.
    """
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError("onnxruntime required. pip install onnxruntime") from e

    model_path = Path(model_path)
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    architecture = detect_architecture(model_path, session)
    logger.info("Using scoring path: %s", architecture)

    if architecture == "mlp_on_oww":
        return architecture, _build_oww_scorer(session, input_name)

    if architecture == "temporal_oww":
        return architecture, _build_temporal_oww_scorer(session, input_name)

    return architecture, _build_cnn_scorer(session, input_name)


# ── Main evaluation function ─────────────────────────────────────────────────


def evaluate_onnx_model(
    model_path: str | Path,
    test_dir: str | Path,
    threshold: float = 0.50,
    dump_scores_csv: str | Path | None = None,
) -> dict:
    """
    Evaluate an ONNX wake word model on a test set.

    Auto-detects model architecture (MLP-on-OWW vs CNN) and uses the
    correct scoring path. For MLP-on-OWW models, extracts OWW embeddings
    (matching train.py exactly). For legacy CNN models, uses mel spectrograms.

    The test_dir must contain:
        positives/  -- WAV files containing the wake word
        negatives/  -- WAV files of background audio (no wake word)

    Args:
        model_path: Path to the .onnx model file.
        test_dir: Directory containing positives/ and negatives/ subdirs.
        threshold: Classification threshold for FAR/FRR computation.
        dump_scores_csv: Optional path to write per-file scores CSV.
            Columns: file, label, score, threshold_pass.
            Useful for debugging false rejects/accepts.

    Returns:
        dict with keys:
            d_prime, far_per_hour, frr, roc_auc,
            tp_scores, fp_scores, tp_mean, fp_mean,
            n_positives, n_negatives, threshold_used,
            architecture,
            optimal_threshold, optimal_far, optimal_frr, eer_approx,
            confusion_matrix (dict with tp, fp, tn, fn, precision, recall, f1)
    """
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError as e:
        raise ImportError("scikit-learn required. pip install 'violawake[training]'") from e

    model_path = Path(model_path)
    test_dir = Path(test_dir)
    pos_dir = test_dir / "positives"
    neg_dir = test_dir / "negatives"

    if not pos_dir.exists():
        raise FileNotFoundError(f"positives/ directory not found in {test_dir}")
    if not neg_dir.exists():
        raise FileNotFoundError(f"negatives/ directory not found in {test_dir}")

    architecture, score_file = build_model_scorer(model_path)

    # Score positive samples
    logger.info("Scoring positive samples in %s", pos_dir)
    pos_files = sorted(list(pos_dir.rglob("*.wav")) + list(pos_dir.rglob("*.flac")))
    pos_scores: list[float] = []
    pos_scored_files: list[Path] = []
    for f in pos_files:
        s = score_file(f)
        if s is not None:
            pos_scores.append(s)
            pos_scored_files.append(f)

    # Score negative samples
    logger.info("Scoring negative samples in %s", neg_dir)
    neg_files = sorted(list(neg_dir.rglob("*.wav")) + list(neg_dir.rglob("*.flac")))
    neg_scores: list[float] = []
    neg_scored_files: list[Path] = []
    for f in neg_files:
        s = score_file(f)
        if s is not None:
            neg_scores.append(s)
            neg_scored_files.append(f)

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
    fpr, tpr, _thresholds = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    # FAR and FRR at the given threshold
    tp_at_thresh = np.sum(pos_arr >= threshold)
    fp_at_thresh = np.sum(neg_arr >= threshold)
    frr = float(1.0 - tp_at_thresh / len(pos_arr)) if len(pos_arr) > 0 else 1.0

    # FAR in events/hour -- assumes each negative file is approximately 1 second
    # (standard test set convention). Adjust if your test set files are longer.
    neg_total_hours = len(neg_arr) / 3600.0
    far_per_hour = float(fp_at_thresh / neg_total_hours) if neg_total_hours > 0 else float("inf")

    # Optimal threshold sweep
    opt = find_optimal_threshold(pos_arr, neg_arr)

    # Confusion matrix at the given threshold
    cm = compute_confusion_matrix(pos_arr, neg_arr, threshold)

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
        "architecture": architecture,
        # Optimal threshold results
        "optimal_threshold": opt["optimal_threshold"],
        "optimal_far": opt["optimal_far"],
        "optimal_frr": opt["optimal_frr"],
        "eer_approx": opt["eer_approx"],
        # Confusion matrix at the given threshold
        "confusion_matrix": cm,
    }

    # Dump per-file scores if requested
    if dump_scores_csv is not None:
        _dump_scores_csv(
            pos_scored_files, pos_scores,
            neg_scored_files, neg_scores,
            threshold, Path(dump_scores_csv),
        )

    logger.info(
        "Evaluation complete: arch=%s, d'=%.2f, FAR=%.2f/hr, FRR=%.1f%%, AUC=%.3f, optimal_thresh=%.2f",
        architecture, d_prime, far_per_hour, frr * 100, roc_auc, opt["optimal_threshold"],
    )

    return results
