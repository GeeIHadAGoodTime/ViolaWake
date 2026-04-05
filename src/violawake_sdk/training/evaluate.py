"""
ViolaWake evaluation helpers.

Supports:
  - ``mlp_on_oww`` models with input shape ``(batch, 96)``
  - ``temporal_oww`` models with input shape ``(batch, seq_len, 96)``
  - legacy mel/CNN models
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_OWW_EMBEDDING_DIM = 96
_DEFAULT_TEMPORAL_SEQ_LEN = 9


def _get_feature_dims(input_shape: list | tuple) -> list:
    """Return non-batch input dimensions from an ONNX input shape."""
    dims = list(input_shape)
    return dims[1:] if len(dims) >= 2 else dims


def _detect_architecture_from_input_shape(input_shape: list | tuple, input_name: str) -> str:
    """Detect evaluation path from ONNX input shape."""
    feature_dims = _get_feature_dims(input_shape)

    if len(feature_dims) == 1 and feature_dims[0] == _OWW_EMBEDDING_DIM:
        logger.info("Architecture detected from input shape %s: mlp_on_oww", input_shape)
        return "mlp_on_oww"

    if len(feature_dims) == 2 and feature_dims[-1] == _OWW_EMBEDDING_DIM:
        logger.info("Architecture detected from input shape %s: temporal_oww", input_shape)
        return "temporal_oww"

    if input_name == "embeddings":
        logger.info(
            "Architecture detected from input name %s with shape %s: temporal_oww",
            input_name,
            input_shape,
        )
        return "temporal_oww"

    logger.info("Architecture detected from input shape %s: cnn", input_shape)
    return "cnn"


def _infer_temporal_seq_len(input_shape: list | tuple) -> int:
    """Infer ``seq_len`` from a temporal OWW ONNX input shape."""
    feature_dims = _get_feature_dims(input_shape)
    if len(feature_dims) >= 2 and isinstance(feature_dims[0], int) and feature_dims[0] > 0:
        return int(feature_dims[0])
    return _DEFAULT_TEMPORAL_SEQ_LEN


def detect_architecture(model_path: Path, session) -> str:
    """
    Auto-detect model architecture from config or ONNX input shape.

    Returns one of ``mlp_on_oww``, ``temporal_oww``, or ``cnn``.
    """
    config_path = model_path.with_suffix(".config.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            config = {}

        arch = config.get("architecture", "")
        if arch in ("mlp_on_oww", "temporal_oww", "temporal_cnn"):
            result = "temporal_oww" if arch == "temporal_cnn" else arch
            logger.info("Architecture detected from config: %s", result)
            return result
        if arch == "cnn":
            logger.info("Architecture detected from config: cnn")
            return "cnn"

    input_info = session.get_inputs()[0]
    return _detect_architecture_from_input_shape(input_info.shape, input_info.name)


def compute_dprime(
    pos_scores: list[float] | np.ndarray,
    neg_scores: list[float] | np.ndarray,
) -> float:
    """Compute the repo's historical Cohen's d-style separation metric."""
    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)

    if len(pos) == 0 or len(neg) == 0:
        return 0.0

    pooled_std = np.sqrt(0.5 * (pos.var() + neg.var()))
    if pooled_std < 1e-10:
        return 0.0

    return float((pos.mean() - neg.mean()) / pooled_std)


def compute_eer(fpr: np.ndarray, tpr: np.ndarray) -> tuple[float, int]:
    """Compute Equal Error Rate from ROC arrays."""
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, idx


def find_optimal_threshold(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    step: float = 0.01,
) -> dict:
    """Sweep thresholds and minimize ``FPR + FNR``."""
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

    for threshold in thresholds:
        far = float(np.sum(neg_scores >= threshold) / n_neg)
        frr = float(np.sum(pos_scores < threshold) / n_pos)
        cost = far + frr
        if cost < best_cost:
            best_cost = cost
            best_thresh = float(threshold)
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
    """Compute confusion-matrix counts and precision/recall/F1."""
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


def _dump_scores_csv(
    pos_files: list[Path],
    pos_scores: list[float],
    neg_files: list[Path],
    neg_scores: list[float],
    threshold: float,
    csv_path: Path,
) -> None:
    """Write per-file scores using the active analysis threshold."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if len(pos_files) != len(pos_scores):
        raise ValueError(f"Positive files/scores mismatch: {len(pos_files)} vs {len(pos_scores)}")
    if len(neg_files) != len(neg_scores):
        raise ValueError(f"Negative files/scores mismatch: {len(neg_files)} vs {len(neg_scores)}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "score", "correct"])
        for fpath, score in zip(pos_files, pos_scores, strict=True):
            writer.writerow([str(fpath), "positive", f"{score:.6f}", score >= threshold])
        for fpath, score in zip(neg_files, neg_scores, strict=True):
            writer.writerow([str(fpath), "negative", f"{score:.6f}", score < threshold])

    logger.info(
        "Score dump written to %s (%d entries)", csv_path, len(pos_scores) + len(neg_scores)
    )


def _extract_oww_frame_embeddings(
    wav_path: Path,
    *,
    preprocessor,
    load_audio,
    center_crop,
    clip_samples: int,
) -> np.ndarray | None:
    """Extract frame embeddings with the same OpenWakeWord path used in training."""
    audio = load_audio(wav_path)
    if audio is None:
        return None

    audio_rms = float(np.sqrt(np.mean(audio**2)))
    if audio_rms < 1e-6:
        logger.warning("Skipping zero-energy file: %s", wav_path)
        return None

    audio = center_crop(audio, clip_samples)
    audio_i16 = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767).astype(np.int16)

    if len(audio_i16) < clip_samples:
        audio_i16 = np.pad(audio_i16, (0, clip_samples - len(audio_i16)))
    else:
        audio_i16 = audio_i16[:clip_samples]

    frame_embeddings = np.asarray(preprocessor.embed_clips(audio_i16.reshape(1, -1), ncpu=1))
    if frame_embeddings.ndim == 3:
        frame_embeddings = frame_embeddings[0]
    else:
        frame_embeddings = np.squeeze(frame_embeddings)

    if frame_embeddings.ndim == 1:
        frame_embeddings = frame_embeddings.reshape(1, -1)

    return frame_embeddings.astype(np.float32, copy=False)


def _build_oww_scorer(session, input_name: str):
    """Build a scorer for mean-pooled OpenWakeWord embedding models."""
    try:
        from openwakeword.model import Model as OWWModel  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "openwakeword required for MLP-on-OWW evaluation. pip install openwakeword"
        ) from e

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    def _score_file_oww(wav_path: Path) -> float | None:
        try:
            embeddings = _extract_oww_frame_embeddings(
                wav_path,
                preprocessor=preprocessor,
                load_audio=load_audio,
                center_crop=center_crop,
                clip_samples=CLIP_SAMPLES,
            )
            if embeddings is None:
                return None

            embedding = embeddings.mean(axis=0).astype(np.float32)
            score = session.run(None, {input_name: embedding.reshape(1, -1)})[0]
            return float(np.asarray(score).flatten()[0])
        except Exception:
            logger.warning("Failed to score file (OWW path): %s", wav_path, exc_info=True)
            return None

    return _score_file_oww


def _build_cnn_scorer(session, input_name: str):
    """Build a scorer for legacy mel/CNN models."""
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
    """Build a scorer for temporal OpenWakeWord embedding models."""
    try:
        from openwakeword.model import Model as OWWModel  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "openwakeword required for temporal OWW evaluation. pip install openwakeword"
        ) from e

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    seq_len = _infer_temporal_seq_len(session.get_inputs()[0].shape)

    def _score_file_temporal(wav_path: Path) -> float | None:
        try:
            emb = _extract_oww_frame_embeddings(
                wav_path,
                preprocessor=preprocessor,
                load_audio=load_audio,
                center_crop=center_crop,
                clip_samples=CLIP_SAMPLES,
            )
            if emb is None:
                return None

            n_frames = emb.shape[0]
            if n_frames < seq_len:
                padded = np.zeros((seq_len, emb.shape[1]), dtype=np.float32)
                padded[:n_frames] = emb
                for idx in range(n_frames, seq_len):
                    padded[idx] = emb[-1]
                window = padded[np.newaxis, :, :].astype(np.float32)
                score = session.run(None, {input_name: window})[0]
                return float(np.asarray(score).flatten()[0])

            max_score = -1.0
            for idx in range(n_frames - seq_len + 1):
                window = emb[idx : idx + seq_len][np.newaxis, :, :].astype(np.float32)
                score = session.run(None, {input_name: window})[0]
                max_score = max(max_score, float(np.asarray(score).flatten()[0]))
            return max_score
        except Exception:
            logger.warning("Failed to score file (temporal OWW path): %s", wav_path, exc_info=True)
            return None

    return _score_file_temporal


def build_model_scorer(model_path: str | Path) -> tuple[str, Callable[[Path], float | None]]:
    """Create a clip scorer for an ONNX wake-word model."""
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


def evaluate_onnx_model(
    model_path: str | Path,
    test_dir: str | Path,
    threshold: float = 0.50,
    dump_scores_csv: str | Path | None = None,
    sweep: bool = True,
) -> dict:
    """
    Evaluate an ONNX wake-word model on a test set.

    When ``sweep`` is enabled, thresholds are scanned from ``0.00`` to ``1.00``
    in ``0.01`` steps and the operating point minimizing ``FPR + FNR`` is used
    for the optimal-threshold analysis outputs.
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

    logger.info("Scoring positive samples in %s", pos_dir)
    pos_files = sorted(list(pos_dir.rglob("*.wav")) + list(pos_dir.rglob("*.flac")))
    pos_scores: list[float] = []
    pos_scored_files: list[Path] = []
    for path in pos_files:
        score = score_file(path)
        if score is not None:
            pos_scores.append(score)
            pos_scored_files.append(path)

    logger.info("Scoring negative samples in %s", neg_dir)
    neg_files = sorted(list(neg_dir.rglob("*.wav")) + list(neg_dir.rglob("*.flac")))
    neg_scores: list[float] = []
    neg_scored_files: list[Path] = []
    for path in neg_files:
        score = score_file(path)
        if score is not None:
            neg_scores.append(score)
            neg_scored_files.append(path)

    if not pos_scores:
        raise RuntimeError(f"No valid positive audio files found in {pos_dir}")
    if not neg_scores:
        raise RuntimeError(f"No valid negative audio files found in {neg_dir}")

    pos_arr = np.asarray(pos_scores, dtype=np.float32)
    neg_arr = np.asarray(neg_scores, dtype=np.float32)

    d_prime = compute_dprime(pos_arr, neg_arr)

    all_scores = np.concatenate([pos_arr, neg_arr])
    all_labels = np.concatenate([np.ones(len(pos_arr)), np.zeros(len(neg_arr))])
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    tp_at_threshold = np.sum(pos_arr >= threshold)
    fp_at_threshold = np.sum(neg_arr >= threshold)
    far = float(fp_at_threshold / len(neg_arr)) if len(neg_arr) > 0 else 1.0
    frr = float(1.0 - tp_at_threshold / len(pos_arr)) if len(pos_arr) > 0 else 1.0

    neg_total_hours = len(neg_arr) / 3600.0
    far_per_hour = float(fp_at_threshold / neg_total_hours) if neg_total_hours > 0 else float("inf")

    if sweep:
        opt = find_optimal_threshold(pos_arr, neg_arr)
    else:
        opt = {
            "optimal_threshold": float(threshold),
            "optimal_far": far,
            "optimal_frr": frr,
            "eer_approx": (far + frr) / 2.0,
        }

    cm = compute_confusion_matrix(pos_arr, neg_arr, threshold)
    optimal_cm = compute_confusion_matrix(pos_arr, neg_arr, opt["optimal_threshold"])

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
        "optimal_threshold": opt["optimal_threshold"],
        "optimal_far": opt["optimal_far"],
        "optimal_frr": opt["optimal_frr"],
        "eer_approx": opt["eer_approx"],
        "confusion_matrix": cm,
        "optimal_confusion_matrix": optimal_cm,
        "threshold_sweep_enabled": sweep,
        "score_dump_threshold": opt["optimal_threshold"],
    }

    if dump_scores_csv is not None:
        _dump_scores_csv(
            pos_scored_files,
            pos_scores,
            neg_scored_files,
            neg_scores,
            opt["optimal_threshold"],
            Path(dump_scores_csv),
        )

    logger.info(
        "Evaluation complete: arch=%s, d'=%.2f, FAR=%.2f/hr, FRR=%.1f%%, AUC=%.3f, opt=%.2f",
        architecture,
        d_prime,
        far_per_hour,
        frr * 100,
        roc_auc,
        opt["optimal_threshold"],
    )

    return results
