#!/usr/bin/env python
"""
J5: Multi-frame temporal model training.

Trains TemporalCNN, TemporalGRU, and TemporalConvGRU on the cached temporal
embeddings (embedding_cache_temporal.npz) and compares against the mean-pool
MLP baseline.

Uses the same training infrastructure as the existing pipeline:
  - FocalLoss for class imbalance
  - AdamW + cosine annealing LR
  - EMA weight averaging
  - Group-aware train/val split (by source file)
  - Early stopping on validation loss

Exports best model to ONNX and verifies it loads with onnxruntime.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from violawake_sdk.training.losses import FocalLoss
from violawake_sdk.training.temporal_model import (
    TemporalCNN,
    TemporalConvGRU,
    TemporalGRU,
    count_parameters,
    export_temporal_onnx,
)
from violawake_sdk.training.weight_averaging import EMATracker, auto_select_averaging


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_PATH = Path(__file__).parent / "embedding_cache_temporal.npz"
OUTPUT_DIR = Path(__file__).parent / "models" / "j5_temporal"
RESULTS_PATH = Path(__file__).parent / "j5_temporal_results.json"

SEEDS = [42, 43, 44]
EPOCHS = 80
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 15
EMA_DECAY = 0.999
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Threshold sweep for evaluation
THRESHOLD_SWEEP = [0.3, 0.5, 0.7, 0.8, 0.9]


# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_temporal_data(cache_path: Path) -> dict:
    """Load the temporal embedding cache."""
    data = np.load(str(cache_path), allow_pickle=True)
    return {
        "temporal": data["temporal"],       # (N, 9, 96)
        "meanpool": data["meanpool"],       # (N, 96) -- for baseline comparison
        "labels": data["labels"],           # (N,) int32
        "tags": data["tags"],               # (N,) str
        "source_idx": data["source_idx"],   # (N,) int32
    }


def make_split(
    labels: np.ndarray,
    source_idx: np.ndarray,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware stratified train/val split.

    All embeddings from the same source file go to the same split
    to prevent data leakage from augmented variants.
    """
    rng = np.random.default_rng(seed)

    # Split positive sources
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_sources = sorted(set(source_idx[pos_mask].tolist()))
    neg_sources = sorted(set(source_idx[neg_mask].tolist()))

    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)

    n_val_pos = max(1, int(len(pos_sources) * val_fraction))
    n_val_neg = max(1, int(len(neg_sources) * val_fraction))

    val_pos_sources = set(pos_sources[:n_val_pos])
    val_neg_sources = set(neg_sources[:n_val_neg])

    val_mask = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 1 and source_idx[i] in val_pos_sources:
            val_mask[i] = True
        elif labels[i] == 0 and source_idx[i] in val_neg_sources:
            val_mask[i] = True

    train_indices = np.where(~val_mask)[0]
    val_indices = np.where(val_mask)[0]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    model_name: str,
    seed: int,
) -> dict:
    """Train a single model and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ema = EMATracker(model, decay=EMA_DECAY)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state = None
    best_ema_state = None

    t0 = time.monotonic()

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update()
            train_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ema_state = ema.state_dict()
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or no_improve == 0:
            marker = " *" if epoch == best_epoch else ""
            print(
                f"  [{model_name} s{seed}] Epoch {epoch:3d}: "
                f"train={avg_train:.4f} val={avg_val:.4f} best={best_val_loss:.4f}{marker}"
            )

        if no_improve >= PATIENCE:
            print(f"  [{model_name} s{seed}] Early stop at epoch {epoch} (best: {best_epoch})")
            break

    train_time = time.monotonic() - t0

    # Restore best and select averaging
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    if best_ema_state is not None:
        ema.load_state_dict(best_ema_state)

    # Evaluate EMA
    ema.apply()
    model.eval()
    ema_val_loss = 0.0
    n_ema = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            pred = model(bx)
            loss = criterion(pred, by)
            ema_val_loss += loss.item()
            n_ema += 1
    ema_val_loss = ema_val_loss / max(n_ema, 1)
    ema.restore()

    method = auto_select_averaging(
        raw_val_loss=best_val_loss,
        ema_val_loss=ema_val_loss,
        swa_val_loss=None,
    )
    if method == "ema":
        ema.apply()

    print(f"  [{model_name} s{seed}] Averaging: {method} (raw={best_val_loss:.4f}, ema={ema_val_loss:.4f})")

    # Score all data for metrics
    model.eval()
    model = model.to(DEVICE)
    return {
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "ema_val_loss": float(ema_val_loss),
        "averaging_method": method,
        "train_time": round(train_time, 1),
    }


def evaluate_scores(
    model: nn.Module,
    X: torch.Tensor,
    labels: np.ndarray,
    tags: np.ndarray,
) -> dict:
    """Compute EER, AUC, d-prime, and per-tag/threshold metrics."""
    from sklearn.metrics import auc, roc_curve

    model.eval()
    model = model.to(DEVICE)

    # Score all samples in batches
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            batch = X[i:i + BATCH_SIZE].to(DEVICE)
            scores = model(batch).cpu().numpy().flatten()
            all_scores.append(scores)
    all_scores = np.concatenate(all_scores)

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_scores = all_scores[pos_mask]
    neg_scores = all_scores[neg_mask]

    # Cohen's d
    pooled_std = np.sqrt(0.5 * (pos_scores.var() + neg_scores.var()))
    dprime = float((pos_scores.mean() - neg_scores.mean()) / pooled_std) if pooled_std > 1e-10 else 0.0

    # ROC/AUC
    fpr, tpr, thresholds = roc_curve(labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    # EER
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5

    # Per-tag scores
    unique_tags = sorted(set(tags.tolist()))
    tag_scores = {}
    for tag in unique_tags:
        tag_mask = tags == tag
        tag_s = all_scores[tag_mask]
        tag_scores[tag] = {
            "mean": round(float(tag_s.mean()), 4),
            "std": round(float(tag_s.std()), 4),
            "min": round(float(tag_s.min()), 4),
            "max": round(float(tag_s.max()), 4),
            "n": int(tag_mask.sum()),
        }

    # Confusable negatives
    conf_mask = np.isin(tags, ["neg_confusable", "neg_confusable_v2"])
    conf_scores = all_scores[conf_mask] if conf_mask.any() else np.array([])

    # Threshold sweep
    threshold_results = {}
    for t in THRESHOLD_SWEEP:
        det_rate = float((pos_scores >= t).mean()) if len(pos_scores) > 0 else 0.0
        fa_rate = float((neg_scores >= t).mean()) if len(neg_scores) > 0 else 0.0
        conf_fa_rate = float((conf_scores >= t).mean()) if len(conf_scores) > 0 else 0.0
        threshold_results[str(t)] = {
            "detection_rate": round(det_rate, 4),
            "false_alarm_rate": round(fa_rate, 4),
            "confusable_fa_rate": round(conf_fa_rate, 4),
        }

    return {
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "auc": round(roc_auc, 4),
        "dprime": round(dprime, 3),
        "pos_mean": round(float(pos_scores.mean()), 4),
        "pos_std": round(float(pos_scores.std()), 4),
        "neg_mean": round(float(neg_scores.mean()), 4),
        "neg_std": round(float(neg_scores.std()), 4),
        "conf_mean": round(float(conf_scores.mean()), 4) if len(conf_scores) > 0 else None,
        "conf_std": round(float(conf_scores.std()), 4) if len(conf_scores) > 0 else None,
        "tag_scores": tag_scores,
        "thresholds": threshold_results,
    }


# ---------------------------------------------------------------------------
# MLP baseline (for fair comparison on same data/splits)
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """Mean-pool MLP baseline matching train.py architecture."""

    def __init__(self, embedding_dim: int = 96, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("J5: Multi-frame Temporal Model Training")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load data
    print(f"Loading temporal embeddings from {CACHE_PATH}...")
    data = load_temporal_data(CACHE_PATH)
    temporal = data["temporal"]  # (N, 9, 96)
    meanpool = data["meanpool"]  # (N, 96)
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    n_total = len(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    seq_len = temporal.shape[1]
    emb_dim = temporal.shape[2]

    print(f"Samples: {n_total} ({n_pos} pos, {n_neg} neg)")
    print(f"Temporal shape: ({seq_len} frames, {emb_dim}-dim embeddings)")
    print()

    # Model configurations to train
    model_configs = {
        "baseline_mlp": {
            "factory": lambda: BaselineMLP(emb_dim, 64),
            "uses_temporal": False,
        },
        "temporal_cnn": {
            "factory": lambda: TemporalCNN(emb_dim, seq_len),
            "uses_temporal": True,
        },
        "temporal_gru": {
            "factory": lambda: TemporalGRU(emb_dim, hidden_dim=32),
            "uses_temporal": True,
        },
        "temporal_convgru": {
            "factory": lambda: TemporalConvGRU(emb_dim, conv_channels=48, gru_hidden=24),
            "uses_temporal": True,
        },
    }

    # Print model sizes
    print("Model sizes:")
    for name, cfg in model_configs.items():
        m = cfg["factory"]()
        n_params = count_parameters(m)
        print(f"  {name}: {n_params:,} params")
    print()

    # Prepare tensors
    X_temporal = torch.tensor(temporal, dtype=torch.float32)
    X_meanpool = torch.tensor(meanpool, dtype=torch.float32)
    y_all = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    # Results storage
    all_results = {
        "experiment": "j5_temporal_models",
        "n_samples": n_total,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "seq_len": seq_len,
        "embedding_dim": emb_dim,
        "device": DEVICE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "patience": PATIENCE,
        "seeds": SEEDS,
        "models": {},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, cfg in model_configs.items():
        print(f"\n{'=' * 60}")
        print(f"Training: {model_name}")
        print(f"{'=' * 60}")

        seed_results = []

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            # Split
            train_idx, val_idx = make_split(labels, source_idx, seed)
            print(f"  Split: {len(train_idx)} train, {len(val_idx)} val")

            # Select data based on model type
            if cfg["uses_temporal"]:
                X_train = X_temporal[train_idx]
                X_val = X_temporal[val_idx]
            else:
                X_train = X_meanpool[train_idx]
                X_val = X_meanpool[val_idx]

            y_train = y_all[train_idx]
            y_val = y_all[val_idx]

            # Create fresh model
            model = cfg["factory"]()

            # Train
            train_info = train_model(
                model, X_train, y_train, X_val, y_val,
                model_name, seed,
            )

            # Evaluate on all data
            if cfg["uses_temporal"]:
                scores = evaluate_scores(model, X_temporal, labels, tags)
            else:
                scores = evaluate_scores(model, X_meanpool, labels, tags)

            result = {**train_info, **scores, "name": f"{model_name}_s{seed}"}
            seed_results.append(result)

            print(
                f"  [{model_name} s{seed}] "
                f"EER={scores['eer']:.4f} AUC={scores['auc']:.4f} "
                f"d'={scores['dprime']:.3f} pos_mean={scores['pos_mean']:.3f}"
            )

            # Export ONNX for best seed (seed 42)
            if seed == 42:
                onnx_path = OUTPUT_DIR / f"{model_name}.onnx"
                if cfg["uses_temporal"]:
                    export_temporal_onnx(model, str(onnx_path), seq_len=seq_len, embedding_dim=emb_dim)
                else:
                    # MLP baseline export
                    model.eval()
                    model = model.cpu()
                    dummy = torch.zeros(1, emb_dim)
                    torch.onnx.export(
                        model,
                        dummy,
                        str(onnx_path),
                        input_names=["embedding"],
                        output_names=["score"],
                        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
                        opset_version=11,
                    )
                print(f"  ONNX exported: {onnx_path}")

                # Save config
                config_path = onnx_path.with_suffix(".config.json")
                config = {
                    "architecture": "temporal_" + model_name.replace("temporal_", "") if cfg["uses_temporal"] else "mlp_on_oww",
                    "model_class": model_name,
                    "embedding_dim": emb_dim,
                    "seq_len": seq_len if cfg["uses_temporal"] else None,
                    "n_params": count_parameters(cfg["factory"]()),
                    "eer": scores["eer"],
                    "auc": scores["auc"],
                    "dprime": scores["dprime"],
                }
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)

        # Aggregate across seeds
        eers = [r["eer"] for r in seed_results]
        aucs = [r["auc"] for r in seed_results]
        dprimes = [r["dprime"] for r in seed_results]
        pos_means = [r["pos_mean"] for r in seed_results]
        neg_means = [r["neg_mean"] for r in seed_results]

        summary = {
            "eer_mean": round(float(np.mean(eers)), 4),
            "eer_std": round(float(np.std(eers)), 4),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "dprime_mean": round(float(np.mean(dprimes)), 3),
            "pos_mean": round(float(np.mean(pos_means)), 4),
            "neg_mean": round(float(np.mean(neg_means)), 4),
            "n_params": count_parameters(cfg["factory"]()),
            "per_seed": seed_results,
        }

        all_results["models"][model_name] = summary

        print(f"\n  {model_name} SUMMARY:")
        print(f"    EER:    {summary['eer_mean']:.4f} +/- {summary['eer_std']:.4f}")
        print(f"    AUC:    {summary['auc_mean']:.4f}")
        print(f"    d':     {summary['dprime_mean']:.3f}")
        print(f"    params: {summary['n_params']:,}")

    # ---------------------------------------------------------------------------
    # Final comparison
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>8} {'EER':>8} {'AUC':>8} {'d-prime':>8} {'pos_mean':>9} {'neg_mean':>9}")
    print("-" * 80)
    for name, summary in all_results["models"].items():
        print(
            f"{name:<20} {summary['n_params']:>8,} "
            f"{summary['eer_mean']:>8.4f} {summary['auc_mean']:>8.4f} "
            f"{summary['dprime_mean']:>8.3f} {summary['pos_mean']:>9.4f} "
            f"{summary['neg_mean']:>9.4f}"
        )

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")

    # ---------------------------------------------------------------------------
    # Verify ONNX models load
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ONNX VERIFICATION")
    print("=" * 70)

    import onnxruntime as ort

    for model_name, cfg in model_configs.items():
        onnx_path = OUTPUT_DIR / f"{model_name}.onnx"
        if not onnx_path.exists():
            print(f"  {model_name}: SKIP (no ONNX file)")
            continue

        try:
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]

            # Run inference
            if cfg["uses_temporal"]:
                dummy = np.random.randn(1, seq_len, emb_dim).astype(np.float32)
            else:
                dummy = np.random.randn(1, emb_dim).astype(np.float32)

            t0 = time.monotonic()
            n_runs = 1000
            for _ in range(n_runs):
                result = session.run(None, {input_info.name: dummy})
            latency_ms = (time.monotonic() - t0) / n_runs * 1000

            score = float(result[0].flatten()[0])
            print(
                f"  {model_name}: OK "
                f"(input={input_info.shape}, output={output_info.shape}, "
                f"score={score:.4f}, latency={latency_ms:.2f}ms)"
            )
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
