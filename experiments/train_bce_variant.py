"""
Train any experiment configuration with BCE loss instead of focal loss.
Uses the same data selection, architecture, and evaluation as the main harness.

Usage:
  python experiments/train_bce_variant.py D_combined
  python experiments/train_bce_variant.py I_full_corpus --seeds 3
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
EVAL_DIR = WAKEWORD / "eval_clean"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"
CONFIG_FILE = EXPERIMENTS / "experiment_config.json"


def train_bce(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    source_idx: np.ndarray,
    output_model: Path,
    config: dict,
    arch_name: str = "default",
    seed: int = 42,
) -> dict:
    """Train MLP with BCE loss from pre-extracted embeddings."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    training = config.get("training_defaults", {})
    epochs = training.get("epochs", 60)
    patience = training.get("patience", 12)
    batch_size = training.get("batch_size", 32)
    lr = training.get("lr", 1e-3)
    weight_decay = training.get("weight_decay", 1e-4)
    ema_decay = training.get("ema_decay", 0.999)

    arch_variants = config.get("architecture_variants", {})
    arch_config = arch_variants.get(arch_name, {"hidden_dims": [64, 32], "dropout": [0.3, 0.2], "activation": "relu"})

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embedding_dim = X.shape[1]

    # Group-aware 80/20 split
    rng = np.random.default_rng(seed)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sources = sorted(set(source_idx[pos_mask]))
    neg_sources = sorted(set(source_idx[neg_mask]))
    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)
    val_pos_set = set(pos_sources[:max(1, len(pos_sources) // 5)])
    val_neg_set = set(neg_sources[:max(1, len(neg_sources) // 5)])

    val_mask = np.zeros(len(labels), dtype=bool)
    train_mask = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 1:
            val_mask[i] = source_idx[i] in val_pos_set
            train_mask[i] = not val_mask[i]
        else:
            val_mask[i] = source_idx[i] in val_neg_set
            train_mask[i] = not val_mask[i]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"  {name}_bce s{seed}: {train_mask.sum()} train / {val_mask.sum()} val "
          f"({n_pos} pos + {n_neg} neg)")

    # Build model
    hidden_dims = arch_config["hidden_dims"]
    dropouts = arch_config.get("dropout", [0.3, 0.2])
    layers = []
    in_dim = embedding_dim
    for i, h in enumerate(hidden_dims):
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        if i < len(dropouts):
            layers.append(nn.Dropout(dropouts[i]))
        in_dim = h
    layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
    model = nn.Sequential(*layers)

    # BCE loss — the key difference
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = best_ema = None
    t0 = time.monotonic()

    for ep in range(1, epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for n, p in model.named_parameters():
                ema[n] = ema_decay * ema[n] + (1 - ema_decay) * p.data
            tl += loss.item(); nb += 1
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                vl += criterion(model(bx), by).item(); vn += 1

        avg_v = vl / max(vn, 1)
        if avg_v < best_val:
            best_val, best_ep, no_imp = avg_v, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema = {k: v.clone() for k, v in ema.items()}
        else:
            no_imp += 1

        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)
    if best_ema:
        for n, p in model.named_parameters():
            p.data.copy_(best_ema[n])

    dur = time.monotonic() - t0

    # Export ONNX
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    import torch as _torch
    _torch.onnx.export(
        model, _torch.zeros(1, embedding_dim), str(output_model),
        input_names=["embedding"], output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    return {"best_epoch": best_ep, "best_val_loss": float(best_val), "time": round(dur, 1)}


def evaluate_onnx(model_path: Path, name: str) -> dict:
    """Standard ONNX evaluation."""
    from violawake_sdk.training.evaluate import (
        evaluate_onnx_model, find_optimal_threshold, compute_dprime,
    )
    csv_path = EXPERIMENTS / f"{name}.scores.csv"
    results = evaluate_onnx_model(
        model_path=str(model_path), test_dir=str(EVAL_DIR),
        threshold=0.50, dump_scores_csv=str(csv_path),
    )
    pos_scores = np.array(results["tp_scores"])
    neg_scores = np.array(results["fp_scores"])

    trained = []
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["label"] == "positive":
                    fn = row["file"].lower()
                    if "viola_wake_up" not in fn and "viola_please" not in fn:
                        trained.append(float(row["score"]))
    trained_arr = np.array(trained) if trained else pos_scores
    opt = find_optimal_threshold(trained_arr, neg_scores)

    return {
        "experiment": name,
        "all_eer": round(results["eer_approx"], 4),
        "all_auc": round(results["roc_auc"], 4),
        "all_dprime": round(results["d_prime"], 3),
        "all_frr_050": round(results["frr"], 4),
        "trained_eer": round(opt["eer_approx"], 4),
        "pos_mean": round(float(pos_scores.mean()), 4),
        "neg_mean": round(float(neg_scores.mean()), 4),
        "n_pos": len(pos_scores),
        "n_neg": len(neg_scores),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment name from config")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--arch", default="default")
    args = parser.parse_args()

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    exp_def = config["experiments"].get(args.experiment)
    if not exp_def:
        print(f"Unknown experiment: {args.experiment}")
        print(f"Available: {list(config['experiments'].keys())}")
        sys.exit(1)

    # Load cache
    data = np.load(CACHE_FILE, allow_pickle=True)
    tags = data["tags"]

    # Build mask for this experiment's sources
    all_tags = exp_def["pos"] + exp_def["neg"]
    mask = np.zeros(len(tags), dtype=bool)
    for t in all_tags:
        mask |= (tags == t)

    embs = data["embeddings"][mask]
    labels = data["labels"][mask]
    source_idx = data["source_idx"][mask]

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"{'='*60}")
    print(f"BCE TRAINING: {args.experiment}")
    print(f"  Sources: {all_tags}")
    print(f"  Data: {n_pos} pos + {n_neg} neg = {len(labels)} total")
    print(f"{'='*60}")

    seeds = list(range(42, 42 + args.seeds))
    results = []

    for seed in seeds:
        model_path = EXPERIMENTS / "models" / f"{args.experiment}_bce_s{seed}.onnx"
        train_info = train_bce(
            args.experiment, embs, labels, source_idx, model_path, config, args.arch, seed
        )
        eval_result = evaluate_onnx(model_path, f"{args.experiment}_bce_s{seed}")
        eval_result.update(train_info)
        results.append(eval_result)
        print(f"  seed={seed}: EER={eval_result['all_eer']:.1%}, AUC={eval_result['all_auc']:.4f}")

    # Summary
    eers = [r["all_eer"] for r in results]
    aucs = [r["all_auc"] for r in results]
    print(f"\n{'='*60}")
    print(f"BCE {args.experiment}: EER {np.mean(eers):.1%} +/- {np.std(eers):.1%}")
    print(f"  AUC {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"{'='*60}")

    # Save
    out = EXPERIMENTS / f"exp_{args.experiment}_bce.json"
    summary = {
        "experiment": args.experiment,
        "loss": "BCE",
        "seeds": seeds,
        "per_seed": results,
        "eer_mean": round(float(np.mean(eers)), 4),
        "eer_std": round(float(np.std(eers)), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
