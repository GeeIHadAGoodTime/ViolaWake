"""
Verify BCE Loss Result — Multi-seed + ONNX Export
===================================================

The loss sweep found BCE weighted gives 11.9% EER vs 14.1% for focal loss.
This script verifies that result with:
1. Multi-seed training (3 seeds)
2. ONNX export + standard evaluate_onnx_model evaluation
3. Direct comparison with D_combined (focal loss) using same methodology
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
EVAL_DIR = WAKEWORD / "eval_clean"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"


def train_d_combined_bce(seed: int = 42) -> tuple:
    """Train D_combined with BCE loss instead of focal loss."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load cache
    data = np.load(CACHE_FILE, allow_pickle=True)
    embs = data["embeddings"]
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    # D_combined sources
    d_tags = ["pos_main", "pos_diverse", "neg_main", "neg_confusable"]
    mask = np.zeros(len(embs), dtype=bool)
    for t in d_tags:
        mask |= (tags == t)

    d_embs = embs[mask]
    d_labels = labels[mask]
    d_source = source_idx[mask]

    n_pos = (d_labels == 1).sum()
    n_neg = (d_labels == 0).sum()

    # Split
    rng = np.random.default_rng(seed)
    pos_mask = d_labels == 1
    neg_mask = d_labels == 0
    pos_sources = sorted(set(d_source[pos_mask]))
    neg_sources = sorted(set(d_source[neg_mask]))
    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)
    val_pos = set(pos_sources[:max(1, len(pos_sources) // 5)])
    val_neg = set(neg_sources[:max(1, len(neg_sources) // 5)])

    val_mask = np.zeros(len(d_labels), dtype=bool)
    train_mask = np.zeros(len(d_labels), dtype=bool)
    for i in range(len(d_labels)):
        if d_labels[i] == 1:
            if d_source[i] in val_pos:
                val_mask[i] = True
            else:
                train_mask[i] = True
        else:
            if d_source[i] in val_neg:
                val_mask[i] = True
            else:
                train_mask[i] = True

    X = torch.tensor(d_embs, dtype=torch.float32)
    y = torch.tensor(d_labels, dtype=torch.float32).unsqueeze(1)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"  seed={seed}: {train_mask.sum()} train / {val_mask.sum()} val "
          f"({n_pos} pos + {n_neg} neg)")

    embedding_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1), nn.Sigmoid(),
    )

    # BCE loss (the winning configuration)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    ema_decay = 0.999
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = best_ema = None
    t0 = time.monotonic()

    for ep in range(1, 61):
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

        if no_imp >= 12:
            break

    if best_state:
        model.load_state_dict(best_state)
    if best_ema:
        for n, p in model.named_parameters():
            p.data.copy_(best_ema[n])

    dur = time.monotonic() - t0
    print(f"  Trained in {dur:.1f}s, best_ep={best_ep}")

    # Export ONNX
    model_path = EXPERIMENTS / "models" / f"D_combined_bce_s{seed}.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model, torch.zeros(1, embedding_dim), str(model_path),
        input_names=["embedding"], output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    return model_path, {"best_epoch": best_ep, "best_val_loss": float(best_val), "time": round(dur, 1)}


def evaluate_onnx(model_path: Path, name: str) -> dict:
    """Standard ONNX evaluation (same as main harness)."""
    from violawake_sdk.training.evaluate import (
        evaluate_onnx_model, find_optimal_threshold, compute_dprime,
    )

    csv_path = EXPERIMENTS / f"{name}.scores.csv"
    results = evaluate_onnx_model(
        model_path=str(model_path),
        test_dir=str(EVAL_DIR),
        threshold=0.50,
        dump_scores_csv=str(csv_path),
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
    print("=" * 60)
    print("BCE LOSS VERIFICATION (Multi-Seed + ONNX)")
    print("=" * 60)

    seeds = [42, 43, 44]
    results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        model_path, train_info = train_d_combined_bce(seed)
        eval_result = evaluate_onnx(model_path, f"D_bce_s{seed}")
        eval_result.update(train_info)
        results.append(eval_result)
        print(f"  EER={eval_result['all_eer']:.1%}, AUC={eval_result['all_auc']:.4f}, "
              f"d'={eval_result['all_dprime']:.3f}, FRR@0.50={eval_result['all_frr_050']:.1%}")

    # Multi-seed summary
    eers = [r["all_eer"] for r in results]
    aucs = [r["all_auc"] for r in results]
    dprimes = [r["all_dprime"] for r in results]

    print("\n" + "=" * 60)
    print("BCE MULTI-SEED RESULTS")
    print("=" * 60)
    print(f"  EER:  {np.mean(eers):.1%} +/- {np.std(eers):.1%}")
    print(f"  AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
    print(f"  d':   {np.mean(dprimes):.3f} +/- {np.std(dprimes):.3f}")
    print(f"\n  D_combined (focal): 13.6% +/- 0.1% (3 seeds)")
    delta = np.mean(eers) - 0.136
    print(f"  BCE improvement: {delta:+.1%} absolute")
    print(f"  Statistically significant: {abs(delta) > 2 * max(np.std(eers), 0.001)}")

    # Also run focal loss baseline for direct comparison
    print("\n--- Focal Loss Reference (seed 42) ---")
    d_focal = EXPERIMENTS / "models" / "D_combined.onnx"
    if d_focal.exists():
        focal_result = evaluate_onnx(d_focal, "D_focal_ref")
        print(f"  Focal EER={focal_result['all_eer']:.1%}")
        print(f"  BCE best={min(eers):.1%}, Focal={focal_result['all_eer']:.1%}")
        print(f"  Delta: {min(eers) - focal_result['all_eer']:+.1%}")

    # Save
    out = EXPERIMENTS / "exp_bce_verification.json"
    summary = {
        "loss": "BCELoss (no focal, no label smoothing)",
        "data": "D_combined (pos_main + pos_diverse + neg_main + neg_confusable)",
        "seeds": seeds,
        "per_seed": results,
        "eer_mean": round(float(np.mean(eers)), 4),
        "eer_std": round(float(np.std(eers)), 4),
        "auc_mean": round(float(np.mean(aucs)), 4),
        "dprime_mean": round(float(np.mean(dprimes)), 3),
        "vs_focal_delta": round(float(np.mean(eers) - 0.136), 4),
    }
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
