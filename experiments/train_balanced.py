"""
Class-balanced training with negative subsampling.
Addresses the massive class imbalance in G_acav and I_full_corpus.

Strategy: Subsample negatives to maintain a target pos:neg ratio,
prioritizing hard negatives (confusable, adversarial) over generic ones.

Usage:
  python experiments/train_balanced.py I_full_corpus --ratio 3
  python experiments/train_balanced.py G_acav --ratio 5 --seeds 3
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

# Priority order for negative sampling (keep hard negatives first)
NEG_PRIORITY = [
    "neg_confusable",      # highest priority — phoneme-similar words
    "neg_confusable_v2",   # phoneme-mined confusables
    "neg_main",            # curated negatives
    "neg_musan_speech",    # diverse speech
    "neg_acav100m",        # large-scale audio (lowest priority)
    "neg_musan_music",
    "neg_musan_noise",
]


def subsample_negatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    tags: np.ndarray,
    source_idx: np.ndarray,
    target_ratio: float = 3.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample negatives to achieve target pos:neg ratio.

    Prioritizes hard negatives over generic ones.
    """
    rng = np.random.default_rng(seed)

    pos_mask = labels == 1
    n_pos = pos_mask.sum()
    target_neg = int(n_pos * target_ratio)

    # Group negatives by tag
    neg_groups = {}
    for i in range(len(labels)):
        if labels[i] == 0:
            t = str(tags[i])
            if t not in neg_groups:
                neg_groups[t] = []
            neg_groups[t].append(i)

    # Select negatives by priority
    selected_neg_indices = []
    remaining = target_neg

    for tag in NEG_PRIORITY:
        if tag not in neg_groups or remaining <= 0:
            continue
        indices = np.array(neg_groups[tag])
        rng.shuffle(indices)
        take = min(len(indices), remaining)
        selected_neg_indices.extend(indices[:take].tolist())
        remaining -= take
        print(f"    {tag}: {take}/{len(indices)}")

    # If still need more, take from remaining tags
    for tag, indices_list in neg_groups.items():
        if tag in NEG_PRIORITY or remaining <= 0:
            continue
        indices = np.array(indices_list)
        rng.shuffle(indices)
        take = min(len(indices), remaining)
        selected_neg_indices.extend(indices[:take].tolist())
        remaining -= take

    # Combine positive + selected negative indices
    pos_indices = np.where(pos_mask)[0].tolist()
    all_indices = pos_indices + selected_neg_indices

    print(f"  Balanced: {n_pos} pos + {len(selected_neg_indices)} neg (ratio {len(selected_neg_indices)/n_pos:.1f}:1)")

    return (
        embeddings[all_indices],
        labels[all_indices],
        source_idx[all_indices],
    )


def train_balanced(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    source_idx: np.ndarray,
    output_model: Path,
    config: dict,
    seed: int = 42,
    use_bce: bool = True,
) -> dict:
    """Train with class-balanced data."""
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

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embedding_dim = X.shape[1]

    # Group-aware split
    rng = np.random.default_rng(seed)
    pos_sources = sorted(set(source_idx[labels == 1]))
    neg_sources = sorted(set(source_idx[labels == 0]))
    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)
    val_pos = set(pos_sources[:max(1, len(pos_sources) // 5)])
    val_neg = set(neg_sources[:max(1, len(neg_sources) // 5)])

    val_mask = np.zeros(len(labels), dtype=bool)
    train_mask = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 1:
            val_mask[i] = source_idx[i] in val_pos
        else:
            val_mask[i] = source_idx[i] in val_neg
        train_mask[i] = not val_mask[i]

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    model = nn.Sequential(
        nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1), nn.Sigmoid(),
    )

    if use_bce:
        criterion = nn.BCELoss()
    else:
        from violawake_sdk.training.losses import FocalLoss
        criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)

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
    parser.add_argument("experiment")
    parser.add_argument("--ratio", type=float, default=3.0, help="Target neg:pos ratio")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--focal", action="store_true", help="Use focal loss instead of BCE")
    args = parser.parse_args()

    with open(CONFIG_FILE) as f:
        config = json.load(f)

    exp_def = config["experiments"][args.experiment]
    data = np.load(CACHE_FILE, allow_pickle=True)
    tags = data["tags"]

    all_tags = exp_def["pos"] + exp_def["neg"]
    mask = np.zeros(len(tags), dtype=bool)
    for t in all_tags:
        mask |= (tags == t)

    embs = data["embeddings"][mask]
    labels = data["labels"][mask]
    source_idx = data["source_idx"][mask]
    tag_subset = tags[mask]

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    loss_name = "focal" if args.focal else "bce"
    suffix = f"bal{args.ratio:.0f}_{loss_name}"

    print(f"{'='*60}")
    print(f"BALANCED TRAINING: {args.experiment} ({suffix})")
    print(f"  Original: {n_pos} pos + {n_neg} neg (ratio {n_neg/n_pos:.1f}:1)")
    print(f"  Target ratio: {args.ratio}:1")
    print(f"{'='*60}")

    seeds = list(range(42, 42 + args.seeds))
    results = []

    for seed in seeds:
        # Subsample for each seed (different random selection)
        bal_embs, bal_labels, bal_sidx = subsample_negatives(
            embs, labels, tag_subset, source_idx, args.ratio, seed
        )
        model_path = EXPERIMENTS / "models" / f"{args.experiment}_{suffix}_s{seed}.onnx"
        train_info = train_balanced(
            args.experiment, bal_embs, bal_labels, bal_sidx, model_path,
            config, seed, use_bce=not args.focal
        )
        eval_result = evaluate_onnx(model_path, f"{args.experiment}_{suffix}_s{seed}")
        eval_result.update(train_info)
        results.append(eval_result)
        print(f"  seed={seed}: EER={eval_result['all_eer']:.1%}, AUC={eval_result['all_auc']:.4f}")

    eers = [r["all_eer"] for r in results]
    print(f"\n{'='*60}")
    print(f"{args.experiment} ({suffix}): EER {np.mean(eers):.1%} +/- {np.std(eers):.1%}")
    print(f"{'='*60}")

    out = EXPERIMENTS / f"exp_{args.experiment}_{suffix}.json"
    with open(out, "w") as f:
        json.dump({"experiment": args.experiment, "variant": suffix, "ratio": args.ratio,
                    "per_seed": results, "eer_mean": round(float(np.mean(eers)), 4)}, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
