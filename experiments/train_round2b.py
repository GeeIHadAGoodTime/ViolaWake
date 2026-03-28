"""
Round 2b: Ablation variants to find best FAPH/TP tradeoff.

Variants:
  A) 10x weight, only top-500 ACAV hard negs (score >0.7)
  B) 5x weight on all hard negatives
  C) 10x weight, only top-1000 ACAV hard negs (score >0.6)
  D) 10x weight, all hard negs, but positives also get 2x weight boost

Seeds 42 and 43 only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"
ACAV_SCORES_FILE = EXPERIMENTS / "acav_hard_neg_scores.npz"
MODEL_DIR = EXPERIMENTS / "models"


def load_data_with_tags():
    data = np.load(CACHE_FILE, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "tags": data["tags"],
        "files": data["files"],
        "source_idx": data["source_idx"],
    }


def train_weighted_bce(
    name, embeddings, labels, source_idx, sample_weights, output_model,
    hidden_dims=(64, 32), dropouts=(0.3, 0.2), seed=42, epochs=80,
    patience=15, batch_size=32, lr=1e-3, weight_decay=1e-4, ema_decay=0.999,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    w = torch.tensor(sample_weights, dtype=torch.float32).unsqueeze(1)
    embedding_dim = X.shape[1]

    rng = np.random.default_rng(seed)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sources = sorted(set(source_idx[pos_mask]))
    neg_sources = sorted(set(source_idx[neg_mask]))
    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)
    val_pos_set = set(pos_sources[: max(1, len(pos_sources) // 5)])
    val_neg_set = set(neg_sources[: max(1, len(neg_sources) // 5)])

    val_mask = np.zeros(len(labels), dtype=bool)
    train_mask = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 1:
            val_mask[i] = source_idx[i] in val_pos_set
        else:
            val_mask[i] = source_idx[i] in val_neg_set
        train_mask[i] = not val_mask[i]

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]
    print(f"  {name}: {train_mask.sum()} train / {val_mask.sum()} val ({n_pos} pos + {n_neg} neg)")

    layers = []
    in_dim = embedding_dim
    for i, h in enumerate(hidden_dims):
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        if i < len(dropouts):
            layers.append(nn.Dropout(dropouts[i]))
        in_dim = h
    layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
    model = nn.Sequential(*layers)

    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    train_ds = TensorDataset(X_train, y_train, w_train)
    val_ds = TensorDataset(X_val, y_val, w_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = best_ema = None
    t0 = time.monotonic()

    for ep in range(1, epochs + 1):
        model.train()
        for bx, by, bw in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = (criterion(pred, by) * bw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for n, p in model.named_parameters():
                ema[n] = ema_decay * ema[n] + (1 - ema_decay) * p.data
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for bx, by, bw in val_loader:
                pred = model(bx)
                vl += (criterion(pred, by) * bw).mean().item()
                vn += 1

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


def main():
    cache = load_data_with_tags()
    embs = cache["embeddings"]
    labels = cache["labels"]
    tags = cache["tags"]
    source_idx = cache["source_idx"]

    # Load ACAV scores for filtering
    acav_data = np.load(ACAV_SCORES_FILE, allow_pickle=True)
    hard_indices = acav_data["hard_indices"]  # absolute cache indices
    hard_scores = acav_data["hard_scores"]

    # Base tags (always included)
    base_tags = ["pos_main", "pos_diverse", "neg_main", "neg_confusable", "neg_confusable_v2",
                 "neg_librispeech_hard", "neg_devclean_hard_r2"]

    base_mask = np.zeros(len(tags), dtype=bool)
    for t in base_tags:
        base_mask |= (tags == t)

    # Variant definitions
    variants = {
        "r2a_top500_10x": {
            "desc": "Top-500 ACAV (>0.7) + all other hard negs, 10x weight",
            "acav_score_thresh": 0.7,
            "hard_weight": 10.0,
            "pos_weight": 1.0,
        },
        "r2b_all_5x": {
            "desc": "All 3978 ACAV hard negs, 5x weight",
            "acav_score_thresh": 0.3,
            "hard_weight": 5.0,
            "pos_weight": 1.0,
        },
        "r2c_top1000_10x": {
            "desc": "Top-1000 ACAV (>0.6) + all other hard negs, 10x weight",
            "acav_score_thresh": 0.6,
            "hard_weight": 10.0,
            "pos_weight": 1.0,
        },
        "r2d_all_10x_pos2x": {
            "desc": "All 3978 ACAV at 10x, positives at 2x (TP protection)",
            "acav_score_thresh": 0.3,
            "hard_weight": 10.0,
            "pos_weight": 2.0,
        },
    }

    seeds = [42, 43]

    for vname, vconf in variants.items():
        print(f"\n{'='*70}")
        print(f"VARIANT: {vname}")
        print(f"  {vconf['desc']}")
        print(f"{'='*70}")

        # Filter ACAV hard negs by score threshold
        acav_thresh = vconf["acav_score_thresh"]
        acav_selected = hard_indices[hard_scores >= acav_thresh]
        print(f"  ACAV hard negs (score >= {acav_thresh}): {len(acav_selected)}")

        # Build selection mask: base + selected ACAV hard negs
        # The neg_acav_hard_r2 tag has ALL 3978 in the cache
        # We need to include only the ones above our threshold
        sel_mask = base_mask.copy()

        # Add the specific ACAV hard neg cache indices
        acav_include = np.zeros(len(tags), dtype=bool)
        acav_include[acav_selected] = True
        sel_mask |= acav_include

        sel_embs = embs[sel_mask]
        sel_labels = labels[sel_mask]
        sel_tags = tags[sel_mask]
        sel_source_idx = source_idx[sel_mask]

        n_pos = (sel_labels == 1).sum()
        n_neg = (sel_labels == 0).sum()
        print(f"  Data: {n_pos} pos + {n_neg} neg = {len(sel_labels)}")

        # Build weights
        weights = np.ones(len(sel_labels), dtype=np.float32)

        # Confusable at 5x
        conf_mask = (sel_tags == "neg_confusable") | (sel_tags == "neg_confusable_v2")
        weights[conf_mask & (sel_labels == 0)] = 5.0

        # Hard negs at specified weight
        hard_mask = np.zeros(len(sel_tags), dtype=bool)
        for t in np.unique(sel_tags):
            if "hard" in str(t).lower():
                hard_mask |= (sel_tags == t)
        # Also mark the ACAV entries selected by index (they have tag neg_acav100m)
        # We need to track which entries in sel_ came from acav_include
        sel_indices = np.where(sel_mask)[0]
        acav_in_sel = np.isin(sel_indices, acav_selected)
        hard_mask |= acav_in_sel

        weights[hard_mask & (sel_labels == 0)] = vconf["hard_weight"]

        # Positive weight boost
        if vconf["pos_weight"] > 1.0:
            weights[sel_labels == 1] = vconf["pos_weight"]

        n_hard = (hard_mask & (sel_labels == 0)).sum()
        n_conf = (conf_mask & (sel_labels == 0)).sum()
        print(f"  Weights: {n_hard} hard at {vconf['hard_weight']}x, {n_conf} confusable at 5x, pos at {vconf['pos_weight']}x")

        for seed in seeds:
            model_name = f"{vname}_s{seed}"
            model_path = MODEL_DIR / f"{model_name}.onnx"
            info = train_weighted_bce(
                model_name, sel_embs, sel_labels, sel_source_idx, weights, model_path, seed=seed,
            )
            print(f"  {model_name}: val_loss={info['best_val_loss']:.6f}, ep={info['best_epoch']}, {info['time']}s")

    print(f"\n{'='*70}")
    print("ALL VARIANTS TRAINED. Run TP eval next.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
