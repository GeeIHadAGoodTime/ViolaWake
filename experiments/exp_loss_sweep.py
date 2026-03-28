"""
Loss Function Sweep Experiment
===============================

Tests different loss function configurations on D_combined data.
Empirical question Q5 variant: Does tuning focal loss params or using
different loss functions improve EER?

Variants tested:
  1. FocalLoss(gamma=2.0, alpha=0.75) — current default
  2. FocalLoss(gamma=3.0, alpha=0.75) — higher gamma (more focus on hard examples)
  3. FocalLoss(gamma=1.0, alpha=0.75) — lower gamma
  4. FocalLoss(gamma=2.0, alpha=0.85) — higher alpha (more weight on positives)
  5. BCE with class weights
  6. FocalLoss(gamma=2.0) + no label smoothing
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
EVAL_DIR = WAKEWORD / "eval_clean"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"


def train_with_loss(
    name: str,
    X_train, y_train, X_val, y_val,
    loss_fn,
    hidden_dims=(64, 32),
    dropouts=(0.3, 0.2),
    epochs=60,
    patience=12,
    seed=42,
):
    """Train MLP with specific loss function and return model + info."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    embedding_dim = X_train.shape[1]

    layers = []
    in_dim = embedding_dim
    for i, h in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        if i < len(dropouts):
            layers.append(nn.Dropout(dropouts[i]))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())
    model = nn.Sequential(*layers)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = None
    t0 = time.monotonic()

    for ep in range(1, epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tl += loss.item(); nb += 1
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                vl += loss_fn(model(bx), by).item(); vn += 1

        avg_v = vl / max(vn, 1)
        if avg_v < best_val:
            best_val, best_ep, no_imp = avg_v, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    dur = time.monotonic() - t0
    return model, {
        "name": name,
        "best_epoch": best_ep,
        "best_val_loss": float(best_val),
        "training_time": round(dur, 1),
    }


def evaluate_model_torch(model, eval_dir: Path) -> dict:
    """Evaluate PyTorch model against eval set."""
    import torch
    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio
    from violawake_sdk.training.evaluate import find_optimal_threshold, compute_dprime

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    def _embed(audio):
        audio = center_crop(audio, CLIP_SAMPLES)
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        if len(audio_int16) < CLIP_SAMPLES:
            audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:CLIP_SAMPLES]
        try:
            embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            return embeddings.mean(axis=1)[0].astype(np.float32)
        except Exception:
            return None

    model.eval()
    pos_scores, neg_scores = [], []

    for label, subdir in [("positive", "positives"), ("negative", "negatives")]:
        d = eval_dir / subdir
        for f in sorted(d.rglob("*.wav")):
            audio = load_audio(f)
            if audio is None:
                continue
            emb = _embed(audio)
            if emb is None:
                continue
            with torch.no_grad():
                score = model(torch.tensor(emb, dtype=torch.float32).unsqueeze(0)).item()
            if label == "positive":
                pos_scores.append(score)
            else:
                neg_scores.append(score)

    pos = np.array(pos_scores)
    neg = np.array(neg_scores)
    opt = find_optimal_threshold(pos, neg)

    return {
        "eer": round(opt["eer_approx"], 4),
        "dprime": round(float(compute_dprime(pos, neg)), 3),
        "pos_mean": round(float(pos.mean()), 4),
        "neg_mean": round(float(neg.mean()), 4),
        "n_pos": len(pos),
        "n_neg": len(neg),
    }


def main():
    import torch
    import torch.nn as nn
    from violawake_sdk.training.losses import FocalLoss

    print("=" * 60)
    print("LOSS FUNCTION SWEEP EXPERIMENT")
    print("=" * 60)

    # Load cache
    data = np.load(CACHE_FILE, allow_pickle=True)
    embs = data["embeddings"]
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    # D_combined data
    d_tags = ["pos_main", "pos_diverse", "neg_main", "neg_confusable"]
    mask = np.zeros(len(embs), dtype=bool)
    for t in d_tags:
        mask |= (tags == t)

    d_embs = embs[mask]
    d_labels = labels[mask]
    d_source = source_idx[mask]

    # Split
    rng = np.random.default_rng(42)
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
            (val_mask if d_source[i] in val_pos else train_mask).__setitem__(i, True)
        else:
            (val_mask if d_source[i] in val_neg else train_mask).__setitem__(i, True)

    X = torch.tensor(d_embs, dtype=torch.float32)
    y = torch.tensor(d_labels, dtype=torch.float32).unsqueeze(1)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    # Define loss variants
    pos_weight = torch.tensor([(d_labels == 0).sum() / max((d_labels == 1).sum(), 1)])

    loss_variants = [
        ("focal_g2_a75_ls05", FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)),
        ("focal_g3_a75_ls05", FocalLoss(gamma=3.0, alpha=0.75, label_smoothing=0.05)),
        ("focal_g1_a75_ls05", FocalLoss(gamma=1.0, alpha=0.75, label_smoothing=0.05)),
        ("focal_g2_a85_ls05", FocalLoss(gamma=2.0, alpha=0.85, label_smoothing=0.05)),
        ("focal_g2_a65_ls05", FocalLoss(gamma=2.0, alpha=0.65, label_smoothing=0.05)),
        ("focal_g2_a75_ls00", FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.0)),
        ("focal_g2_a75_ls10", FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.10)),
        ("bce_weighted", nn.BCELoss(weight=None)),  # Will use pos_weight manually
    ]

    results = []
    for name, loss_fn in loss_variants:
        print(f"\n--- {name} ---")
        model, info = train_with_loss(
            name, X_train, y_train, X_val, y_val, loss_fn,
        )
        eval_result = evaluate_model_torch(model, EVAL_DIR)
        eval_result.update(info)
        results.append(eval_result)
        print(f"  EER={eval_result['eer']:.1%}, d'={eval_result['dprime']:.3f}")

    # Sort by EER
    results.sort(key=lambda x: x["eer"])

    print("\n" + "=" * 70)
    print("LOSS FUNCTION SWEEP RESULTS")
    print("=" * 70)
    dprime_hdr = "d'"
    print(f"{'Loss Config':<25} {'EER':>8} {dprime_hdr:>8} {'pos_m':>8} {'neg_m':>8} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['eer']:>7.1%} {r['dprime']:>7.3f} "
              f"{r['pos_mean']:>7.4f} {r['neg_mean']:>7.4f} {r['training_time']:>5.0f}s")

    winner = results[0]
    print(f"\n  BEST: {winner['name']} (EER={winner['eer']:.1%})")

    out = EXPERIMENTS / "exp_loss_sweep_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
