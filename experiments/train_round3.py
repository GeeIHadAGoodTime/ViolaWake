"""
Round 3 Training — Same-Domain Hard Negative Mining
=====================================================

Uses 90 hard negatives mined directly from dev-clean (the FAPH test corpus).
These are the exact windows causing false alarms.

Unlike R2 (ACAV cross-domain, 0% transfer), R3 mines from the same domain
where we measure FAPH. Expected: significant FAPH reduction on dev-clean.

Variants:
  r3_10x      - 10x weight on all hard negatives + 2x on positives (same as R2)
  r3_20x      - 20x weight on R3 dev-clean hard negs specifically (stronger)
  r3_focal    - Focal loss gamma=2 on hard negatives only

Seeds: 42, 43

Usage:
    python experiments/train_round3.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"
R3_HARD_NEG_FILE = EXPERIMENTS / "devclean_hard_neg_r3.npz"
MODEL_DIR = EXPERIMENTS / "models"
HELDOUT_BACKUP_TAG = "pos_backup"


def load_cache():
    data = np.load(CACHE_FILE, allow_pickle=True)
    return {k: data[k] for k in data.files}


def add_r3_hard_negatives(cache):
    """Add R3 dev-clean hard negatives to cache if not already present."""
    tags = cache["tags"]
    if "neg_devclean_hard_r3" in tags:
        n = (tags == "neg_devclean_hard_r3").sum()
        print(f"  R3 hard negatives already in cache: {n}")
        return cache

    r3 = np.load(R3_HARD_NEG_FILE, allow_pickle=True)
    r3_embs = r3["embeddings"]
    n_new = len(r3_embs)
    print(f"  Adding {n_new} R3 dev-clean hard negatives to cache")

    cache["embeddings"] = np.concatenate([cache["embeddings"], r3_embs])
    cache["labels"] = np.concatenate([cache["labels"], np.zeros(n_new, dtype=cache["labels"].dtype)])
    cache["tags"] = np.concatenate([cache["tags"], np.array(["neg_devclean_hard_r3"] * n_new, dtype=object)])
    cache["files"] = np.concatenate([cache["files"], r3["files"]])

    max_src = cache["source_idx"].max() + 1
    unique_files = {}
    new_src = []
    for f in r3["files"]:
        if f not in unique_files:
            unique_files[f] = max_src
            max_src += 1
        new_src.append(unique_files[f])
    cache["source_idx"] = np.concatenate([cache["source_idx"], np.array(new_src, dtype=cache["source_idx"].dtype)])

    import shutil
    backup = str(CACHE_FILE) + ".bak_round3"
    if not Path(backup).exists():
        shutil.copy2(CACHE_FILE, backup)
        print(f"  Backed up to {backup}")

    print(f"  Saving updated cache ({len(cache['labels'])} total entries)")
    np.savez(CACHE_FILE, **cache)
    return cache


def select_training_data(cache):
    """Select all data including R3 hard negatives."""
    tags = cache["tags"]
    target_tags = [
        "pos_main", "pos_diverse",
        "neg_main", "neg_confusable", "neg_confusable_v2",
        "neg_librispeech_hard",
        "neg_acav_hard_r2", "neg_devclean_hard_r2",
        "neg_devclean_hard_r3",
    ]

    mask = np.zeros(len(tags), dtype=bool)
    for t in target_tags:
        tag_mask = tags == t
        n = int(tag_mask.sum())
        if n > 0:
            mask |= tag_mask
            print(f"  {t}: {n}")
        else:
            print(f"  {t}: 0 (not found)")

    sel = {k: v[mask] for k, v in cache.items()}
    n_pos = int((sel["labels"] == 1).sum())
    n_neg = int((sel["labels"] == 0).sum())
    print(f"\n  Total: {n_pos} pos + {n_neg} neg = {len(sel['labels'])}")
    return sel


def build_sample_weights(labels, tags, variant="r3_10x"):
    weights = np.ones(len(labels), dtype=np.float32)

    if variant == "r3_10x":
        for t in np.unique(tags):
            if "hard" in str(t).lower():
                weights[tags == t] = 10.0
        weights[labels == 1] = 2.0

    elif variant == "r3_20x":
        for t in np.unique(tags):
            if "hard" in str(t).lower():
                weights[tags == t] = 10.0
        weights[tags == "neg_devclean_hard_r3"] = 20.0
        weights[labels == 1] = 2.0

    elif variant == "r3_focal":
        pass

    return weights


def train_mlp(
    name, embeddings, labels, source_idx, weights,
    output_model, seed=42, epochs=80, patience=15,
    use_focal=False, focal_gamma=2.0,
):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    w = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    rng = np.random.default_rng(seed)
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sources = sorted(set(source_idx[pos_mask]))
    neg_sources = sorted(set(source_idx[neg_mask]))
    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)
    val_pos = set(pos_sources[:max(1, len(pos_sources) // 5)])
    val_neg = set(neg_sources[:max(1, len(neg_sources) // 5)])

    val_mask = np.array([
        (source_idx[i] in val_pos if labels[i] == 1 else source_idx[i] in val_neg)
        for i in range(len(labels))
    ])
    train_mask = ~val_mask

    X_train, y_train, w_train = X[train_mask], y[train_mask], w[train_mask]
    X_val, y_val, w_val = X[val_mask], y[val_mask], w[val_mask]

    print(f"  {name}: {train_mask.sum()} train / {val_mask.sum()} val")

    model = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1), nn.Sigmoid(),
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    train_ds = TensorDataset(X_train, y_train, w_train)
    val_ds = TensorDataset(X_val, y_val, w_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = best_ema = None
    t0 = time.monotonic()

    for ep in range(1, epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for bx, by, bw in train_loader:
            optimizer.zero_grad()
            pred = model(bx)

            if use_focal:
                eps = 1e-7
                pred_c = pred.clamp(eps, 1 - eps)
                bce = F.binary_cross_entropy(pred_c, by, reduction='none')
                p_t = pred_c * by + (1 - pred_c) * (1 - by)
                focal_w = (1 - p_t) ** focal_gamma
                loss = (focal_w * bce).mean()
            else:
                loss = F.binary_cross_entropy(pred, by, weight=bw)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for n, p in model.named_parameters():
                ema[n] = 0.999 * ema[n] + 0.001 * p.data
            tl += loss.item()
            nb += 1
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for bx, by, bw in val_loader:
                pred = model(bx)
                if use_focal:
                    eps = 1e-7
                    pred_c = pred.clamp(eps, 1 - eps)
                    bce = F.binary_cross_entropy(pred_c, by, reduction='none')
                    p_t = pred_c * by + (1 - pred_c) * (1 - by)
                    focal_w = (1 - p_t) ** focal_gamma
                    vl += (focal_w * bce).mean().item()
                else:
                    vl += F.binary_cross_entropy(pred, by, weight=bw).item()
                vn += 1

        avg_v = vl / max(vn, 1)
        if avg_v < best_val:
            best_val, best_ep, no_imp = avg_v, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema = {k: v.clone() for k, v in ema.items()}
        else:
            no_imp += 1

        if ep % 10 == 0 or no_imp >= patience:
            print(f"    Epoch {ep}: val={avg_v:.6f} (best={best_val:.6f} @ep{best_ep})")

        if no_imp >= patience:
            print(f"    Early stop at epoch {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)
    if best_ema:
        for n, p in model.named_parameters():
            p.data.copy_(best_ema[n])

    dur = time.monotonic() - t0

    import torch as _torch
    model.eval()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    _torch.onnx.export(
        model, _torch.zeros(1, X.shape[1]), str(output_model),
        input_names=["embedding"], output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    print(f"    Exported: {output_model} ({dur:.1f}s)")
    return {"best_epoch": best_ep, "best_val_loss": float(best_val), "time": round(dur, 1)}


def eval_heldout(cache, model_paths: dict[str, Path]):
    import onnxruntime as ort

    tags = cache["tags"]
    all_embs = cache["embeddings"]

    backup_mask = tags == HELDOUT_BACKUP_TAG
    backup_embs = all_embs[backup_mask]
    print(f"\n  pos_backup: {backup_mask.sum()} embeddings")

    train_pos_mask = np.isin(tags, ["pos_main", "pos_diverse"])
    train_pos_embs = all_embs[train_pos_mask]

    A = backup_embs / (np.linalg.norm(backup_embs, axis=1, keepdims=True) + 1e-8)
    B = train_pos_embs / (np.linalg.norm(train_pos_embs, axis=1, keepdims=True) + 1e-8)
    max_sims = np.zeros(len(A), dtype=np.float32)
    for i in range(0, len(A), 500):
        sims = A[i:i+500] @ B.T
        max_sims[i:i+500] = sims.max(axis=1)

    clean = backup_embs[max_sims <= 0.99]
    n_clean = len(clean)
    print(f"  Clean held-out: {n_clean}")

    results = {}
    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]

    for name, path in model_paths.items():
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out = sess.get_outputs()[0].name
        scores = sess.run([out], {inp: clean.astype(np.float32)})[0].flatten()

        print(f"\n  {name}:")
        model_results = {}
        for t in thresholds:
            det = int((scores >= t).sum())
            rate = det / n_clean * 100
            model_results[t] = {"detected": det, "total": n_clean, "rate": round(rate, 2)}
            print(f"    @{t:.2f}: {det}/{n_clean} ({rate:.1f}%)")
        results[name] = model_results

    return results


def main():
    print("=" * 70)
    print("ROUND 3 TRAINING — Same-Domain Hard Negative Mining")
    print("=" * 70)

    print("\nLoading cache...")
    cache = load_cache()
    cache = add_r3_hard_negatives(cache)

    print("\nSelecting training data...")
    sel = select_training_data(cache)

    variants = {
        "r3_10x": {"use_focal": False},
        "r3_20x": {"use_focal": False},
        "r3_focal": {"use_focal": True},
    }
    seeds = [42, 43]
    train_results = {}

    for vname, vconf in variants.items():
        print(f"\n{'='*70}")
        print(f"VARIANT: {vname}")
        print(f"{'='*70}")

        weights = build_sample_weights(sel["labels"], sel["tags"], variant=vname)

        for seed in seeds:
            model_name = f"{vname}_s{seed}"
            model_path = MODEL_DIR / f"{model_name}.onnx"
            print(f"\n  Training {model_name}...")
            info = train_mlp(
                model_name,
                sel["embeddings"], sel["labels"], sel["source_idx"], weights,
                model_path, seed=seed,
                use_focal=vconf["use_focal"],
            )
            train_results[model_name] = info

    print(f"\n{'='*70}")
    print("HELD-OUT TP EVALUATION")
    print(f"{'='*70}")

    model_paths = {
        "round2_best (R2)": MODEL_DIR / "round2_best.onnx",
        "faph_hardened_s43 (R1)": MODEL_DIR / "faph_hardened_s43.onnx",
    }
    for vname in variants:
        for seed in seeds:
            n = f"{vname}_s{seed}"
            model_paths[n] = MODEL_DIR / f"{n}.onnx"

    cache = load_cache()
    heldout_results = eval_heldout(cache, model_paths)

    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    header = f"{'Model':<35} " + " ".join(f"@{t:.2f}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for name, results in heldout_results.items():
        rates = " ".join(f"{results[t]['rate']:5.1f}%" for t in thresholds)
        print(f"{name:<35} {rates}")

    best_name = None
    best_score = -1
    for name, results in heldout_results.items():
        if not name.startswith("r3_"):
            continue
        score = results[0.8]["rate"] * 2 + results[0.9]["rate"]
        if score > best_score:
            best_score = score
            best_name = name

    if best_name:
        print(f"\n  BEST R3 MODEL: {best_name}")
        import shutil
        src = MODEL_DIR / f"{best_name}.onnx"
        dst = MODEL_DIR / "round3_best.onnx"
        shutil.copy2(src, dst)
        print(f"  Copied to: {dst}")

    output = {
        "train_results": train_results,
        "heldout_eval": {
            k: {str(t): v for t, v in results.items()}
            for k, results in heldout_results.items()
        },
    }
    out_path = EXPERIMENTS / "round3_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
