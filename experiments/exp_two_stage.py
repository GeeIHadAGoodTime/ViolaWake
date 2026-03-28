"""
Two-Stage Wake Word Detection Experiment
=========================================

Architecture:
  Stage 1: Primary MLP (broad detector, high recall, low threshold)
  Stage 2: Verifier MLP (trained on confusable negatives only, high precision)

Both must fire for a wake detection. This specifically targets false accepts
from confusable words while maintaining recall.

Tests empirical question Q6: Does a two-stage verifier reduce FAR without FRR cost?
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
VIOLA = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA")
EXPERIMENTS = WAKEWORD / "experiments"
EVAL_DIR = WAKEWORD / "eval_clean"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"


def train_mlp(
    name: str,
    X_train: "torch.Tensor",
    y_train: "torch.Tensor",
    X_val: "torch.Tensor",
    y_val: "torch.Tensor",
    hidden_dims: list[int],
    dropouts: list[float],
    epochs: int = 60,
    patience: int = 12,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple:
    """Train a single MLP stage and return (model, train_info)."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from violawake_sdk.training.losses import FocalLoss

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

    criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
        else:
            no_imp += 1

        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    print(f"  {name}: best_ep={best_ep}, best_val={best_val:.4f}")
    return model, {"best_epoch": best_ep, "best_val_loss": float(best_val)}


def evaluate_two_stage(
    primary_model,
    verifier_model,
    eval_dir: Path,
    primary_thresholds: list[float],
    verifier_thresholds: list[float],
) -> dict:
    """Evaluate two-stage detection on eval set."""
    import torch
    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    def _embed(audio: np.ndarray) -> np.ndarray | None:
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

    # Collect embeddings + scores from both stages
    pos_dir = eval_dir / "positives"
    neg_dir = eval_dir / "negatives"

    pos_files = sorted(list(pos_dir.rglob("*.wav")))
    neg_files = sorted(list(neg_dir.rglob("*.wav")))

    print(f"  Evaluating: {len(pos_files)} pos + {len(neg_files)} neg clips...")

    primary_model.eval()
    verifier_model.eval()

    pos_primary_scores = []
    pos_verifier_scores = []
    neg_primary_scores = []
    neg_verifier_scores = []

    with torch.no_grad():
        for fpath in pos_files:
            audio = load_audio(fpath)
            if audio is None:
                continue
            emb = _embed(audio)
            if emb is None:
                continue
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            s1 = primary_model(x).item()
            s2 = verifier_model(x).item()
            pos_primary_scores.append(s1)
            pos_verifier_scores.append(s2)

        for fpath in neg_files:
            audio = load_audio(fpath)
            if audio is None:
                continue
            emb = _embed(audio)
            if emb is None:
                continue
            x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            s1 = primary_model(x).item()
            s2 = verifier_model(x).item()
            neg_primary_scores.append(s1)
            neg_verifier_scores.append(s2)

    pos_p = np.array(pos_primary_scores)
    pos_v = np.array(pos_verifier_scores)
    neg_p = np.array(neg_primary_scores)
    neg_v = np.array(neg_verifier_scores)

    # Sweep threshold combinations
    best_result = None
    best_eer = 1.0
    all_combos = []

    for pt in primary_thresholds:
        for vt in verifier_thresholds:
            # Two-stage: both must pass
            pos_detected = (pos_p >= pt) & (pos_v >= vt)
            neg_detected = (neg_p >= pt) & (neg_v >= vt)

            tp = pos_detected.sum()
            fn = len(pos_p) - tp
            fp = neg_detected.sum()
            tn = len(neg_p) - fp

            frr = fn / max(len(pos_p), 1)
            far = fp / max(len(neg_p), 1)
            eer_approx = (frr + far) / 2  # Simplified EER approximation

            combo = {
                "primary_threshold": pt,
                "verifier_threshold": vt,
                "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
                "frr": round(frr, 4),
                "far": round(far, 4),
                "eer_approx": round(eer_approx, 4),
            }
            all_combos.append(combo)

            if eer_approx < best_eer:
                best_eer = eer_approx
                best_result = combo

    # Also compute proper EER by sweeping
    # Use the combined score: min(primary_score, verifier_score)
    pos_combined = np.minimum(pos_p, pos_v)
    neg_combined = np.minimum(neg_p, neg_v)

    from violawake_sdk.training.evaluate import find_optimal_threshold, compute_dprime
    opt = find_optimal_threshold(pos_combined, neg_combined)

    return {
        "n_pos": len(pos_p),
        "n_neg": len(neg_p),
        "combined_eer": round(opt["eer_approx"], 4),
        "combined_optimal_threshold": round(opt["optimal_threshold"], 4),
        "combined_dprime": round(float(compute_dprime(pos_combined, neg_combined)), 3),
        "best_grid_result": best_result,
        "pos_primary_mean": round(float(pos_p.mean()), 4),
        "neg_primary_mean": round(float(neg_p.mean()), 4),
        "pos_verifier_mean": round(float(pos_v.mean()), 4),
        "neg_verifier_mean": round(float(neg_v.mean()), 4),
        "top5_combos": sorted(all_combos, key=lambda x: x["eer_approx"])[:5],
    }


def main():
    import torch

    print("=" * 60)
    print("TWO-STAGE WAKE WORD DETECTION EXPERIMENT")
    print("=" * 60)

    # Load cached embeddings
    print("Loading embedding cache...")
    data = np.load(CACHE_FILE, allow_pickle=True)
    embs = data["embeddings"]
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    # ── Stage 1: Primary detector (broad, trained on all D_combined data) ──
    print("\n--- Stage 1: Primary Detector ---")
    d_tags = ["pos_main", "pos_diverse", "neg_main", "neg_confusable"]
    d_mask = np.zeros(len(embs), dtype=bool)
    for t in d_tags:
        d_mask |= (tags == t)

    d_embs = embs[d_mask]
    d_labels = labels[d_mask]
    d_source = source_idx[d_mask]

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

    primary_model, primary_info = train_mlp(
        "Primary (64-32)",
        X[train_mask], y[train_mask], X[val_mask], y[val_mask],
        hidden_dims=[64, 32], dropouts=[0.3, 0.2],
    )

    # ── Stage 2: Verifier (precision-focused, trained with emphasis on confusables) ──
    print("\n--- Stage 2: Verifier (confusable-focused) ---")
    # Verifier training data: positives + confusable negatives only
    # This makes the verifier specifically good at distinguishing wake word from confusables
    v_pos_tags = ["pos_main", "pos_diverse"]
    v_neg_tags = ["neg_confusable"]  # Only confusable negatives

    v_mask = np.zeros(len(embs), dtype=bool)
    for t in v_pos_tags + v_neg_tags:
        v_mask |= (tags == t)

    v_embs = embs[v_mask]
    v_labels = labels[v_mask]
    v_source = source_idx[v_mask]

    # Split verifier data
    v_pos_mask = v_labels == 1
    v_neg_mask = v_labels == 0
    v_pos_sources = sorted(set(v_source[v_pos_mask]))
    v_neg_sources = sorted(set(v_source[v_neg_mask]))
    rng2 = np.random.default_rng(123)
    rng2.shuffle(v_pos_sources)
    rng2.shuffle(v_neg_sources)
    v_val_pos = set(v_pos_sources[:max(1, len(v_pos_sources) // 5)])
    v_val_neg = set(v_neg_sources[:max(1, len(v_neg_sources) // 5)])

    v_val_mask = np.zeros(len(v_labels), dtype=bool)
    v_train_mask = np.zeros(len(v_labels), dtype=bool)
    for i in range(len(v_labels)):
        if v_labels[i] == 1:
            (v_val_mask if v_source[i] in v_val_pos else v_train_mask).__setitem__(i, True)
        else:
            (v_val_mask if v_source[i] in v_val_neg else v_train_mask).__setitem__(i, True)

    Xv = torch.tensor(v_embs, dtype=torch.float32)
    yv = torch.tensor(v_labels, dtype=torch.float32).unsqueeze(1)

    verifier_model, verifier_info = train_mlp(
        "Verifier (32-16, confusable-only negs)",
        Xv[v_train_mask], yv[v_train_mask], Xv[v_val_mask], yv[v_val_mask],
        hidden_dims=[32, 16], dropouts=[0.2, 0.1],
        seed=123,
    )

    # ── Evaluate two-stage system ──
    print("\n--- Two-Stage Evaluation ---")
    primary_thresholds = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    verifier_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    results = evaluate_two_stage(
        primary_model, verifier_model, EVAL_DIR,
        primary_thresholds, verifier_thresholds,
    )

    # Compare with single-stage D_combined
    d_model_path = EXPERIMENTS / "models" / "D_combined.onnx"
    if d_model_path.exists():
        from violawake_sdk.training.evaluate import evaluate_onnx_model, find_optimal_threshold
        d_results = evaluate_onnx_model(str(d_model_path), str(EVAL_DIR), threshold=0.50)
        d_eer = d_results["eer_approx"]
        results["single_stage_eer"] = round(d_eer, 4)
        delta = results["combined_eer"] - d_eer
        results["vs_single_stage_delta"] = round(delta, 4)
        results["beats_single_stage"] = delta < 0

    # Print results
    print("\n" + "=" * 60)
    print("TWO-STAGE RESULTS")
    print("=" * 60)
    print(f"  Combined EER:           {results['combined_eer']:.1%}")
    print(f"  Combined d':            {results['combined_dprime']:.3f}")
    print(f"  Optimal threshold:      {results['combined_optimal_threshold']:.4f}")
    if "single_stage_eer" in results:
        print(f"  Single-stage D EER:     {results['single_stage_eer']:.1%}")
        print(f"  Delta:                  {results['vs_single_stage_delta']:+.1%}")
        print(f"  Beats single-stage:     {results['beats_single_stage']}")
    print(f"\n  Primary stage:  pos_mean={results['pos_primary_mean']:.4f}, neg_mean={results['neg_primary_mean']:.4f}")
    print(f"  Verifier stage: pos_mean={results['pos_verifier_mean']:.4f}, neg_mean={results['neg_verifier_mean']:.4f}")
    print(f"\n  Top 5 threshold combos:")
    for c in results["top5_combos"]:
        print(f"    pt={c['primary_threshold']:.2f} vt={c['verifier_threshold']:.2f} "
              f"FRR={c['frr']:.1%} FAR={c['far']:.1%} EER~={c['eer_approx']:.1%}")

    # Save
    out = EXPERIMENTS / "exp_two_stage_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
