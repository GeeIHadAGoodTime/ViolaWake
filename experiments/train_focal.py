"""
Focal Loss Training Experiment
==============================

Hypothesis: Focal loss preserves true positive detection better at high
thresholds compared to weighted BCE with 10x hard negative weight.

Current problem: weighted BCE 10x degrades detection 99.1% -> 96.0% at
threshold 0.80 on held-out positives. Focal loss focuses more on the
*hardest* examples (via (1-p_t)^gamma) rather than uniformly upweighting
all hard negatives, which may avoid degrading positive confidence.

Variants:
  focal_g2      - gamma=2, no class weighting
  focal_g2_a    - gamma=2, alpha=[1.0, 2.0] (2x positive weight)
  focal_g3      - gamma=3, more aggressive focusing
  focal_g2_hardonly - gamma=2 on hard negatives, standard BCE on regular negatives

Seeds: 42 and 43
Architecture: 64->32->1 MLP (same as all prior experiments)

Usage:
    python experiments/train_focal.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"
MODEL_DIR = EXPERIMENTS / "models"
LIBRISPEECH_DEVCLEAN = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "dev-clean"


# ── Data loading ──────────────────────────────────────────────────────────

def load_data_with_tags():
    """Load embedding cache and return data with tag info preserved."""
    data = np.load(CACHE_FILE, allow_pickle=True)
    return {
        "embeddings": data["embeddings"],
        "labels": data["labels"],
        "tags": data["tags"],
        "files": data["files"],
        "source_idx": data["source_idx"],
    }


def select_r2d_data(cache):
    """Select same data as round2 r2d variant: all positives + all negatives including hard."""
    tags = cache["tags"]
    target_tags = [
        "pos_main",
        "pos_diverse",
        "neg_main",
        "neg_confusable",
        "neg_confusable_v2",
        "neg_librispeech_hard",
        "neg_acav_hard_r2",
        "neg_devclean_hard_r2",
    ]

    mask = np.zeros(len(tags), dtype=bool)
    for t in target_tags:
        tag_mask = tags == t
        n = tag_mask.sum()
        if n > 0:
            mask |= tag_mask
            print(f"  {t}: {n}")
        else:
            print(f"  {t}: 0 (NOT FOUND)")

    sel = {k: v[mask] for k, v in cache.items()}
    n_pos = (sel["labels"] == 1).sum()
    n_neg = (sel["labels"] == 0).sum()
    print(f"\n  Total: {n_pos} pos + {n_neg} neg = {len(sel['labels'])}")
    return sel


def get_hard_neg_mask(tags):
    """Return boolean mask for hard negative tags."""
    hard = np.zeros(len(tags), dtype=bool)
    for t in np.unique(tags):
        if "hard" in str(t).lower():
            hard |= (tags == t)
    return hard


# ── Focal Loss ────────────────────────────────────────────────────────────

def train_focal(
    name,
    embeddings,
    labels,
    source_idx,
    tags,
    output_model,
    gamma=2.0,
    alpha=None,            # None or [neg_weight, pos_weight]
    hard_only_focal=False, # If True, focal only on hard negs, BCE on rest
    hidden_dims=(64, 32),
    dropouts=(0.3, 0.2),
    seed=42,
    epochs=80,
    patience=15,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    ema_decay=0.999,
):
    """Train MLP with focal loss."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embedding_dim = X.shape[1]

    # Build per-sample flags for hard_only_focal mode
    hard_mask = get_hard_neg_mask(tags)
    is_hard = torch.tensor(hard_mask, dtype=torch.float32).unsqueeze(1)

    # Group-aware 80/20 split
    rng = np.random.default_rng(seed)
    pos_mask = labels == 1
    neg_mask_np = labels == 0
    pos_sources = sorted(set(source_idx[pos_mask]))
    neg_sources = sorted(set(source_idx[neg_mask_np]))
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

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    h_train, h_val = is_hard[train_mask], is_hard[val_mask]

    print(
        f"  {name} s{seed}: {train_mask.sum()} train / {val_mask.sum()} val "
        f"({n_pos} pos + {n_neg} neg)"
    )

    # Build model
    layers = []
    in_dim = embedding_dim
    for i, h in enumerate(hidden_dims):
        layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
        if i < len(dropouts):
            layers.append(nn.Dropout(dropouts[i]))
        in_dim = h
    layers.extend([nn.Linear(in_dim, 1), nn.Sigmoid()])
    model = nn.Sequential(*layers)

    # Alpha tensor
    alpha_t = None
    if alpha is not None:
        alpha_t = torch.tensor(alpha, dtype=torch.float32)  # [neg_weight, pos_weight]

    def focal_loss(pred, target, is_hard_flag=None):
        """Focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)"""
        eps = 1e-7
        pred = pred.clamp(eps, 1.0 - eps)
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        focal_weight = (1.0 - p_t) ** gamma

        if alpha_t is not None:
            # alpha_t[1] for positives, alpha_t[0] for negatives
            aw = alpha_t[1] * target + alpha_t[0] * (1.0 - target)
            focal_weight = aw * focal_weight

        if hard_only_focal and is_hard_flag is not None:
            # For hard negatives: use focal loss
            # For everything else: use standard BCE (focal_weight=1)
            focal_weight = torch.where(is_hard_flag > 0.5, focal_weight, torch.ones_like(focal_weight))

        return (focal_weight * bce).mean()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    train_ds = TensorDataset(X_train, y_train, h_train)
    val_ds = TensorDataset(X_val, y_val, h_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = best_ema = None
    t0 = time.monotonic()

    for ep in range(1, epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for bx, by, bh in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = focal_loss(pred, by, bh)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for n, p in model.named_parameters():
                ema[n] = ema_decay * ema[n] + (1 - ema_decay) * p.data
            tl += loss.item()
            nb += 1
        scheduler.step()

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for bx, by, bh in val_loader:
                pred = model(bx)
                vl += focal_loss(pred, by, bh).item()
                vn += 1

        avg_v = vl / max(vn, 1)
        if avg_v < best_val:
            best_val, best_ep, no_imp = avg_v, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema = {k: v.clone() for k, v in ema.items()}
        else:
            no_imp += 1

        if ep % 10 == 0 or no_imp >= patience:
            print(f"    Epoch {ep}: val_loss={avg_v:.6f} (best={best_val:.6f} @ep{best_ep})")

        if no_imp >= patience:
            print(f"    Early stop at epoch {ep}")
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
        model,
        _torch.zeros(1, embedding_dim),
        str(output_model),
        input_names=["embedding"],
        output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    print(f"    Exported: {output_model}")
    return {"best_epoch": best_ep, "best_val_loss": float(best_val), "time": round(dur, 1)}


# ── Held-out evaluation ──────────────────────────────────────────────────

def cosine_sim_matrix(A, B):
    """Max cosine similarity between each row of A and all rows of B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    max_sims = np.zeros(A.shape[0], dtype=np.float32)
    chunk = 500
    for i in range(0, A.shape[0], chunk):
        sims = A_norm[i:i+chunk] @ B_norm.T
        max_sims[i:i+chunk] = sims.max(axis=1)
    return max_sims


def score_with_model(model_path, embeddings):
    """Run embeddings through an ONNX model, return scores."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    scores = []
    batch = 1024
    for i in range(0, len(embeddings), batch):
        out = sess.run([output_name], {input_name: embeddings[i:i+batch].astype(np.float32)})[0]
        scores.append(out.flatten())
    return np.concatenate(scores)


def eval_heldout(model_path, model_name, clean_embs, n_clean):
    """Evaluate detection rates at various thresholds on held-out positives."""
    scores = score_with_model(model_path, clean_embs)
    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    results = {}
    print(f"\n  {model_name}:")
    print(f"    Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
          f"mean={scores.mean():.4f}, median={np.median(scores):.4f}")
    for t in thresholds:
        detected = (scores >= t).sum()
        rate = detected / n_clean * 100
        results[t] = {"detected": int(detected), "total": n_clean, "rate": round(rate, 2)}
        print(f"    Threshold {t:.2f}: {detected}/{n_clean} ({rate:.1f}%)")
    return results, scores


# ── Quick FAPH estimate ──────────────────────────────────────────────────

def quick_faph_estimate(model_path, model_name, n_files=500):
    """Run through first n_files of dev-clean to estimate FAPH."""
    import onnxruntime as ort
    import soundfile as sf
    from openwakeword.utils import AudioFeatures

    SAMPLE_RATE = 16000
    CLIP_SAMPLES = 24000   # 1.5s
    STEP_SAMPLES = 1600    # 100ms

    print(f"\n  Quick FAPH estimate: {model_name} ({n_files} files)")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    preprocessor = AudioFeatures()

    flac_files = sorted(LIBRISPEECH_DEVCLEAN.rglob("*.flac"))[:n_files]
    total_audio_sec = 0.0
    thresholds = [0.80, 0.85, 0.90, 0.95]
    triggers = {t: 0 for t in thresholds}

    t0 = time.time()
    for fi, fpath in enumerate(flac_files):
        try:
            audio, sr = sf.read(fpath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except Exception:
            continue

        total_audio_sec += len(audio) / SAMPLE_RATE
        pos = 0
        # Debounce: track last trigger time per threshold
        last_trigger = {t: -999.0 for t in thresholds}
        debounce_sec = 2.0

        while pos + CLIP_SAMPLES <= len(audio):
            window = audio[pos:pos + CLIP_SAMPLES]
            window_int16 = (window * 32767).clip(-32768, 32767).astype(np.int16)
            embs = preprocessor.embed_clips(window_int16.reshape(1, -1), ncpu=1)
            emb = embs.mean(axis=1)[0].astype(np.float32)
            score = float(session.run(None, {input_name: emb.reshape(1, -1)})[0][0][0])

            time_sec = pos / SAMPLE_RATE
            for t in thresholds:
                if score >= t and (time_sec - last_trigger[t]) > debounce_sec:
                    triggers[t] += 1
                    last_trigger[t] = time_sec

            pos += STEP_SAMPLES

        if (fi + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"    [{fi+1}/{n_files}] {total_audio_sec/3600:.2f}h processed ({elapsed:.0f}s)")

    total_hours = total_audio_sec / 3600
    elapsed = time.time() - t0
    print(f"    Done: {total_hours:.2f}h of audio in {elapsed:.0f}s")

    faph_results = {}
    for t in thresholds:
        faph = triggers[t] / total_hours if total_hours > 0 else 0
        faph_results[t] = {"triggers": triggers[t], "hours": round(total_hours, 3), "faph": round(faph, 2)}
        print(f"    Threshold {t:.2f}: {triggers[t]} triggers / {total_hours:.2f}h = {faph:.2f} FAPH")

    return faph_results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Focal loss training experiment")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just evaluate")
    parser.add_argument("--skip-faph", action="store_true", help="Skip FAPH estimation")
    parser.add_argument("--faph-files", type=int, default=500, help="Number of dev-clean files for FAPH")
    args = parser.parse_args()

    print("="*70)
    print("FOCAL LOSS TRAINING EXPERIMENT")
    print("="*70)

    # Load data
    print("\nLoading embedding cache...")
    cache = load_data_with_tags()
    sel = select_r2d_data(cache)

    seeds = [42, 43]
    train_results = {}

    # ── Variant definitions ───────────────────────────────────────────
    variants = {
        "focal_g2": {
            "desc": "Focal loss, gamma=2, no class weighting",
            "gamma": 2.0,
            "alpha": None,
            "hard_only_focal": False,
        },
        "focal_g2_a": {
            "desc": "Focal loss, gamma=2, alpha=[1.0, 2.0] (2x positive weight)",
            "gamma": 2.0,
            "alpha": [1.0, 2.0],
            "hard_only_focal": False,
        },
        "focal_g3": {
            "desc": "Focal loss, gamma=3 (more aggressive focusing)",
            "gamma": 3.0,
            "alpha": None,
            "hard_only_focal": False,
        },
        "focal_g2_hardonly": {
            "desc": "Focal gamma=2 on hard negatives only, standard BCE on rest",
            "gamma": 2.0,
            "alpha": None,
            "hard_only_focal": True,
        },
    }

    if not args.skip_train:
        for vname, vconf in variants.items():
            print(f"\n{'='*70}")
            print(f"VARIANT: {vname}")
            print(f"  {vconf['desc']}")
            print(f"{'='*70}")

            for seed in seeds:
                model_name = f"{vname}_s{seed}"
                model_path = MODEL_DIR / f"{model_name}.onnx"
                print(f"\n  Training {model_name}...")
                info = train_focal(
                    model_name,
                    sel["embeddings"],
                    sel["labels"],
                    sel["source_idx"],
                    sel["tags"],
                    model_path,
                    gamma=vconf["gamma"],
                    alpha=vconf["alpha"],
                    hard_only_focal=vconf["hard_only_focal"],
                    seed=seed,
                )
                train_results[model_name] = info
                print(f"  {model_name}: val_loss={info['best_val_loss']:.6f}, "
                      f"ep={info['best_epoch']}, {info['time']}s")

    # ── Held-out evaluation ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print("HELD-OUT POSITIVE EVALUATION (pos_backup)")
    print(f"{'='*70}")

    # Get clean held-out positives
    all_tags = cache["tags"]
    all_embs = cache["embeddings"]

    backup_mask = all_tags == "pos_backup"
    backup_embs = all_embs[backup_mask]
    n_backup = backup_mask.sum()
    print(f"\n  pos_backup: {n_backup} embeddings")

    # Deduplicate against training positives
    train_pos_mask = np.isin(all_tags, ["pos_main", "pos_diverse"])
    train_pos_embs = all_embs[train_pos_mask]
    print(f"  Training positives: {train_pos_embs.shape[0]}")

    print("  Checking contamination...")
    max_sims = cosine_sim_matrix(backup_embs, train_pos_embs)
    contaminated = max_sims > 0.99
    n_contaminated = contaminated.sum()
    clean_embs = backup_embs[~contaminated]
    n_clean = len(clean_embs)
    print(f"  Contaminated: {n_contaminated}, Clean: {n_clean}")

    if n_clean == 0:
        print("  ERROR: No clean held-out positives!")
        sys.exit(1)

    # Models to evaluate
    eval_models = {}

    # Baselines
    baseline_models = {
        "D_combined_bce_s42 (baseline)": MODEL_DIR / "D_combined_bce_s42.onnx",
        "faph_hardened_s43 (R1)": MODEL_DIR / "faph_hardened_s43.onnx",
        "round2_s42 (R2 weighted BCE)": MODEL_DIR / "round2_s42.onnx",
        "round2_s43 (R2 weighted BCE)": MODEL_DIR / "round2_s43.onnx",
    }
    for name, path in baseline_models.items():
        if path.exists():
            eval_models[name] = path

    # Focal models
    for vname in variants:
        for seed in seeds:
            model_name = f"{vname}_s{seed}"
            model_path = MODEL_DIR / f"{model_name}.onnx"
            if model_path.exists():
                eval_models[model_name] = model_path

    all_eval_results = {}
    for model_name, model_path in eval_models.items():
        results, scores = eval_heldout(model_path, model_name, clean_embs, n_clean)
        all_eval_results[model_name] = results

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("DETECTION RATE COMPARISON TABLE")
    print(f"{'='*70}")
    thresholds = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
    header = f"{'Model':<40} " + " ".join(f"@{t:.2f}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for model_name, results in all_eval_results.items():
        rates = " ".join(f"{results[t]['rate']:5.1f}%" for t in thresholds)
        print(f"{model_name:<40} {rates}")

    # ── Quick FAPH estimate ───────────────────────────────────────────
    if not args.skip_faph:
        print(f"\n{'='*70}")
        print(f"QUICK FAPH ESTIMATE ({args.faph_files} dev-clean files)")
        print(f"{'='*70}")

        # Find best focal model: highest detection at 0.85
        best_focal_name = None
        best_focal_rate = -1
        for model_name, results in all_eval_results.items():
            if model_name.startswith("focal_"):
                rate_85 = results[0.85]["rate"]
                if rate_85 > best_focal_rate:
                    best_focal_rate = rate_85
                    best_focal_name = model_name

        faph_models = {}
        # Always test round2_s42 for comparison
        if (MODEL_DIR / "round2_s42.onnx").exists():
            faph_models["round2_s42 (weighted BCE)"] = MODEL_DIR / "round2_s42.onnx"
        # Best focal
        if best_focal_name:
            faph_models[f"{best_focal_name} (best focal)"] = eval_models[best_focal_name]
            print(f"  Best focal model: {best_focal_name} ({best_focal_rate:.1f}% at 0.85)")

        # Also test the focal_g2_a models (likely best TP preservation)
        for seed in seeds:
            name = f"focal_g2_a_s{seed}"
            if name in eval_models and name != best_focal_name:
                faph_models[name] = eval_models[name]

        all_faph_results = {}
        for model_name, model_path in faph_models.items():
            faph_results = quick_faph_estimate(model_path, model_name, n_files=args.faph_files)
            all_faph_results[model_name] = faph_results

        # FAPH comparison table
        print(f"\n{'='*70}")
        print("FAPH COMPARISON TABLE")
        print(f"{'='*70}")
        faph_thresholds = [0.80, 0.85, 0.90, 0.95]
        header = f"{'Model':<40} " + " ".join(f"@{t:.2f}" for t in faph_thresholds)
        print(header)
        print("-" * len(header))
        for model_name, results in all_faph_results.items():
            faphs = " ".join(f"{results[t]['faph']:6.2f}" for t in faph_thresholds)
            print(f"{model_name:<40} {faphs}")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "train_results": train_results,
        "heldout_eval": {k: {str(t): v for t, v in results.items()} for k, results in all_eval_results.items()},
        "n_clean_heldout": n_clean,
    }
    if not args.skip_faph:
        output["faph_estimate"] = {k: {str(t): v for t, v in results.items()} for k, results in all_faph_results.items()}

    out_path = EXPERIMENTS / "focal_loss_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\nKey question: Does focal loss give >95% detection at threshold 0.85 or 0.90?")
    for model_name, results in all_eval_results.items():
        r85 = results[0.85]["rate"]
        r90 = results[0.9]["rate"]
        marker = " <-- WINNER" if r85 > 95 and model_name.startswith("focal_") else ""
        print(f"  {model_name}: @0.85={r85:.1f}%, @0.90={r90:.1f}%{marker}")

    # Find best model
    best_name = None
    best_score = -1
    for model_name, results in all_eval_results.items():
        if not model_name.startswith("focal_"):
            continue
        # Score: prioritize high detection at 0.85, then 0.90
        score = results[0.85]["rate"] * 2 + results[0.9]["rate"]
        if score > best_score:
            best_score = score
            best_name = model_name

    if best_name:
        print(f"\n  BEST FOCAL MODEL: {best_name}")
        r = all_eval_results[best_name]
        print(f"    Detection: @0.80={r[0.8]['rate']:.1f}%, @0.85={r[0.85]['rate']:.1f}%, "
              f"@0.90={r[0.9]['rate']:.1f}%, @0.95={r[0.95]['rate']:.1f}%")


if __name__ == "__main__":
    main()
