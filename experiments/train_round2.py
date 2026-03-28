"""
Round 2 Hard Negative Mining Training
======================================

Pipeline:
  Step 1: Mine dev-clean hard negatives from audio (scan files from FAPH results)
  Step 2: Add ACAV hard negatives (score >0.3) to embedding cache
  Step 3: Add dev-clean hard negatives to embedding cache
  Step 4: Retrain with 10x weight on ALL hard negative tags
  Step 5: TP eval (run separately via hardened_tp_eval.py)

Hard negative tags getting 10x weight:
  - neg_librispeech_hard  (629, round 1)
  - neg_acav_hard_r2      (3,978, ACAV score >0.3)
  - neg_devclean_hard_r2  (deduped from dev-clean FAPH scan)

Seeds: 42 and 43 only (s44 fails TP safety).

Usage:
    python experiments/train_round2.py
    python experiments/train_round2.py --skip-mining   # skip steps 1-3, just retrain
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
ACAV_SCORES_FILE = EXPERIMENTS / "acav_hard_neg_scores.npz"
DEVCLEAN_FAPH_FILE = EXPERIMENTS / "faph_devclean_s43_debounced.json"
LIBRISPEECH_DEVCLEAN = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "dev-clean"
MODEL_DIR = EXPERIMENTS / "models"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000       # 1.5s
STEP_SAMPLES = 1600        # 100ms


# ── Step 1: Mine dev-clean hard negatives ─────────────────────────────────

def mine_devclean_hard_negatives(model_path: Path, threshold: float = 0.5, dedupe_window_sec: float = 2.0):
    """Scan dev-clean audio through the model to find all windows scoring > threshold.

    Deduplicates: for clusters of overlapping windows (same file, within dedupe_window_sec),
    keeps only the peak-scoring window.

    Returns list of dicts: {file, time_sec, score, embedding}
    """
    import onnxruntime as ort
    import soundfile as sf
    from openwakeword.utils import AudioFeatures

    print(f"\n{'='*70}")
    print("STEP 1: Mining dev-clean hard negatives")
    print(f"{'='*70}")
    print(f"  Model: {model_path.name}")
    print(f"  Threshold: {threshold}")
    print(f"  Dedupe window: {dedupe_window_sec}s")

    # Load model
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Init preprocessor
    print("  Initializing OWW preprocessor...")
    preprocessor = AudioFeatures()

    # Find all flac files
    flac_files = sorted(LIBRISPEECH_DEVCLEAN.rglob("*.flac"))
    print(f"  Found {len(flac_files)} flac files in dev-clean")

    all_triggers = []  # (file, time_sec, score, embedding)
    dedupe_samples = int(dedupe_window_sec * SAMPLE_RATE)
    t0 = time.time()

    for fi, fpath in enumerate(flac_files):
        try:
            audio, sr = sf.read(fpath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  WARN: Failed to load {fpath.name}: {e}")
            continue

        n_samples = len(audio)
        file_triggers = []  # (pos, score, embedding)

        pos = 0
        while pos + CLIP_SAMPLES <= n_samples:
            window = audio[pos:pos + CLIP_SAMPLES]
            window_int16 = (window * 32767).clip(-32768, 32767).astype(np.int16)

            # Extract embedding
            embs = preprocessor.embed_clips(window_int16.reshape(1, -1), ncpu=1)
            emb = embs.mean(axis=1)[0].astype(np.float32)

            # Score
            score = float(session.run(None, {input_name: emb.reshape(1, -1)})[0][0][0])

            if score >= threshold:
                file_triggers.append((pos, score, emb))

            pos += STEP_SAMPLES

        # Deduplicate: cluster overlapping triggers, keep peak
        if file_triggers:
            clusters = []
            current_cluster = [file_triggers[0]]
            for t in file_triggers[1:]:
                if t[0] - current_cluster[-1][0] <= dedupe_samples:
                    current_cluster.append(t)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [t]
            clusters.append(current_cluster)

            rel_path = str(fpath.relative_to(LIBRISPEECH_DEVCLEAN))
            for cluster in clusters:
                # Keep peak-scoring window
                peak = max(cluster, key=lambda x: x[1])
                all_triggers.append({
                    "file": rel_path,
                    "time_sec": round(peak[0] / SAMPLE_RATE, 2),
                    "score": round(peak[1], 6),
                    "embedding": peak[2],
                })

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    [{fi+1}/{len(flac_files)}] {len(all_triggers)} unique triggers so far ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  Done: {len(all_triggers)} unique hard negatives from dev-clean ({elapsed:.0f}s)")
    print(f"  Score range: {min(t['score'] for t in all_triggers):.4f} - {max(t['score'] for t in all_triggers):.4f}")

    return all_triggers


def mine_devclean_from_top_scores():
    """Fast path: extract embeddings for just the top-20 windows from the saved FAPH results.

    Used when we don't want to re-scan all of dev-clean (saves 40+ minutes).
    Returns list of dicts with embeddings.
    """
    import soundfile as sf
    from openwakeword.utils import AudioFeatures

    print(f"\n{'='*70}")
    print("STEP 1: Mining dev-clean hard negatives (from saved top scores)")
    print(f"{'='*70}")

    with open(DEVCLEAN_FAPH_FILE) as f:
        faph_data = json.load(f)

    top_scores = faph_data["top_scores"]
    print(f"  {len(top_scores)} top scoring windows to process")

    # Deduplicate: same file within 2s → keep peak
    # Group by file
    by_file = {}
    for entry in top_scores:
        fname = entry["file"]
        if fname not in by_file:
            by_file[fname] = []
        by_file[fname].append(entry)

    deduped = []
    for fname, entries in by_file.items():
        entries.sort(key=lambda x: x["time_sec"])
        clusters = [[entries[0]]]
        for e in entries[1:]:
            if e["time_sec"] - clusters[-1][-1]["time_sec"] <= 2.0:
                clusters[-1].append(e)
            else:
                clusters.append([e])
        for cluster in clusters:
            peak = max(cluster, key=lambda x: x["score"])
            deduped.append(peak)

    print(f"  After deduplication: {len(deduped)} unique windows")

    # Extract embeddings
    print("  Initializing OWW preprocessor...")
    preprocessor = AudioFeatures()

    results = []
    for entry in deduped:
        fname = entry["file"].replace("\\", "/")
        fpath = LIBRISPEECH_DEVCLEAN / fname
        if not fpath.exists():
            # Try with backslash conversion
            fpath = LIBRISPEECH_DEVCLEAN / entry["file"]
        if not fpath.exists():
            print(f"  WARN: File not found: {fpath}")
            continue

        try:
            audio, sr = sf.read(fpath, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  WARN: Failed to load {fpath.name}: {e}")
            continue

        # Extract the window at the given time
        start_sample = int(entry["time_sec"] * SAMPLE_RATE)
        end_sample = start_sample + CLIP_SAMPLES
        if end_sample > len(audio):
            # Pad if needed
            audio = np.pad(audio, (0, max(0, end_sample - len(audio))))
        window = audio[start_sample:end_sample]
        window_int16 = (window * 32767).clip(-32768, 32767).astype(np.int16)

        embs = preprocessor.embed_clips(window_int16.reshape(1, -1), ncpu=1)
        emb = embs.mean(axis=1)[0].astype(np.float32)

        results.append({
            "file": entry["file"],
            "time_sec": entry["time_sec"],
            "score": entry["score"],
            "embedding": emb,
        })
        print(f"    {entry['file']} @{entry['time_sec']}s score={entry['score']:.4f} OK")

    print(f"\n  Extracted {len(results)} dev-clean hard negative embeddings")
    return results


# ── Step 2: Add ACAV hard negatives to cache ─────────────────────────────

def prepare_acav_hard_negatives():
    """Load ACAV hard negatives (score >0.3) and return their embeddings."""
    print(f"\n{'='*70}")
    print("STEP 2: Preparing ACAV hard negatives")
    print(f"{'='*70}")

    # Load scores
    acav_data = np.load(ACAV_SCORES_FILE, allow_pickle=True)
    hard_indices = acav_data["hard_indices"]  # indices into the neg_acav100m embeddings
    hard_scores = acav_data["hard_scores"]

    print(f"  {len(hard_indices)} ACAV embeddings scoring >0.3")
    print(f"  Score range: {hard_scores.min():.4f} - {hard_scores.max():.4f}")
    print(f"  Mean score: {hard_scores.mean():.4f}")

    # Load the full cache to get embeddings
    cache = np.load(CACHE_FILE, allow_pickle=True)
    all_embs = cache["embeddings"]
    print(f"  Cache size: {len(all_embs)} entries")

    # hard_indices are absolute cache indices (into the full cache array)
    # Verify they point to neg_acav100m entries
    all_tags = cache["tags"]
    sample_tags = [str(all_tags[i]) for i in hard_indices[:5]]
    print(f"  Sample tags at hard_indices: {sample_tags}")

    hard_embeddings = all_embs[hard_indices]
    print(f"  Extracted {len(hard_embeddings)} hard negative embeddings, shape: {hard_embeddings.shape}")

    return hard_embeddings, hard_scores


# ── Step 3: Update embedding cache ───────────────────────────────────────

def update_cache(devclean_results, acav_hard_embeddings):
    """Add new hard negative entries to the embedding cache."""
    print(f"\n{'='*70}")
    print("STEP 3: Updating embedding cache")
    print(f"{'='*70}")

    # Load existing cache
    cache = np.load(CACHE_FILE, allow_pickle=True)
    existing_embs = cache["embeddings"]
    existing_labels = cache["labels"]
    existing_tags = cache["tags"]
    existing_files = cache["files"]
    existing_source_idx = cache["source_idx"]

    print(f"  Existing cache: {len(existing_embs)} entries")
    max_source_idx = int(existing_source_idx.max())
    print(f"  Max source_idx: {max_source_idx}")

    # Check if tags already exist (idempotency)
    existing_tag_set = set(existing_tags)

    new_embs = []
    new_labels = []
    new_tags = []
    new_files = []
    new_source_idx = []
    next_idx = max_source_idx + 1

    # Add ACAV hard negatives (if not already present)
    if "neg_acav_hard_r2" not in existing_tag_set:
        n_acav = len(acav_hard_embeddings)
        new_embs.append(acav_hard_embeddings)
        new_labels.append(np.zeros(n_acav, dtype=np.int32))
        new_tags.extend(["neg_acav_hard_r2"] * n_acav)
        new_files.extend([f"acav_hard_{i}" for i in range(n_acav)])
        new_source_idx.extend(list(range(next_idx, next_idx + n_acav)))
        next_idx += n_acav
        print(f"  Adding {n_acav} ACAV hard negatives (tag: neg_acav_hard_r2)")
    else:
        print(f"  SKIP: neg_acav_hard_r2 already exists in cache ({(existing_tags == 'neg_acav_hard_r2').sum()} entries)")

    # Add dev-clean hard negatives (if not already present)
    if "neg_devclean_hard_r2" not in existing_tag_set:
        n_dc = len(devclean_results)
        dc_embs = np.array([r["embedding"] for r in devclean_results], dtype=np.float32)
        new_embs.append(dc_embs)
        new_labels.append(np.zeros(n_dc, dtype=np.int32))
        new_tags.extend(["neg_devclean_hard_r2"] * n_dc)
        new_files.extend([f"{r['file']}@{r['time_sec']}s" for r in devclean_results])
        new_source_idx.extend(list(range(next_idx, next_idx + n_dc)))
        next_idx += n_dc
        print(f"  Adding {n_dc} dev-clean hard negatives (tag: neg_devclean_hard_r2)")
    else:
        print(f"  SKIP: neg_devclean_hard_r2 already exists in cache ({(existing_tags == 'neg_devclean_hard_r2').sum()} entries)")

    if not new_embs:
        print("  No new entries to add. Cache unchanged.")
        return

    # Concatenate
    all_new_embs = np.concatenate(new_embs, axis=0)
    all_new_labels = np.concatenate(new_labels, axis=0)
    all_new_tags = np.array(new_tags, dtype=object)
    all_new_files = np.array(new_files, dtype=object)
    all_new_source_idx = np.array(new_source_idx, dtype=np.int32)

    final_embs = np.concatenate([existing_embs, all_new_embs], axis=0)
    final_labels = np.concatenate([existing_labels, all_new_labels], axis=0)
    final_tags = np.concatenate([existing_tags, all_new_tags], axis=0)
    final_files = np.concatenate([existing_files, all_new_files], axis=0)
    final_source_idx = np.concatenate([existing_source_idx, all_new_source_idx], axis=0)

    print(f"  New cache size: {len(final_embs)} entries (+{len(all_new_embs)})")

    # Save (backup first)
    backup_path = CACHE_FILE.with_suffix(".npz.bak_round2")
    if not backup_path.exists():
        import shutil
        shutil.copy2(CACHE_FILE, backup_path)
        print(f"  Backup saved: {backup_path.name}")

    np.savez_compressed(
        CACHE_FILE,
        embeddings=final_embs,
        labels=final_labels,
        tags=final_tags,
        files=final_files,
        source_idx=final_source_idx,
    )
    print(f"  Cache saved: {CACHE_FILE}")

    # Verify
    verify = np.load(CACHE_FILE, allow_pickle=True)
    vtags = verify["tags"]
    unique_tags, counts = np.unique(vtags, return_counts=True)
    print(f"\n  Verified cache tags:")
    for t, c in zip(unique_tags, counts):
        print(f"    {t}: {c}")


# ── Step 4: Train round 2 models ─────────────────────────────────────────

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


def build_round2_weights(labels, tags, hard_weight=10.0, confusable_weight=5.0):
    """Build per-sample weights: hard negatives at 10x, confusable at 5x."""
    weights = np.ones(len(labels), dtype=np.float32)

    # Confusable negatives at 5x
    confusable_mask = (tags == "neg_confusable") | (tags == "neg_confusable_v2")
    weights[confusable_mask & (labels == 0)] = confusable_weight
    n_conf = (confusable_mask & (labels == 0)).sum()

    # Hard negatives at 10x (overrides confusable weight if both apply)
    hard_mask = np.zeros(len(tags), dtype=bool)
    for t in np.unique(tags):
        if "hard" in str(t).lower():
            hard_mask |= (tags == t)
    weights[hard_mask & (labels == 0)] = hard_weight
    n_hard = (hard_mask & (labels == 0)).sum()

    print(f"  Sample weights: {n_conf} confusable at {confusable_weight}x, {n_hard} hard negatives at {hard_weight}x")
    return weights


def train_weighted_bce(
    name,
    embeddings,
    labels,
    source_idx,
    sample_weights,
    output_model,
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
    """Train MLP with per-sample weighted BCE loss."""
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

    # Group-aware 80/20 split
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

    # Weighted BCE loss
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
        tl, nb = 0.0, 0
        for bx, by, bw in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss_per_sample = criterion(pred, by)
            loss = (loss_per_sample * bw).mean()
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
            for bx, by, bw in val_loader:
                pred = model(bx)
                loss_per_sample = criterion(pred, by)
                vl += (loss_per_sample * bw).mean().item()
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


def train_round2():
    """Step 4: Train round 2 models with all hard negatives at 10x weight."""
    print(f"\n{'='*70}")
    print("STEP 4: Training round 2 models")
    print(f"{'='*70}")

    cache = load_data_with_tags()
    embs = cache["embeddings"]
    labels = cache["labels"]
    tags = cache["tags"]
    source_idx = cache["source_idx"]

    # Select training tags
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

    sel_embs = embs[mask]
    sel_labels = labels[mask]
    sel_tags = tags[mask]
    sel_source_idx = source_idx[mask]

    n_pos = (sel_labels == 1).sum()
    n_neg = (sel_labels == 0).sum()
    print(f"\n  Total: {n_pos} pos + {n_neg} neg = {len(sel_labels)}")

    # Build weights
    sample_weights = build_round2_weights(sel_labels, sel_tags, hard_weight=10.0, confusable_weight=5.0)

    seeds = [42, 43]
    results = {}

    for seed in seeds:
        model_name = f"round2_s{seed}"
        model_path = MODEL_DIR / f"{model_name}.onnx"
        print(f"\n  Training {model_name}...")
        train_info = train_weighted_bce(
            model_name,
            sel_embs,
            sel_labels,
            sel_source_idx,
            sample_weights,
            model_path,
            seed=seed,
        )
        results[model_name] = train_info
        print(f"  {model_name}: best_epoch={train_info['best_epoch']}, val_loss={train_info['best_val_loss']:.6f}, time={train_info['time']}s")

    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Round 2 hard negative mining training")
    parser.add_argument("--skip-mining", action="store_true", help="Skip steps 1-3 (mining + cache update)")
    parser.add_argument("--full-scan", action="store_true", help="Full dev-clean scan (slow, ~40min) instead of top-20 only")
    args = parser.parse_args()

    if not args.skip_mining:
        # Step 1: Mine dev-clean hard negatives
        if args.full_scan:
            model_path = MODEL_DIR / "faph_hardened_s43.onnx"
            devclean_results = mine_devclean_hard_negatives(model_path, threshold=0.5)
        else:
            devclean_results = mine_devclean_from_top_scores()

        # Step 2: Prepare ACAV hard negatives
        acav_hard_embs, acav_hard_scores = prepare_acav_hard_negatives()

        # Step 3: Update cache
        update_cache(devclean_results, acav_hard_embs)

    # Step 4: Train
    train_results = train_round2()

    # Summary
    print(f"\n{'='*70}")
    print("ROUND 2 TRAINING COMPLETE")
    print(f"{'='*70}")
    for name, info in train_results.items():
        print(f"  {name}: val_loss={info['best_val_loss']:.6f}, best_epoch={info['best_epoch']}, time={info['time']}s")
    print(f"\n  Models saved to: {MODEL_DIR}")
    print(f"    round2_s42.onnx")
    print(f"    round2_s43.onnx")
    print(f"\n  Next: Run TP eval:")
    print(f"    python experiments/hardened_tp_eval.py")


if __name__ == "__main__":
    main()
