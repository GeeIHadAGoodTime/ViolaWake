#!/usr/bin/env python3
"""
Experiment: Temporal Concatenation vs Mean-Pool
================================================

Hypothesis: Mean-pooling across 9 temporal frames (9x96 -> 96-dim) destroys
temporal order information. Concatenating all frames (9x96 = 864-dim) may
improve wake word discrimination, especially against confusable words.

This experiment:
  1. Re-extracts temporal embeddings (9, 96) from audio files (skips precomputed NPZ)
  2. Caches both mean-pooled (96-dim) and concatenated (864-dim) versions
  3. Trains identical MLP architectures on both representations
  4. Compares EER, AUC, d-prime, and confusable-word scores

Usage:
    python experiments/exp_temporal_concat.py                    # Full run
    python experiments/exp_temporal_concat.py --skip-extract     # Use cached temporal embeddings
    python experiments/exp_temporal_concat.py --max-samples 5000 # Quick test with subset
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
EVAL_DIR = WAKEWORD / "eval_clean"
CACHE_96 = EXPERIMENTS / "embedding_cache.npz"
CACHE_TEMPORAL = EXPERIMENTS / "embedding_cache_temporal.npz"
CONFIG_FILE = EXPERIMENTS / "experiment_config.json"
RESULTS_FILE = EXPERIMENTS / "exp_temporal_concat_results.json"

# Tags used by train_hardened.py (the strongest dataset)
TARGET_TAGS = [
    "pos_main",
    "pos_diverse",
    "neg_main",
    "neg_confusable",
    "neg_confusable_v2",
]

N_FRAMES = 9  # OWW always produces 9 frames for 1.5s clips
EMB_DIM = 96
CONCAT_DIM = N_FRAMES * EMB_DIM  # 864


def extract_temporal_embeddings(max_samples: int | None = None) -> None:
    """Re-extract embeddings keeping full temporal structure (9, 96)."""
    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    # Load config to get audio file paths
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Load existing 96-dim cache for metadata (labels, tags, files, source_idx)
    old_data = np.load(CACHE_96, allow_pickle=True)
    old_tags = old_data["tags"]
    old_files = old_data["files"]
    old_labels = old_data["labels"]
    old_source_idx = old_data["source_idx"]

    # Filter to target tags only
    mask = np.zeros(len(old_tags), dtype=bool)
    for t in TARGET_TAGS:
        mask |= old_tags == t

    target_files = old_files[mask]
    target_labels = old_labels[mask]
    target_tags = old_tags[mask]
    target_source_idx = old_source_idx[mask]

    if max_samples and len(target_files) > max_samples:
        # Stratified subsample
        rng = np.random.default_rng(42)
        idx = rng.choice(len(target_files), max_samples, replace=False)
        idx.sort()
        target_files = target_files[idx]
        target_labels = target_labels[idx]
        target_tags = target_tags[idx]
        target_source_idx = target_source_idx[idx]

    n_total = len(target_files)
    print(f"Extracting temporal embeddings for {n_total} samples...")
    for t in TARGET_TAGS:
        print(f"  {t}: {(target_tags == t).sum()}")

    # Init OWW preprocessor
    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    # Extract
    all_temporal = []   # (9, 96) per sample
    all_meanpool = []   # (96,) per sample
    all_labels = []
    all_tags = []
    all_files = []
    all_source_idx = []
    failed = 0

    t0 = time.monotonic()
    for i, (fpath, label, tag, sidx) in enumerate(
        zip(target_files, target_labels, target_tags, target_source_idx)
    ):
        fpath_str = str(fpath)

        # Skip precomputed NPZ entries (they have # in filename)
        if "#" in fpath_str:
            failed += 1
            continue

        audio = load_audio(Path(fpath_str))
        if audio is None:
            failed += 1
            continue

        audio = center_crop(audio, CLIP_SAMPLES)
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        if len(audio_int16) < CLIP_SAMPLES:
            audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:CLIP_SAMPLES]

        try:
            embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            # embeddings shape: (1, T, 96), T should be 9
            temporal = embeddings[0]  # (T, 96)

            if temporal.shape[0] != N_FRAMES:
                # Pad or truncate to exactly 9 frames
                if temporal.shape[0] < N_FRAMES:
                    pad = np.zeros((N_FRAMES - temporal.shape[0], EMB_DIM), dtype=np.float32)
                    temporal = np.concatenate([temporal, pad], axis=0)
                else:
                    temporal = temporal[:N_FRAMES]

            all_temporal.append(temporal.astype(np.float32))
            all_meanpool.append(temporal.mean(axis=0).astype(np.float32))
            all_labels.append(label)
            all_tags.append(tag)
            all_files.append(fpath_str)
            all_source_idx.append(sidx)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Failed on {fpath_str}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (n_total - i - 1) / rate
            print(f"  {i+1}/{n_total} ({rate:.0f}/s, ETA {eta:.0f}s) [{len(all_temporal)} ok, {failed} failed]")

    elapsed = time.monotonic() - t0
    print(f"\nExtraction complete: {len(all_temporal)} samples in {elapsed:.0f}s ({failed} failed)")

    # Save temporal cache
    temporal_arr = np.array(all_temporal, dtype=np.float32)  # (N, 9, 96)
    meanpool_arr = np.array(all_meanpool, dtype=np.float32)  # (N, 96)
    concat_arr = temporal_arr.reshape(len(temporal_arr), -1)  # (N, 864)

    np.savez_compressed(
        CACHE_TEMPORAL,
        temporal=temporal_arr,
        meanpool=meanpool_arr,
        concat=concat_arr,
        labels=np.array(all_labels, dtype=np.int32),
        tags=np.array(all_tags, dtype=object),
        files=np.array(all_files, dtype=object),
        source_idx=np.array(all_source_idx, dtype=np.int32),
    )
    print(f"Saved temporal cache: {CACHE_TEMPORAL}")
    print(f"  temporal: {temporal_arr.shape}")
    print(f"  meanpool: {meanpool_arr.shape}")
    print(f"  concat:   {concat_arr.shape}")


def train_mlp(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    tags: np.ndarray,
    source_idx: np.ndarray,
    hidden_dims: tuple[int, ...] = (128, 64),
    dropouts: tuple[float, ...] = (0.3, 0.2),
    seed: int = 42,
    epochs: int = 80,
    patience: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    confusable_weight: float = 5.0,
) -> dict:
    """Train MLP classifier and return metrics. Almost identical to train_hardened.py."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    embedding_dim = embeddings.shape[1]

    # Build sample weights (confusable negatives get extra weight)
    weights = np.ones(len(labels), dtype=np.float32)
    confusable_mask = (tags == "neg_confusable") | (tags == "neg_confusable_v2")
    weights[confusable_mask & (labels == 0)] = confusable_weight

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    w = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

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

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"  {name}: {train_mask.sum()} train / {val_mask.sum()} val "
          f"({n_pos} pos + {n_neg} neg), input_dim={embedding_dim}")

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

    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_ds = TensorDataset(X_train, y_train, w_train)
    val_ds = TensorDataset(X_val, y_val, w_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = None
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
        else:
            no_imp += 1

        if no_imp >= patience:
            break

    if best_state:
        model.load_state_dict(best_state)

    dur = time.monotonic() - t0

    # Score all data for metrics
    model.eval()
    with torch.no_grad():
        all_scores = model(X).squeeze(1).numpy()

    pos_scores = all_scores[labels == 1]
    neg_scores = all_scores[labels == 0]
    conf_scores = all_scores[confusable_mask & (labels == 0)]

    # EER
    from sklearn.metrics import auc, roc_curve
    all_labels_np = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    all_scores_np = np.concatenate([pos_scores, neg_scores])
    fpr, tpr, thresholds = roc_curve(all_labels_np, all_scores_np)
    roc_auc = float(auc(fpr, tpr))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])

    # d-prime
    pooled_std = np.sqrt(0.5 * (pos_scores.var() + neg_scores.var()))
    dprime = float((pos_scores.mean() - neg_scores.mean()) / pooled_std) if pooled_std > 1e-10 else 0.0

    # Per-tag score analysis
    tag_scores = {}
    for t in np.unique(tags):
        t_mask = tags == t
        t_scores = all_scores[t_mask]
        tag_scores[str(t)] = {
            "mean": round(float(t_scores.mean()), 4),
            "std": round(float(t_scores.std()), 4),
            "min": round(float(t_scores.min()), 4),
            "max": round(float(t_scores.max()), 4),
            "n": int(t_mask.sum()),
        }

    # Threshold analysis
    thresholds_analysis = {}
    for thr in [0.3, 0.5, 0.7, 0.8, 0.9]:
        det_rate = float((pos_scores >= thr).mean())
        fa_rate = float((neg_scores >= thr).mean())
        conf_fa_rate = float((conf_scores >= thr).mean()) if len(conf_scores) > 0 else 0.0
        thresholds_analysis[str(thr)] = {
            "detection_rate": round(det_rate, 4),
            "false_alarm_rate": round(fa_rate, 4),
            "confusable_fa_rate": round(conf_fa_rate, 4),
        }

    return {
        "name": name,
        "input_dim": embedding_dim,
        "hidden_dims": list(hidden_dims),
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
        "best_epoch": best_ep,
        "best_val_loss": round(float(best_val), 6),
        "train_time": round(dur, 1),
        "tag_scores": tag_scores,
        "thresholds": thresholds_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description="Temporal concat vs mean-pool experiment")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Use cached temporal embeddings (skip re-extraction)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples for quick testing")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of random seeds (default: 3)")
    args = parser.parse_args()

    # Step 1: Extract temporal embeddings
    if not args.skip_extract or not CACHE_TEMPORAL.exists():
        print("=" * 70)
        print("STEP 1: Extracting temporal embeddings (9 x 96)")
        print("=" * 70)
        sys.path.insert(0, str(WAKEWORD / "src"))
        extract_temporal_embeddings(max_samples=args.max_samples)
    else:
        print("Using cached temporal embeddings from", CACHE_TEMPORAL)

    # Step 2: Load temporal cache
    print("\n" + "=" * 70)
    print("STEP 2: Loading embeddings")
    print("=" * 70)

    data = np.load(CACHE_TEMPORAL, allow_pickle=True)
    meanpool = data["meanpool"]   # (N, 96)
    concat = data["concat"]       # (N, 864)
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_conf = ((tags == "neg_confusable") | (tags == "neg_confusable_v2")).sum()
    print(f"  Samples: {len(labels)} ({n_pos} pos + {n_neg} neg, {n_conf} confusable)")
    print(f"  Mean-pooled shape: {meanpool.shape}")
    print(f"  Concatenated shape: {concat.shape}")

    # Step 3: Train and compare
    print("\n" + "=" * 70)
    print("STEP 3: Training models")
    print("=" * 70)

    seeds = list(range(42, 42 + args.seeds))
    all_results = {"meanpool_96": [], "concat_864": [], "concat_864_wide": []}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")

        # Baseline: mean-pooled 96-dim -> [64, 32] -> 1
        print("\n  [A] Mean-pooled 96-dim (baseline)")
        r = train_mlp(
            f"meanpool_96_s{seed}",
            meanpool, labels, tags, source_idx,
            hidden_dims=(64, 32),
            dropouts=(0.3, 0.2),
            seed=seed,
        )
        all_results["meanpool_96"].append(r)
        print(f"    EER={r['eer']:.4f}, AUC={r['auc']:.4f}, d'={r['dprime']:.2f}, "
              f"conf_mean={r['conf_mean']}")

        # Experiment: concatenated 864-dim -> [128, 64] -> 1
        print("\n  [B] Concatenated 864-dim (temporal)")
        r = train_mlp(
            f"concat_864_s{seed}",
            concat, labels, tags, source_idx,
            hidden_dims=(128, 64),
            dropouts=(0.3, 0.2),
            seed=seed,
        )
        all_results["concat_864"].append(r)
        print(f"    EER={r['eer']:.4f}, AUC={r['auc']:.4f}, d'={r['dprime']:.2f}, "
              f"conf_mean={r['conf_mean']}")

        # Experiment: concatenated 864-dim -> [256, 128, 64] -> 1 (wider)
        print("\n  [C] Concatenated 864-dim (wider MLP)")
        r = train_mlp(
            f"concat_864_wide_s{seed}",
            concat, labels, tags, source_idx,
            hidden_dims=(256, 128, 64),
            dropouts=(0.3, 0.2, 0.1),
            seed=seed,
        )
        all_results["concat_864_wide"].append(r)
        print(f"    EER={r['eer']:.4f}, AUC={r['auc']:.4f}, d'={r['dprime']:.2f}, "
              f"conf_mean={r['conf_mean']}")

    # Step 4: Summary
    print("\n\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Variant':<25} {'EER':>8} {'AUC':>8} {'d-prime':>8} "
          f"{'conf_mean':>10} {'pos_mean':>10} {'neg_mean':>10}")
    print("-" * 85)

    summary = {}
    for variant, results in all_results.items():
        eers = [r["eer"] for r in results]
        aucs = [r["auc"] for r in results]
        dprimes = [r["dprime"] for r in results]
        conf_means = [r["conf_mean"] for r in results if r["conf_mean"] is not None]
        pos_means = [r["pos_mean"] for r in results]
        neg_means = [r["neg_mean"] for r in results]

        summary[variant] = {
            "eer_mean": round(float(np.mean(eers)), 4),
            "eer_std": round(float(np.std(eers)), 4),
            "auc_mean": round(float(np.mean(aucs)), 4),
            "dprime_mean": round(float(np.mean(dprimes)), 3),
            "conf_mean": round(float(np.mean(conf_means)), 4) if conf_means else None,
            "pos_mean": round(float(np.mean(pos_means)), 4),
            "neg_mean": round(float(np.mean(neg_means)), 4),
            "per_seed": results,
        }

        eer_str = f"{np.mean(eers):.4f}+/-{np.std(eers):.4f}"
        auc_str = f"{np.mean(aucs):.4f}"
        dp_str = f"{np.mean(dprimes):.2f}"
        conf_str = f"{np.mean(conf_means):.4f}" if conf_means else "N/A"
        pos_str = f"{np.mean(pos_means):.4f}"
        neg_str = f"{np.mean(neg_means):.4f}"

        print(f"{variant:<25} {eer_str:>16} {auc_str:>8} {dp_str:>8} "
              f"{conf_str:>10} {pos_str:>10} {neg_str:>10}")

    # Delta analysis
    print("\n" + "=" * 70)
    print("DELTA ANALYSIS (vs baseline mean-pool)")
    print("=" * 70)

    baseline_eer = summary["meanpool_96"]["eer_mean"]
    baseline_auc = summary["meanpool_96"]["auc_mean"]
    baseline_dp = summary["meanpool_96"]["dprime_mean"]
    baseline_conf = summary["meanpool_96"]["conf_mean"]

    for variant in ["concat_864", "concat_864_wide"]:
        s = summary[variant]
        eer_delta = s["eer_mean"] - baseline_eer
        auc_delta = s["auc_mean"] - baseline_auc
        dp_delta = s["dprime_mean"] - baseline_dp
        conf_delta = (s["conf_mean"] - baseline_conf) if s["conf_mean"] and baseline_conf else None

        print(f"\n  {variant}:")
        print(f"    EER:     {s['eer_mean']:.4f} ({eer_delta:+.4f}, {'BETTER' if eer_delta < 0 else 'WORSE'})")
        print(f"    AUC:     {s['auc_mean']:.4f} ({auc_delta:+.4f}, {'BETTER' if auc_delta > 0 else 'WORSE'})")
        print(f"    d-prime: {s['dprime_mean']:.2f} ({dp_delta:+.2f}, {'BETTER' if dp_delta > 0 else 'WORSE'})")
        if conf_delta is not None:
            print(f"    conf_mean: {s['conf_mean']:.4f} ({conf_delta:+.4f}, {'BETTER' if conf_delta < 0 else 'WORSE'})")

    # Threshold comparison
    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON (best seed per variant)")
    print("=" * 70)

    for variant, results in all_results.items():
        best = min(results, key=lambda r: r["eer"])
        print(f"\n  {variant} (seed with best EER={best['eer']:.4f}):")
        for thr, info in best["thresholds"].items():
            print(f"    @{thr}: detect={info['detection_rate']:.1%}, "
                  f"FA={info['false_alarm_rate']:.1%}, "
                  f"conf_FA={info['confusable_fa_rate']:.1%}")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    best_variant = min(summary.items(), key=lambda x: x[1]["eer_mean"])
    worst_variant = max(summary.items(), key=lambda x: x[1]["eer_mean"])

    print(f"\n  Best EER:  {best_variant[0]} ({best_variant[1]['eer_mean']:.4f})")
    print(f"  Worst EER: {worst_variant[0]} ({worst_variant[1]['eer_mean']:.4f})")

    if best_variant[0] == "meanpool_96":
        print("\n  CONCLUSION: Mean-pooling is NOT the bottleneck.")
        print("  Temporal order information does not help discrimination.")
        print("  The 'OWW embedding ceiling' is a true representational limit,")
        print("  not an information-loss artifact from mean-pooling.")
    else:
        eer_improvement = baseline_eer - best_variant[1]["eer_mean"]
        print(f"\n  CONCLUSION: Temporal concatenation IMPROVES discrimination!")
        print(f"  EER improvement: {eer_improvement:.4f} ({eer_improvement/baseline_eer*100:.1f}% relative)")
        print(f"  The 'OWW embedding ceiling' was partly a 'mean-pool ceiling'.")
        print(f"  Consider integrating {best_variant[0]} into the production pipeline.")

    # Save results
    output = {
        "experiment": "temporal_concat_vs_meanpool",
        "hypothesis": "Mean-pooling destroys temporal information that helps wake word discrimination",
        "n_samples": int(len(labels)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_confusable": int(n_conf),
        "n_seeds": args.seeds,
        "summary": summary,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
