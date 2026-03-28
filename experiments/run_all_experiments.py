"""
ViolaWake Experiments — Config-Driven Test Harness
===================================================

JSON-driven experiment system. Reads experiment_config.json for:
- Data source registry (audio dirs + pre-computed NPZ files)
- Experiment definitions (which sources to combine)
- Architecture variants (hidden dims, dropout, activation)
- Training hyperparameters

Extracts all embeddings ONCE, caches to .npz, then runs all experiments
from cache. ~45 min first run, ~2 min per experiment after.

Usage:
  python run_all_experiments.py                     # Run all experiments in config
  python run_all_experiments.py --experiments F G    # Run specific experiments
  python run_all_experiments.py --force-extract      # Re-extract embeddings
  python run_all_experiments.py --arch wide          # Use wide architecture variant
  python run_all_experiments.py --seeds 3            # Multi-seed runs (mean +/- std)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
VIOLA = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA")
EXPERIMENTS = WAKEWORD / "experiments"
EVAL_DIR = WAKEWORD / "eval_clean"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"
CONFIG_FILE = EXPERIMENTS / "experiment_config.json"
BASELINE_MODEL = VIOLA / "violawake_data" / "trained_models" / "viola_mlp_oww.onnx"


def load_config() -> dict:
    """Load experiment config from JSON."""
    with open(CONFIG_FILE) as f:
        return json.load(f)


def resolve_source_path(path_str: str) -> Path:
    """Resolve a source path — absolute or relative to WAKEWORD."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return WAKEWORD / p


def build_sources_from_config(config: dict) -> list[tuple[str, Path, int, dict]]:
    """Build (tag, path, label, meta) tuples from config data_sources."""
    sources = []
    for tag, info in config["data_sources"].items():
        path = resolve_source_path(info["path"])
        label = info["label"]
        meta = {k: v for k, v in info.items() if k not in ("path", "label")}
        sources.append((tag, path, label, meta))
    return sources


def extract_all_embeddings(
    config: dict,
    force: bool = False,
    augment_factor: int | None = None,
) -> dict:
    """Extract OWW embeddings from all sources and cache to .npz.

    Supports both audio directories and pre-computed NPZ files.
    """
    if CACHE_FILE.exists() and not force:
        print(f"Loading cached embeddings from {CACHE_FILE}")
        data = np.load(CACHE_FILE, allow_pickle=True)
        return {
            "embeddings": data["embeddings"],
            "labels": data["labels"],
            "tags": data["tags"],
            "files": data["files"],
            "source_idx": data["source_idx"],
        }

    print("Extracting embeddings from all sources...")

    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio
    from violawake_sdk.training.augment import AugmentationPipeline

    training_defaults = config.get("training_defaults", {})
    aug_factor = augment_factor or training_defaults.get("augment_factor", 8)
    seed = training_defaults.get("seed", 42)

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

    all_embeddings = []
    all_labels = []
    all_tags = []
    all_files = []
    all_source_idx = []

    pipeline = AugmentationPipeline(seed=seed)
    sources = build_sources_from_config(config)

    for tag, directory, label, meta in sources:
        source_type = meta.get("type", "audio_dir")

        # ── Pre-computed NPZ embeddings ──
        if source_type == "precomputed_npz":
            if not directory.exists():
                print(f"  SKIP {tag}: {directory} not found")
                continue
            print(f"  {tag}: loading pre-computed NPZ...", end="", flush=True)
            t0 = time.time()
            npz = np.load(directory, allow_pickle=True)

            # Support different NPZ formats
            if "embeddings" in npz:
                embs = npz["embeddings"].astype(np.float32)
            elif "x" in npz:
                embs = npz["x"].astype(np.float32)
            else:
                # Try first array
                keys = list(npz.keys())
                embs = npz[keys[0]].astype(np.float32)

            n = len(embs)
            all_embeddings.extend(embs)
            all_labels.extend([label] * n)
            all_tags.extend([tag] * n)
            all_files.extend([f"{directory}#{i}" for i in range(n)])
            all_source_idx.extend(list(range(n)))

            elapsed = time.time() - t0
            print(f" -> {n} embeddings in {elapsed:.0f}s")
            continue

        # ── Standard audio directory ──
        if not directory.exists():
            print(f"  SKIP {tag}: {directory} not found")
            continue

        exclude_subdirs = meta.get("exclude_subdirs", [])
        files = sorted(
            list(directory.rglob("*.wav")) + list(directory.rglob("*.flac"))
        )
        # Apply exclude_subdirs
        for excl in exclude_subdirs:
            files = [f for f in files if excl not in f.parts]

        if not files:
            print(f"  SKIP {tag}: no audio files")
            continue

        print(f"  {tag}: {len(files)} files (label={label})...", end="", flush=True)
        t0 = time.time()
        count = 0

        for fi, fpath in enumerate(files):
            audio = load_audio(fpath)
            if audio is None:
                continue

            # Original embedding
            emb = _embed(audio)
            if emb is not None:
                all_embeddings.append(emb)
                all_labels.append(label)
                all_tags.append(tag)
                all_files.append(str(fpath))
                all_source_idx.append(fi)
                count += 1

            # Augment positives only
            if label == 1 and training_defaults.get("augment_positives", True):
                variants = pipeline.augment_clip(audio, factor=aug_factor)
                for v in variants:
                    emb = _embed(v)
                    if emb is not None:
                        all_embeddings.append(emb)
                        all_labels.append(label)
                        all_tags.append(tag)
                        all_files.append(str(fpath) + "_aug")
                        all_source_idx.append(fi)
                        count += 1

            if (fi + 1) % 500 == 0:
                print(f" {fi+1}", end="", flush=True)

        elapsed = time.time() - t0
        print(f" -> {count} embeddings in {elapsed:.0f}s")

    result = {
        "embeddings": np.array(all_embeddings, dtype=np.float32),
        "labels": np.array(all_labels, dtype=np.int32),
        "tags": np.array(all_tags),
        "files": np.array(all_files),
        "source_idx": np.array(all_source_idx, dtype=np.int32),
    }

    # Save cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE_FILE, **result)
    print(f"\nCached {len(all_embeddings)} embeddings to {CACHE_FILE}")
    print(f"  Positives: {sum(1 for l in all_labels if l == 1)}")
    print(f"  Negatives: {sum(1 for l in all_labels if l == 0)}")

    return result


def build_model(arch_config: dict, embedding_dim: int):
    """Build MLP model from architecture config."""
    import torch.nn as nn

    hidden_dims = arch_config.get("hidden_dims", [64, 32])
    dropouts = arch_config.get("dropout", [0.3, 0.2])
    activation_name = arch_config.get("activation", "relu")

    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    activation_cls = activation_map.get(activation_name, nn.ReLU)

    layers = []
    in_dim = embedding_dim
    for i, h_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(activation_cls())
        if i < len(dropouts):
            layers.append(nn.Dropout(dropouts[i]))
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def train_from_cache(
    name: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    source_idx: np.ndarray,
    output_model: Path,
    config: dict,
    arch_name: str = "default",
    seed: int = 42,
) -> dict:
    """Train MLP from pre-extracted embeddings."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from violawake_sdk.training.losses import FocalLoss

    training = config.get("training_defaults", {})
    epochs = training.get("epochs", 60)
    patience = training.get("patience", 12)
    batch_size = training.get("batch_size", 32)
    lr = training.get("lr", 1e-3)
    weight_decay = training.get("weight_decay", 1e-4)
    focal_gamma = training.get("focal_gamma", 2.0)
    focal_alpha = training.get("focal_alpha", 0.75)
    label_smoothing = training.get("label_smoothing", 0.05)
    ema_decay = training.get("ema_decay", 0.999)

    # Get architecture config
    arch_variants = config.get("architecture_variants", {})
    arch_config = arch_variants.get(arch_name, arch_variants.get("default", {
        "hidden_dims": [64, 32], "dropout": [0.3, 0.2], "activation": "relu"
    }))

    print(f"\n{'='*60}")
    print(f"TRAINING: {name} (arch={arch_name}, seed={seed})")
    print(f"  Architecture: {arch_config['hidden_dims']}")
    print(f"{'='*60}")

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"Data: {n_pos} positives + {n_neg} negatives = {len(labels)} total")

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embedding_dim = X.shape[1]

    # Group-aware 80/20 split by source file
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
            if source_idx[i] in val_pos_set:
                val_mask[i] = True
            else:
                train_mask[i] = True
        else:
            if source_idx[i] in val_neg_set:
                val_mask[i] = True
            else:
                train_mask[i] = True

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    print(f"Split: {train_mask.sum()} train / {val_mask.sum()} val")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False
    )

    model = build_model(arch_config, embedding_dim)
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Best':>10}")
    print("-" * 40)

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

        avg_t, avg_v = tl / max(nb, 1), vl / max(vn, 1)
        if avg_v < best_val:
            best_val, best_ep, no_imp = avg_v, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema = {k: v.clone() for k, v in ema.items()}
        else:
            no_imp += 1

        if ep % 10 == 0 or ep == 1 or no_imp == 0:
            m = " *" if ep == best_ep else ""
            print(f"{ep:>6} {avg_t:>10.4f} {avg_v:>10.4f} {best_val:>10.4f}{m}")

        if no_imp >= patience:
            print(f"\nEarly stop ep {ep}. Best: {best_ep}")
            break

    if best_state:
        model.load_state_dict(best_state)
    if best_ema:
        for n, p in model.named_parameters():
            p.data.copy_(best_ema[n])

    dur = time.monotonic() - t0
    print(f"Training: {dur:.1f}s")

    # Export ONNX
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    import torch
    torch.onnx.export(
        model, torch.zeros(1, embedding_dim), str(output_model),
        input_names=["embedding"], output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    print(f"Saved: {output_model}")
    return {
        "best_epoch": best_ep,
        "best_val_loss": float(best_val),
        "training_time": round(dur, 1),
        "architecture": arch_name,
        "hidden_dims": arch_config["hidden_dims"],
        "seed": seed,
    }


def evaluate_model(model_path: Path, eval_dir: Path, name: str) -> dict:
    """Evaluate model against clean eval set."""
    from violawake_sdk.training.evaluate import (
        evaluate_onnx_model, find_optimal_threshold, compute_dprime,
    )

    csv_path = EXPERIMENTS / f"{name}.scores.csv"
    results = evaluate_onnx_model(
        model_path=str(model_path),
        test_dir=str(eval_dir),
        threshold=0.50,
        dump_scores_csv=str(csv_path),
    )

    pos_scores = np.array(results["tp_scores"])
    neg_scores = np.array(results["fp_scores"])

    # Trained-phrases-only (exclude "viola_wake_up" and "viola_please")
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
        "trained_dprime": round(float(compute_dprime(trained_arr, neg_scores)), 3),
        "pos_mean": round(float(pos_scores.mean()), 4),
        "neg_mean": round(float(neg_scores.mean()), 4),
        "n_pos": len(pos_scores),
        "n_neg": len(neg_scores),
    }


def print_table(results: list[dict]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON")
    print("=" * 100)
    print(
        f"{'Experiment':<20} {'Arch':<8} {'EER(all)':>10} {'EER(trn)':>10} "
        f"{'AUC':>8} {'d(trn)':>8} {'FRR@.5':>8} {'pos_m':>7} {'neg_m':>7}"
    )
    print("-" * 100)

    base_eer = None
    for r in results:
        if r["experiment"] == "BASELINE":
            base_eer = r["all_eer"]

    for r in results:
        delta = ""
        if base_eer is not None and r["experiment"] != "BASELINE":
            d = r["all_eer"] - base_eer
            delta = f" ({d:+.1%})"
        arch = r.get("architecture", "—")
        print(
            f"{r['experiment']:<20} "
            f"{arch:<8} "
            f"{r['all_eer']:>8.1%}{delta:>6} "
            f"{r['trained_eer']:>8.1%}  "
            f"{r['all_auc']:>7.4f} "
            f"{r['trained_dprime']:>7.2f} "
            f"{r['all_frr_050']:>7.1%} "
            f"{r['pos_mean']:>7.3f} "
            f"{r['neg_mean']:>7.3f}"
        )
    print()

    # Champion detection
    trained_eers = [(r["experiment"], r["all_eer"]) for r in results if r["experiment"] != "BASELINE"]
    if trained_eers:
        champion = min(trained_eers, key=lambda x: x[1])
        print(f"  CHAMPION: {champion[0]} (EER={champion[1]:.1%})")
        if base_eer:
            improvement = (base_eer - champion[1]) / base_eer
            print(f"  vs BASELINE: {improvement:+.1%} relative improvement")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ViolaWake Experiment Harness")
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract even if cache exists")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Which experiments to run (default: all non-pending)")
    parser.add_argument("--arch", default="default",
                        help="Architecture variant (default, wide, deep, narrow)")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Number of seeds for multi-seed runs")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline evaluation")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only run these exact experiment names")
    args = parser.parse_args()

    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    config = load_config()
    experiments = config["experiments"]

    # Step 1: Extract all embeddings (or load cache)
    data = extract_all_embeddings(config, force=args.force_extract)
    embs = data["embeddings"]
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    print(f"\nTotal embeddings: {len(embs)}")
    for tag in sorted(set(tags)):
        mask = tags == tag
        n = mask.sum()
        lbl = "pos" if labels[mask][0] == 1 else "neg"
        print(f"  {tag}: {n} ({lbl})")

    results = []

    # Step 2: Baseline evaluation
    if not args.skip_baseline:
        print("\n" + "=" * 60)
        print("BASELINE (current production model)")
        print("=" * 60)
        baseline = evaluate_model(BASELINE_MODEL, EVAL_DIR, "BASELINE")
        results.append(baseline)

    # Step 3: Determine which experiments to run
    if args.only:
        exp_names = args.only
    elif args.experiments:
        # Map short names (B, C, D, F) to full names
        exp_map = {}
        for name in experiments:
            short = name.split("_")[0] if "_" in name else name
            exp_map[short] = name
            exp_map[name] = name
        exp_names = [exp_map.get(e, e) for e in args.experiments]
    else:
        # Run all non-pending experiments
        exp_names = [
            name for name, defn in experiments.items()
            if defn.get("status") != "pending_data"
        ]

    # Step 4: Run experiments
    base_seed = config.get("training_defaults", {}).get("seed", 42)
    seeds = [base_seed + i for i in range(args.seeds)]

    for exp_name in exp_names:
        if exp_name not in experiments:
            print(f"\nSKIP {exp_name}: not in config")
            continue

        defn = experiments[exp_name]
        if defn.get("status") == "pending_data":
            # Check if data actually exists despite pending status
            all_tags_needed = defn.get("pos", []) + defn.get("neg", [])
            missing = [t for t in all_tags_needed if t not in set(tags)]
            if missing:
                print(f"\nSKIP {exp_name}: missing sources {missing} (status=pending_data)")
                continue
            print(f"\n  Note: {exp_name} was marked pending_data but all sources found — running!")

        all_tags_needed = defn.get("pos", []) + defn.get("neg", [])
        missing = [t for t in all_tags_needed if t not in set(tags)]
        if missing:
            print(f"\nSKIP {exp_name}: missing tags {missing}")
            continue

        # Select embeddings for this experiment
        mask = np.zeros(len(embs), dtype=bool)
        for t in all_tags_needed:
            mask |= (tags == t)

        exp_embs = embs[mask]
        exp_labels = labels[mask]
        exp_source = source_idx[mask]

        seed_results = []
        for s_idx, seed in enumerate(seeds):
            suffix = f"_s{seed}" if len(seeds) > 1 else ""
            model_name = f"{exp_name}{suffix}"
            if args.arch != "default":
                model_name = f"{exp_name}_{args.arch}{suffix}"

            model_path = EXPERIMENTS / "models" / f"{model_name}.onnx"
            train_info = train_from_cache(
                model_name, exp_embs, exp_labels, exp_source,
                model_path, config, arch_name=args.arch, seed=seed,
            )

            eval_result = evaluate_model(model_path, EVAL_DIR, model_name)
            eval_result.update(train_info)
            seed_results.append(eval_result)

        if len(seeds) > 1:
            # Multi-seed: report mean +/- std
            eers = [r["all_eer"] for r in seed_results]
            aucs = [r["all_auc"] for r in seed_results]
            dprimes = [r["all_dprime"] for r in seed_results]
            print(f"\n  {exp_name} multi-seed ({len(seeds)} runs):")
            print(f"    EER:  {np.mean(eers):.1%} +/- {np.std(eers):.1%}")
            print(f"    AUC:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")
            print(f"    d':   {np.mean(dprimes):.3f} +/- {np.std(dprimes):.3f}")
            # Use best seed for results table
            best_idx = np.argmin(eers)
            best = seed_results[best_idx]
            best["experiment"] = exp_name  # Clean name for table
            best["seed_stats"] = {
                "eer_mean": float(np.mean(eers)), "eer_std": float(np.std(eers)),
                "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                "n_seeds": len(seeds),
            }
            results.append(best)
        else:
            seed_results[0]["experiment"] = exp_name
            results.append(seed_results[0])

    # Step 5: Compare
    print_table(results)

    # Save
    out = EXPERIMENTS / "all_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out}")

    # Regression guard
    champion_name = config.get("_champion", "D_combined")
    champion_results = [r for r in results if r["experiment"] == champion_name]
    other_results = [r for r in results if r["experiment"] not in ("BASELINE", champion_name)]
    if champion_results and other_results:
        champ_eer = champion_results[0]["all_eer"]
        for r in other_results:
            if r["all_eer"] < champ_eer:
                print(f"  NEW CHAMPION CANDIDATE: {r['experiment']} "
                      f"(EER={r['all_eer']:.1%} < {champ_eer:.1%})")


if __name__ == "__main__":
    main()
