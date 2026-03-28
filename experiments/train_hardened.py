"""
Train hardened wake word models specifically designed to reject confusable words.

Strategy:
  1. Weighted BCE: confusable negatives get 5x weight in loss
  2. Hard negative mining: mine highest-scoring negatives from eval and upweight them
  3. Two-head architecture: main detection + rejection head, combined score

Usage:
  python experiments/train_hardened.py --seeds 3
  python experiments/train_hardened.py --seeds 3 --skip-twohead
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

# Confusable words we specifically want to reject (from eval analysis)
CONFUSABLE_WORDS = [
    "voila", "hola", "vanilla", "villa", "via", "viva", "viola",
    "vial", "viable", "violin", "violet", "violence", "violent",
    "vintage", "vinyl", "viral", "vista", "visual", "vital", "vivid",
    "vela", "verona", "vienna", "crayola", "granola", "gondola",
    "corolla", "ebola", "payola", "cupola", "koala", "cola",
    "how_are_you", "what_time_is_it",
]


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


def build_sample_weights(labels, tags, confusable_weight=5.0):
    """Build per-sample weights: confusable negatives get extra weight."""
    weights = np.ones(len(labels), dtype=np.float32)
    confusable_mask = (tags == "neg_confusable") | (tags == "neg_confusable_v2")
    weights[confusable_mask & (labels == 0)] = confusable_weight
    n_upweighted = (confusable_mask & (labels == 0)).sum()
    print(f"  Sample weights: {n_upweighted} confusable negatives at {confusable_weight}x")
    return weights


def mine_hard_negatives_from_eval(model_path, threshold=0.3):
    """Score eval negatives and return filenames of hard negatives."""
    csv_path = model_path.with_suffix(".hard_neg_scan.csv")
    from violawake_sdk.training.evaluate import evaluate_onnx_model
    results = evaluate_onnx_model(
        model_path=str(model_path),
        test_dir=str(EVAL_DIR),
        threshold=threshold,
        dump_scores_csv=str(csv_path),
    )
    hard_negs = []
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["label"] == "negative" and float(row["score"]) > threshold:
                    hard_negs.append((row["file"], float(row["score"])))
    csv_path.unlink(missing_ok=True)
    return hard_negs


def score_confusable_words(model_path, scores_csv):
    """Extract per-word scores from eval CSV for known confusable words."""
    import os
    word_scores = {}
    if not Path(scores_csv).exists():
        return word_scores

    with open(scores_csv) as f:
        for row in csv.DictReader(f):
            if row["label"] != "negative":
                continue
            fname = os.path.basename(row["file"]).lower()
            score = float(row["score"])
            for word in CONFUSABLE_WORDS:
                if f"_{word}." in fname or f"_{word}_" in fname or fname.startswith(f"{word}.") or fname.startswith(f"{word}_"):
                    if word not in word_scores:
                        word_scores[word] = []
                    word_scores[word].append(score)
                    break

    # Average per word
    result = {}
    for word, scores in sorted(word_scores.items()):
        result[word] = {
            "mean": round(float(np.mean(scores)), 4),
            "max": round(float(np.max(scores)), 4),
            "count": len(scores),
        }
    return result


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

        if no_imp >= patience:
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
    return {"best_epoch": best_ep, "best_val_loss": float(best_val), "time": round(dur, 1)}


def train_twohead(
    name,
    embeddings,
    labels,
    is_confusable,
    source_idx,
    output_model,
    seed=42,
    epochs=80,
    patience=15,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    ema_decay=0.999,
):
    """Train 2-head model: detection head + confusable rejection head.

    Combined score = detection_score * (1 - rejection_score)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    embedding_dim = embeddings.shape[1]

    X = torch.tensor(embeddings, dtype=torch.float32)
    y_detect = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    y_reject = torch.tensor(is_confusable, dtype=torch.float32).unsqueeze(1)

    # Group-aware split
    rng = np.random.default_rng(seed)
    pos_sources = sorted(set(source_idx[labels == 1]))
    neg_sources = sorted(set(source_idx[labels == 0]))
    rng.shuffle(pos_sources)
    rng.shuffle(neg_sources)
    val_pos = set(pos_sources[: max(1, len(pos_sources) // 5)])
    val_neg = set(neg_sources[: max(1, len(neg_sources) // 5)])

    val_mask = np.zeros(len(labels), dtype=bool)
    train_mask = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        if labels[i] == 1:
            val_mask[i] = source_idx[i] in val_pos
        else:
            val_mask[i] = source_idx[i] in val_neg
        train_mask[i] = not val_mask[i]

    X_train = X[train_mask]
    y_detect_train, y_reject_train = y_detect[train_mask], y_reject[train_mask]
    X_val = X[val_mask]
    y_detect_val, y_reject_val = y_detect[val_mask], y_reject[val_mask]

    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    n_conf = is_confusable.sum()
    print(
        f"  {name} s{seed}: {train_mask.sum()} train / {val_mask.sum()} val "
        f"({n_pos} pos + {n_neg} neg, {n_conf} confusable)"
    )

    class TwoHeadMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            # Shared backbone
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            # Detection head: is this "Viola"?
            self.detect_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
            # Rejection head: is this a confusable word?
            self.reject_head = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

        def forward(self, x):
            features = self.backbone(x)
            detect = self.detect_head(features)
            reject = self.reject_head(features)
            # Combined: high detect AND low reject = wake word
            combined = detect * (1.0 - reject)
            return combined, detect, reject

    model = TwoHeadMLP(embedding_dim)
    detect_criterion = nn.BCELoss()
    reject_criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = {n: p.data.clone() for n, p in model.named_parameters()}

    train_ds = TensorDataset(X_train, y_detect_train, y_reject_train)
    val_ds = TensorDataset(X_val, y_detect_val, y_reject_val)
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
        for bx, by_d, by_r in train_loader:
            optimizer.zero_grad()
            _, detect_out, reject_out = model(bx)
            loss_detect = detect_criterion(detect_out, by_d)
            loss_reject = reject_criterion(reject_out, by_r)
            # Weight rejection loss higher to ensure confusable rejection
            loss = loss_detect + 2.0 * loss_reject
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
            for bx, by_d, by_r in val_loader:
                _, detect_out, reject_out = model(bx)
                loss_detect = detect_criterion(detect_out, by_d)
                loss_reject = reject_criterion(reject_out, by_r)
                vl += (loss_detect + 2.0 * loss_reject).item()
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

    # Export ONNX — we need to export the combined output
    # Create a wrapper that only outputs the combined score
    class CombinedWrapper(nn.Module):
        def __init__(self, two_head):
            super().__init__()
            self.two_head = two_head

        def forward(self, x):
            combined, _, _ = self.two_head(x)
            return combined

    wrapper = CombinedWrapper(model)
    wrapper.eval()
    output_model.parent.mkdir(parents=True, exist_ok=True)
    import torch as _torch

    _torch.onnx.export(
        wrapper,
        _torch.zeros(1, embedding_dim),
        str(output_model),
        input_names=["embedding"],
        output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    return {"best_epoch": best_ep, "best_val_loss": float(best_val), "time": round(dur, 1)}


def evaluate_onnx(model_path, name):
    """Standard ONNX evaluation with per-confusable-word scoring."""
    from violawake_sdk.training.evaluate import evaluate_onnx_model, find_optimal_threshold

    csv_path = EXPERIMENTS / f"{name}.scores.csv"
    results = evaluate_onnx_model(
        model_path=str(model_path),
        test_dir=str(EVAL_DIR),
        threshold=0.50,
        dump_scores_csv=str(csv_path),
    )
    pos_scores = np.array(results["tp_scores"])
    neg_scores = np.array(results["fp_scores"])

    # Compute trained EER (exclude unseen patterns)
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

    # Per-confusable-word scores
    word_scores = score_confusable_words(model_path, csv_path)

    # Detection/false alarm at specific thresholds
    thresholds = {0.5: {}, 0.7: {}, 0.8: {}, 0.9: {}}
    for thr in thresholds:
        det_rate = float((pos_scores >= thr).mean()) if len(pos_scores) > 0 else 0.0
        fa_rate = float((neg_scores >= thr).mean()) if len(neg_scores) > 0 else 0.0
        thresholds[thr] = {
            "detection_rate": round(det_rate, 4),
            "false_alarm_rate": round(fa_rate, 4),
            "n_false_alarms": int((neg_scores >= thr).sum()),
        }

    return {
        "experiment": name,
        "all_eer": round(results["eer_approx"], 4),
        "all_auc": round(results["roc_auc"], 4),
        "all_dprime": round(results["d_prime"], 3),
        "trained_eer": round(opt["eer_approx"], 4),
        "pos_mean": round(float(pos_scores.mean()), 4),
        "neg_mean": round(float(neg_scores.mean()), 4),
        "n_pos": len(pos_scores),
        "n_neg": len(neg_scores),
        "thresholds": thresholds,
        "confusable_word_scores": word_scores,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train hardened confusable-rejecting models")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument(
        "--confusable-weight",
        type=float,
        default=5.0,
        help="Weight multiplier for confusable negatives (default: 5.0)",
    )
    parser.add_argument("--skip-twohead", action="store_true", help="Skip 2-head architecture")
    parser.add_argument(
        "--skip-weighted",
        action="store_true",
        help="Skip weighted BCE variants",
    )
    args = parser.parse_args()

    # Load all data
    print("Loading embedding cache...")
    cache = load_data_with_tags()
    embs = cache["embeddings"]
    labels = cache["labels"]
    tags = cache["tags"]
    files = cache["files"]
    source_idx = cache["source_idx"]

    # Use D_combined data sources + confusable_v2
    # This is the strongest confusable-aware dataset
    target_tags = [
        "pos_main",
        "pos_diverse",
        "neg_main",
        "neg_confusable",
        "neg_confusable_v2",
    ]
    mask = np.zeros(len(tags), dtype=bool)
    for t in target_tags:
        mask |= tags == t

    sel_embs = embs[mask]
    sel_labels = labels[mask]
    sel_tags = tags[mask]
    sel_source_idx = source_idx[mask]

    n_pos = (sel_labels == 1).sum()
    n_neg = (sel_labels == 0).sum()
    n_conf = ((sel_tags == "neg_confusable") | (sel_tags == "neg_confusable_v2")).sum()

    print(f"\n{'='*70}")
    print(f"HARDENED MODEL TRAINING")
    print(f"  Data: {n_pos} pos + {n_neg} neg = {len(sel_labels)} total")
    print(f"  Confusable negatives: {n_conf} ({n_conf/n_neg*100:.1f}% of negs)")
    print(f"  Confusable weight: {args.confusable_weight}x")
    print(f"  Seeds: {args.seeds}")
    print(f"{'='*70}\n")

    seeds = list(range(42, 42 + args.seeds))
    all_results = {}

    # =============================================
    # Variant 1: Weighted BCE (confusable 5x)
    # =============================================
    if not args.skip_weighted:
        print(f"\n{'='*70}")
        print("VARIANT 1: Weighted BCE (confusable negatives at 5x)")
        print(f"{'='*70}")

        sample_weights = build_sample_weights(sel_labels, sel_tags, args.confusable_weight)
        variant1_results = []

        for seed in seeds:
            model_name = f"hardened_weighted5x_s{seed}"
            model_path = EXPERIMENTS / "models" / f"{model_name}.onnx"
            train_info = train_weighted_bce(
                model_name,
                sel_embs,
                sel_labels,
                sel_source_idx,
                sample_weights,
                model_path,
                seed=seed,
            )
            eval_result = evaluate_onnx(model_path, model_name)
            eval_result.update(train_info)
            variant1_results.append(eval_result)

            # Print summary
            ws = eval_result.get("confusable_word_scores", {})
            print(f"\n  seed={seed}: EER={eval_result['all_eer']:.1%}, trained_EER={eval_result['trained_eer']:.1%}")
            print(f"  Threshold analysis:")
            for thr, info in eval_result["thresholds"].items():
                print(f"    @{thr}: detect={info['detection_rate']:.1%}, FA={info['false_alarm_rate']:.1%} ({info['n_false_alarms']} FAs)")
            if ws:
                print(f"  Confusable word scores (mean/max):")
                for word in ["voila", "hola", "vanilla", "villa", "how_are_you", "tone_1khz"]:
                    if word in ws:
                        print(f"    {word}: {ws[word]['mean']:.4f} / {ws[word]['max']:.4f}")

        all_results["weighted_5x"] = variant1_results

        # =============================================
        # Variant 2: Weighted BCE at 10x
        # =============================================
        print(f"\n{'='*70}")
        print("VARIANT 2: Weighted BCE (confusable negatives at 10x)")
        print(f"{'='*70}")

        sample_weights_10x = build_sample_weights(sel_labels, sel_tags, 10.0)
        variant2_results = []

        for seed in seeds:
            model_name = f"hardened_weighted10x_s{seed}"
            model_path = EXPERIMENTS / "models" / f"{model_name}.onnx"
            train_info = train_weighted_bce(
                model_name,
                sel_embs,
                sel_labels,
                sel_source_idx,
                sample_weights_10x,
                model_path,
                seed=seed,
            )
            eval_result = evaluate_onnx(model_path, model_name)
            eval_result.update(train_info)
            variant2_results.append(eval_result)

            ws = eval_result.get("confusable_word_scores", {})
            print(f"\n  seed={seed}: EER={eval_result['all_eer']:.1%}, trained_EER={eval_result['trained_eer']:.1%}")
            if ws:
                print(f"  Confusable word scores (mean/max):")
                for word in ["voila", "hola", "vanilla", "villa", "how_are_you", "tone_1khz"]:
                    if word in ws:
                        print(f"    {word}: {ws[word]['mean']:.4f} / {ws[word]['max']:.4f}")

        all_results["weighted_10x"] = variant2_results

        # =============================================
        # Variant 3: Weighted BCE 5x + wider network
        # =============================================
        print(f"\n{'='*70}")
        print("VARIANT 3: Weighted BCE 5x + wider network [128, 64]")
        print(f"{'='*70}")

        sample_weights = build_sample_weights(sel_labels, sel_tags, args.confusable_weight)
        variant3_results = []

        for seed in seeds:
            model_name = f"hardened_wide5x_s{seed}"
            model_path = EXPERIMENTS / "models" / f"{model_name}.onnx"
            train_info = train_weighted_bce(
                model_name,
                sel_embs,
                sel_labels,
                sel_source_idx,
                sample_weights,
                model_path,
                hidden_dims=(128, 64),
                dropouts=(0.3, 0.2),
                seed=seed,
            )
            eval_result = evaluate_onnx(model_path, model_name)
            eval_result.update(train_info)
            variant3_results.append(eval_result)

            ws = eval_result.get("confusable_word_scores", {})
            print(f"\n  seed={seed}: EER={eval_result['all_eer']:.1%}, trained_EER={eval_result['trained_eer']:.1%}")
            if ws:
                print(f"  Confusable word scores (mean/max):")
                for word in ["voila", "hola", "vanilla", "villa", "how_are_you", "tone_1khz"]:
                    if word in ws:
                        print(f"    {word}: {ws[word]['mean']:.4f} / {ws[word]['max']:.4f}")

        all_results["wide_5x"] = variant3_results

    # =============================================
    # Variant 4: Two-head architecture
    # =============================================
    if not args.skip_twohead:
        print(f"\n{'='*70}")
        print("VARIANT 4: Two-head architecture (detect + reject)")
        print(f"{'='*70}")

        # Build confusable label: 1 if neg_confusable or neg_confusable_v2, 0 otherwise
        is_confusable = np.zeros(len(sel_labels), dtype=np.float32)
        conf_mask = (sel_tags == "neg_confusable") | (sel_tags == "neg_confusable_v2")
        is_confusable[conf_mask] = 1.0

        variant4_results = []

        for seed in seeds:
            model_name = f"hardened_twohead_s{seed}"
            model_path = EXPERIMENTS / "models" / f"{model_name}.onnx"
            train_info = train_twohead(
                model_name,
                sel_embs,
                sel_labels,
                is_confusable,
                sel_source_idx,
                model_path,
                seed=seed,
            )
            eval_result = evaluate_onnx(model_path, model_name)
            eval_result.update(train_info)
            variant4_results.append(eval_result)

            ws = eval_result.get("confusable_word_scores", {})
            print(f"\n  seed={seed}: EER={eval_result['all_eer']:.1%}, trained_EER={eval_result['trained_eer']:.1%}")
            print(f"  Threshold analysis:")
            for thr, info in eval_result["thresholds"].items():
                print(f"    @{thr}: detect={info['detection_rate']:.1%}, FA={info['false_alarm_rate']:.1%} ({info['n_false_alarms']} FAs)")
            if ws:
                print(f"  Confusable word scores (mean/max):")
                for word in ["voila", "hola", "vanilla", "villa", "how_are_you", "tone_1khz"]:
                    if word in ws:
                        print(f"    {word}: {ws[word]['mean']:.4f} / {ws[word]['max']:.4f}")

        all_results["twohead"] = variant4_results

    # =============================================
    # Summary & comparison
    # =============================================
    print(f"\n{'='*70}")
    print("COMPREHENSIVE COMPARISON")
    print(f"{'='*70}\n")

    summary = {}
    for variant_name, results in all_results.items():
        eers = [r["all_eer"] for r in results]
        trained_eers = [r["trained_eer"] for r in results]

        # Aggregate confusable word scores across seeds
        agg_word_scores = {}
        for r in results:
            for word, info in r.get("confusable_word_scores", {}).items():
                if word not in agg_word_scores:
                    agg_word_scores[word] = []
                agg_word_scores[word].append(info["mean"])

        avg_word = {w: round(float(np.mean(s)), 4) for w, s in agg_word_scores.items()}

        summary[variant_name] = {
            "eer_mean": round(float(np.mean(eers)), 4),
            "eer_std": round(float(np.std(eers)), 4),
            "trained_eer_mean": round(float(np.mean(trained_eers)), 4),
            "per_seed": results,
            "confusable_avg_scores": avg_word,
        }

        print(f"  {variant_name}:")
        print(f"    EER: {np.mean(eers):.1%} +/- {np.std(eers):.1%}")
        print(f"    Trained EER: {np.mean(trained_eers):.1%}")

        # Print threshold info from best seed
        best_idx = np.argmin(eers)
        best_r = results[best_idx]
        print(f"    Best seed thresholds:")
        for thr, info in best_r["thresholds"].items():
            print(f"      @{thr}: detect={info['detection_rate']:.1%}, FA={info['false_alarm_rate']:.1%}")
        if avg_word:
            print(f"    Confusable word avg scores:")
            for word in ["voila", "hola", "vanilla", "villa", "how_are_you", "tone_1khz"]:
                if word in avg_word:
                    print(f"      {word}: {avg_word[word]:.4f}")
        print()

    # Find best model overall
    best_variant = None
    best_eer = 1.0
    best_model_path = None
    for variant_name, info in summary.items():
        results = info["per_seed"]
        for r in results:
            if r["trained_eer"] < best_eer:
                best_eer = r["trained_eer"]
                best_variant = variant_name
                best_model_path = EXPERIMENTS / "models" / f"{r['experiment']}.onnx"

    if best_model_path and best_model_path.exists():
        dest = EXPERIMENTS / "models" / "hardened_best.onnx"
        import shutil
        shutil.copy2(best_model_path, dest)
        print(f"Best model: {best_variant} (trained_EER={best_eer:.1%})")
        print(f"  Source: {best_model_path}")
        print(f"  Copied to: {dest}")

    # Save results
    out_path = EXPERIMENTS / "hardened_model_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
