"""
ViolaWake FAPH Test — 864-dim Temporal Concatenation Model
==========================================================

Same as faph_test.py but uses concatenated 9x96=864-dim embeddings
instead of mean-pooled 96-dim. This tests whether temporal information
reduces false alarms.

Step 1: Train concat_864_wide model and export ONNX
Step 2: Run full sliding-window FAPH on test-clean

Usage:
    python experiments/faph_test_864.py
    python experiments/faph_test_864.py --skip-train
    python experiments/faph_test_864.py --corpus dev-clean
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = WAKEWORD_ROOT / "experiments"
MODEL_DIR = EXPERIMENTS / "models"
CACHE_TEMPORAL = EXPERIMENTS / "embedding_cache_temporal.npz"
LIBRISPEECH_BASE = WAKEWORD_ROOT / "corpus" / "librispeech" / "LibriSpeech"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000          # 1.5s at 16kHz
STEP_SAMPLES = 1600           # 100ms step
DEBOUNCE_SECONDS = 2.0
DEBOUNCE_SAMPLES = int(DEBOUNCE_SECONDS * SAMPLE_RATE)
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N_FRAMES = 9
EMB_DIM = 96
CONCAT_DIM = N_FRAMES * EMB_DIM  # 864


# ---------------------------------------------------------------------------
# Step 1: Train and export 864-dim model
# ---------------------------------------------------------------------------

def train_and_export(seed: int = 42) -> Path:
    """Train concat_864_wide model and export to ONNX. Returns model path."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    model_path = MODEL_DIR / f"concat_864_wide_s{seed}.onnx"
    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return model_path

    print("Loading temporal cache...")
    data = np.load(CACHE_TEMPORAL, allow_pickle=True)
    concat = data["concat"]       # (N, 864)
    labels = data["labels"]
    tags = data["tags"]
    source_idx = data["source_idx"]

    # Filter to training tags (same as original experiment)
    target_tags = ["pos_main", "pos_diverse", "neg_main", "neg_confusable", "neg_confusable_v2"]
    mask = np.zeros(len(tags), dtype=bool)
    for t in target_tags:
        mask |= (tags == t)

    concat = concat[mask]
    labels = labels[mask]
    tags = tags[mask]
    source_idx = source_idx[mask]

    print(f"Training data: {len(labels)} samples ({(labels==1).sum()} pos + {(labels==0).sum()} neg)")

    # Weights: confusable negatives get 5x
    weights = np.ones(len(labels), dtype=np.float32)
    conf_mask = (tags == "neg_confusable") | (tags == "neg_confusable_v2")
    weights[conf_mask & (labels == 0)] = 5.0

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = torch.tensor(concat, dtype=torch.float32)
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
    val_pos = set(pos_sources[:max(1, len(pos_sources) // 5)])
    val_neg = set(neg_sources[:max(1, len(neg_sources) // 5)])

    val_mask_arr = np.array([
        (source_idx[i] in val_pos if labels[i] == 1 else source_idx[i] in val_neg)
        for i in range(len(labels))
    ])
    train_mask_arr = ~val_mask_arr

    X_train, y_train, w_train = X[train_mask_arr], y[train_mask_arr], w[train_mask_arr]
    X_val, y_val, w_val = X[val_mask_arr], y[val_mask_arr], w[val_mask_arr]
    print(f"Split: {train_mask_arr.sum()} train / {val_mask_arr.sum()} val")

    # concat_864_wide architecture: (256, 128, 64) with dropouts (0.3, 0.2, 0.1)
    model = nn.Sequential(
        nn.Linear(864, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(64, 1), nn.Sigmoid(),
    )

    criterion = nn.BCELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    train_ds = TensorDataset(X_train, y_train, w_train)
    val_ds = TensorDataset(X_val, y_val, w_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_val = float("inf")
    best_ep = 0
    no_imp = 0
    best_state = None
    t0 = time.monotonic()

    for ep in range(1, 81):
        model.train()
        for bx, by, bw in train_loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = (criterion(pred, by) * bw).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
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
        else:
            no_imp += 1

        if ep % 10 == 0 or no_imp >= 15:
            print(f"  Epoch {ep}: val={avg_v:.6f} (best={best_val:.6f} @ep{best_ep})")
        if no_imp >= 15:
            print(f"  Early stop at epoch {ep}")
            break

    if best_state:
        model.load_state_dict(best_state)

    dur = time.monotonic() - t0
    print(f"Training done in {dur:.1f}s, best epoch {best_ep}")

    # Export ONNX
    import torch as _torch
    model.eval()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    _torch.onnx.export(
        model, _torch.zeros(1, 864), str(model_path),
        input_names=["embedding"], output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )
    print(f"Exported: {model_path}")
    return model_path


# ---------------------------------------------------------------------------
# Step 2: FAPH test with 864-dim concat embeddings
# ---------------------------------------------------------------------------

def init_oww_preprocessor():
    from openwakeword.utils import AudioFeatures
    return AudioFeatures()


def extract_embedding_864(preprocessor, audio_int16: np.ndarray) -> np.ndarray:
    """Extract 864-dim concatenated embedding (9 frames x 96 dims)."""
    embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    # embeddings shape: (1, T, 96)
    temporal = embeddings[0]  # (T, 96)
    if temporal.shape[0] < N_FRAMES:
        pad = np.zeros((N_FRAMES - temporal.shape[0], EMB_DIM), dtype=np.float32)
        temporal = np.concatenate([temporal, pad], axis=0)
    elif temporal.shape[0] > N_FRAMES:
        temporal = temporal[:N_FRAMES]
    return temporal.reshape(-1).astype(np.float32)  # (864,)


def load_audio_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def run_faph_864(
    model_path: Path,
    librispeech_dir: Path,
    thresholds: list[float] | None = None,
    top_k: int = 50,
) -> dict:
    if thresholds is None:
        thresholds = THRESHOLDS

    print("=" * 60)
    print("ViolaWake FAPH Test — 864-dim Temporal Concat")
    print("=" * 60)

    flac_files = sorted(librispeech_dir.rglob("*.flac"))
    if not flac_files:
        raise FileNotFoundError(f"No .flac files in {librispeech_dir}")
    print(f"Found {len(flac_files)} .flac files")

    print(f"Loading model: {model_path.name}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    inp_shape = session.get_inputs()[0].shape
    print(f"Model input: {inp_name} shape={inp_shape}")

    print("Initializing OWW preprocessor...")
    preprocessor = init_oww_preprocessor()

    total_windows = 0
    total_audio_samples = 0
    trigger_counts_raw = {t: 0 for t in thresholds}
    trigger_counts_debounced = {t: 0 for t in thresholds}
    last_trigger_pos = {t: -999999 for t in thresholds}
    top_scores: list[tuple[float, str, float]] = []
    min_top_score = 0.0

    t_start = time.time()
    files_processed = 0

    for i, fpath in enumerate(flac_files):
        try:
            audio = load_audio_16k(fpath)
        except Exception as e:
            print(f"  WARN: Failed to load {fpath.name}: {e}")
            continue

        n_samples = len(audio)
        total_audio_samples += n_samples

        for t in thresholds:
            last_trigger_pos[t] = -999999

        pos = 0
        while pos + CLIP_SAMPLES <= n_samples:
            window = audio[pos : pos + CLIP_SAMPLES]
            window_int16 = float_to_int16(window)

            embedding = extract_embedding_864(preprocessor, window_int16)
            result = session.run([out_name], {inp_name: embedding.reshape(1, -1)})
            score = float(result[0][0][0])
            total_windows += 1

            for t in thresholds:
                if score >= t:
                    trigger_counts_raw[t] += 1
                    if pos - last_trigger_pos[t] >= DEBOUNCE_SAMPLES:
                        trigger_counts_debounced[t] += 1
                        last_trigger_pos[t] = pos

            if score > min_top_score or len(top_scores) < top_k:
                window_start_sec = pos / SAMPLE_RATE
                rel_path = str(fpath.relative_to(librispeech_dir))
                top_scores.append((score, rel_path, window_start_sec))
                top_scores.sort(key=lambda x: x[0], reverse=True)
                if len(top_scores) > top_k:
                    top_scores = top_scores[:top_k]
                min_top_score = top_scores[-1][0] if top_scores else 0.0

            pos += STEP_SAMPLES

        files_processed += 1
        if files_processed % 100 == 0:
            elapsed = time.time() - t_start
            hours_processed = total_audio_samples / SAMPLE_RATE / 3600
            print(
                f"  [{files_processed}/{len(flac_files)}] "
                f"{hours_processed:.2f}h audio, "
                f"{total_windows} windows, "
                f"elapsed {elapsed:.0f}s"
            )

    elapsed = time.time() - t_start
    total_hours = total_audio_samples / SAMPLE_RATE / 3600

    print()
    print("=" * 60)
    print("RESULTS — 864-dim Temporal Concat")
    print("=" * 60)
    print(f"Total audio:    {total_hours:.3f} hours")
    print(f"Total windows:  {total_windows}")
    print(f"Total files:    {files_processed}")
    print(f"Elapsed time:   {elapsed:.1f}s")
    print()

    faph_results = {}
    print(f"{'Threshold':<12} {'Raw Trig':<12} {'Raw FAPH':<12} {'Debounced':<12} {'Deb FAPH':<12}")
    print("-" * 60)
    for t in thresholds:
        faph_raw = trigger_counts_raw[t] / total_hours if total_hours > 0 else 0
        faph_deb = trigger_counts_debounced[t] / total_hours if total_hours > 0 else 0
        faph_results[str(t)] = {
            "threshold": t,
            "triggers_raw": trigger_counts_raw[t],
            "faph_raw": round(faph_raw, 2),
            "triggers_debounced": trigger_counts_debounced[t],
            "faph_debounced": round(faph_deb, 2),
        }
        print(f"{t:<12.2f} {trigger_counts_raw[t]:<12} {faph_raw:<12.2f} {trigger_counts_debounced[t]:<12} {faph_deb:<12.2f}")

    print()
    print(f"TOP {top_k} highest-scoring windows:")
    print(f"{'Rank':<6} {'Score':<10} {'File':<50} {'Time (s)':<10}")
    print("-" * 76)
    top_scores_list = []
    for rank, (score, fpath_rel, t_sec) in enumerate(top_scores, 1):
        print(f"{rank:<6} {score:<10.4f} {fpath_rel:<50} {t_sec:<10.1f}")
        top_scores_list.append({
            "rank": rank,
            "score": round(score, 6),
            "file": fpath_rel,
            "time_sec": round(t_sec, 2),
        })

    return {
        "model": model_path.name,
        "embedding_type": "temporal_concat_864",
        "architecture": "864→256→128→64→1",
        "dataset": f"LibriSpeech {librispeech_dir.name}",
        "total_hours": round(total_hours, 4),
        "total_windows": total_windows,
        "total_files": files_processed,
        "elapsed_seconds": round(elapsed, 1),
        "clip_samples": CLIP_SAMPLES,
        "step_samples": STEP_SAMPLES,
        "pooling": "concat (9x96=864)",
        "debounce_seconds": DEBOUNCE_SECONDS,
        "faph": faph_results,
        "top_scores": top_scores_list,
    }


def main():
    parser = argparse.ArgumentParser(description="FAPH test with 864-dim temporal concat model")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing model")
    parser.add_argument("--corpus", default="test-clean", choices=["test-clean", "dev-clean"],
                        help="LibriSpeech corpus to test on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    model_path = MODEL_DIR / f"concat_864_wide_s{args.seed}.onnx"

    if not args.skip_train:
        print("=" * 60)
        print("STEP 1: Training 864-dim wide model")
        print("=" * 60)
        model_path = train_and_export(seed=args.seed)
    else:
        if not model_path.exists():
            print(f"ERROR: Model not found: {model_path}")
            print("Run without --skip-train first.")
            return

    print()
    print("=" * 60)
    print(f"STEP 2: FAPH test on {args.corpus}")
    print("=" * 60)

    librispeech_dir = LIBRISPEECH_BASE / args.corpus
    if not librispeech_dir.exists():
        print(f"ERROR: LibriSpeech dir not found: {librispeech_dir}")
        return

    results = run_faph_864(model_path=model_path, librispeech_dir=librispeech_dir)

    if args.output is None:
        args.output = EXPERIMENTS / f"faph_864_{args.corpus.replace('-','_')}_s{args.seed}.json"

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
