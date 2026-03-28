"""
Unified FAPH Comparison: MLP vs Temporal Models
=================================================

Fair head-to-head streaming FAPH comparison on LibriSpeech test-clean (~5.4h).

MLP models: take mean-pooled 96-dim OWW embedding
Temporal models: take (batch, 9, 96) sequence of OWW embedding frames
concat_864: takes (batch, 864) = 9 frames concatenated

All models evaluated with identical:
  - 1.5s sliding window, 100ms step
  - 2.0s debounce
  - Multiwindow confirmation: 1-of-1, 2-of-2, 3-of-3
  - Thresholds: 0.50, 0.60, 0.70, 0.80, 0.90, 0.95
  - LibriSpeech test-clean corpus (2620 files, ~5.4h)
  - TP eval on eval_fresh positives (60 files)

Usage:
    python experiments/faph_unified_comparison.py
    python experiments/faph_unified_comparison.py --max-files 100   # quick spot check
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# ---------------------------------------------------------------------------
WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
LIBRISPEECH_DIR = WAKEWORD_ROOT / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"
POS_DIR = WAKEWORD_ROOT / "experiments" / "eval_fresh" / "positives"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000       # 1.5s
STEP_SAMPLES = 1600        # 100ms
DEBOUNCE_SECONDS = 2.0
DEBOUNCE_SAMPLES = int(DEBOUNCE_SECONDS * SAMPLE_RATE)

THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
MW_CONFIGS = [
    ("1-of-1", 1, 1),
    ("2-of-2", 2, 2),
    ("3-of-3", 3, 3),
]

# Models to compare
MODELS = {
    "r3_10x_s42 [PROD]": {
        "path": WAKEWORD_ROOT / "experiments" / "models" / "r3_10x_s42.onnx",
        "type": "mlp",  # mean-pooled 96-dim input
    },
    "j5_baseline_mlp": {
        "path": WAKEWORD_ROOT / "experiments" / "models" / "j5_temporal" / "baseline_mlp.onnx",
        "type": "mlp",
    },
    "j5_temporal_cnn": {
        "path": WAKEWORD_ROOT / "experiments" / "models" / "j5_temporal" / "temporal_cnn.onnx",
        "type": "temporal",  # (batch, 9, 96) input
    },
    "j5_temporal_convgru": {
        "path": WAKEWORD_ROOT / "experiments" / "models" / "j5_temporal" / "temporal_convgru.onnx",
        "type": "temporal",
    },
    "j5_temporal_gru": {
        "path": WAKEWORD_ROOT / "experiments" / "models" / "j5_temporal" / "temporal_gru.onnx",
        "type": "temporal",
    },
    "concat_864_wide_s42": {
        "path": WAKEWORD_ROOT / "experiments" / "models" / "concat_864_wide_s42.onnx",
        "type": "concat",  # (batch, 864) = 9 * 96 concatenated
    },
}


class MultiWindowDetector:
    """N-of-M consecutive windows above threshold with debounce."""

    def __init__(self, n_required: int, window_size: int, debounce_samples: int):
        self.n_required = n_required
        self.window_size = window_size
        self.debounce_samples = debounce_samples
        self.triggers_debounced = 0
        self.history: deque[bool] = deque(maxlen=self.window_size)
        self.last_trigger_pos = -999999

    def reset_file(self):
        """Reset per-file state."""
        self.history.clear()
        self.last_trigger_pos = -999999

    def update(self, above_threshold: bool, pos: int) -> bool:
        self.history.append(above_threshold)
        if len(self.history) == self.window_size:
            if sum(self.history) >= self.n_required:
                if pos - self.last_trigger_pos >= self.debounce_samples:
                    self.triggers_debounced += 1
                    self.last_trigger_pos = pos
                    return True
        return False


def init_oww_preprocessor():
    from openwakeword.utils import AudioFeatures
    return AudioFeatures()


def extract_embedding_frames(preprocessor, audio_int16: np.ndarray) -> np.ndarray:
    """Extract per-frame OWW embeddings from a 1.5s int16 clip.

    Returns shape (num_frames, 96). For 1.5s at 16kHz, typically 9 frames
    (depending on OWW's internal mel stride).
    """
    embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    # embeddings shape: (1, num_frames, 96)
    return embeddings[0].astype(np.float32)  # (num_frames, 96)


def score_mlp(session: ort.InferenceSession, frames: np.ndarray) -> float:
    """Score mean-pooled embedding through MLP. frames: (N, 96)."""
    mean_emb = frames.mean(axis=0).reshape(1, -1).astype(np.float32)
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    result = session.run([out_name], {inp_name: mean_emb})
    return float(result[0][0][0])


def score_temporal(session: ort.InferenceSession, frames: np.ndarray, seq_len: int = 9) -> float:
    """Score temporal model with (batch, seq_len, 96) input.

    If fewer frames than seq_len, pad with zeros at the start.
    If more, take the last seq_len frames (most recent context).
    """
    n_frames = frames.shape[0]
    if n_frames == 0:
        return 0.0

    if n_frames < seq_len:
        # Pad at start with zeros
        pad = np.zeros((seq_len - n_frames, 96), dtype=np.float32)
        seq = np.concatenate([pad, frames], axis=0)
    elif n_frames > seq_len:
        # Take last seq_len frames
        seq = frames[-seq_len:]
    else:
        seq = frames

    inp = seq.reshape(1, seq_len, 96).astype(np.float32)
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    result = session.run([out_name], {inp_name: inp})
    return float(result[0][0][0])


def score_concat(session: ort.InferenceSession, frames: np.ndarray, seq_len: int = 9) -> float:
    """Score concat model with (batch, 864) = 9*96 concatenated input."""
    n_frames = frames.shape[0]
    if n_frames == 0:
        return 0.0

    if n_frames < seq_len:
        pad = np.zeros((seq_len - n_frames, 96), dtype=np.float32)
        seq = np.concatenate([pad, frames], axis=0)
    elif n_frames > seq_len:
        seq = frames[-seq_len:]
    else:
        seq = frames

    concat = seq.flatten().reshape(1, -1).astype(np.float32)
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    result = session.run([out_name], {inp_name: concat})
    return float(result[0][0][0])


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def load_audio_16k(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def score_clip(model_type: str, session: ort.InferenceSession, frames: np.ndarray) -> float:
    if model_type == "mlp":
        return score_mlp(session, frames)
    elif model_type == "temporal":
        return score_temporal(session, frames)
    elif model_type == "concat":
        return score_concat(session, frames)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_faph_single_model(
    model_name: str,
    model_type: str,
    session: ort.InferenceSession,
    preprocessor,
    flac_files: list[Path],
    librispeech_dir: Path,
) -> dict:
    """Run streaming FAPH for one model. Returns results dict."""

    # Create detectors for each (mw_config, threshold) combo
    detectors: dict[tuple[str, float], MultiWindowDetector] = {}
    for cfg_name, n_req, win_size in MW_CONFIGS:
        for t in THRESHOLDS:
            detectors[(cfg_name, t)] = MultiWindowDetector(n_req, win_size, DEBOUNCE_SAMPLES)

    total_windows = 0
    total_audio_samples = 0
    total_infer_ns = 0
    t_start = time.time()

    for fi, fpath in enumerate(flac_files):
        try:
            audio, sr = sf.read(fpath, dtype="float32")
        except Exception:
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        n_samples = len(audio)
        total_audio_samples += n_samples

        # Reset detectors for new file
        for det in detectors.values():
            det.reset_file()

        pos = 0
        while pos + CLIP_SAMPLES <= n_samples:
            window = audio[pos:pos + CLIP_SAMPLES]
            window_int16 = float_to_int16(window)

            # Extract per-frame embeddings (shared across all model types)
            frames = extract_embedding_frames(preprocessor, window_int16)

            # Score through model
            t0 = time.perf_counter_ns()
            score = score_clip(model_type, session, frames)
            total_infer_ns += time.perf_counter_ns() - t0

            total_windows += 1

            # Update all multiwindow detectors
            for t in THRESHOLDS:
                above = score >= t
                for cfg_name, n_req, win_size in MW_CONFIGS:
                    detectors[(cfg_name, t)].update(above, pos)

            pos += STEP_SAMPLES

        if (fi + 1) % 200 == 0:
            elapsed = time.time() - t_start
            hours = total_audio_samples / SAMPLE_RATE / 3600
            print(f"    [{fi+1}/{len(flac_files)}] {hours:.2f}h, {total_windows} win, {elapsed:.0f}s")

    elapsed = time.time() - t_start
    total_hours = total_audio_samples / SAMPLE_RATE / 3600
    avg_infer_us = (total_infer_ns / total_windows / 1000) if total_windows > 0 else 0

    # Collect results
    results = {
        "model": model_name,
        "model_type": model_type,
        "total_hours": round(total_hours, 4),
        "total_windows": total_windows,
        "elapsed_seconds": round(elapsed, 1),
        "avg_infer_us": round(avg_infer_us, 1),
        "faph": {},
    }

    for cfg_name, n_req, win_size in MW_CONFIGS:
        for t in THRESHOLDS:
            det = detectors[(cfg_name, t)]
            faph = det.triggers_debounced / total_hours if total_hours > 0 else 0
            key = f"{cfg_name}@{t}"
            results["faph"][key] = {
                "config": cfg_name,
                "threshold": t,
                "triggers": det.triggers_debounced,
                "faph": round(faph, 2),
            }

    return results


def run_tp_single_model(
    model_name: str,
    model_type: str,
    session: ort.InferenceSession,
    preprocessor,
    pos_files: list[Path],
) -> dict:
    """Run TP eval for one model on positive audio files."""

    tp_results = {}
    for cfg_name, n_req, win_size in MW_CONFIGS:
        tp_results[cfg_name] = {t: 0 for t in THRESHOLDS}

    max_scores = []

    for pf in pos_files:
        try:
            audio = load_audio_16k(pf)
        except Exception:
            continue

        n_samp = len(audio)
        if n_samp < CLIP_SAMPLES:
            audio = np.pad(audio, (0, CLIP_SAMPLES - n_samp))
            n_samp = len(audio)

        # Create fresh detectors per file
        detectors: dict[tuple[str, float], MultiWindowDetector] = {}
        for cfg_name, n_req, win_size in MW_CONFIGS:
            for t in THRESHOLDS:
                detectors[(cfg_name, t)] = MultiWindowDetector(n_req, win_size, DEBOUNCE_SAMPLES)

        file_scores = []
        pos = 0
        while pos + CLIP_SAMPLES <= n_samp:
            window = audio[pos:pos + CLIP_SAMPLES]
            window_int16 = float_to_int16(window)
            frames = extract_embedding_frames(preprocessor, window_int16)
            sc = score_clip(model_type, session, frames)
            file_scores.append(sc)

            for t in THRESHOLDS:
                above = sc >= t
                for cfg_name, n_req, win_size in MW_CONFIGS:
                    detectors[(cfg_name, t)].update(above, pos)

            pos += STEP_SAMPLES

        if file_scores:
            max_scores.append((pf.name, max(file_scores)))

        # Check if any detector triggered
        for cfg_name, n_req, win_size in MW_CONFIGS:
            for t in THRESHOLDS:
                if detectors[(cfg_name, t)].triggers_debounced > 0:
                    tp_results[cfg_name][t] += 1

    n_files = len(max_scores)
    result = {"n_files": n_files, "configs": {}}
    for cfg_name, _, _ in MW_CONFIGS:
        result["configs"][cfg_name] = {}
        for t in THRESHOLDS:
            d = tp_results[cfg_name][t]
            rate = d / n_files * 100 if n_files > 0 else 0
            result["configs"][cfg_name][str(t)] = {
                "detected": d,
                "rate_pct": round(rate, 1),
            }

    return result


def main():
    parser = argparse.ArgumentParser(description="Unified FAPH model comparison")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit number of LibriSpeech files (0 = all)")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "faph_unified_results.json")
    args = parser.parse_args()

    print("=" * 80)
    print("ViolaWake Unified FAPH Comparison")
    print("=" * 80)

    # Verify corpus
    flac_files = sorted(LIBRISPEECH_DIR.rglob("*.flac"))
    if not flac_files:
        print(f"ERROR: No .flac files found in {LIBRISPEECH_DIR}")
        return
    if args.max_files > 0:
        flac_files = flac_files[:args.max_files]
    print(f"Corpus: {len(flac_files)} files from LibriSpeech test-clean")

    # Find positive eval files
    pos_files = sorted(POS_DIR.glob("*.wav")) if POS_DIR.exists() else []
    print(f"Positive eval: {len(pos_files)} files from {POS_DIR.name}")

    # Init preprocessor (shared across all models)
    print("Initializing OWW preprocessor...")
    preprocessor = init_oww_preprocessor()

    # Load all models
    print("Loading models...")
    sessions = {}
    for name, info in MODELS.items():
        if info["path"].exists():
            sessions[name] = ort.InferenceSession(
                str(info["path"]), providers=["CPUExecutionProvider"]
            )
            print(f"  Loaded: {name} ({info['type']}, {info['path'].name})")
        else:
            print(f"  SKIP: {name} (file not found)")

    all_results = {}

    # Run FAPH eval for each model
    for name, session in sessions.items():
        model_type = MODELS[name]["type"]
        print(f"\n{'='*80}")
        print(f"FAPH EVAL: {name} (type={model_type})")
        print(f"{'='*80}")

        faph_result = run_faph_single_model(
            name, model_type, session, preprocessor, flac_files, LIBRISPEECH_DIR
        )

        # TP eval
        tp_result = None
        if pos_files:
            print(f"  Running TP eval on {len(pos_files)} positive files...")
            tp_result = run_tp_single_model(
                name, model_type, session, preprocessor, pos_files
            )

        all_results[name] = {
            "faph": faph_result,
            "tp": tp_result,
        }

        # Print summary
        print(f"\n  Results for {name}:")
        print(f"  {'Config':<12} {'Thresh':>6} {'Trig':>6} {'FAPH':>8}")
        print(f"  {'-'*36}")
        for key, data in sorted(faph_result["faph"].items()):
            print(f"  {data['config']:<12} {data['threshold']:>6.2f} {data['triggers']:>6} {data['faph']:>8.2f}")
        print(f"  Avg inference: {faph_result['avg_infer_us']:.0f} us")

    # Print comparison table
    print(f"\n\n{'='*80}")
    print("COMPARISON TABLE: FAPH (debounced) by model")
    print(f"{'='*80}")

    for cfg_name, _, _ in MW_CONFIGS:
        print(f"\n--- {cfg_name} ---")
        header = f"  {'Model':<28}"
        for t in THRESHOLDS:
            header += f" {'@'+str(t):>8}"
        header += f" {'Infer':>8}"
        print(header)
        print(f"  {'-'*(28 + 8*len(THRESHOLDS) + 8)}")

        for name in all_results:
            row = f"  {name:<28}"
            faph_data = all_results[name]["faph"]["faph"]
            for t in THRESHOLDS:
                key = f"{cfg_name}@{t}"
                if key in faph_data:
                    row += f" {faph_data[key]['faph']:>8.2f}"
                else:
                    row += f" {'N/A':>8}"
            row += f" {all_results[name]['faph']['avg_infer_us']:>6.0f}us"
            print(row)

    # TP comparison
    if any(all_results[n]["tp"] for n in all_results):
        print(f"\n\n{'='*80}")
        print("COMPARISON TABLE: TP Detection Rate (%)")
        print(f"{'='*80}")

        for cfg_name, _, _ in MW_CONFIGS:
            print(f"\n--- {cfg_name} ---")
            header = f"  {'Model':<28}"
            for t in THRESHOLDS:
                header += f" {'@'+str(t):>8}"
            print(header)
            print(f"  {'-'*(28 + 8*len(THRESHOLDS))}")

            for name in all_results:
                tp = all_results[name]["tp"]
                if tp is None:
                    continue
                row = f"  {name:<28}"
                for t in THRESHOLDS:
                    st = str(t)
                    if cfg_name in tp["configs"] and st in tp["configs"][cfg_name]:
                        rate = tp["configs"][cfg_name][st]["rate_pct"]
                        row += f" {rate:>7.1f}%"
                    else:
                        row += f" {'N/A':>8}"
                print(row)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")

    # Final recommendation
    print(f"\n{'='*80}")
    print("KEY METRICS FOR PRODUCTION DECISION")
    print(f"{'='*80}")
    print("Production config: 3-of-3 @ threshold 0.80")
    print()
    for name in all_results:
        faph_data = all_results[name]["faph"]["faph"]
        key = "3-of-3@0.8"
        if key in faph_data:
            faph_val = faph_data[key]["faph"]
            tp = all_results[name]["tp"]
            tp_val = "N/A"
            if tp and "3-of-3" in tp["configs"] and "0.8" in tp["configs"]["3-of-3"]:
                tp_val = f"{tp['configs']['3-of-3']['0.8']['rate_pct']:.1f}%"
            infer = all_results[name]["faph"]["avg_infer_us"]
            print(f"  {name:<28}: FAPH={faph_val:.2f}  TP={tp_val}  Infer={infer:.0f}us")


if __name__ == "__main__":
    main()
