"""
Quick TP detection rate comparison: faph_hardened seeds vs baseline.
Scores all eval positives through each model and reports detection rates.
"""
from __future__ import annotations

import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# openwakeword for preprocessor
from openwakeword.model import Model as OWWModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from violawake_sdk.audio import load_audio, center_crop
from violawake_sdk._constants import CLIP_SAMPLES

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA/violawake_data")
MODELS_DIR = Path(__file__).parent / "models"

MODELS = {
    "D_combined_bce_s42 (baseline)": MODELS_DIR / "D_combined_bce_s42.onnx",
    "faph_hardened_s42": MODELS_DIR / "faph_hardened_s42.onnx",
    "faph_hardened_s43": MODELS_DIR / "faph_hardened_s43.onnx",
    "faph_hardened_s44": MODELS_DIR / "faph_hardened_s44.onnx",
    "round2_best (r2d_s43)": MODELS_DIR / "round2_best.onnx",
}

EVAL_POS_DIRS = {
    "jihad_music":   DATA_ROOT / "eval_real" / "positives" / "jihad_music",
    "jihad_normal":  DATA_ROOT / "eval_real" / "positives" / "jihad_normal",
    "jihad_whisper": DATA_ROOT / "eval_real" / "positives" / "jihad_whisper",
    "sierra_music":  DATA_ROOT / "eval_real" / "positives" / "sierra_music",
    "sierra_normal": DATA_ROOT / "eval_real" / "positives" / "sierra_normal",
    "sierra_whisper":DATA_ROOT / "eval_real" / "positives" / "sierra_whisper",
}

THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95]


def load_preprocessor():
    """Load OWW preprocessor (shared across models)."""
    oww = OWWModel()
    return oww.preprocessor


def load_onnx(model_path):
    """Load ONNX session."""
    sess = ort.InferenceSession(str(model_path))
    input_name = sess.get_inputs()[0].name
    return sess, input_name


def score_file(path, preprocessor, sess, input_name):
    """Score a single audio file. Returns probability [0, 1]."""
    audio = load_audio(path)
    audio = center_crop(audio, CLIP_SAMPLES)
    audio_int16 = np.clip(audio, -1, 1)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    if len(audio_int16) < CLIP_SAMPLES:
        audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
    embs = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    emb = embs.mean(axis=1).astype(np.float32)
    logit = sess.run(None, {input_name: emb})[0][0][0]
    if 0 <= logit <= 1:
        return float(logit)
    return float(1.0 / (1.0 + np.exp(-logit)))


def collect_eval_files():
    """Collect all eval positive wav files."""
    files = []
    categories = {}
    for name, dir_path in EVAL_POS_DIRS.items():
        if not dir_path.exists():
            print(f"  SKIP {name}: not found")
            continue
        wavs = sorted(glob.glob(str(dir_path / "*.wav")))
        files.extend(wavs)
        categories[name] = len(wavs)
    return files, categories


def main():
    print("Loading preprocessor...")
    preprocessor = load_preprocessor()

    print("Collecting eval positive files...")
    files, categories = collect_eval_files()
    print(f"  Total: {len(files)} files")
    for name, count in categories.items():
        print(f"    {name}: {count}")

    # Pre-compute embeddings (shared across models)
    print("\nPre-computing embeddings...")
    t0 = time.time()
    embeddings = []
    valid_files = []
    for f in files:
        try:
            audio = load_audio(f)
            audio = center_crop(audio, CLIP_SAMPLES)
            audio_int16 = np.clip(audio, -1, 1)
            audio_int16 = (audio_int16 * 32767).astype(np.int16)
            if len(audio_int16) < CLIP_SAMPLES:
                audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
            embs = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            emb = embs.mean(axis=1).astype(np.float32)
            embeddings.append(emb)
            valid_files.append(f)
        except Exception as e:
            print(f"  ERROR on {os.path.basename(f)}: {e}")
    print(f"  {len(embeddings)} embeddings computed in {time.time()-t0:.1f}s")

    # Score each model
    all_results = {}
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\nSKIP {model_name}: {model_path} not found")
            continue
        print(f"\nScoring with {model_name}...")
        sess, input_name = load_onnx(model_path)
        scores = []
        for emb in embeddings:
            logit = sess.run(None, {input_name: emb})[0][0][0]
            if 0 <= logit <= 1:
                scores.append(float(logit))
            else:
                scores.append(float(1.0 / (1.0 + np.exp(-logit))))
        all_results[model_name] = scores

    # ── Results table ──────────────────────────────────────────────────
    n = len(embeddings)
    print("\n" + "=" * 80)
    print(f"  DETECTION RATE COMPARISON ({n} eval positives)")
    print("=" * 80)

    # Header
    header = f"{'Model':<35}"
    for t in THRESHOLDS:
        header += f"  {'@'+str(t):>7}"
    header += f"  {'Mean':>7}  {'Min':>7}"
    print(header)
    print("-" * len(header))

    for model_name, scores in all_results.items():
        row = f"{model_name:<35}"
        for t in THRESHOLDS:
            rate = sum(1 for s in scores if s >= t) / n * 100
            row += f"  {rate:6.1f}%"
        row += f"  {np.mean(scores):7.4f}  {np.min(scores):7.4f}"
        print(row)

    # Baseline comparison
    baseline_name = "D_combined_bce_s42 (baseline)"
    if baseline_name in all_results:
        print("\n" + "=" * 80)
        print("  DELTA vs BASELINE (positive = improvement)")
        print("=" * 80)
        header = f"{'Model':<35}"
        for t in THRESHOLDS:
            header += f"  {'@'+str(t):>7}"
        print(header)
        print("-" * len(header))

        baseline_scores = all_results[baseline_name]
        for model_name, scores in all_results.items():
            if model_name == baseline_name:
                continue
            row = f"{model_name:<35}"
            for t in THRESHOLDS:
                base_rate = sum(1 for s in baseline_scores if s >= t) / n * 100
                this_rate = sum(1 for s in scores if s >= t) / n * 100
                delta = this_rate - base_rate
                sign = "+" if delta >= 0 else ""
                row += f"  {sign}{delta:5.1f}%"
            print(row)

    # Per-condition breakdown for hardened s43
    target_model = "faph_hardened_s43"
    if target_model in all_results:
        print("\n" + "=" * 80)
        print(f"  PER-CONDITION BREAKDOWN: {target_model}")
        print("=" * 80)

        sess, input_name = load_onnx(MODELS[target_model])
        for cond_name, dir_path in EVAL_POS_DIRS.items():
            if not dir_path.exists():
                continue
            wavs = sorted(glob.glob(str(dir_path / "*.wav")))
            cond_scores = []
            for f in wavs:
                try:
                    audio = load_audio(f)
                    audio = center_crop(audio, CLIP_SAMPLES)
                    audio_int16 = np.clip(audio, -1, 1)
                    audio_int16 = (audio_int16 * 32767).astype(np.int16)
                    if len(audio_int16) < CLIP_SAMPLES:
                        audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
                    embs = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
                    emb = embs.mean(axis=1).astype(np.float32)
                    logit = sess.run(None, {input_name: emb})[0][0][0]
                    if 0 <= logit <= 1:
                        cond_scores.append(float(logit))
                    else:
                        cond_scores.append(float(1.0 / (1.0 + np.exp(-logit))))
                except Exception:
                    pass
            if cond_scores:
                row = f"  {cond_name:<20} ({len(cond_scores):3d} files)"
                for t in THRESHOLDS:
                    rate = sum(1 for s in cond_scores if s >= t) / len(cond_scores) * 100
                    row += f"  {rate:6.1f}%"
                row += f"  mean={np.mean(cond_scores):.4f} min={np.min(cond_scores):.4f}"
                print(row)

    # Reference baseline numbers
    print("\n" + "=" * 80)
    print("  REFERENCE: Expected Baseline (D_combined_bce_s42)")
    print("=" * 80)
    print("  @0.8: 98.2%  @0.9: 97.3%  @0.95: 96.5%")
    print("  (If actual baseline differs, the eval set may have changed)")

    print("\nDone.")


if __name__ == "__main__":
    main()
