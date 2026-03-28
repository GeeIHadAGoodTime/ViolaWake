"""Quick FAPH spot-check: compare embed_clips vs direct 2-model pipeline on LibriSpeech.

Runs a subset of test-clean (first 100 files) to verify false alarm rates are
comparable between the two embedding paths. Full FAPH takes ~45min; this takes ~5min.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

WAKEWORD = Path(__file__).resolve().parent.parent
SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000  # 1.5s
STEP_SAMPLES = 1600   # 100ms
DEBOUNCE_SEC = 2.0
THRESHOLDS = [0.50, 0.80, 0.90, 0.95]
MAX_FILES = 100


def load_models():
    """Load mel, embedding, and MLP models."""
    import openwakeword
    oww_dir = Path(openwakeword.__file__).parent / "resources"
    mel_path = list(oww_dir.rglob("melspectrogram*.onnx"))[0]
    emb_path = list(oww_dir.rglob("embedding_model*.onnx"))[0]

    mel_sess = ort.InferenceSession(str(mel_path), providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(str(emb_path), providers=["CPUExecutionProvider"])

    mlp_path = WAKEWORD / "experiments" / "models" / "r3_10x_s42.onnx"
    mlp_sess = ort.InferenceSession(str(mlp_path), providers=["CPUExecutionProvider"])

    return mel_sess, emb_sess, mlp_sess


def score_clip_direct(audio_f32: np.ndarray, mel_sess, emb_sess, mlp_sess) -> float:
    """Score a 1.5s clip using the direct 2-model pipeline (what SDK will use)."""
    mel_inp = mel_sess.get_inputs()[0].name
    emb_inp = emb_sess.get_inputs()[0].name
    mlp_inp = mlp_sess.get_inputs()[0].name

    # mel extraction
    mel_out = mel_sess.run(None, {mel_inp: audio_f32.reshape(1, -1)})[0]
    mel_data = mel_out.squeeze()  # (T_mel, 32)

    if mel_data.shape[0] < 76:
        return 0.0

    # Sliding window embedding extraction (stride=8, matching embed_clips behavior)
    embeddings = []
    for start in range(0, mel_data.shape[0] - 75, 8):
        chunk = mel_data[start:start + 76].reshape(1, 76, 32, 1).astype(np.float32)
        emb = emb_sess.run(None, {emb_inp: chunk})[0].flatten()
        embeddings.append(emb)

    if not embeddings:
        return 0.0

    # Mean-pool (same as embed_clips path)
    mean_emb = np.stack(embeddings).mean(axis=0).reshape(1, -1).astype(np.float32)
    score = float(mlp_sess.run(None, {mlp_inp: mean_emb})[0][0][0])
    return score


def score_clip_embed_clips(audio_i16: np.ndarray, mlp_sess) -> float:
    """Score using embed_clips (the FAPH measurement path)."""
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures()
    embeddings = af.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
    mean_emb = embeddings[0].mean(axis=0).reshape(1, -1).astype(np.float32)
    mlp_inp = mlp_sess.get_inputs()[0].name
    return float(mlp_sess.run(None, {mlp_inp: mean_emb})[0][0][0])


def main():
    import soundfile as sf

    ls_dir = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"
    if not ls_dir.exists():
        print(f"ERROR: LibriSpeech not found at {ls_dir}")
        return

    mel_sess, emb_sess, mlp_sess = load_models()

    flac_files = sorted(ls_dir.rglob("*.flac"))[:MAX_FILES]
    print(f"Spot-checking {len(flac_files)} files from test-clean")

    # Track triggers per threshold for both paths
    triggers_direct = {t: [] for t in THRESHOLDS}
    triggers_embed = {t: [] for t in THRESHOLDS}
    total_windows = 0
    total_hours = 0.0

    t0 = time.time()
    for fi, fpath in enumerate(flac_files):
        audio, sr = sf.read(fpath, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        audio_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        duration_h = len(audio) / SAMPLE_RATE / 3600
        total_hours += duration_h

        # Sliding window FAPH
        n_windows = max(0, (len(audio) - CLIP_SAMPLES) // STEP_SAMPLES + 1)
        total_windows += n_windows

        scores_direct = []
        scores_embed = []

        for w in range(n_windows):
            start = w * STEP_SAMPLES
            clip_f32 = audio[start:start + CLIP_SAMPLES].astype(np.float32)
            clip_i16 = audio_i16[start:start + CLIP_SAMPLES]

            if len(clip_f32) < CLIP_SAMPLES:
                clip_f32 = np.pad(clip_f32, (0, CLIP_SAMPLES - len(clip_f32)))
                clip_i16 = np.pad(clip_i16, (0, CLIP_SAMPLES - len(clip_i16)))

            sd = score_clip_direct(clip_f32, mel_sess, emb_sess, mlp_sess)
            se = score_clip_embed_clips(clip_i16, mlp_sess)
            scores_direct.append(sd)
            scores_embed.append(se)

        # Count triggers with debounce
        for thr in THRESHOLDS:
            last_t = -999.0
            for w, (sd, se) in enumerate(zip(scores_direct, scores_embed)):
                t_sec = w * STEP_SAMPLES / SAMPLE_RATE
                if sd >= thr and (t_sec - last_t) >= DEBOUNCE_SEC:
                    triggers_direct[thr].append((fpath.name, t_sec, sd))
                    last_t = t_sec

            last_t = -999.0
            for w, (sd, se) in enumerate(zip(scores_direct, scores_embed)):
                t_sec = w * STEP_SAMPLES / SAMPLE_RATE
                if se >= thr and (t_sec - last_t) >= DEBOUNCE_SEC:
                    triggers_embed[thr].append((fpath.name, t_sec, se))
                    last_t = t_sec

        if (fi + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{fi+1}/{len(flac_files)}] {total_hours:.2f}h, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    print(f"\nDone: {len(flac_files)} files, {total_hours:.3f}h, {total_windows} windows, {elapsed:.0f}s")

    print(f"\n{'Threshold':>10}  {'Direct FAPH':>12}  {'Embed FAPH':>12}  {'Ratio':>8}")
    print("-" * 50)
    for thr in THRESHOLDS:
        nd = len(triggers_direct[thr])
        ne = len(triggers_embed[thr])
        faph_d = nd / total_hours if total_hours > 0 else 0
        faph_e = ne / total_hours if total_hours > 0 else 0
        ratio = faph_d / faph_e if faph_e > 0 else float('inf')
        print(f"  {thr:>8.2f}  {faph_d:>10.2f}/h  {faph_e:>10.2f}/h  {ratio:>7.2f}x")

    # Show top direct triggers
    for thr in [0.80, 0.90]:
        print(f"\n  Direct triggers @{thr}: {len(triggers_direct[thr])}")
        for name, t, s in triggers_direct[thr][:10]:
            print(f"    {name} t={t:.1f}s score={s:.6f}")

    # Save results
    result = {
        "files": len(flac_files),
        "total_hours": round(total_hours, 4),
        "total_windows": total_windows,
        "elapsed_seconds": round(elapsed, 1),
        "comparison": {}
    }
    for thr in THRESHOLDS:
        result["comparison"][str(thr)] = {
            "direct_triggers": len(triggers_direct[thr]),
            "direct_faph": round(len(triggers_direct[thr]) / total_hours, 2) if total_hours > 0 else 0,
            "embed_triggers": len(triggers_embed[thr]),
            "embed_faph": round(len(triggers_embed[thr]) / total_hours, 2) if total_hours > 0 else 0,
        }

    out_path = WAKEWORD / "experiments" / "faph_spot_check_streaming.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
