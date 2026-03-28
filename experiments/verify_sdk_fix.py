"""Verify SDK pipeline matches embed_clips after normalization fix.

The critical fix: melspectrogram.onnx expects int16-range float32 values,
and output must be transformed with mel/10+2.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

WAKEWORD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAKEWORD / "src"))

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000


def main():
    import openwakeword
    from openwakeword.utils import AudioFeatures

    # Load test audio
    test_dir = WAKEWORD / "eval_clean" / "positives"
    wavs = sorted(test_dir.rglob("*viola*.wav"))
    if not wavs:
        print("No test audio found")
        return

    import soundfile as sf
    audio, sr = sf.read(wavs[0], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    else:
        audio = audio[:CLIP_SAMPLES]

    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    audio_int16_as_f32 = audio_int16.astype(np.float32)  # int16 range, float32 dtype

    print(f"Test audio: {wavs[0].name}")
    print(f"  int16 range: [{audio_int16.min()}, {audio_int16.max()}]")
    print(f"  float32 range: [{audio_int16_as_f32.min():.0f}, {audio_int16_as_f32.max():.0f}]")

    # Path A: embed_clips (ground truth)
    af = AudioFeatures()
    embeddings_a = af.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    temporal_a = embeddings_a[0].astype(np.float32)
    mean_a = temporal_a.mean(axis=0)
    print(f"\nPath A (embed_clips): {temporal_a.shape} frames")
    print(f"  Mean emb: min={mean_a.min():.4f} max={mean_a.max():.4f}")

    # Path B: Fixed direct pipeline (int16 range + mel/10+2)
    oww_dir = Path(openwakeword.__file__).parent / "resources"
    mel_sess = ort.InferenceSession(str(list(oww_dir.rglob("melspectrogram*.onnx"))[0]), providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(str(list(oww_dir.rglob("embedding_model*.onnx"))[0]), providers=["CPUExecutionProvider"])

    mel_inp = mel_sess.get_inputs()[0].name
    emb_inp = emb_sess.get_inputs()[0].name

    # Extract mel with int16-range input
    mel_out = mel_sess.run(None, {mel_inp: audio_int16_as_f32.reshape(1, -1)})[0]
    mel_raw = mel_out.squeeze().reshape(-1, 32).astype(np.float32)

    # Apply the critical transform
    mel_transformed = mel_raw / 10.0 + 2.0
    print(f"\nMel (raw): min={mel_raw.min():.4f} max={mel_raw.max():.4f}")
    print(f"Mel (transformed): min={mel_transformed.min():.4f} max={mel_transformed.max():.4f}")
    print(f"Mel frames: {mel_transformed.shape[0]}")

    # Extract embeddings with stride 8 (matching embed_clips)
    embeddings_b = []
    for start in range(0, mel_transformed.shape[0] - 75, 8):
        chunk = mel_transformed[start:start + 76].reshape(1, 76, 32, 1).astype(np.float32)
        emb = emb_sess.run(None, {emb_inp: chunk})[0].flatten()
        embeddings_b.append(emb)

    temporal_b = np.stack(embeddings_b).astype(np.float32)
    mean_b = temporal_b.mean(axis=0)
    print(f"\nPath B (fixed direct): {temporal_b.shape} frames")
    print(f"  Mean emb: min={mean_b.min():.4f} max={mean_b.max():.4f}")

    # Compare
    def cosine(a, b):
        a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
        norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
        return float(np.dot(a_f, b_f) / norm) if norm > 1e-10 else 0.0

    cos_mean = cosine(mean_a, mean_b)
    l2_mean = float(np.linalg.norm(mean_a - mean_b))
    print(f"\n=== COMPARISON (mean-pooled) ===")
    print(f"  Cosine similarity: {cos_mean:.6f}")
    print(f"  L2 distance:       {l2_mean:.6f}")

    # Per-frame comparison
    min_frames = min(temporal_a.shape[0], temporal_b.shape[0])
    print(f"\n  Per-frame cosine ({min_frames} frames):")
    for i in range(min_frames):
        cs = cosine(temporal_a[i], temporal_b[i])
        print(f"    Frame {i}: {cs:.6f}")

    # MLP scoring
    mlp_path = WAKEWORD / "experiments" / "models" / "r3_10x_s42.onnx"
    if mlp_path.exists():
        mlp_sess = ort.InferenceSession(str(mlp_path), providers=["CPUExecutionProvider"])
        mlp_inp = mlp_sess.get_inputs()[0].name

        score_a = float(mlp_sess.run(None, {mlp_inp: mean_a.reshape(1, -1)})[0][0][0])
        score_b = float(mlp_sess.run(None, {mlp_inp: mean_b.reshape(1, -1)})[0][0][0])
        print(f"\n=== MLP SCORES ===")
        print(f"  Path A (embed_clips): {score_a:.6f}")
        print(f"  Path B (fixed direct): {score_b:.6f}")
        print(f"  Delta: {abs(score_a - score_b):.6f}")

    # Also test on a negative (LibriSpeech)
    ls_dir = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"
    if ls_dir.exists():
        flacs = sorted(ls_dir.rglob("*.flac"))[:3]
        print(f"\n=== NEGATIVE SAMPLES ===")
        for fpath in flacs:
            a, sr2 = sf.read(fpath, dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr2 != SAMPLE_RATE:
                import librosa
                a = librosa.resample(a, orig_sr=sr2, target_sr=SAMPLE_RATE)
            if len(a) < CLIP_SAMPLES:
                a = np.pad(a, (0, CLIP_SAMPLES - len(a)))
            else:
                a = a[:CLIP_SAMPLES]
            a_i16 = (a * 32767).clip(-32768, 32767).astype(np.int16)
            a_i16_f32 = a_i16.astype(np.float32)

            # embed_clips
            embs_a = af.embed_clips(a_i16.reshape(1, -1), ncpu=1)
            mean_ea = embs_a[0].mean(axis=0).reshape(1, -1).astype(np.float32)

            # direct pipeline
            mel_out2 = mel_sess.run(None, {mel_inp: a_i16_f32.reshape(1, -1)})[0]
            mel2 = mel_out2.squeeze().reshape(-1, 32).astype(np.float32) / 10.0 + 2.0
            embs_b2 = []
            for start in range(0, mel2.shape[0] - 75, 8):
                chunk = mel2[start:start + 76].reshape(1, 76, 32, 1).astype(np.float32)
                emb = emb_sess.run(None, {emb_inp: chunk})[0].flatten()
                embs_b2.append(emb)
            mean_eb = np.stack(embs_b2).mean(axis=0).reshape(1, -1).astype(np.float32)

            if mlp_path.exists():
                sa = float(mlp_sess.run(None, {mlp_inp: mean_ea})[0][0][0])
                sb = float(mlp_sess.run(None, {mlp_inp: mean_eb})[0][0][0])
                cs = cosine(mean_ea.flatten(), mean_eb.flatten())
                print(f"  {fpath.name}: embed={sa:.6f} direct={sb:.6f} cos={cs:.6f} delta={abs(sa-sb):.6f}")

    # Verdict
    print(f"\n=== VERDICT ===")
    if cos_mean > 0.999:
        print(f"  PERFECT MATCH (cosine={cos_mean:.6f})")
        print(f"  SDK pipeline is now equivalent to embed_clips")
        print(f"  FAPH measurements TRANSFER to shipped SDK")
    elif cos_mean > 0.99:
        print(f"  NEAR-PERFECT MATCH (cosine={cos_mean:.6f})")
        print(f"  Minor numerical differences only — FAPH measurements transfer")
    elif cos_mean > 0.95:
        print(f"  GOOD MATCH (cosine={cos_mean:.6f})")
        print(f"  Some divergence — verify FAPH with SDK pipeline")
    else:
        print(f"  MISMATCH (cosine={cos_mean:.6f})")
        print(f"  FAPH measurements may NOT transfer!")


if __name__ == "__main__":
    main()
