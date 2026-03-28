"""Compare OWW embed_clips vs streaming embedding paths.

CRITICAL GATE: If these produce different embeddings, our FAPH measurements
(which use embed_clips) are INVALID for the shipped SDK (which must use streaming).

Test:
1. Load a known positive audio clip (1.5s, 16kHz)
2. Path A: embed_clips() -> (1, 9, 96) -> mean-pool -> 96-dim
3. Path B: OWW streaming via AudioFeatures (frame-by-frame) -> embeddings
4. Compare: cosine similarity, L2 distance, MLP score correlation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

WAKEWORD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WAKEWORD / "src"))

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000  # 1.5s


def find_test_audio() -> Path:
    """Find a test audio file."""
    for d in [
        WAKEWORD / "eval_fresh" / "positives",
        WAKEWORD / "eval_clean" / "positives",
        WAKEWORD / "violawake_data" / "positives",
    ]:
        if d.exists():
            wavs = sorted(d.rglob("*.wav"))
            if wavs:
                return wavs[0]
    # Try LibriSpeech negative (any speech works for embedding comparison)
    ls_dir = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"
    if ls_dir.exists():
        flacs = sorted(ls_dir.rglob("*.flac"))
        if flacs:
            return flacs[0]
    raise FileNotFoundError("No test audio found")


def load_audio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load audio, return (float32, int16) both padded/cropped to 1.5s."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
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
    return audio.astype(np.float32), audio_int16


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_f, b_f = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(a_f, b_f) / norm)


def main():
    import onnxruntime as ort

    audio_path = find_test_audio()
    audio_f32, audio_i16 = load_audio(audio_path)
    print(f"Test audio: {audio_path.name} ({len(audio_f32)} samples)")

    # ================================================================
    # PATH A: embed_clips (FAPH measurement path)
    # ================================================================
    print("\n=== PATH A: embed_clips ===")
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures()

    embeddings_a = af.embed_clips(audio_i16.reshape(1, -1), ncpu=1)
    # shape: (1, T, 96)
    temporal_a = embeddings_a[0].astype(np.float32)  # (T, 96)
    mean_a = temporal_a.mean(axis=0)  # (96,)
    print(f"  Temporal shape: {temporal_a.shape}")
    print(f"  Mean-pooled: min={mean_a.min():.4f} max={mean_a.max():.4f} mean={mean_a.mean():.4f}")

    # ================================================================
    # PATH B: OWW streaming (SDK-like frame-by-frame)
    # ================================================================
    print("\n=== PATH B: OWW streaming ===")

    # Use OWW's internal models directly
    import openwakeword
    oww_dir = Path(openwakeword.__file__).parent / "resources"

    mel_path = list(oww_dir.rglob("melspectrogram*.onnx"))[0]
    emb_path = list(oww_dir.rglob("embedding_model*.onnx"))[0]
    print(f"  Mel model: {mel_path.name}")
    print(f"  Emb model: {emb_path.name}")

    mel_sess = ort.InferenceSession(str(mel_path), providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(str(emb_path), providers=["CPUExecutionProvider"])

    mel_inp = mel_sess.get_inputs()[0]
    emb_inp = emb_sess.get_inputs()[0]
    print(f"  Mel input: {mel_inp.name} shape={mel_inp.shape}")
    print(f"  Emb input: {emb_inp.name} shape={emb_inp.shape}")

    # Method B1: Feed entire 1.5s clip through mel then embedding (batch-like)
    print("\n--- B1: Full clip through mel+emb ---")
    mel_out = mel_sess.run(None, {mel_inp.name: audio_f32.reshape(1, -1)})[0]
    print(f"  Mel output shape: {mel_out.shape}")  # expect (1, 1, T_mel, 32)

    # embedding_model expects (batch, 76, 32, 1)
    # mel output is (1, 1, T_mel, 32) - need 76 mel frames per embedding
    mel_data = mel_out.squeeze()  # (T_mel, 32)
    print(f"  Mel frames: {mel_data.shape[0]}")

    n_emb_frames = mel_data.shape[0] // 76
    remainder = mel_data.shape[0] % 76
    print(f"  Can extract {n_emb_frames} embeddings ({remainder} mel frames leftover)")

    embeddings_b1 = []
    for i in range(n_emb_frames):
        chunk_76 = mel_data[i * 76:(i + 1) * 76]  # (76, 32)
        inp = chunk_76.reshape(1, 76, 32, 1).astype(np.float32)
        emb = emb_sess.run(None, {emb_inp.name: inp})[0]  # (1, 1, 1, 96)
        embeddings_b1.append(emb.flatten())

    if not embeddings_b1:
        print("  ERROR: No embeddings extracted!")
        return

    temporal_b1 = np.stack(embeddings_b1).astype(np.float32)  # (N, 96)
    mean_b1 = temporal_b1.mean(axis=0)
    print(f"  B1 temporal shape: {temporal_b1.shape}")
    print(f"  B1 mean-pooled: min={mean_b1.min():.4f} max={mean_b1.max():.4f}")

    # Method B2: Sliding window of 76 mel frames with stride
    print("\n--- B2: Sliding window (76 mel, stride 8) ---")
    stride = 8  # OWW default stride for streaming
    embeddings_b2 = []
    for start in range(0, mel_data.shape[0] - 75, stride):
        chunk_76 = mel_data[start:start + 76]
        inp = chunk_76.reshape(1, 76, 32, 1).astype(np.float32)
        emb = emb_sess.run(None, {emb_inp.name: inp})[0]
        embeddings_b2.append(emb.flatten())

    temporal_b2 = np.stack(embeddings_b2).astype(np.float32)
    mean_b2 = temporal_b2.mean(axis=0)
    print(f"  B2 temporal shape: {temporal_b2.shape} (stride={stride})")
    print(f"  B2 mean-pooled: min={mean_b2.min():.4f} max={mean_b2.max():.4f}")

    # ================================================================
    # COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # A vs B1 (non-overlapping chunks)
    cos_ab1 = cosine_sim(mean_a, mean_b1)
    l2_ab1 = float(np.linalg.norm(mean_a - mean_b1))
    print(f"\n  A (embed_clips) vs B1 (non-overlap mel chunks):")
    print(f"    Cosine similarity: {cos_ab1:.6f}")
    print(f"    L2 distance:       {l2_ab1:.6f}")
    print(f"    A frames: {temporal_a.shape[0]}, B1 frames: {temporal_b1.shape[0]}")

    # A vs B2 (sliding window)
    cos_ab2 = cosine_sim(mean_a, mean_b2)
    l2_ab2 = float(np.linalg.norm(mean_a - mean_b2))
    print(f"\n  A (embed_clips) vs B2 (sliding window stride={stride}):")
    print(f"    Cosine similarity: {cos_ab2:.6f}")
    print(f"    L2 distance:       {l2_ab2:.6f}")
    print(f"    A frames: {temporal_a.shape[0]}, B2 frames: {temporal_b2.shape[0]}")

    # Per-frame comparison (A vs B1, matching count)
    min_frames = min(temporal_a.shape[0], temporal_b1.shape[0])
    print(f"\n  Per-frame cosine (first {min_frames} frames):")
    for i in range(min_frames):
        cs = cosine_sim(temporal_a[i], temporal_b1[i])
        print(f"    Frame {i}: cos={cs:.6f}")

    # B1 vs B2
    cos_b1b2 = cosine_sim(mean_b1, mean_b2)
    print(f"\n  B1 (non-overlap) vs B2 (sliding): cosine={cos_b1b2:.6f}")

    # ================================================================
    # MLP SCORING
    # ================================================================
    mlp_path = WAKEWORD / "experiments" / "models" / "r3_10x_s42.onnx"
    if mlp_path.exists():
        print("\n" + "=" * 70)
        print("MLP SCORING")
        print("=" * 70)
        mlp_sess = ort.InferenceSession(str(mlp_path), providers=["CPUExecutionProvider"])
        mlp_inp_name = mlp_sess.get_inputs()[0].name

        score_a = float(mlp_sess.run(None, {mlp_inp_name: mean_a.reshape(1, -1)})[0][0][0])
        score_b1 = float(mlp_sess.run(None, {mlp_inp_name: mean_b1.reshape(1, -1)})[0][0][0])
        score_b2 = float(mlp_sess.run(None, {mlp_inp_name: mean_b2.reshape(1, -1)})[0][0][0])

        print(f"  Path A  (embed_clips mean-pool): {score_a:.6f}")
        print(f"  Path B1 (non-overlap mean-pool): {score_b1:.6f}")
        print(f"  Path B2 (sliding mean-pool):     {score_b2:.6f}")

        # Per-frame MLP scores
        print(f"\n  Per-frame MLP scores:")
        print(f"  {'Frame':>5}  {'Path A':>10}  {'Path B1':>10}")
        for i in range(min_frames):
            sa = float(mlp_sess.run(None, {mlp_inp_name: temporal_a[i].reshape(1, -1)})[0][0][0])
            sb = float(mlp_sess.run(None, {mlp_inp_name: temporal_b1[i].reshape(1, -1)})[0][0][0])
            print(f"  {i:>5}  {sa:>10.6f}  {sb:>10.6f}")

        # Score each B2 frame individually (this is what SDK streaming would do)
        print(f"\n  B2 per-frame scores (what SDK streaming sees):")
        for i, emb in enumerate(embeddings_b2):
            s = float(mlp_sess.run(None, {mlp_inp_name: emb.reshape(1, -1)})[0][0][0])
            if s > 0.3 or i < 5 or i >= len(embeddings_b2) - 3:
                print(f"    Frame {i:>3}: {s:.6f} {'<-- TRIGGER' if s > 0.8 else ''}")

    # ================================================================
    # VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if cos_ab1 > 0.99:
        print("  PASS: embed_clips and direct mel+emb produce near-identical embeddings")
        print("  SDK can use the 2-model pipeline with confidence that FAPH results apply")
    elif cos_ab1 > 0.95:
        print("  CAUTION: Embeddings are similar but not identical")
        print("  MLP scores should be compared to assess practical impact")
    else:
        print("  FAIL: Embedding paths diverge significantly!")
        print("  FAPH measurements may not apply to SDK inference path")
        print("  Consider retraining on streaming embeddings or using embed_clips in SDK")

    print(f"\n  Key metric: cosine(embed_clips, direct_pipeline) = {cos_ab1:.6f}")


if __name__ == "__main__":
    main()
