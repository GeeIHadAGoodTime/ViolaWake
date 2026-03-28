"""
SDK Inference Path Validation — CRITICAL GATE
==============================================

Answers Q26: Does oww_backbone.onnx (SDK 20ms frame path) produce the same
embeddings as embed_clips() (FAPH measurement path)?

If they differ, ALL our accuracy measurements are INVALID for the shipped SDK.

Test:
1. Load a known audio clip (1.5s, 16kHz)
2. Path A: embed_clips() → mean-pool → 96-dim (our FAPH/training path)
3. Path B: oww_backbone.onnx frame-by-frame → ??? → MLP input
4. Compare: cosine similarity, L2 distance, MLP score correlation

Usage:
    python experiments/verify_sdk_path.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

WAKEWORD = Path(__file__).resolve().parent.parent
EXPERIMENTS = WAKEWORD / "experiments"
SDK_SRC = WAKEWORD / "src"
sys.path.insert(0, str(SDK_SRC))

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320  # 20ms
CLIP_SAMPLES = 24000  # 1.5s


def get_embed_clips_embedding(audio_int16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Path A: embed_clips (FAPH measurement path). Returns (raw_temporal, mean_pooled)."""
    from openwakeword.utils import AudioFeatures
    preprocessor = AudioFeatures()
    embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    # embeddings shape: (1, T, 96)
    temporal = embeddings[0]  # (T, 96)
    mean_pooled = temporal.mean(axis=0)  # (96,)
    return temporal.astype(np.float32), mean_pooled.astype(np.float32)


def get_backbone_embedding(audio_float32: np.ndarray, backbone_path: Path) -> np.ndarray:
    """Path B: oww_backbone.onnx frame-by-frame (SDK path). Returns raw output."""
    sess = ort.InferenceSession(str(backbone_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    inp_type = sess.get_inputs()[0].type
    out_name = sess.get_outputs()[0].name
    out_shape = sess.get_outputs()[0].shape

    print(f"\n  OWW Backbone ONNX model:")
    print(f"    Input:  name={inp_name}, shape={inp_shape}, type={inp_type}")
    print(f"    Output: name={out_name}, shape={out_shape}")
    print(f"    Num inputs: {len(sess.get_inputs())}")
    print(f"    Num outputs: {len(sess.get_outputs())}")
    for i, out in enumerate(sess.get_outputs()):
        print(f"    Output[{i}]: name={out.name}, shape={out.shape}, type={out.type}")

    # Try feeding full 1.5s clip
    print(f"\n  Testing with full 1.5s clip ({CLIP_SAMPLES} samples)...")
    try:
        full_input = audio_float32.reshape(1, -1)
        result = sess.run(None, {inp_name: full_input})
        print(f"    Full clip result: {[r.shape for r in result]}")
        return result[0]
    except Exception as e:
        print(f"    Full clip FAILED: {e}")

    # Try feeding 20ms frame
    print(f"\n  Testing with single 20ms frame ({FRAME_SAMPLES} samples)...")
    try:
        frame = audio_float32[:FRAME_SAMPLES].reshape(1, -1)
        result = sess.run(None, {inp_name: frame})
        print(f"    Single frame result: {[r.shape for r in result]}")
    except Exception as e:
        print(f"    Single frame FAILED: {e}")

    # Try feeding frames sequentially (accumulating state)
    print(f"\n  Testing sequential frames (entire 1.5s as 75 x 20ms)...")
    n_frames = len(audio_float32) // FRAME_SAMPLES
    all_outputs = []
    for i in range(n_frames):
        frame = audio_float32[i*FRAME_SAMPLES:(i+1)*FRAME_SAMPLES].reshape(1, -1)
        try:
            result = sess.run(None, {inp_name: frame})
            all_outputs.append(result[0])
        except Exception as e:
            print(f"    Frame {i} FAILED: {e}")
            break

    if all_outputs:
        print(f"    Got {len(all_outputs)} frame outputs, shape={all_outputs[0].shape}")
        stacked = np.stack([o.flatten() for o in all_outputs])
        print(f"    Stacked shape: {stacked.shape}")
        return stacked

    return np.array([])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat, b_flat = a.flatten(), b.flatten()
    if len(a_flat) != len(b_flat):
        return -999.0
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(a_flat, b_flat) / norm)


def main():
    print("=" * 70)
    print("SDK INFERENCE PATH VALIDATION — CRITICAL GATE")
    print("=" * 70)

    # Find a test audio file
    test_files = []

    # Try eval_fresh positives
    eval_dirs = [
        WAKEWORD / "eval_fresh" / "positives",
        WAKEWORD / "eval_clean" / "positives",
        WAKEWORD / "violawake_data" / "positives",
    ]
    for d in eval_dirs:
        if d.exists():
            for f in sorted(d.glob("*.wav"))[:3]:
                test_files.append(f)
            break

    # Also grab a LibriSpeech negative
    ls_dir = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"
    if ls_dir.exists():
        for f in sorted(ls_dir.rglob("*.flac"))[:2]:
            test_files.append(f)

    if not test_files:
        print("ERROR: No test audio files found")
        return

    print(f"\nTest files: {[f.name for f in test_files]}")

    # Find backbone model
    backbone_paths = [
        WAKEWORD / "src" / "violawake_sdk" / "models" / "oww_backbone.onnx",
        Path.home() / ".violawake" / "models" / "oww_backbone.onnx",
    ]
    # Also check if openwakeword has it
    try:
        import openwakeword
        oww_dir = Path(openwakeword.__file__).parent / "resources"
        if oww_dir.exists():
            for f in oww_dir.rglob("*.onnx"):
                backbone_paths.append(f)
            # Check for melspectrogram model
            for f in oww_dir.rglob("*melspectrogram*"):
                backbone_paths.append(f)
            for f in oww_dir.rglob("*embedding*"):
                backbone_paths.append(f)
    except ImportError:
        pass

    backbone = None
    for p in backbone_paths:
        if p.exists():
            backbone = p
            break

    # Try to find via model download
    if backbone is None:
        print("\nSearching for OWW backbone model...")
        try:
            from violawake_sdk.models import get_model_path
            backbone = get_model_path("oww_backbone")
        except Exception as e:
            print(f"  get_model_path failed: {e}")

    # List what OWW has
    print("\nChecking openwakeword resources...")
    try:
        import openwakeword
        oww_dir = Path(openwakeword.__file__).parent
        print(f"  OWW installed at: {oww_dir}")
        for f in sorted(oww_dir.rglob("*.onnx")):
            print(f"    {f.relative_to(oww_dir)} ({f.stat().st_size / 1024:.0f} KB)")
        for f in sorted(oww_dir.rglob("*.tflite")):
            print(f"    {f.relative_to(oww_dir)} ({f.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"  Error: {e}")

    # Try using OWW's AudioFeatures directly to understand the pipeline
    print("\n" + "=" * 70)
    print("ANALYZING OWW EMBEDDING PIPELINE")
    print("=" * 70)

    import soundfile as sf
    audio, sr = sf.read(test_files[0], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Pad/crop to 1.5s
    if len(audio) < CLIP_SAMPLES:
        audio = np.pad(audio, (0, CLIP_SAMPLES - len(audio)))
    else:
        audio = audio[:CLIP_SAMPLES]

    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    print(f"\nTest audio: {test_files[0].name}")
    print(f"  Shape: {audio.shape}, SR: {SAMPLE_RATE}")

    # Path A: embed_clips
    print("\n--- PATH A: embed_clips (FAPH measurement path) ---")
    temporal_a, mean_a = get_embed_clips_embedding(audio_int16)
    print(f"  Temporal shape: {temporal_a.shape}")  # Should be (9, 96)
    print(f"  Mean-pooled shape: {mean_a.shape}")  # Should be (96,)
    print(f"  Mean-pooled stats: min={mean_a.min():.4f}, max={mean_a.max():.4f}, mean={mean_a.mean():.4f}")

    # Understand how AudioFeatures works internally
    print("\n--- Inspecting AudioFeatures internals ---")
    from openwakeword.utils import AudioFeatures
    af = AudioFeatures()
    print(f"  AudioFeatures attributes: {[a for a in dir(af) if not a.startswith('_')]}")

    # Check if AudioFeatures uses ONNX models internally
    for attr in dir(af):
        obj = getattr(af, attr, None)
        if obj is not None and hasattr(obj, 'get_inputs'):
            # It's an ONNX session
            print(f"  Found ONNX session: {attr}")
            for inp in obj.get_inputs():
                print(f"    Input: {inp.name}, shape={inp.shape}, type={inp.type}")
            for out in obj.get_outputs():
                print(f"    Output: {out.name}, shape={out.shape}, type={out.type}")

    # Try the per-frame streaming approach used by OWW Model
    print("\n--- Testing OWW streaming (per-frame) approach ---")
    try:
        from openwakeword.model import Model as OWWModel
        # Initialize with no pretrained models
        oww_model = OWWModel(wakeword_models=[], inference_framework="onnx")
        print(f"  OWW Model attributes: {[a for a in dir(oww_model) if not a.startswith('_')]}")

        # Check preprocessor
        if hasattr(oww_model, 'preprocessor'):
            pp = oww_model.preprocessor
            print(f"  Preprocessor type: {type(pp)}")
            print(f"  Preprocessor attrs: {[a for a in dir(pp) if not a.startswith('_')]}")

        # Feed audio frame-by-frame
        if hasattr(oww_model, 'predict'):
            # OWW's predict() takes raw audio and returns prediction dict
            # Feed 1.5s of audio in 80ms chunks (OWW default)
            chunk_size = 1280  # 80ms at 16kHz
            n_chunks = len(audio_int16) // chunk_size
            print(f"  Feeding {n_chunks} chunks of {chunk_size} samples ({chunk_size/16}ms each)")

            for i in range(n_chunks):
                chunk = audio_int16[i*chunk_size:(i+1)*chunk_size]
                result = oww_model.predict(chunk)

            # After feeding all chunks, check if we can extract the embedding
            if hasattr(oww_model, 'preprocessor') and hasattr(oww_model.preprocessor, 'get_features'):
                features = oww_model.preprocessor.get_features()
                print(f"  Features after streaming: shape={features.shape if hasattr(features, 'shape') else type(features)}")

            # Check raw embeddings buffer
            if hasattr(oww_model, 'preprocessor'):
                pp = oww_model.preprocessor
                if hasattr(pp, 'raw_data_buffer'):
                    print(f"  raw_data_buffer: {len(pp.raw_data_buffer)} samples")
                if hasattr(pp, 'melspectrogram_buffer'):
                    buf = pp.melspectrogram_buffer
                    print(f"  melspectrogram_buffer: shape={buf.shape if hasattr(buf, 'shape') else type(buf)}")
                if hasattr(pp, 'accumulated_samples'):
                    print(f"  accumulated_samples: {pp.accumulated_samples}")
                if hasattr(pp, 'feature_buffer'):
                    buf = pp.feature_buffer
                    print(f"  feature_buffer: shape={np.array(buf).shape if buf else 'empty'}")
                if hasattr(pp, 'embedding_buffer'):
                    print(f"  embedding_buffer exists")

                # Dump all non-private attrs
                for attr in sorted(dir(pp)):
                    if attr.startswith('_'):
                        continue
                    val = getattr(pp, attr, None)
                    if callable(val):
                        continue
                    if isinstance(val, np.ndarray):
                        print(f"  pp.{attr}: ndarray shape={val.shape}, dtype={val.dtype}")
                    elif isinstance(val, (list, tuple)):
                        print(f"  pp.{attr}: {type(val).__name__} len={len(val)}")
                    elif isinstance(val, (int, float, str, bool)):
                        print(f"  pp.{attr}: {val}")

    except Exception as e:
        print(f"  OWW Model test failed: {e}")
        import traceback
        traceback.print_exc()

    # Now try backbone directly if found
    if backbone:
        print(f"\n--- PATH B: oww_backbone.onnx ({backbone}) ---")
        result_b = get_backbone_embedding(audio.astype(np.float32), backbone)
        if result_b.size > 0:
            print(f"  Result shape: {result_b.shape}")
    else:
        print("\n  WARNING: oww_backbone.onnx not found")
        print("  The SDK WakeDetector requires this model.")
        print("  Checking if it can be downloaded...")

    # Score through MLP to compare end-to-end
    print("\n" + "=" * 70)
    print("MLP SCORING COMPARISON")
    print("=" * 70)

    mlp_path = EXPERIMENTS / "models" / "r3_10x_s42.onnx"
    if mlp_path.exists():
        mlp_sess = ort.InferenceSession(str(mlp_path), providers=["CPUExecutionProvider"])
        mlp_inp = mlp_sess.get_inputs()[0].name

        # Score path A embedding
        score_a = float(mlp_sess.run(None, {mlp_inp: mean_a.reshape(1, -1)})[0][0][0])
        print(f"  Path A (embed_clips → mean-pool → MLP): score = {score_a:.6f}")

        # Score each temporal frame
        print(f"  Per-frame scores from temporal embedding:")
        for i, frame_emb in enumerate(temporal_a):
            s = float(mlp_sess.run(None, {mlp_inp: frame_emb.reshape(1, -1)})[0][0][0])
            print(f"    Frame {i}: score = {s:.6f}")

    # Test on multiple files
    if len(test_files) > 1:
        print("\n" + "=" * 70)
        print("MULTI-FILE COMPARISON")
        print("=" * 70)
        for tf in test_files:
            try:
                a, sr = sf.read(tf, dtype="float32")
                if a.ndim > 1:
                    a = a.mean(axis=1)
                if sr != SAMPLE_RATE:
                    import librosa
                    a = librosa.resample(a, orig_sr=sr, target_sr=SAMPLE_RATE)
                if len(a) < CLIP_SAMPLES:
                    a = np.pad(a, (0, CLIP_SAMPLES - len(a)))
                else:
                    a = a[:CLIP_SAMPLES]
                a_int16 = (a * 32767).clip(-32768, 32767).astype(np.int16)

                _, mean_emb = get_embed_clips_embedding(a_int16)
                if mlp_path.exists():
                    score = float(mlp_sess.run(None, {mlp_inp: mean_emb.reshape(1, -1)})[0][0][0])
                    print(f"  {tf.name}: score={score:.6f} ({'POS' if score > 0.5 else 'NEG'})")
            except Exception as e:
                print(f"  {tf.name}: ERROR {e}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("See output above. Key question: does the SDK's frame-by-frame path")
    print("produce embeddings compatible with our MLP?")
    print("If OWW backbone outputs differ from embed_clips, we need to fix the SDK.")


if __name__ == "__main__":
    main()
