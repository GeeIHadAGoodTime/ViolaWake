"""Streaming FAPH evaluation for temporal_cnn on LibriSpeech test-clean.

Uses the same OWW pipeline as live_compare.py (verified cosine=1.0 with embed_clips).
Processes all FLAC files frame-by-frame to measure false accepts per hour.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

WAKEWORD = Path(__file__).resolve().parent.parent

# Constants
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20ms at 16kHz
MEL_BINS = 32
MEL_FRAMES_PER_EMBEDDING = 76
MEL_STRIDE = 8
EMBEDDING_DIM = 96
CLIP_SAMPLES = SAMPLE_RATE * 2  # 2s ring buffer

# Model paths
TEMPORAL_CNN = WAKEWORD / "experiments" / "models" / "j5_temporal" / "temporal_cnn.onnx"
SEQ_LEN = 9  # temporal_cnn expects (batch, 9, 96)

# Thresholds to test
THRESHOLDS = [0.80, 0.85, 0.90, 0.95]
CONFIRM_COUNTS = [1, 2, 3]
COOLDOWN_S = 2.0


def find_oww_models():
    import openwakeword
    oww_dir = Path(openwakeword.__file__).parent / "resources"
    mel = list(oww_dir.rglob("melspectrogram*.onnx"))[0]
    emb = list(oww_dir.rglob("embedding_model*.onnx"))[0]
    return mel, emb


def main():
    testclean = WAKEWORD / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"
    if not testclean.exists():
        print(f"ERROR: {testclean} not found")
        sys.exit(1)

    flacs = sorted(testclean.rglob("*.flac"))
    print(f"Found {len(flacs)} FLAC files in test-clean")

    # Load OWW backbone
    mel_path, emb_path = find_oww_models()
    mel_sess = ort.InferenceSession(str(mel_path), providers=["CPUExecutionProvider"])
    emb_sess = ort.InferenceSession(str(emb_path), providers=["CPUExecutionProvider"])
    mel_inp = mel_sess.get_inputs()[0].name
    emb_inp = emb_sess.get_inputs()[0].name

    # Load temporal_cnn
    tcnn_sess = ort.InferenceSession(str(TEMPORAL_CNN), providers=["CPUExecutionProvider"])
    tcnn_inp = tcnn_sess.get_inputs()[0].name
    print(f"Loaded temporal_cnn: {TEMPORAL_CNN.name}")
    print(f"  Input shape: {tcnn_sess.get_inputs()[0].shape}")

    # Process all files
    total_seconds = 0.0
    all_scores: list[float] = []
    # Track triggers per threshold+confirm combo
    triggers: dict[str, list[float]] = {}
    for th in THRESHOLDS:
        for cc in CONFIRM_COUNTS:
            triggers[f"{th}_{cc}"] = []

    t0 = time.time()
    files_done = 0

    for fi, fpath in enumerate(flacs):
        audio, sr = sf.read(fpath, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            # Resample
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        duration_s = len(audio) / SAMPLE_RATE
        total_seconds += duration_s

        # Convert to int16-range float32 (CRITICAL normalization)
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.float32)

        # Extract mel spectrogram
        mel_out = mel_sess.run(None, {mel_inp: audio_int16.reshape(1, -1)})[0]
        mel_raw = mel_out.squeeze().reshape(-1, MEL_BINS).astype(np.float32)
        mel = mel_raw / 10.0 + 2.0

        # Extract embeddings with stride 8
        embeddings = []
        for start in range(0, mel.shape[0] - MEL_FRAMES_PER_EMBEDDING + 1, MEL_STRIDE):
            chunk = mel[start:start + MEL_FRAMES_PER_EMBEDDING].reshape(
                1, MEL_FRAMES_PER_EMBEDDING, MEL_BINS, 1
            )
            emb_out = emb_sess.run(None, {emb_inp: chunk})[0].flatten()
            embeddings.append(emb_out)

        if len(embeddings) < SEQ_LEN:
            continue

        # Score with temporal_cnn using rolling window
        file_scores = []
        for i in range(SEQ_LEN - 1, len(embeddings)):
            window = np.stack(embeddings[i - SEQ_LEN + 1:i + 1])
            temporal_input = window.reshape(1, SEQ_LEN, EMBEDDING_DIM).astype(np.float32)
            score = float(tcnn_sess.run(None, {tcnn_inp: temporal_input})[0].flatten()[0])
            file_scores.append(score)
            all_scores.append(score)

        # Check triggers for each threshold+confirm combo
        for th in THRESHOLDS:
            for cc in CONFIRM_COUNTS:
                key = f"{th}_{cc}"
                confirm_counter = 0
                last_trigger_idx = -999
                for idx, s in enumerate(file_scores):
                    if s >= th:
                        confirm_counter += 1
                    else:
                        confirm_counter = 0
                    if confirm_counter >= cc:
                        # Cooldown: ~50 frames per second at embedding rate
                        emb_per_sec = SAMPLE_RATE / (MEL_STRIDE * 160)  # approx
                        cooldown_frames = int(COOLDOWN_S * emb_per_sec)
                        if idx - last_trigger_idx > cooldown_frames:
                            triggers[key].append(total_seconds - duration_s + idx * 0.16)
                            last_trigger_idx = idx
                            confirm_counter = 0

        files_done += 1
        if files_done % 100 == 0:
            elapsed = time.time() - t0
            hours = total_seconds / 3600
            print(f"  [{files_done}/{len(flacs)}] {hours:.2f}h processed, {elapsed:.0f}s elapsed")

    total_hours = total_seconds / 3600
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"TEMPORAL_CNN STREAMING FAPH — LibriSpeech test-clean")
    print(f"{'='*60}")
    print(f"Total audio: {total_hours:.2f}h ({total_seconds:.0f}s)")
    print(f"Files: {files_done}")
    print(f"Total scores: {len(all_scores)}")
    print(f"Processing time: {elapsed:.0f}s ({elapsed/total_seconds:.2f}x realtime)")
    print()

    # Score distribution
    scores_arr = np.array(all_scores)
    print(f"Score distribution:")
    print(f"  min={scores_arr.min():.6f} max={scores_arr.max():.6f}")
    print(f"  mean={scores_arr.mean():.6f} std={scores_arr.std():.6f}")
    print(f"  p50={np.percentile(scores_arr, 50):.6f}")
    print(f"  p90={np.percentile(scores_arr, 90):.6f}")
    print(f"  p95={np.percentile(scores_arr, 95):.6f}")
    print(f"  p99={np.percentile(scores_arr, 99):.6f}")
    print(f"  p99.9={np.percentile(scores_arr, 99.9):.6f}")
    print()

    # FAPH results
    print(f"FAPH Results (temporal_cnn on test-clean, {total_hours:.2f}h):")
    print(f"{'Confirm':>8} | {'@0.80':>8} | {'@0.85':>8} | {'@0.90':>8} | {'@0.95':>8}")
    print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for cc in CONFIRM_COUNTS:
        row = f"{cc}-of-{cc:>2} |"
        for th in THRESHOLDS:
            key = f"{th}_{cc}"
            n = len(triggers[key])
            faph = n / total_hours if total_hours > 0 else 0
            row += f" {faph:>7.2f} |"
        print(row)
    print()

    # Detailed trigger info for the production config (0.80, 3-of-3)
    prod_key = "0.8_3"
    prod_triggers = triggers[prod_key]
    print(f"Production config (0.80, 3-of-3): {len(prod_triggers)} triggers = {len(prod_triggers)/total_hours:.2f} FAPH")
    if prod_triggers:
        print(f"  Trigger timestamps: {[f'{t:.1f}s' for t in prod_triggers[:20]]}")

    # Save results
    results = {
        "model": "temporal_cnn",
        "corpus": "librispeech-test-clean",
        "total_hours": round(total_hours, 4),
        "total_files": files_done,
        "total_scores": len(all_scores),
        "score_stats": {
            "min": float(scores_arr.min()),
            "max": float(scores_arr.max()),
            "mean": float(scores_arr.mean()),
            "std": float(scores_arr.std()),
            "p99": float(np.percentile(scores_arr, 99)),
            "p999": float(np.percentile(scores_arr, 99.9)),
        },
        "faph": {},
    }
    for cc in CONFIRM_COUNTS:
        results["faph"][f"{cc}-of-{cc}"] = {}
        for th in THRESHOLDS:
            key = f"{th}_{cc}"
            n = len(triggers[key])
            faph = n / total_hours if total_hours > 0 else 0
            results["faph"][f"{cc}-of-{cc}"][str(th)] = {
                "triggers": n,
                "faph": round(faph, 2),
                "timestamps": [round(t, 1) for t in triggers[key]],
            }

    out_path = WAKEWORD / "experiments" / "faph_temporal_cnn_testclean.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
