"""
Mine hard negatives from LibriSpeech dev-clean.

Slides a 1.5s window (24000 samples) with 100ms step across ALL dev-clean files.
For each window scoring >0.3 through round2_best.onnx, extracts the OWW embedding.
Deduplicates overlapping windows within 2 seconds in same file (keeps peak).
Saves hard negative embeddings to experiments/devclean_hard_neg_r3.npz.

Usage:
    python experiments/mine_devclean.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# ---------------------------------------------------------------------------
WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = WAKEWORD_ROOT / "experiments" / "models" / "round2_best.onnx"
LIBRISPEECH_DIR = WAKEWORD_ROOT / "corpus" / "librispeech" / "LibriSpeech" / "dev-clean"
OUTPUT_PATH = WAKEWORD_ROOT / "experiments" / "devclean_hard_neg_r3.npz"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000       # 1.5s
STEP_SAMPLES = 1600        # 100ms
SCORE_THRESHOLD = 0.3
DEDUP_SECONDS = 2.0
DEDUP_SAMPLES = int(DEDUP_SECONDS * SAMPLE_RATE)


def init_oww_preprocessor():
    from openwakeword.utils import AudioFeatures
    return AudioFeatures()


def extract_embedding(preprocessor, audio_int16: np.ndarray) -> np.ndarray:
    embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    return embeddings.mean(axis=1)[0].astype(np.float32)


def load_model(model_path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def score_embedding(session: ort.InferenceSession, embedding: np.ndarray) -> float:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: embedding.reshape(1, -1)})
    return float(result[0][0][0])


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


def find_flac_files(d: Path) -> list[Path]:
    files = sorted(d.rglob("*.flac"))
    print(f"Found {len(files)} .flac files in {d}")
    return files


def deduplicate_hits(hits: list[dict]) -> list[dict]:
    """Deduplicate: for overlapping windows from same file within 2s, keep peak."""
    if not hits:
        return []

    # Group by file
    by_file: dict[str, list[dict]] = {}
    for h in hits:
        by_file.setdefault(h["file"], []).append(h)

    deduped = []
    for fname, file_hits in by_file.items():
        # Sort by position
        file_hits.sort(key=lambda x: x["pos"])
        # Greedy merge: walk through, merge overlapping within 2s, keep peak
        groups = []
        current_group = [file_hits[0]]
        for h in file_hits[1:]:
            if h["pos"] - current_group[-1]["pos"] < DEDUP_SAMPLES:
                current_group.append(h)
            else:
                groups.append(current_group)
                current_group = [h]
        groups.append(current_group)

        for group in groups:
            best = max(group, key=lambda x: x["score"])
            deduped.append(best)

    return deduped


def main():
    print("=" * 60)
    print("Mining dev-clean hard negatives for Round 3")
    print("=" * 60)
    print(f"Model: {MODEL_PATH.name}")
    print(f"Threshold: {SCORE_THRESHOLD}")
    print()

    flac_files = find_flac_files(LIBRISPEECH_DIR)
    session = load_model(MODEL_PATH)

    print("Initializing OWW preprocessor...")
    preprocessor = init_oww_preprocessor()

    hits: list[dict] = []  # Each: {file, pos, score, embedding}
    total_windows = 0
    total_audio_samples = 0
    t_start = time.time()

    for i, fpath in enumerate(flac_files):
        try:
            audio, sr = sf.read(fpath, dtype="float32")
        except Exception as e:
            print(f"  WARN: Failed to load {fpath.name}: {e}")
            continue

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        n_samples = len(audio)
        total_audio_samples += n_samples
        rel_path = str(fpath.relative_to(LIBRISPEECH_DIR))

        pos = 0
        while pos + CLIP_SAMPLES <= n_samples:
            window = audio[pos:pos + CLIP_SAMPLES]
            window_int16 = float_to_int16(window)

            embedding = extract_embedding(preprocessor, window_int16)
            score = score_embedding(session, embedding)
            total_windows += 1

            if score > SCORE_THRESHOLD:
                hits.append({
                    "file": rel_path,
                    "pos": pos,
                    "score": score,
                    "embedding": embedding.copy(),
                })

            pos += STEP_SAMPLES

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            hours = total_audio_samples / SAMPLE_RATE / 3600
            print(
                f"  [{i+1}/{len(flac_files)}] "
                f"{hours:.2f}h audio, {total_windows} windows, "
                f"{len(hits)} hits, elapsed {elapsed:.0f}s"
            )

    elapsed = time.time() - t_start
    total_hours = total_audio_samples / SAMPLE_RATE / 3600

    print()
    print(f"Total audio: {total_hours:.3f} hours")
    print(f"Total windows: {total_windows}")
    print(f"Raw hits (score > {SCORE_THRESHOLD}): {len(hits)}")
    print(f"Elapsed: {elapsed:.0f}s")

    # Deduplicate
    deduped = deduplicate_hits(hits)
    print(f"After deduplication (2s window): {len(deduped)}")

    # Score distribution
    if deduped:
        scores = np.array([h["score"] for h in deduped])
        for thresh in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            count = (scores >= thresh).sum()
            print(f"  score >= {thresh}: {count}")

    # Save
    if deduped:
        embeddings = np.array([h["embedding"] for h in deduped], dtype=np.float32)
        scores_arr = np.array([h["score"] for h in deduped], dtype=np.float32)
        files_arr = np.array([h["file"] for h in deduped], dtype=object)

        np.savez(
            OUTPUT_PATH,
            embeddings=embeddings,
            scores=scores_arr,
            files=files_arr,
        )
        print(f"\nSaved {len(deduped)} hard negative embeddings to {OUTPUT_PATH}")
    else:
        print("\nNo hard negatives found!")


if __name__ == "__main__":
    main()
