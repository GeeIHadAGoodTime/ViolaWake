"""
ViolaWake False Alarms Per Hour (FAPH) Test
============================================

Runs the ViolaWake MLP classifier on LibriSpeech test-clean (~5.4 hours
of read English speech containing NO wake words) to measure false alarm rate.

Simulates the production sliding window: 1.5s window, 100ms step.
For each window, extracts OWW embedding, mean-pools, and scores through MLP.

Reports FAPH at multiple thresholds and logs the top-scoring windows.

Usage:
    python experiments/faph_test.py
    python experiments/faph_test.py --threshold 0.5
    python experiments/faph_test.py --top-k 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WAKEWORD_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = WAKEWORD_ROOT / "experiments" / "models" / "D_combined_bce_s42.onnx"
LIBRISPEECH_DIR = WAKEWORD_ROOT / "corpus" / "librispeech" / "LibriSpeech" / "test-clean"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 24000          # 1.5s at 16kHz
STEP_SAMPLES = 1600           # 100ms at 16kHz (production infer_interval)
DEBOUNCE_SECONDS = 2.0        # Production debounce — triggers within 2s of a previous trigger are suppressed
DEBOUNCE_SAMPLES = int(DEBOUNCE_SECONDS * SAMPLE_RATE)
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def init_oww_preprocessor():
    """Initialize the openwakeword preprocessor for embedding extraction.

    Uses AudioFeatures directly to avoid downloading all OWW wake word models.
    """
    from openwakeword.utils import AudioFeatures
    preprocessor = AudioFeatures()
    return preprocessor


def extract_embedding(preprocessor, audio_int16: np.ndarray) -> np.ndarray:
    """Extract 96-dim OWW embedding from a 1.5s int16 audio clip.

    Uses mean pooling across time frames to match training/eval pipeline.
    """
    # embed_clips expects shape (1, samples)
    embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
    # embeddings shape: (1, num_frames, 96) — mean pool across time
    return embeddings.mean(axis=1)[0].astype(np.float32)


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_path: Path) -> ort.InferenceSession:
    """Load the ONNX MLP model."""
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return session


def score_embedding(session: ort.InferenceSession, embedding: np.ndarray) -> float:
    """Score a 96-dim embedding through the MLP. Returns probability 0-1."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run(
        [output_name],
        {input_name: embedding.reshape(1, -1)},
    )
    return float(result[0][0][0])


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio_16k(path: Path) -> np.ndarray:
    """Load audio file, resample to 16kHz mono if needed, return float32."""
    audio, sr = sf.read(path, dtype="float32")
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample if not 16kHz
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio [-1, 1] to int16."""
    return (audio * 32767).clip(-32768, 32767).astype(np.int16)


# ---------------------------------------------------------------------------
# Main FAPH test
# ---------------------------------------------------------------------------

def find_flac_files(librispeech_dir: Path) -> list[Path]:
    """Find all .flac files in the LibriSpeech directory."""
    files = sorted(librispeech_dir.rglob("*.flac"))
    if not files:
        print(f"ERROR: No .flac files found in {librispeech_dir}")
        print("Make sure LibriSpeech test-clean is extracted there.")
        sys.exit(1)
    return files


def run_faph_test(
    model_path: Path = MODEL_PATH,
    librispeech_dir: Path = LIBRISPEECH_DIR,
    thresholds: list[float] | None = None,
    top_k: int = 20,
) -> dict:
    """Run the FAPH test and return results."""
    if thresholds is None:
        thresholds = THRESHOLDS

    print("=" * 60)
    print("ViolaWake FAPH Test — LibriSpeech test-clean")
    print("=" * 60)

    # Find audio files
    flac_files = find_flac_files(librispeech_dir)
    print(f"Found {len(flac_files)} .flac files")

    # Load model
    print(f"Loading model: {model_path.name}")
    session = load_model(model_path)

    # Init OWW preprocessor
    print("Initializing OWW preprocessor...")
    preprocessor = init_oww_preprocessor()

    # Tracking
    total_windows = 0
    total_audio_samples = 0
    trigger_counts_raw = {t: 0 for t in thresholds}       # raw (no debounce)
    trigger_counts_debounced = {t: 0 for t in thresholds}  # with 2s debounce
    last_trigger_pos = {t: -999999 for t in thresholds}    # per-file debounce tracking
    # Top-K tracking: list of (score, file, window_start_sec)
    top_scores: list[tuple[float, str, float]] = []
    min_top_score = 0.0  # minimum score currently in top-K

    t_start = time.time()
    files_processed = 0

    for i, fpath in enumerate(flac_files):
        # Load audio
        try:
            audio = load_audio_16k(fpath)
        except Exception as e:
            print(f"  WARN: Failed to load {fpath.name}: {e}")
            continue

        n_samples = len(audio)
        total_audio_samples += n_samples

        # Reset debounce tracking per file (different files are independent events)
        for t in thresholds:
            last_trigger_pos[t] = -999999

        # Slide window through this file
        pos = 0
        while pos + CLIP_SAMPLES <= n_samples:
            window = audio[pos : pos + CLIP_SAMPLES]
            window_int16 = float_to_int16(window)

            # Extract embedding and score
            embedding = extract_embedding(preprocessor, window_int16)
            score = score_embedding(session, embedding)
            total_windows += 1

            # Check thresholds (both raw and debounced)
            for t in thresholds:
                if score >= t:
                    trigger_counts_raw[t] += 1
                    # Debounce: only count if >2s since last trigger at this threshold
                    if pos - last_trigger_pos[t] >= DEBOUNCE_SAMPLES:
                        trigger_counts_debounced[t] += 1
                        last_trigger_pos[t] = pos

            # Track top-K
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

    # Build results
    print()
    print("=" * 60)
    print("RESULTS")
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

    results = {
        "model": model_path.name,
        "dataset": "LibriSpeech test-clean",
        "total_hours": round(total_hours, 4),
        "total_windows": total_windows,
        "total_files": files_processed,
        "elapsed_seconds": round(elapsed, 1),
        "clip_samples": CLIP_SAMPLES,
        "step_samples": STEP_SAMPLES,
        "pooling": "mean",
        "debounce_seconds": DEBOUNCE_SECONDS,
        "faph": faph_results,
        "top_scores": top_scores_list,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="ViolaWake FAPH test on LibriSpeech")
    parser.add_argument(
        "--model", type=Path, default=MODEL_PATH,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--librispeech-dir", type=Path, default=LIBRISPEECH_DIR,
        help="Path to LibriSpeech test-clean directory",
    )
    parser.add_argument(
        "--threshold", type=float, nargs="*", default=None,
        help="Thresholds to evaluate (default: 0.5 0.6 0.7 0.8 0.9 0.95)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top-scoring windows to report",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "faph_results.json",
        help="Output JSON file",
    )
    args = parser.parse_args()

    thresholds = args.threshold if args.threshold else THRESHOLDS

    results = run_faph_test(
        model_path=args.model,
        librispeech_dir=args.librispeech_dir,
        thresholds=thresholds,
        top_k=args.top_k,
    )

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
