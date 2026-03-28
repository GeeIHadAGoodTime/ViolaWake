"""
Download ACAV100M pre-computed OWW embeddings from HuggingFace.

The openWakeWord project provides pre-extracted embeddings from ACAV100M
(AudioSet-like dataset). These are 32-dim OWW embeddings ready for training.

Usage:
  python experiments/download_acav100m.py
  python experiments/download_acav100m.py --max-files 10  # Limit for testing
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
CORPUS_DIR = WAKEWORD / "corpus"
OUTPUT_FILE = CORPUS_DIR / "acav100m_embeddings.npz"


def download_from_huggingface():
    """Download ACAV100M embeddings from openWakeWord HuggingFace repo."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download, list_repo_files

    # openWakeWord stores pre-computed embeddings in their HuggingFace repo
    repo_id = "dscripka/openwakeword"
    print(f"Checking repository: {repo_id}")

    try:
        files = list_repo_files(repo_id)
        print(f"Found {len(files)} files in repo")

        # Look for embedding files (typically .npy or .npz)
        embedding_files = [f for f in files if "acav" in f.lower() or "embedding" in f.lower() or f.endswith(".npy") or f.endswith(".npz")]
        if embedding_files:
            print(f"Potential embedding files: {embedding_files}")
        else:
            print(f"No ACAV embedding files found directly. Listing all files:")
            for f in sorted(files):
                print(f"  {f}")
    except Exception as e:
        print(f"Error listing repo files: {e}")

    # Try direct download of known embedding paths
    known_paths = [
        "embeddings/acav100m_embeddings.npy",
        "acav100m_embeddings.npy",
        "embeddings/acav100m.npz",
        "acav100m_embeddings.npz",
    ]

    for path in known_paths:
        try:
            print(f"  Trying: {path}...")
            local = hf_hub_download(repo_id=repo_id, filename=path)
            print(f"  Downloaded: {local}")
            return local
        except Exception:
            continue

    return None


def try_datasets_approach():
    """Try using the datasets library to find ACAV100M embeddings."""
    try:
        from datasets import load_dataset
        print("Trying datasets approach...")
        # Try loading as a dataset
        ds = load_dataset("dscripka/openwakeword", split="train", streaming=True)
        sample = next(iter(ds))
        print(f"Dataset sample keys: {sample.keys()}")
        return ds
    except Exception as e:
        print(f"Datasets approach failed: {e}")
        return None


def generate_alternative_negatives():
    """If ACAV100M isn't available, generate alternative negative embeddings
    from existing audio sources using our embedding pipeline."""
    print("\nFallback: Generating negative embeddings from available audio sources...")

    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    def _embed(audio):
        audio = center_crop(audio, CLIP_SAMPLES)
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        if len(audio_int16) < CLIP_SAMPLES:
            audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:CLIP_SAMPLES]
        try:
            embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            return embeddings.mean(axis=1)[0].astype(np.float32)
        except Exception:
            return None

    # Check MUSAN if available
    musan_speech = WAKEWORD / "corpus" / "musan" / "musan" / "speech"
    if musan_speech.exists():
        files = sorted(musan_speech.rglob("*.wav"))
        print(f"Found {len(files)} MUSAN speech files")
        embeddings = []
        for i, f in enumerate(files):
            audio = load_audio(f)
            if audio is None:
                continue
            emb = _embed(audio)
            if emb is not None:
                embeddings.append(emb)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(files)}...", flush=True)
        if embeddings:
            arr = np.array(embeddings, dtype=np.float32)
            np.savez_compressed(OUTPUT_FILE, embeddings=arr)
            print(f"Saved {len(arr)} MUSAN speech embeddings to {OUTPUT_FILE}")
            return OUTPUT_FILE
    else:
        print("MUSAN not yet available")

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fallback", action="store_true",
                        help="Use fallback embedding generation instead of download")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    if args.fallback:
        generate_alternative_negatives()
        return

    print("=" * 60)
    print("ACAV100M Embedding Download")
    print("=" * 60)

    result = download_from_huggingface()
    if result:
        # Convert to our format
        print(f"\nConverting to standard format...")
        if result.endswith(".npy"):
            data = np.load(result)
        else:
            data = np.load(result, allow_pickle=True)
            if "embeddings" in data:
                data = data["embeddings"]
            else:
                keys = list(data.keys())
                data = data[keys[0]]

        print(f"Shape: {data.shape}, dtype: {data.dtype}")
        np.savez_compressed(OUTPUT_FILE, embeddings=data.astype(np.float32))
        print(f"Saved to {OUTPUT_FILE}")
    else:
        print("\nACAV100M embeddings not directly available from HuggingFace.")
        print("Trying fallback: generate embeddings from available audio...")
        generate_alternative_negatives()


if __name__ == "__main__":
    main()
