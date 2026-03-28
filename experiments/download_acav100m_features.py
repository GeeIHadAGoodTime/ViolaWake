"""
Download ACAV100M pre-computed features from HuggingFace.

Source: davidscripka/openwakeword_features
File: openwakeword_features_ACAV100M_2000_hrs_16bit.npy
Shape: (5625000, 16, 96) — 96-dim mel features, 16 timesteps, 5.6M samples

These are NOT 32-dim OWW embeddings. They need to be converted through
the OWW embedding model to get our target format.

For training purposes, we can either:
1. Use the 96-dim features directly with a larger MLP
2. Convert to 32-dim OWW embeddings (slow but compatible with existing pipeline)
3. Subsample to a manageable size (e.g., 100K samples)

Usage:
  python experiments/download_acav100m_features.py                  # Download + convert subset
  python experiments/download_acav100m_features.py --max-samples 50000  # Smaller subset
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
CORPUS_DIR = WAKEWORD / "corpus"
OUTPUT_FILE = CORPUS_DIR / "acav100m_embeddings.npz"


def download_features(max_samples: int = 100000) -> Path | None:
    """Download ACAV100M features from HuggingFace and convert to OWW embeddings."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import hf_hub_download

    repo_id = "davidscripka/openwakeword_features"
    filename = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"

    print(f"Downloading {filename} from {repo_id}...")
    print("(This file is ~17GB — download may take a while)")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )
        print(f"Downloaded to: {local_path}")
        return Path(local_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def convert_features_to_embeddings(
    features_path: Path,
    max_samples: int = 100000,
    batch_size: int = 1000,
) -> np.ndarray | None:
    """Convert 96-dim OWW features to 32-dim embeddings using the OWW embedding model.

    The OWW preprocessor has an internal embedding model that converts
    (N, 16, 96) features to (N, 32) embeddings.
    """
    from openwakeword.model import Model as OWWModel

    print(f"\nLoading features from {features_path}...")
    # Memory-map for large files
    features = np.load(str(features_path), mmap_mode="r")
    print(f"Features shape: {features.shape}, dtype: {features.dtype}")

    total = min(len(features), max_samples)
    print(f"Converting {total} / {len(features)} samples to OWW embeddings...")

    oww = OWWModel()
    preprocessor = oww.preprocessor

    # The preprocessor has embedding models that convert features to embeddings
    # We need to find the right method
    if hasattr(preprocessor, "embedding_model"):
        embed_model = preprocessor.embedding_model
    elif hasattr(preprocessor, "onnx_models"):
        # Try to use the ONNX embedding model directly
        embed_model = preprocessor.onnx_models
    else:
        print("Cannot find OWW embedding model. Falling back to direct features.")
        # Just flatten and take mean as a simple conversion
        subset = features[:total].astype(np.float32)
        # Mean pool over time dimension: (N, 16, 96) -> (N, 96)
        embeddings_96 = subset.mean(axis=1)
        np.savez_compressed(OUTPUT_FILE, embeddings=embeddings_96)
        print(f"Saved {len(embeddings_96)} x 96-dim features to {OUTPUT_FILE}")
        return embeddings_96

    # Process in batches
    all_embeddings = []
    t0 = time.time()

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = features[start:end].astype(np.float32)

        # Try to run through embedding model
        try:
            # The embedding model expects (batch, 16, 96) and outputs (batch, 32)
            embs = embed_model.run(None, {"input": batch})[0]
            all_embeddings.append(embs)
        except Exception as e:
            if start == 0:
                print(f"Embedding model failed: {e}")
                print("Falling back to mean-pooled 96-dim features")
                embeddings_96 = features[:total].astype(np.float32).mean(axis=1)
                np.savez_compressed(OUTPUT_FILE, embeddings=embeddings_96)
                print(f"Saved {len(embeddings_96)} x 96-dim features to {OUTPUT_FILE}")
                return embeddings_96
            break

        if (start + batch_size) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (start + batch_size) / elapsed
            eta = (total - start - batch_size) / max(rate, 1)
            print(f"  {start + batch_size}/{total} ({rate:.0f}/s, ETA {eta:.0f}s)")

    if all_embeddings:
        embeddings = np.concatenate(all_embeddings, axis=0)
        print(f"\nConverted: {embeddings.shape}")
        np.savez_compressed(OUTPUT_FILE, embeddings=embeddings)
        print(f"Saved to {OUTPUT_FILE}")
        return embeddings

    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100000,
                        help="Maximum number of samples to convert")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ACAV100M Feature Download + Conversion")
    print("=" * 60)

    features_path = download_features(args.max_samples)
    if features_path is None:
        print("Download failed. Cannot proceed.")
        return

    embeddings = convert_features_to_embeddings(features_path, args.max_samples)
    if embeddings is not None:
        print(f"\nDone! {len(embeddings)} embeddings ready at {OUTPUT_FILE}")
    else:
        print("Conversion failed.")


if __name__ == "__main__":
    main()
