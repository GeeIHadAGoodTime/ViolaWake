"""Check for train/eval data contamination (overlap between training and evaluation sets).

Detects three types of overlap:
  - filename: Identical filenames across directories
  - hash: Identical file content via SHA-256
  - embedding: Near-duplicate OWW embeddings (cosine similarity > 0.99)

Usage::

    # Quick filename check
    python -m violawake_sdk.tools.contamination_check \\
        --train data/train/ --eval data/eval/ --method filename

    # Thorough embedding-based check
    python -m violawake_sdk.tools.contamination_check \\
        --train data/train/ --eval data/eval/ --method embedding
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_AUDIO_EXTENSIONS = {"*.wav", "*.flac", "*.mp3", "*.ogg"}


def _collect_audio_files(directory: str) -> list[Path]:
    """Recursively collect audio files from a directory."""
    d = Path(directory)
    files: list[Path] = []
    for ext in _AUDIO_EXTENSIONS:
        files.extend(d.rglob(ext))
    return sorted(files)


def _check_filename_overlap(train_files: list[Path], eval_files: list[Path]) -> dict:
    """Check for identical filenames across directories."""
    train_names = {f.name for f in train_files}
    eval_names = {f.name for f in eval_files}
    overlap = train_names & eval_names

    return {
        "method": "filename",
        "train_files": len(train_files),
        "eval_files": len(eval_files),
        "overlap_count": len(overlap),
        "overlapping_files": sorted(overlap),
        "contamination_rate": len(overlap) / max(len(eval_names), 1),
    }


def _check_hash_overlap(train_files: list[Path], eval_files: list[Path]) -> dict:
    """Check for identical file content via SHA-256."""

    def _file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    train_hashes: dict[str, Path] = {}
    for f in train_files:
        train_hashes[_file_hash(f)] = f

    overlapping: list[tuple[str, str]] = []
    eval_hashes_seen: set[str] = set()
    for f in eval_files:
        h = _file_hash(f)
        eval_hashes_seen.add(h)
        if h in train_hashes:
            overlapping.append((str(train_hashes[h]), str(f)))

    return {
        "method": "hash",
        "train_files": len(train_files),
        "eval_files": len(eval_files),
        "overlap_count": len(overlapping),
        "overlapping_files": overlapping,
        "contamination_rate": len(overlapping) / max(len(eval_files), 1),
    }


def _check_embedding_overlap(
    train_files: list[Path],
    eval_files: list[Path],
    cosine_threshold: float = 0.99,
) -> dict:
    """Check for near-duplicate embeddings (cosine similarity > threshold).

    Loads openwakeword to extract embeddings, then computes pairwise cosine
    similarity between all train and eval embeddings.
    """
    import numpy as np

    try:
        from openwakeword.model import Model as OWWModel
    except ImportError:
        return {
            "method": "embedding",
            "error": "openwakeword not installed. Install with: pip install openwakeword",
            "overlap_count": -1,
        }

    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor

    def _embed_file(path: Path) -> np.ndarray | None:
        audio = load_audio(path)
        if audio is None:
            return None
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

    print(f"Extracting train embeddings ({len(train_files)} files)...", file=sys.stderr)
    train_embs: list[tuple[Path, np.ndarray]] = []
    for f in train_files:
        emb = _embed_file(f)
        if emb is not None:
            train_embs.append((f, emb))

    print(f"Extracting eval embeddings ({len(eval_files)} files)...", file=sys.stderr)
    eval_embs: list[tuple[Path, np.ndarray]] = []
    for f in eval_files:
        emb = _embed_file(f)
        if emb is not None:
            eval_embs.append((f, emb))

    if not train_embs or not eval_embs:
        return {
            "method": "embedding",
            "train_files": len(train_files),
            "eval_files": len(eval_files),
            "train_embeddings": len(train_embs),
            "eval_embeddings": len(eval_embs),
            "overlap_count": 0,
            "overlapping_files": [],
            "contamination_rate": 0.0,
        }

    # Build matrices for batch cosine similarity
    train_matrix = np.array([e for _, e in train_embs])
    eval_matrix = np.array([e for _, e in eval_embs])

    # Normalize
    train_norms = np.linalg.norm(train_matrix, axis=1, keepdims=True)
    eval_norms = np.linalg.norm(eval_matrix, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    eval_norms[eval_norms == 0] = 1.0
    train_normed = train_matrix / train_norms
    eval_normed = eval_matrix / eval_norms

    # Cosine similarity matrix: (n_eval, n_train)
    sim_matrix = eval_normed @ train_normed.T

    overlapping: list[dict] = []
    for i in range(len(eval_embs)):
        for j in range(len(train_embs)):
            if sim_matrix[i, j] >= cosine_threshold:
                overlapping.append(
                    {
                        "train_file": str(train_embs[j][0]),
                        "eval_file": str(eval_embs[i][0]),
                        "cosine_similarity": float(sim_matrix[i, j]),
                    }
                )

    return {
        "method": "embedding",
        "cosine_threshold": cosine_threshold,
        "train_files": len(train_files),
        "eval_files": len(eval_files),
        "train_embeddings": len(train_embs),
        "eval_embeddings": len(eval_embs),
        "overlap_count": len(overlapping),
        "overlapping_files": overlapping,
        "contamination_rate": len(overlapping) / max(len(eval_embs), 1),
    }


def check_contamination(
    train_dir: str,
    eval_dir: str,
    method: str = "filename",
    cosine_threshold: float = 0.99,
) -> dict:
    """Detect overlap between training and evaluation datasets.

    Methods:
    - filename: Check for identical filenames across directories
    - hash: Check for identical file content via SHA-256
    - embedding: Check for near-duplicate embeddings (cosine > threshold)

    Args:
        train_dir: Path to training data directory.
        eval_dir: Path to evaluation data directory.
        method: Detection method — "filename", "hash", or "embedding".
        cosine_threshold: Similarity threshold for embedding method (default 0.99).

    Returns:
        Dict with overlap_count, overlapping_files, and contamination_rate.
    """
    train_files = _collect_audio_files(train_dir)
    eval_files = _collect_audio_files(eval_dir)

    if method == "filename":
        return _check_filename_overlap(train_files, eval_files)
    elif method == "hash":
        return _check_hash_overlap(train_files, eval_files)
    elif method == "embedding":
        return _check_embedding_overlap(train_files, eval_files, cosine_threshold)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'filename', 'hash', or 'embedding'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-contamination-check",
        description="Check for train/eval data contamination.",
    )
    parser.add_argument("--train", required=True, help="Training data directory")
    parser.add_argument("--eval", required=True, help="Evaluation data directory")
    parser.add_argument(
        "--method",
        default="filename",
        choices=["filename", "hash", "embedding"],
        help="Detection method (default: filename)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.99,
        help="Cosine similarity threshold for embedding method (default: 0.99)",
    )

    args = parser.parse_args()

    if not Path(args.train).is_dir():
        print(f"ERROR: Training directory not found: {args.train}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.eval).is_dir():
        print(f"ERROR: Evaluation directory not found: {args.eval}", file=sys.stderr)
        sys.exit(1)

    import json

    result = check_contamination(
        args.train,
        args.eval,
        args.method,
        args.cosine_threshold,
    )

    print(json.dumps(result, indent=2))

    if result.get("overlap_count", 0) > 0:
        rate = result.get("contamination_rate", 0)
        print(
            f"\nWARNING: {result['overlap_count']} overlapping items found "
            f"({rate * 100:.1f}% contamination rate)",
            file=sys.stderr,
        )
        sys.exit(1)
    else:
        print("\nNo contamination detected.", file=sys.stderr)


if __name__ == "__main__":
    main()
