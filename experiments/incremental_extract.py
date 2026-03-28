"""
Incremental embedding extraction — add new sources to existing cache.
Avoids re-extracting the 35K+ embeddings already cached.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
EXPERIMENTS = WAKEWORD / "experiments"
CACHE_FILE = EXPERIMENTS / "embedding_cache.npz"
CONFIG_FILE = EXPERIMENTS / "experiment_config.json"


def extract_new_sources(target_tags: list[str] | None = None) -> None:
    """Extract embeddings for sources not already in the cache."""
    # Load existing cache
    data = np.load(CACHE_FILE, allow_pickle=True)
    existing_tags = set(np.unique(data["tags"]))
    print(f"Existing cache: {len(data['embeddings'])} embeddings, tags: {sorted(existing_tags)}")

    # Load config
    with open(CONFIG_FILE) as f:
        config = json.load(f)

    # Find missing sources
    all_source_tags = set(config["data_sources"].keys())
    if target_tags:
        missing = [t for t in target_tags if t not in existing_tags]
    else:
        missing = sorted(all_source_tags - existing_tags)

    if not missing:
        print("No new sources to extract!")
        return

    print(f"Will extract: {missing}")

    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    def _embed(audio: np.ndarray) -> np.ndarray | None:
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

    # Start with existing data
    all_embs = list(data["embeddings"])
    all_labels = list(data["labels"])
    all_tags = list(data["tags"])
    all_files = list(data["files"])
    all_source_idx = list(data["source_idx"])

    for tag in missing:
        info = config["data_sources"][tag]
        path_str = info["path"]
        label = info["label"]
        source_type = info.get("type", "audio_dir")

        # Resolve path
        path = Path(path_str)
        if not path.is_absolute():
            path = WAKEWORD / path

        if source_type == "precomputed_npz":
            if not path.exists():
                print(f"  SKIP {tag}: {path} not found")
                continue
            print(f"  {tag}: loading pre-computed NPZ...", end="", flush=True)
            t0 = time.time()
            npz = np.load(path, allow_pickle=True)
            if "embeddings" in npz:
                embs = npz["embeddings"].astype(np.float32)
            elif "x" in npz:
                embs = npz["x"].astype(np.float32)
            else:
                keys = list(npz.keys())
                embs = npz[keys[0]].astype(np.float32)

            n = len(embs)
            all_embs.extend(embs)
            all_labels.extend([label] * n)
            all_tags.extend([tag] * n)
            all_files.extend([f"{path}#{i}" for i in range(n)])
            all_source_idx.extend(list(range(n)))
            print(f" -> {n} embeddings in {time.time()-t0:.0f}s")
            continue

        if not path.exists():
            print(f"  SKIP {tag}: {path} not found")
            continue

        files = sorted(list(path.rglob("*.wav")) + list(path.rglob("*.flac")))
        if not files:
            print(f"  SKIP {tag}: no audio files")
            continue

        print(f"  {tag}: {len(files)} files (label={label})...", flush=True)
        t0 = time.time()
        count = 0

        for fi, fpath in enumerate(files):
            audio = load_audio(fpath)
            if audio is None:
                continue
            emb = _embed(audio)
            if emb is not None:
                all_embs.append(emb)
                all_labels.append(label)
                all_tags.append(tag)
                all_files.append(str(fpath))
                all_source_idx.append(fi)
                count += 1

            if (fi + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (fi + 1) / elapsed
                print(f"    {fi+1}/{len(files)} ({rate:.0f}/s)")

        elapsed = time.time() - t0
        print(f"  {tag}: {count} embeddings in {elapsed:.0f}s")

    # Save updated cache
    old_count = len(data["embeddings"])  # capture before overwriting file
    embs_arr = np.array(all_embs, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)
    tags_arr = np.array(all_tags, dtype=object)
    files_arr = np.array(all_files, dtype=object)
    source_arr = np.array(all_source_idx, dtype=np.int32)

    np.savez_compressed(
        CACHE_FILE,
        embeddings=embs_arr,
        labels=labels_arr,
        tags=tags_arr,
        files=files_arr,
        source_idx=source_arr,
    )
    new_total = len(embs_arr)
    print(f"\nSaved updated cache: {new_total} embeddings ({new_total - old_count} new)")

    # Summary
    unique_tags, counts = np.unique(tags_arr, return_counts=True)
    for t, c in sorted(zip(unique_tags, counts), key=lambda x: -x[1]):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    import sys
    tags = sys.argv[1:] if len(sys.argv) > 1 else None
    extract_new_sources(tags)
