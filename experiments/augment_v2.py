"""
audiomentations v2 Augmentation Pipeline
=========================================

Uses the audiomentations library (43+ transforms) to create more realistic
augmentations than our basic noise+reverb pipeline.

This re-extracts positive embeddings with the v2 augmentation chain,
creating a new cache that can be used for training experiments.

Usage:
  python experiments/augment_v2.py                  # Generate augmented embeddings
  python experiments/augment_v2.py --factor 12      # More augmentations per sample
  python experiments/augment_v2.py --preview 5      # Preview augmentation on 5 samples
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import audiomentations as am
import numpy as np

WAKEWORD = Path("J:/CLAUDE/PROJECTS/Wakeword")
VIOLA = Path("J:/PROJECTS/NOVVIOLA_fixed3_patched/NOVVIOLA")
EXPERIMENTS = WAKEWORD / "experiments"
SAMPLE_RATE = 16000


def build_augmentation_chain(seed: int = 42) -> am.Compose:
    """Build a comprehensive augmentation chain using audiomentations.

    Designed for wake word detection: simulates real-world conditions
    (background noise, room acoustics, device proximity, mic quality).
    """
    return am.Compose([
        # ── Environmental noise ──
        am.OneOf([
            am.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
            am.AddGaussianSNR(min_snr_db=10.0, max_snr_db=35.0, p=1.0),
        ], p=0.6),

        # ── Room acoustics ──
        am.OneOf([
            am.RoomSimulator(
                min_size_x=2.0, max_size_x=6.0,
                min_size_y=2.0, max_size_y=5.0,
                min_size_z=2.2, max_size_z=3.5,
                min_absorption_value=0.2, max_absorption_value=0.8,
                p=1.0,
            ),
        ], p=0.4),

        # ── Gain variations (different distances from mic) ──
        am.Gain(min_gain_db=-12.0, max_gain_db=6.0, p=0.5),

        # ── Pitch shifting (natural voice variation) ──
        am.PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.3),

        # ── Time stretching (speaking speed) ──
        am.TimeStretch(min_rate=0.85, max_rate=1.15, p=0.3),

        # ── Frequency masking (simulates phone/speaker bandwidth) ──
        am.OneOf([
            am.LowPassFilter(min_cutoff_freq=2000, max_cutoff_freq=7500, p=1.0),
            am.HighPassFilter(min_cutoff_freq=100, max_cutoff_freq=500, p=1.0),
            am.BandPassFilter(
                min_center_freq=200, max_center_freq=4000,
                min_bandwidth_fraction=0.5, max_bandwidth_fraction=1.99,
                p=1.0,
            ),
        ], p=0.3),

        # ── Clipping (loud environments, mic saturation) ──
        am.Clip(a_min=-0.8, a_max=0.8, p=0.1),

        # ── Time shift (alignment jitter) ──
        am.Shift(min_shift=-0.1, max_shift=0.1, rollover=False, p=0.3),

        # ── Normalize to prevent level creep ──
        am.Normalize(p=0.3),

    ], shuffle=False)


def augment_and_extract(
    audio_dirs: list[tuple[str, Path]],
    output_npz: Path,
    factor: int = 8,
    seed: int = 42,
):
    """Augment positive audio files and extract OWW embeddings.

    Args:
        audio_dirs: List of (tag, directory) tuples
        output_npz: Where to save the augmented embeddings
        factor: Number of augmented variants per original
        seed: Random seed
    """
    from openwakeword.model import Model as OWWModel
    from violawake_sdk._constants import CLIP_SAMPLES
    from violawake_sdk.audio import center_crop, load_audio

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    chain = build_augmentation_chain(seed=seed)

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

    all_embeddings = []
    all_labels = []
    all_tags = []
    all_files = []
    all_source_idx = []

    for tag, directory in audio_dirs:
        if not directory.exists():
            print(f"  SKIP {tag}: {directory} not found")
            continue

        files = sorted(
            list(directory.rglob("*.wav")) + list(directory.rglob("*.flac"))
        )
        if tag == "pos_main":
            files = [f for f in files if "_excluded" not in str(f)]

        print(f"  {tag}: {len(files)} files x {factor+1} (orig + {factor} aug)...",
              end="", flush=True)
        t0 = time.time()
        count = 0

        for fi, fpath in enumerate(files):
            audio = load_audio(fpath)
            if audio is None:
                continue

            # Ensure float32 for audiomentations
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Original embedding
            emb = _embed(audio)
            if emb is not None:
                all_embeddings.append(emb)
                all_labels.append(1)
                all_tags.append(tag)
                all_files.append(str(fpath))
                all_source_idx.append(fi)
                count += 1

            # Augmented variants
            for aug_i in range(factor):
                try:
                    augmented = chain(samples=audio, sample_rate=SAMPLE_RATE)
                    emb = _embed(augmented)
                    if emb is not None:
                        all_embeddings.append(emb)
                        all_labels.append(1)
                        all_tags.append(tag)
                        all_files.append(f"{fpath}_augv2_{aug_i}")
                        all_source_idx.append(fi)
                        count += 1
                except Exception:
                    pass

            if (fi + 1) % 100 == 0:
                print(f" {fi+1}", end="", flush=True)

        elapsed = time.time() - t0
        print(f" -> {count} embeddings in {elapsed:.0f}s")

    if not all_embeddings:
        print("ERROR: No embeddings extracted!")
        return

    result = {
        "embeddings": np.array(all_embeddings, dtype=np.float32),
        "labels": np.array(all_labels, dtype=np.int32),
        "tags": np.array(all_tags),
        "files": np.array(all_files),
        "source_idx": np.array(all_source_idx, dtype=np.int32),
    }

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **result)
    n = len(all_embeddings)
    print(f"\nSaved {n} augmented embeddings to {output_npz}")
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=int, default=8,
                        help="Augmentation factor per positive sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", type=int, default=0,
                        help="Preview mode: just augment N files and report")
    args = parser.parse_args()

    # Positive sources to augment with v2 pipeline
    pos_sources = [
        ("pos_main", VIOLA / "violawake_data" / "positives"),
        ("pos_diverse", EXPERIMENTS / "training_data" / "diverse_positives"),
    ]

    output = EXPERIMENTS / "augmented_v2_cache.npz"

    if args.preview:
        print(f"Preview mode: augmenting {args.preview} files...")
        chain = build_augmentation_chain(args.seed)
        from violawake_sdk.audio import load_audio
        src = pos_sources[0][1]
        files = sorted(src.rglob("*.wav"))[:args.preview]
        for f in files:
            audio = load_audio(f)
            if audio is not None:
                aug = chain(samples=audio.astype(np.float32), sample_rate=SAMPLE_RATE)
                print(f"  {f.name}: orig={audio.shape}, aug={aug.shape}, "
                      f"orig_rms={np.sqrt(np.mean(audio**2)):.4f}, "
                      f"aug_rms={np.sqrt(np.mean(aug**2)):.4f}")
        return

    print("=" * 60)
    print("audiomentations v2 Augmentation Pipeline")
    print(f"Factor: {args.factor}, Seed: {args.seed}")
    print("=" * 60)

    augment_and_extract(pos_sources, output, factor=args.factor, seed=args.seed)


if __name__ == "__main__":
    main()
