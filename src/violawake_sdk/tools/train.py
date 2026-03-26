"""
violawake-train CLI -- Train a custom wake word model.

Entry point: ``violawake-train`` (declared in pyproject.toml).

Architecture: MLP classifier head on top of frozen OpenWakeWord (OWW) audio
embeddings. This is the same architecture used in production Viola
(Cohen's d 15.10 on a synthetic-negative benchmark; speech-negative d-prime TBD).

Training pipeline (from Viola's violawake/training/trainer.py):
  - FocalLoss for class imbalance handling
  - AdamW optimizer with cosine annealing LR schedule
  - Exponential Moving Average (EMA) of model weights
  - 80/20 train/validation split with early stopping
  - Multi-source negative embedding generation
  - Audio-level data augmentation before embedding extraction

Negative sources (in priority order):
  A. Real audio files via --negatives directory (best quality)
  B. Diverse synthetic negatives when no directory provided:
     white noise, pink noise, silence, sine tones, chirps, mixed
  C. (Future) Production training should use MUSAN + Common Voice corpus negatives

Usage::

    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --output models/jarvis.onnx \\
      --epochs 50 \\
      --augment

    # With real negative samples and evaluation:
    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --negatives data/jarvis/negatives/ \\
      --output models/jarvis.onnx \\
      --eval-dir data/jarvis/test/

Minimum: 20 positive samples. Recommended: 50+ samples.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _generate_synthetic_negatives(
    n_samples: int,
    clip_samples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Generate diverse synthetic negative audio clips.

    Covers more of the acoustic space than uniform random noise:
    white noise, pink noise, silence with DC offset, sine tones at
    speech frequencies, chirps (frequency sweeps), and tone+noise mixes.

    Args:
        n_samples: Number of synthetic clips to generate.
        clip_samples: Samples per clip (e.g. 24000 for 1.5s @ 16kHz).
        rng: Numpy random generator.

    Returns:
        List of float32 audio arrays, each of length clip_samples.

    Note:
        For production training, use real negative audio (MUSAN dataset,
        Common Voice non-target utterances, environmental recordings)
        via the --negatives flag. Synthetic negatives are a reasonable
        fallback but cannot match the diversity of real-world audio.
    """
    import numpy as np

    clips: list[np.ndarray] = []
    t = np.arange(clip_samples, dtype=np.float32) / 16000.0  # time axis in seconds

    # Distribute clips across types roughly evenly
    types = ["white", "pink", "silence", "tone", "chirp", "mixed"]
    per_type = max(1, n_samples // len(types))

    for i in range(n_samples):
        noise_type = types[i % len(types)] if i < per_type * len(types) else rng.choice(types)

        if noise_type == "white":
            # White noise at various amplitudes (0.01 to 0.5)
            amplitude = rng.uniform(0.01, 0.5)
            clip = rng.standard_normal(clip_samples).astype(np.float32) * amplitude

        elif noise_type == "pink":
            # Pink noise (1/f spectrum) via spectral shaping
            white = rng.standard_normal(clip_samples).astype(np.float32)
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(clip_samples, d=1.0 / 16000.0)
            freqs[0] = 1.0  # avoid division by zero
            fft /= np.sqrt(freqs)
            clip = np.fft.irfft(fft, n=clip_samples).astype(np.float32)
            amplitude = rng.uniform(0.01, 0.3)
            clip_std = clip.std()
            if clip_std > 1e-9:
                clip = clip / clip_std * amplitude

        elif noise_type == "silence":
            # Silence with small random DC offsets (simulates real silence)
            dc_offset = rng.uniform(-0.005, 0.005)
            clip = np.full(clip_samples, dc_offset, dtype=np.float32)
            # Add very low-level noise
            clip += rng.standard_normal(clip_samples).astype(np.float32) * 0.001

        elif noise_type == "tone":
            # Sine tones at common speech frequencies (100-3000 Hz)
            freq = rng.uniform(100.0, 3000.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amplitude = rng.uniform(0.05, 0.4)
            clip = (amplitude * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)

        elif noise_type == "chirp":
            # Frequency sweep (chirp) across speech band
            f_start = rng.uniform(80.0, 500.0)
            f_end = rng.uniform(1000.0, 4000.0)
            amplitude = rng.uniform(0.05, 0.3)
            instantaneous_freq = f_start + (f_end - f_start) * t / t[-1]
            phase = 2.0 * np.pi * np.cumsum(instantaneous_freq) / 16000.0
            clip = (amplitude * np.sin(phase)).astype(np.float32)

        else:  # mixed: tone + noise
            freq = rng.uniform(100.0, 2000.0)
            tone_amp = rng.uniform(0.05, 0.3)
            noise_amp = rng.uniform(0.01, 0.15)
            tone = tone_amp * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
            noise = rng.standard_normal(clip_samples).astype(np.float32) * noise_amp
            clip = (tone + noise).astype(np.float32)

        clips.append(np.clip(clip, -1.0, 1.0).astype(np.float32))

    return clips


ProgressCallback = Callable[[dict[str, Any]], None]


def _train_mlp_on_oww(
    positives_dir: Path,
    output_path: Path,
    epochs: int = 50,
    augment: bool = True,
    eval_dir: Path | None = None,
    negatives_dir: Path | None = None,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    neg_ratio: int = 5,
    patience: int = 10,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """
    Train an MLP classifier on OWW embeddings.

    This is the production architecture from Viola: MLP head on top of
    frozen OpenWakeWord audio embeddings (96-dim, 10ms frame rate).

    Steps:
      1. Extract OWW embeddings from all positive samples
      2. Optionally augment positive audio before embedding extraction
      3. Generate negative embeddings (real audio or synthetic fallback)
      4. 80/20 train/validation split
      5. Train MLP with FocalLoss + AdamW + cosine LR + EMA
      6. Early stopping on validation loss (patience-based)
      7. Export best model to ONNX

    Args:
        positives_dir: Directory of WAV/FLAC positive samples.
        output_path: Path to save the .onnx model.
        epochs: Training epochs (default 50).
        augment: Apply audio-level augmentation to positives (default True).
        eval_dir: Optional test set for Cohen's d / FAR / FRR evaluation during training.
        negatives_dir: Optional directory of negative WAV/FLAC files.
        batch_size: Mini-batch size (default 32).
        lr: Learning rate (default 1e-3).
        hidden_dim: Hidden layer dimension (default 64).
        neg_ratio: Negatives per positive (default 5).
        patience: Early stopping patience in epochs (default 10).
        verbose: Print training progress (default True).
        progress_callback: Optional callback invoked each epoch with a dict
            containing: epoch, total_epochs, train_loss, val_loss,
            best_val_loss, lr. Used by the Console website for SSE progress.
    """
    training_start = time.monotonic()

    # -- Imports ----------------------------------------------------------------
    import logging

    logger = logging.getLogger("violawake.train")

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        print(f"ERROR: PyTorch required for training: {e}", file=sys.stderr)
        print("Install with: pip install 'violawake[training]'", file=sys.stderr)
        sys.exit(1)

    try:
        from openwakeword.model import Model as OWWModel  # type: ignore[import]
    except ImportError as e:
        print(f"ERROR: openwakeword required for OWW embeddings: {e}", file=sys.stderr)
        print("Install with: pip install openwakeword", file=sys.stderr)
        sys.exit(1)

    try:
        import numpy as np
    except ImportError as e:
        print(f"ERROR: numpy required: {e}", file=sys.stderr)
        sys.exit(1)

    from violawake_sdk._constants import CLIP_SAMPLES, get_feature_config
    from violawake_sdk.audio import center_crop, load_audio
    from violawake_sdk.training.augment import AugmentationPipeline
    from violawake_sdk.training.losses import FocalLoss

    # -- Collect positive files -------------------------------------------------
    pos_files = sorted(
        list(positives_dir.rglob("*.wav")) + list(positives_dir.rglob("*.flac"))
    )
    if len(pos_files) < 5:
        print(
            f"ERROR: Found only {len(pos_files)} positive samples in {positives_dir}.",
            file=sys.stderr,
        )
        print(
            "Minimum 20 samples required. Use 'violawake-collect' to record more.",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"Found {len(pos_files)} positive samples")

    # -- Load OWW model for embedding extraction --------------------------------
    if verbose:
        print("Loading OpenWakeWord backbone for embedding extraction...")

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    # -- Embedding extraction helper -------------------------------------------
    def _audio_to_embedding(audio_float32: np.ndarray) -> np.ndarray | None:
        """Convert a float32 audio clip to an OWW embedding vector."""
        audio = center_crop(audio_float32, CLIP_SAMPLES)
        audio_int16 = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio_int16 * 32767).astype(np.int16)
        if len(audio_int16) < CLIP_SAMPLES:
            audio_int16 = np.pad(audio_int16, (0, CLIP_SAMPLES - len(audio_int16)))
        else:
            audio_int16 = audio_int16[:CLIP_SAMPLES]
        try:
            embeddings = preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
            # Mean-pool across time axis (production Viola approach used for the
            # synthetic-benchmark Cohen's d reference model).
            return embeddings.mean(axis=1)[0].astype(np.float32)
        except Exception as e:
            logger.warning("Failed to extract embedding: %s", e)
            return None

    def _extract_embedding_from_file(wav_path: Path) -> np.ndarray | None:
        """Load a WAV/FLAC file and return its OWW embedding."""
        audio = load_audio(wav_path)
        if audio is None:
            return None
        return _audio_to_embedding(audio)

    # -- Load positive audio and optionally augment ----------------------------
    if verbose:
        aug_label = " (with augmentation)" if augment else ""
        print(f"Extracting embeddings from positive samples{aug_label}...")

    pos_embeddings: list[np.ndarray] = []
    # Track which source file each embedding came from (index into pos_files).
    # This prevents data leakage: augmented variants of the same source file
    # must stay together in the same train/val split.
    pos_source_file_idx: list[int] = []
    augment_factor = 0

    embed_failures = 0
    embed_total = 0

    if augment:
        # Load raw audio first, augment, then extract embeddings
        pipeline = AugmentationPipeline(seed=42)
        augment_factor = 10  # each sample -> 10 augmented variants
        n_originals = 0
        n_augmented = 0

        for file_idx, f in enumerate(pos_files):
            audio = load_audio(f)
            if audio is None:
                continue

            # Original embedding
            embed_total += 1
            emb = _audio_to_embedding(audio)
            if emb is not None:
                pos_embeddings.append(emb)
                pos_source_file_idx.append(file_idx)
                n_originals += 1
            else:
                embed_failures += 1

            # Augmented variants
            variants = pipeline.augment_clip(audio, factor=augment_factor)
            for variant in variants:
                embed_total += 1
                emb = _audio_to_embedding(variant)
                if emb is not None:
                    pos_embeddings.append(emb)
                    pos_source_file_idx.append(file_idx)
                    n_augmented += 1
                else:
                    embed_failures += 1

        if verbose:
            print(
                f"  {n_originals} originals + {n_augmented} augmented "
                f"= {len(pos_embeddings)} total positive embeddings"
            )
    else:
        # No augmentation -- extract embeddings directly
        for file_idx, f in enumerate(pos_files):
            embed_total += 1
            emb = _extract_embedding_from_file(f)
            if emb is not None:
                pos_embeddings.append(emb)
                pos_source_file_idx.append(file_idx)
            else:
                embed_failures += 1

    n_pos_success = embed_total - embed_failures
    logger.info(
        "Extracted %d/%d positive embeddings (%d failures)",
        n_pos_success, embed_total, embed_failures,
    )

    if len(pos_embeddings) < 5:
        print(
            "ERROR: Too few usable positive samples (embedding extraction failed).",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Generate / load negative embeddings -----------------------------------
    n_negatives = len(pos_embeddings) * neg_ratio
    neg_source = "synthetic"  # default, overridden if directory succeeds
    neg_embeddings: list[np.ndarray] = []
    # Track which source file each negative embedding came from.
    # Same anti-leakage logic as positives: all embeddings from a source file
    # stay together in train or val.
    neg_source_file_idx: list[int] = []

    if negatives_dir is not None and negatives_dir.exists():
        # Option A: Real negative audio files
        neg_files = sorted(
            list(negatives_dir.rglob("*.wav")) + list(negatives_dir.rglob("*.flac"))
        )
        if len(neg_files) == 0:
            print(
                f"WARNING: No WAV/FLAC files in {negatives_dir}. "
                "Falling back to synthetic negatives.",
                file=sys.stderr,
            )
        else:
            if verbose:
                print(
                    f"Extracting embeddings from {len(neg_files)} negative files "
                    f"(target: {n_negatives})..."
                )

            # Extract embeddings from each negative file
            neg_embed_failures = 0
            for file_idx, f in enumerate(neg_files):
                embed_total += 1
                emb = _extract_embedding_from_file(f)
                if emb is not None:
                    neg_embeddings.append(emb)
                    neg_source_file_idx.append(file_idx)
                else:
                    neg_embed_failures += 1

            # If we have fewer negative files than needed and augmentation
            # is enabled, augment negatives to increase diversity instead
            # of cycling through identical files (which adds zero diversity)
            if len(neg_embeddings) < n_negatives and augment:
                aug_pipeline = AugmentationPipeline(seed=99)
                remaining = n_negatives - len(neg_embeddings)
                # Augment each negative file with variants, cycling through
                # files, until we have enough
                aug_per_file = max(1, remaining // max(len(neg_files), 1))
                if verbose:
                    print(
                        f"  Augmenting {len(neg_files)} negative files "
                        f"(~{aug_per_file} variants each) to reach {n_negatives}..."
                    )
                neg_file_iter = 0
                while len(neg_embeddings) < n_negatives:
                    source_idx = neg_file_iter % len(neg_files)
                    f = neg_files[source_idx]
                    audio = load_audio(f)
                    if audio is not None:
                        variants = aug_pipeline.augment_clip(
                            audio, factor=min(aug_per_file, n_negatives - len(neg_embeddings))
                        )
                        for variant in variants:
                            if len(neg_embeddings) >= n_negatives:
                                break
                            embed_total += 1
                            emb = _audio_to_embedding(variant)
                            if emb is not None:
                                neg_embeddings.append(emb)
                                neg_source_file_idx.append(source_idx)
                            else:
                                neg_embed_failures += 1
                    neg_file_iter += 1
                    # Safety: stop if we've cycled many times without progress
                    if neg_file_iter > n_negatives * 2:
                        break
            elif len(neg_embeddings) < n_negatives:
                # No augmentation -- fall back to cycling through files
                file_iter = 0
                while len(neg_embeddings) < n_negatives and file_iter < n_negatives:
                    source_idx = file_iter % len(neg_files)
                    f = neg_files[source_idx]
                    embed_total += 1
                    emb = _extract_embedding_from_file(f)
                    if emb is not None:
                        neg_embeddings.append(emb)
                        neg_source_file_idx.append(source_idx)
                    else:
                        neg_embed_failures += 1
                    file_iter += 1

            logger.info(
                "Negative embedding extraction: %d failures",
                neg_embed_failures,
            )

            if len(neg_embeddings) >= 5:
                neg_source = "directory"
            else:
                print(
                    "WARNING: Too few usable negatives from directory. "
                    "Falling back to synthetic.",
                    file=sys.stderr,
                )
                neg_embeddings = []
                neg_source_file_idx = []

    if neg_source != "directory":
        # Option B: Synthetic negatives (diverse, not just uniform noise)
        # Each synthetic clip is its own "source" -- no grouping needed,
        # but we assign unique indices for consistency.
        if verbose:
            print(f"Generating {n_negatives} synthetic negative embeddings...")
            print(
                "  (types: white noise, pink noise, silence, tones, chirps, mixed)"
            )

        rng = np.random.default_rng(42)
        synthetic_clips = _generate_synthetic_negatives(n_negatives, CLIP_SAMPLES, rng)

        neg_embeddings = []
        neg_source_file_idx = []
        for synth_idx, clip in enumerate(synthetic_clips):
            emb = _audio_to_embedding(clip)
            if emb is not None:
                neg_embeddings.append(emb)
                neg_source_file_idx.append(synth_idx)

    if len(neg_embeddings) < 5:
        print("ERROR: Could not generate enough negative embeddings.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"  {len(neg_embeddings)} negative embeddings ({neg_source})")

    # -- Build dataset with train/val split ------------------------------------
    pos_tensor = torch.tensor(np.array(pos_embeddings), dtype=torch.float32)
    neg_tensor = torch.tensor(np.array(neg_embeddings), dtype=torch.float32)

    pos_labels = torch.ones(len(pos_embeddings), 1)
    neg_labels = torch.zeros(len(neg_embeddings), 1)

    X = torch.cat([pos_tensor, neg_tensor], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    embedding_dim = X.shape[1]  # Typically 96 for OWW

    # 80/20 stratified train/validation split -- GROUP-AWARE
    #
    # When augmentation is enabled, each source file produces multiple
    # embeddings (1 original + N augmented). A naive embedding-level split
    # would leak augmented variants of the same source into both train and
    # val, inflating validation metrics. Instead, we split at the SOURCE
    # FILE level: all embeddings from a given file go to train OR val,
    # never both.
    rng_split = np.random.default_rng(42)

    # --- Positive split (group-aware by source file) ---
    unique_pos_sources = sorted(set(pos_source_file_idx))
    rng_split.shuffle(unique_pos_sources)
    n_val_pos_sources = max(1, len(unique_pos_sources) // 5)  # 20% of source files
    val_pos_sources = set(unique_pos_sources[:n_val_pos_sources])

    # Map source-file split back to embedding indices in the combined tensor.
    # Positive embeddings are indices 0..len(pos_embeddings)-1 in X.
    val_pos = [i for i, src in enumerate(pos_source_file_idx) if src in val_pos_sources]
    train_pos = [i for i, src in enumerate(pos_source_file_idx) if src not in val_pos_sources]

    # --- Negative split (group-aware by source file) ---
    unique_neg_sources = sorted(set(neg_source_file_idx))
    rng_split.shuffle(unique_neg_sources)
    n_val_neg_sources = max(1, len(unique_neg_sources) // 5)  # 20% of source files
    val_neg_sources = set(unique_neg_sources[:n_val_neg_sources])

    # Negative embeddings start at offset len(pos_embeddings) in the combined tensor.
    neg_offset = len(pos_embeddings)
    val_neg = [neg_offset + i for i, src in enumerate(neg_source_file_idx) if src in val_neg_sources]
    train_neg = [neg_offset + i for i, src in enumerate(neg_source_file_idx) if src not in val_neg_sources]

    val_indices = val_pos + val_neg
    train_indices = train_pos + train_neg

    # Shuffle within each set so batches mix classes
    rng_split.shuffle(train_indices)
    rng_split.shuffle(val_indices)

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    n_train = len(train_indices)
    n_val = len(val_indices)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    if verbose:
        n_pos_total = len(pos_embeddings)
        n_neg_total = len(neg_embeddings)
        print(
            f"Dataset: {n_pos_total} pos + {n_neg_total} neg | "
            f"embedding_dim={embedding_dim}"
        )
        train_pos_count = int(y_train.sum().item())
        val_pos_count = int(y_val.sum().item())
        n_train_pos_sources = len(unique_pos_sources) - n_val_pos_sources
        n_train_neg_sources = len(unique_neg_sources) - n_val_neg_sources
        print(
            f"Group-aware split: "
            f"{len(unique_pos_sources)} pos sources -> "
            f"{n_train_pos_sources} train / {n_val_pos_sources} val | "
            f"{len(unique_neg_sources)} neg sources -> "
            f"{n_train_neg_sources} train / {n_val_neg_sources} val"
        )
        print(
            f"  Embeddings: {n_train} train ({train_pos_count} pos / "
            f"{n_train - train_pos_count} neg) / "
            f"{n_val} val ({val_pos_count} pos / "
            f"{n_val - val_pos_count} neg)"
        )

    # -- Build MLP model -------------------------------------------------------
    model = nn.Sequential(
        nn.Linear(embedding_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim // 2, 1),
        nn.Sigmoid(),
    )

    criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing LR schedule (from production Viola)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # EMA of model weights (from production Viola -- improves final accuracy)
    ema_decay = 0.999
    ema_params: dict[str, torch.Tensor] = {
        name: param.data.clone() for name, param in model.named_parameters()
    }

    def _update_ema() -> None:
        for name, param in model.named_parameters():
            ema_params[name] = ema_decay * ema_params[name] + (1 - ema_decay) * param.data

    # -- Training loop with early stopping -------------------------------------
    if verbose:
        print(f"\nTraining MLP for up to {epochs} epochs (patience={patience})...")
        print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'Best':>10} {'LR':>10}")
        print("-" * 50)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_ema_params: dict[str, torch.Tensor] | None = None
    early_stopped = False

    for epoch in range(1, epochs + 1):
        # --- Train phase ---
        model.train()
        train_loss = 0.0
        n_train_batches = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            _update_ema()
            train_loss += float(loss.item())
            n_train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(n_train_batches, 1)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                val_loss += float(loss.item())
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)

        # --- Early stopping check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Save best model state
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ema_params = {k: v.clone() for k, v in ema_params.items()}
        else:
            epochs_without_improvement += 1

        # --- Progress reporting ---
        current_lr = scheduler.get_last_lr()[0]

        if progress_callback is not None:
            progress_callback({
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "best_val_loss": best_val_loss,
                "lr": current_lr,
            })

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs
                        or epochs_without_improvement == 0):
            marker = " *" if epoch == best_epoch else ""
            print(
                f"{epoch:>6} {avg_train_loss:>10.4f} {avg_val_loss:>10.4f} "
                f"{best_val_loss:>10.4f} {current_lr:>10.6f}{marker}"
            )

        # --- Stop if no improvement for `patience` epochs ---
        if epochs_without_improvement >= patience:
            early_stopped = True
            if verbose:
                print(
                    f"\nEarly stopping at epoch {epoch} "
                    f"(no improvement for {patience} epochs). "
                    f"Best epoch: {best_epoch}"
                )
            break

    # -- Restore best model weights -------------------------------------------
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    if best_ema_params is not None:
        ema_params = best_ema_params

    # Apply EMA weights to model
    for name, param in model.named_parameters():
        param.data.copy_(ema_params[name])

    training_duration = time.monotonic() - training_start

    if verbose:
        print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        print(f"Training duration: {training_duration:.1f}s")

    # -- Export to ONNX --------------------------------------------------------
    if verbose:
        print(f"\nExporting model to ONNX: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.zeros(1, embedding_dim)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["embedding"],
        output_names=["score"],
        dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
        opset_version=11,
    )

    # -- Evaluate if test set provided -----------------------------------------
    d_prime_result: float | None = None
    if eval_dir and eval_dir.exists():
        if verbose:
            print(f"\nEvaluating on test set: {eval_dir}")
        try:
            from violawake_sdk.training.evaluate import evaluate_onnx_model

            results = evaluate_onnx_model(output_path, eval_dir)
            d_prime_result = results["d_prime"]
            far = results["far_per_hour"]
            frr = results["frr"] * 100
            print(f"Cohen's d: {d_prime_result:.2f}  FAR: {far:.2f}/hr  FRR: {frr:.1f}%")
            if d_prime_result < 10.0:
                print(
                    "WARNING: Cohen's d < 10.0 on this benchmark. "
                    "Consider collecting more positive samples or validating against harder negatives."
                )
        except Exception as e:
            print(f"Evaluation failed: {e}")

    # -- Save checkpoint config ------------------------------------------------
    config_path = output_path.with_suffix(".config.json")
    config = get_feature_config()
    config.update({
        "architecture": "mlp_on_oww",
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "n_pos_samples": len(pos_embeddings),
        "n_neg_samples": len(neg_embeddings),
        "neg_source": neg_source,
        "neg_ratio": neg_ratio,
        "augmented": augment,
        "augment_factor": augment_factor if augment else 0,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "early_stopped": early_stopped,
        "training_duration_s": round(training_duration, 2),
        "patience": patience,
    })
    if d_prime_result is not None:
        config["d_prime"] = round(d_prime_result, 2)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"Checkpoint config saved: {config_path}")
        print(f"\nModel saved: {output_path}")
        print(f"Load with:  WakeDetector(model='{output_path}')")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-train",
        description="Train a custom wake word model (MLP on OWW embeddings).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--word",
        required=True,
        metavar="WORD",
        help="The wake word name (used for display only, e.g. 'jarvis')",
    )
    parser.add_argument(
        "--positives",
        required=True,
        metavar="DIR",
        help="Directory containing positive WAV/FLAC samples of the wake word",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output path for the trained ONNX model (e.g., models/jarvis.onnx)",
    )
    parser.add_argument(
        "--negatives",
        metavar="DIR",
        default=None,
        help="Optional directory of negative WAV/FLAC files (speech, music, "
             "environmental sounds). If not provided, synthetic negatives are "
             "generated automatically.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Maximum training epochs (default: 50). May stop earlier via "
             "early stopping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="Mini-batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="RATE",
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        metavar="N",
        help="Hidden layer dimension in the MLP (default: 64)",
    )
    parser.add_argument(
        "--neg-ratio",
        type=int,
        default=5,
        metavar="N",
        help="Number of negatives per positive sample (default: 5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        metavar="N",
        help="Early stopping patience: stop after N epochs without validation "
             "loss improvement (default: 10)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable audio-level data augmentation (default: True)",
    )
    parser.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--eval-dir",
        metavar="DIR",
        help="Optional test set directory for Cohen's d / FAR / FRR evaluation after training. "
             "Must contain positives/ and negatives/ subdirectories.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress training progress output",
    )

    args = parser.parse_args()

    positives_dir = Path(args.positives)
    output_path = Path(args.output)
    eval_dir = Path(args.eval_dir) if args.eval_dir else None
    negatives_dir = Path(args.negatives) if args.negatives else None

    if not positives_dir.exists():
        print(f"ERROR: Positives directory not found: {positives_dir}", file=sys.stderr)
        sys.exit(1)

    if negatives_dir and not negatives_dir.exists():
        print(f"ERROR: Negatives directory not found: {negatives_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Training wake word: '{args.word}'")
        print(f"Positives:          {positives_dir}")
        if negatives_dir:
            print(f"Negatives:          {negatives_dir}")
        print(f"Output:             {output_path}")
        print(f"Epochs:             {args.epochs} (patience={args.patience})")
        print(f"Batch size:         {args.batch_size}")
        print(f"Learning rate:      {args.lr}")
        print(f"Hidden dim:         {args.hidden_dim}")
        print(f"Neg ratio:          {args.neg_ratio}x")
        print(f"Augmentation:       {'enabled' if args.augment else 'disabled'}")
        if eval_dir:
            print(f"Eval set:           {eval_dir}")
        print()

    _train_mlp_on_oww(
        positives_dir=positives_dir,
        output_path=output_path,
        epochs=args.epochs,
        augment=args.augment,
        eval_dir=eval_dir,
        negatives_dir=negatives_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        neg_ratio=args.neg_ratio,
        patience=args.patience,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
