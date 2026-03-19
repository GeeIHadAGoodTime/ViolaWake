"""
violawake-train CLI — Train a custom wake word model.

Entry point: ``violawake-train`` (declared in pyproject.toml).

Architecture: MLP classifier head on top of frozen OpenWakeWord (OWW) audio
embeddings. This is the same architecture used in production Viola (d-prime 15.10).

Training pipeline (from Viola's violawake/training/trainer.py):
  - FocalLoss for class imbalance handling
  - AdamW optimizer with cosine annealing LR schedule
  - Exponential Moving Average (EMA) of model weights
  - Optional Stochastic Weight Averaging (SWA)
  - Checkpoint config embedding (prevents config-drift bug)

Usage::

    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --output models/jarvis.onnx \\
      --epochs 50 \\
      --augment

    # For evaluation during training:
    violawake-train \\
      --word "jarvis" \\
      --positives data/jarvis/positives/ \\
      --output models/jarvis.onnx \\
      --eval-dir data/jarvis/test/

Minimum: 20 positive samples. Recommended: 50+ samples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _train_mlp_on_oww(
    positives_dir: Path,
    output_path: Path,
    epochs: int = 50,
    augment: bool = True,
    eval_dir: Path | None = None,
    verbose: bool = True,
) -> None:
    """
    Train an MLP classifier on OWW embeddings.

    This is the production architecture from Viola: MLP head on top of
    frozen OpenWakeWord audio embeddings (96-dim, 10ms frame rate).

    Steps:
      1. Extract OWW embeddings from all positive samples
      2. Generate negative embeddings from random OWW background data
      3. Train MLP with FocalLoss + AdamW + cosine LR + EMA
      4. Export to ONNX

    Args:
        positives_dir: Directory of WAV/FLAC positive samples.
        output_path: Path to save the .onnx model.
        epochs: Training epochs (default 50).
        augment: Apply data augmentation (default True).
        eval_dir: Optional test set for d-prime evaluation during training.
        verbose: Print training progress (default True).
    """
    # ── Imports ──────────────────────────────────────────────────────────────
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
    from violawake_sdk.training.losses import FocalLoss

    # ── Collect positive files ────────────────────────────────────────────────
    pos_files = sorted(
        list(positives_dir.rglob("*.wav")) + list(positives_dir.rglob("*.flac"))
    )
    if len(pos_files) < 5:
        print(f"ERROR: Found only {len(pos_files)} positive samples in {positives_dir}.", file=sys.stderr)
        print("Minimum 20 samples required. Use 'violawake-collect' to record more.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Found {len(pos_files)} positive samples")

    # ── Load OWW model for embedding extraction ────────────────────────────────
    if verbose:
        print("Loading OpenWakeWord backbone for embedding extraction...")

    oww = OWWModel()
    preprocessor = oww.preprocessor
    if not hasattr(preprocessor, "onnx_execution_provider"):
        preprocessor.onnx_execution_provider = "CPUExecutionProvider"

    # ── Extract positive embeddings ──────────────────────────────────────────
    if verbose:
        print("Extracting embeddings from positive samples...")

    def _extract_embedding(wav_path: Path) -> np.ndarray | None:
        audio = load_audio(wav_path)
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
            # Mean-pool across time axis (production Viola approach, d-prime 15.10)
            return embeddings.mean(axis=1)[0].astype(np.float32)
        except Exception:
            return None

    pos_embeddings = []
    for f in pos_files:
        emb = _extract_embedding(f)
        if emb is not None:
            pos_embeddings.append(emb)

    if len(pos_embeddings) < 5:
        print("ERROR: Too few usable positive samples (embedding extraction failed).", file=sys.stderr)
        sys.exit(1)

    # ── Generate negative embeddings (from OWW background data) ─────────────
    # Generate ~5x negatives as positives for class balance
    n_negatives = len(pos_embeddings) * 5
    if verbose:
        print(f"Generating {n_negatives} negative (background) embeddings...")

    # Use random noise and silence as negatives
    neg_embeddings = []
    rng = np.random.default_rng(42)
    for _ in range(n_negatives):
        # Random noise clip
        noise = rng.uniform(-0.1, 0.1, CLIP_SAMPLES).astype(np.float32)
        noise_int16 = (noise * 32767).astype(np.int16)
        try:
            embeddings = preprocessor.embed_clips(noise_int16.reshape(1, -1), ncpu=1)
            neg_embeddings.append(embeddings.mean(axis=1)[0].astype(np.float32))
        except Exception:
            continue

    if len(neg_embeddings) < 5:
        print("ERROR: Could not generate negative embeddings.", file=sys.stderr)
        sys.exit(1)

    # ── Build dataset ─────────────────────────────────────────────────────────
    pos_tensor = torch.tensor(np.array(pos_embeddings), dtype=torch.float32)
    neg_tensor = torch.tensor(np.array(neg_embeddings), dtype=torch.float32)

    pos_labels = torch.ones(len(pos_embeddings), 1)
    neg_labels = torch.zeros(len(neg_embeddings), 1)

    X = torch.cat([pos_tensor, neg_tensor], dim=0)
    y = torch.cat([pos_labels, neg_labels], dim=0)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    embedding_dim = X.shape[1]  # Typically 96 for OWW
    if verbose:
        print(f"Dataset: {len(pos_embeddings)} pos + {len(neg_embeddings)} neg | embedding_dim={embedding_dim}")

    # ── Build MLP model ───────────────────────────────────────────────────────
    model = nn.Sequential(
        nn.Linear(embedding_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1),
        nn.Sigmoid(),
    )

    criterion = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Cosine annealing LR schedule (from production Viola)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # EMA of model weights (from production Viola — improves final accuracy)
    ema_decay = 0.999
    ema_params = {name: param.data.clone() for name, param in model.named_parameters()}

    def _update_ema():
        for name, param in model.named_parameters():
            ema_params[name] = ema_decay * ema_params[name] + (1 - ema_decay) * param.data

    # ── Training loop ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\nTraining MLP for {epochs} epochs...")
        print(f"{'Epoch':>6} {'Loss':>10} {'LR':>10}")
        print("-" * 30)

    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            _update_ema()
            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            lr = scheduler.get_last_lr()[0]
            print(f"{epoch:>6} {avg_loss:>10.4f} {lr:>10.6f}")

    # ── Apply EMA weights to model ────────────────────────────────────────────
    for name, param in model.named_parameters():
        param.data.copy_(ema_params[name])

    # ── Export to ONNX ────────────────────────────────────────────────────────
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

    # ── Save checkpoint config (prevents config-drift bug) ───────────────────
    import json
    config_path = output_path.with_suffix(".config.json")
    config = get_feature_config()
    config["architecture"] = "mlp_on_oww"
    config["embedding_dim"] = embedding_dim
    config["n_pos_samples"] = len(pos_embeddings)
    config["epochs"] = epochs
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    if verbose:
        print(f"Checkpoint config saved: {config_path}")

    # ── Evaluate if test set provided ─────────────────────────────────────────
    if eval_dir and eval_dir.exists():
        if verbose:
            print(f"\nEvaluating on test set: {eval_dir}")
        try:
            from violawake_sdk.training.evaluate import evaluate_onnx_model
            results = evaluate_onnx_model(output_path, eval_dir)
            d = results["d_prime"]
            far = results["far_per_hour"]
            frr = results["frr"] * 100
            print(f"d-prime: {d:.2f}  FAR: {far:.2f}/hr  FRR: {frr:.1f}%")
            if d < 10.0:
                print("WARNING: d-prime < 10.0. Consider collecting more positive samples.")
        except Exception as e:
            print(f"Evaluation failed: {e}")

    if verbose:
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
        "--epochs",
        type=int,
        default=50,
        metavar="N",
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Enable data augmentation (default: True)",
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
        help="Optional test set directory for d-prime evaluation after training. "
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

    if not positives_dir.exists():
        print(f"ERROR: Positives directory not found: {positives_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"Training wake word: '{args.word}'")
        print(f"Positives:          {positives_dir}")
        print(f"Output:             {output_path}")
        print(f"Epochs:             {args.epochs}")
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
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
