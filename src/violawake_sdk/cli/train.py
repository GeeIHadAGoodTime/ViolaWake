"""violawake-train CLI (cli module) — Train a custom wake word model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-train",
        description="Train a custom wake word model (MLP on OWW embeddings).",
    )
    parser.add_argument(
        "--positive-dir",
        required=True,
        metavar="DIR",
        help="Directory containing positive WAV samples",
    )
    parser.add_argument(
        "--negative-dir", metavar="DIR", help="Directory containing negative WAV samples (optional)"
    )
    parser.add_argument(
        "--output-model",
        required=True,
        metavar="PATH",
        help="Output path for the trained ONNX model",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, metavar="N", help="Training epochs (default: 50)"
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        default=False,
        help="Disable data augmentation (augmentation is on by default)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress training progress output")
    args = parser.parse_args()

    positive_dir = Path(args.positive_dir)
    if not positive_dir.exists():
        print(f"ERROR: Positives directory not found: {positive_dir}", file=sys.stderr)
        sys.exit(1)

    # Delegate to tools.train implementation
    from violawake_sdk.tools.train import _train_mlp_on_oww

    output_path = Path(args.output_model)
    negative_dir = Path(args.negative_dir) if args.negative_dir else None
    _train_mlp_on_oww(
        positives_dir=positive_dir,
        output_path=output_path,
        epochs=args.epochs,
        augment=not args.no_augment,
        eval_dir=negative_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
