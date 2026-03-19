"""
violawake-download CLI — Download ViolaWake models to local cache.

Entry point: ``violawake-download`` (declared in pyproject.toml).

Usage::

    violawake-download                          # Download default models
    violawake-download --model viola_mlp_oww    # Download a specific model
    violawake-download --list                   # Show available models
    violawake-download --list-cached            # Show locally cached models
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-download",
        description="Download ViolaWake models to local cache (~/.violawake/models/).",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Model name to download (see --list for available names). "
             "Downloads required defaults if omitted.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models in the registry",
    )
    parser.add_argument(
        "--list-cached",
        action="store_true",
        help="List models already in the local cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already cached",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip SHA-256 verification after download (not recommended)",
    )

    args = parser.parse_args()

    from violawake_sdk.models import MODEL_REGISTRY, download_model, list_cached_models

    if args.list:
        print("Available models:")
        print()
        for name, spec in MODEL_REGISTRY.items():
            size_mb = spec.size_bytes / 1_000_000
            print(f"  {name:<30} {size_mb:>6.0f} MB  {spec.description}")
        print()
        print("Download with: violawake-download --model <name>")
        return

    if args.list_cached:
        cached = list_cached_models()
        if not cached:
            print("No models cached. Run: violawake-download")
            return
        print("Cached models:")
        for name, path, size_mb in cached:
            print(f"  {name:<30} {size_mb:>6.1f} MB  {path}")
        return

    # Download specific model or defaults
    models_to_download = []
    if args.model:
        if args.model not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            print(f"ERROR: Unknown model '{args.model}'", file=sys.stderr)
            print(f"Available: {available}", file=sys.stderr)
            sys.exit(1)
        models_to_download = [args.model]
    else:
        # Default: download the required models for basic wake word detection
        models_to_download = ["viola_mlp_oww", "oww_backbone"]
        print("Downloading default models (viola_mlp_oww + oww_backbone)...")
        print("For TTS, run: violawake-download --model kokoro_v1_0")
        print()

    verify = not args.no_verify
    success = True
    for model_name in models_to_download:
        try:
            path = download_model(model_name, force=args.force, verify=verify)
            spec = MODEL_REGISTRY[model_name]
            size_mb = path.stat().st_size / 1_000_000
            print(f"  {model_name}: {path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"ERROR downloading {model_name}: {e}", file=sys.stderr)
            success = False

    if success:
        print()
        print("Done. Models cached to ~/.violawake/models/")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
