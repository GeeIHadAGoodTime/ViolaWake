"""
violawake-download CLI — Download ViolaWake models to local cache.

Entry point: ``violawake-download`` (declared in pyproject.toml).

Usage::

    violawake-download                          # Download default models
    violawake-download --model temporal_cnn     # Download a specific model
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
    parser.add_argument(
        "--format",
        choices=["onnx", "tflite"],
        default="onnx",
        help="Model format to download or convert to (default: onnx). "
        "When 'tflite' is specified, downloads the ONNX model first then "
        "converts it locally. Requires: pip install onnx2tf tensorflow",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic-range quantization when converting to TFLite (~4x smaller)",
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
        models_to_download = ["temporal_cnn"]
        print("Downloading default wake word model (temporal_cnn)...")
        print("For TTS, run: violawake-download --model kokoro_v1_0")
        print()

    verify = not args.no_verify
    success = True
    for model_name in models_to_download:
        try:
            path = download_model(
                model_name,
                force=args.force,
                verify=verify,
                skip_verify=args.no_verify,
            )
            spec = MODEL_REGISTRY[model_name]
            size_mb = path.stat().st_size / 1_000_000
            print(f"  {model_name}: {path} ({size_mb:.1f} MB)")

            # Convert to TFLite if requested
            if args.format == "tflite":
                _convert_to_tflite(path, quantize=args.quantize)

        except Exception as e:
            print(f"ERROR downloading {model_name}: {e}", file=sys.stderr)
            success = False

    if success:
        print()
        print("Done. Models cached to ~/.violawake/models/")
    else:
        sys.exit(1)


def _convert_to_tflite(onnx_path, *, quantize: bool = False) -> None:
    """Convert a downloaded ONNX model to TFLite format."""
    from pathlib import Path

    onnx_path = Path(onnx_path)
    tflite_path = onnx_path.with_suffix(".tflite")

    if tflite_path.exists():
        size_mb = tflite_path.stat().st_size / 1_000_000
        print(f"  TFLite already exists: {tflite_path} ({size_mb:.1f} MB)")
        return

    print(f"  Converting {onnx_path.name} -> TFLite...", end="", flush=True)
    try:
        from violawake_sdk.backends.tflite_backend import convert_onnx_to_tflite

        result = convert_onnx_to_tflite(onnx_path, tflite_path, quantize=quantize)
        size_mb = result.stat().st_size / 1_000_000
        extra = " (quantized)" if quantize else ""
        print(f" done: {result} ({size_mb:.1f} MB{extra})")
    except ImportError as e:
        print(f" FAILED", file=sys.stderr)
        print(
            f"  TFLite conversion requires additional dependencies:\n"
            f"    pip install onnx2tf tensorflow\n"
            f"  Or: pip install onnx onnx-tf tensorflow\n"
            f"  Error: {e}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f" FAILED", file=sys.stderr)
        print(f"  Conversion error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
