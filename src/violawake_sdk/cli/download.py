"""violawake-download CLI (cli module) — Download ViolaWake models."""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_CACHE_DIR = Path.home() / ".violawake" / "models"

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-download",
        description="Download ViolaWake ONNX models to local cache (~/.violawake/models/).",
    )
    parser.add_argument("--model", metavar="NAME",
                        help="Model name to download (see --list for options)")
    parser.add_argument("--list", action="store_true",
                        help="List all available models")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if already cached")
    parser.add_argument("--cache-dir", metavar="DIR", default=str(DEFAULT_CACHE_DIR),
                        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})")
    args = parser.parse_args()

    # Delegate to tools.download_model implementation
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = ["violawake-download"]
    if args.model:
        _sys.argv += ["--model", args.model]
    if args.list:
        _sys.argv.append("--list")
    if args.force:
        _sys.argv.append("--force")
    try:
        from violawake_sdk.tools.download_model import main as _main
        _main()
    finally:
        _sys.argv = old_argv

if __name__ == "__main__":
    main()
