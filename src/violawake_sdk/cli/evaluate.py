"""violawake-eval CLI (cli module) — Evaluate a wake word model."""
from __future__ import annotations
import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="violawake-eval",
        description="Evaluate a ViolaWake ONNX model using Cohen's d plus FAR/FRR.",
    )
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="Path to the ONNX model file")
    parser.add_argument("--test-dir", required=True, metavar="DIR",
                        help="Test directory with positives/ and negatives/ subdirs")
    parser.add_argument("--threshold", type=float, default=0.50, metavar="FLOAT",
                        help="Detection threshold (default: 0.50)")
    parser.add_argument("--report", action="store_true",
                        help="Print full evaluation report")
    args = parser.parse_args()

    model_path = Path(args.model)
    test_dir = Path(args.test_dir)

    # Delegate to tools.evaluate implementation
    import sys as _sys
    old_argv = _sys.argv
    _sys.argv = [
        "violawake-eval",
        "--model", str(model_path),
        "--test-dir", str(test_dir),
        "--threshold", str(args.threshold),
    ]
    if args.report:
        _sys.argv.append("--report")
    try:
        from violawake_sdk.tools.evaluate import main as _main
        _main()
    finally:
        _sys.argv = old_argv

if __name__ == "__main__":
    main()
