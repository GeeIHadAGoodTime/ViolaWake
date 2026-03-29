#!/usr/bin/env python3
"""Verify all models in the ViolaWake registry.

Downloads each model, checks SHA-256 when a real hash is present, loads ONNX
models with onnxruntime, and validates input/output tensor shapes.

Usage:
    python scripts/verify_models.py          # Interactive output
    python scripts/verify_models.py --ci     # CI mode: JSON report + nonzero exit on failure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

# Expected ONNX model input/output shapes.
# Keys are model names from MODEL_REGISTRY.
# Shapes use None for dynamic/batch dimensions.
EXPECTED_SHAPES: dict[str, dict[str, list[list[int | None]]]] = {
    # viola_mlp_oww and viola_cnn_v4 removed — never uploaded to GitHub Releases.
    # Add expected shapes for new models as needed.
}


def shapes_match(
    actual: list[int | str | None],
    expected: list[int | None],
) -> bool:
    """Check if actual tensor shape matches expected."""
    if len(actual) != len(expected):
        return False

    for actual_dim, expected_dim in zip(actual, expected):
        if expected_dim is None:
            continue
        if isinstance(actual_dim, str):
            return False
        if actual_dim != expected_dim:
            return False

    return True


def verify_model(
    name: str,
    spec: object,
    model_dir: Path,
) -> dict:
    """Verify a single model and return a structured result."""
    from violawake_sdk.models import _verify_sha256  # noqa: PLC2701

    result: dict = {
        "model": name,
        "status": "unknown",
        "checks": {},
        "errors": [],
        "warnings": [],
        "duration_s": 0.0,
    }
    t0 = time.monotonic()

    ext = Path(spec.url).suffix or ".onnx"
    model_path = model_dir / f"{spec.name}{ext}"

    # Check 1: download
    try:
        from violawake_sdk.models import download_model

        download_model(name, force=False, verify=False)
        result["checks"]["download"] = "pass"
    except Exception as exc:  # pragma: no cover - exercised in CI/runtime
        result["checks"]["download"] = "fail"
        result["errors"].append(f"Download failed: {exc}")
        result["status"] = "fail"
        result["duration_s"] = round(time.monotonic() - t0, 2)
        return result

    if not model_path.exists():
        result["checks"]["download"] = "fail"
        result["errors"].append(f"Model file not found at {model_path} after download")
        result["status"] = "fail"
        result["duration_s"] = round(time.monotonic() - t0, 2)
        return result

    # Check 2: SHA-256
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _verify_sha256(model_path, spec.sha256, name)

        if caught:
            result["checks"]["sha256"] = "warn"
            for caught_warning in caught:
                result["warnings"].append(str(caught_warning.message))
        else:
            result["checks"]["sha256"] = "pass"
    except ValueError as exc:
        result["checks"]["sha256"] = "fail"
        result["errors"].append(str(exc))
        result["status"] = "fail"
        result["duration_s"] = round(time.monotonic() - t0, 2)
        return result

    # Check 3: file size sanity
    actual_size = model_path.stat().st_size
    low = int(spec.size_bytes * 0.8)
    high = int(spec.size_bytes * 1.2)
    if low <= actual_size <= high:
        result["checks"]["size"] = "pass"
    else:
        result["checks"]["size"] = "warn"
        result["warnings"].append(
            f"Size mismatch: expected ~{spec.size_bytes} bytes, got {actual_size} bytes"
        )

    # Check 4: ONNX load and shape validation
    if ext == ".onnx":
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            result["checks"]["onnx_load"] = "pass"

            inputs = session.get_inputs()
            outputs = session.get_outputs()
            input_shapes = [inp.shape for inp in inputs]
            output_shapes = [out.shape for out in outputs]

            result["checks"]["input_shapes"] = [str(shape) for shape in input_shapes]
            result["checks"]["output_shapes"] = [str(shape) for shape in output_shapes]

            if name in EXPECTED_SHAPES:
                expected = EXPECTED_SHAPES[name]

                if len(input_shapes) != len(expected["inputs"]):
                    result["checks"]["shape_match_inputs"] = "fail"
                    result["errors"].append(
                        f"Expected {len(expected['inputs'])} inputs, got {len(input_shapes)}"
                    )
                else:
                    inputs_match = True
                    for index, (actual_shape, expected_shape) in enumerate(
                        zip(input_shapes, expected["inputs"])
                    ):
                        if not shapes_match(actual_shape, expected_shape):
                            inputs_match = False
                            result["errors"].append(
                                f"Input[{index}] shape mismatch: "
                                f"expected {expected_shape}, got {actual_shape}"
                            )
                    result["checks"]["shape_match_inputs"] = (
                        "pass" if inputs_match else "fail"
                    )

                if len(output_shapes) != len(expected["outputs"]):
                    result["checks"]["shape_match_outputs"] = "fail"
                    result["errors"].append(
                        f"Expected {len(expected['outputs'])} outputs, got {len(output_shapes)}"
                    )
                else:
                    outputs_match = True
                    for index, (actual_shape, expected_shape) in enumerate(
                        zip(output_shapes, expected["outputs"])
                    ):
                        if not shapes_match(actual_shape, expected_shape):
                            outputs_match = False
                            result["errors"].append(
                                f"Output[{index}] shape mismatch: "
                                f"expected {expected_shape}, got {actual_shape}"
                            )
                    result["checks"]["shape_match_outputs"] = (
                        "pass" if outputs_match else "fail"
                    )
            else:
                result["checks"]["shape_match"] = "skip (no expected shapes defined)"

        except ImportError:
            result["checks"]["onnx_load"] = "skip (onnxruntime not installed)"
        except Exception as exc:  # pragma: no cover - exercised in CI/runtime
            result["checks"]["onnx_load"] = "fail"
            result["errors"].append(f"ONNX load failed: {exc}")
    else:
        result["checks"]["onnx_load"] = "skip (not an .onnx file)"

    has_fail = any(value == "fail" for value in result["checks"].values())
    result["status"] = "fail" if has_fail else "pass"
    result["duration_s"] = round(time.monotonic() - t0, 2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify ViolaWake model registry")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: write JSON report and exit nonzero on failure",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Verify a single model instead of all",
    )
    args = parser.parse_args()

    from violawake_sdk.models import MODEL_REGISTRY, get_model_dir

    model_dir = get_model_dir()
    print(f"Model directory: {model_dir}")
    print()

    models_to_verify: list[tuple[str, object]] = []
    if args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"ERROR: Model '{args.model}' not found", file=sys.stderr)
            sys.exit(1)
        models_to_verify.append((args.model, MODEL_REGISTRY[args.model]))
    else:
        seen_names: set[str] = set()
        for name, spec in MODEL_REGISTRY.items():
            if spec.name in seen_names:
                print(f"  SKIP  {name:<30} (alias for {spec.name})")
                continue
            seen_names.add(spec.name)
            models_to_verify.append((name, spec))

    print()
    print(f"Verifying {len(models_to_verify)} model(s)...")
    print("=" * 70)

    results = []
    all_passed = True

    for name, spec in models_to_verify:
        print(f"\n--- {name} ---")
        result = verify_model(name, spec, model_dir)
        results.append(result)

        status_icon = "PASS" if result["status"] == "pass" else "FAIL"
        print(f"  Status: {status_icon}")
        for check_name, check_val in result["checks"].items():
            print(f"    {check_name}: {check_val}")
        for error in result["errors"]:
            print(f"    ERROR: {error}")
        for warning in result["warnings"]:
            print(f"    WARNING: {warning}")
        print(f"  Duration: {result['duration_s']}s")

        if result["status"] != "pass":
            all_passed = False

    print()
    print("=" * 70)
    passed = sum(1 for result in results if result["status"] == "pass")
    failed = sum(1 for result in results if result["status"] != "pass")
    print(f"Results: {passed} passed, {failed} failed, out of {len(results)} verified")

    report = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
        "results": results,
    }
    report_path = Path("model-verify-report.json")
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report written to: {report_path}")

    if not all_passed:
        print("\nFAILED: One or more models failed verification.", file=sys.stderr)
        if args.ci:
            sys.exit(1)
    else:
        print("\nAll models verified successfully.")


if __name__ == "__main__":
    main()
