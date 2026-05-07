#!/usr/bin/env python3
"""Verify downloadable models in the ViolaWake registry.

The CI path downloads each non-deprecated registry entry, computes SHA-256,
and compares it to the hash declared in ``src/violawake_sdk/models.py``.
Package-managed models, such as the OpenWakeWord backbone, are skipped by
default because they are not release assets owned by ViolaWake.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
REPORT_PATH = REPO_ROOT / "model-verify-report.json"
CHUNK_SIZE = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ViolaWake release model hashes")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exit nonzero when any verified model fails.",
    )
    parser.add_argument(
        "--model",
        metavar="NAME",
        help="Verify one registry key instead of all downloadable models.",
    )
    parser.add_argument(
        "--skip-deprecated",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip models whose description marks them deprecated (default: true).",
    )
    parser.add_argument(
        "--skip-package-managed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip models managed by upstream packages instead of ViolaWake releases (default: true).",
    )
    parser.add_argument(
        "--report",
        default=str(REPORT_PATH),
        help="JSON report path (default: model-verify-report.json).",
    )
    return parser.parse_args()


def ensure_src_importable() -> None:
    src_path = str(SRC_ROOT)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_deprecated(spec: Any) -> bool:
    description = str(getattr(spec, "description", ""))
    return "DEPRECATED" in description.upper()


def target_path(model_dir: Path, spec: Any) -> Path:
    ext = Path(str(getattr(spec, "url", ""))).suffix or ".onnx"
    return model_dir / f"{getattr(spec, 'name')}{ext}"


def download_to_path(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    os.close(fd)

    try:
        request = urllib.request.Request(url, headers={"User-Agent": "violawake-model-verify"})
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def verify_one(name: str, spec: Any, model_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "model": name,
        "status": "unknown",
        "checks": {},
        "errors": [],
        "warnings": [],
        "duration_s": 0.0,
    }

    url = str(getattr(spec, "url", ""))
    expected_sha256 = str(getattr(spec, "sha256", ""))
    if not url.startswith("https://"):
        result["checks"]["url"] = "fail"
        result["errors"].append(f"Refusing non-HTTPS URL: {url}")
        result["status"] = "fail"
        result["duration_s"] = round(time.monotonic() - started, 2)
        return result

    if not expected_sha256 or "placeholder" in expected_sha256.lower():
        result["checks"]["sha256"] = "fail"
        result["errors"].append("Registry SHA-256 is missing or still a placeholder.")
        result["status"] = "fail"
        result["duration_s"] = round(time.monotonic() - started, 2)
        return result

    path = target_path(model_dir, spec)
    if path.exists():
        result["checks"]["download"] = "cached"
    else:
        try:
            download_to_path(url, path)
            result["checks"]["download"] = "pass"
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            result["checks"]["download"] = "fail"
            result["errors"].append(f"Download failed: {exc}")
            result["status"] = "fail"
            result["duration_s"] = round(time.monotonic() - started, 2)
            return result

    actual_sha256 = sha256_file(path)
    result["checks"]["sha256"] = "pass" if actual_sha256 == expected_sha256 else "fail"
    result["actual_sha256"] = actual_sha256
    result["expected_sha256"] = expected_sha256
    result["size_bytes"] = path.stat().st_size

    if actual_sha256 != expected_sha256:
        result["errors"].append(
            f"SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )

    expected_size = int(getattr(spec, "size_bytes", 0) or 0)
    if expected_size > 0:
        lower = int(expected_size * 0.8)
        upper = int(expected_size * 1.2)
        result["checks"]["size"] = "pass" if lower <= path.stat().st_size <= upper else "warn"
        if result["checks"]["size"] == "warn":
            result["warnings"].append(
                f"Size differs from registry: expected about {expected_size}, got {path.stat().st_size}"
            )

    result["status"] = "fail" if result["errors"] else "pass"
    result["duration_s"] = round(time.monotonic() - started, 2)
    return result


def iter_registry(args: argparse.Namespace) -> list[tuple[str, Any, str | None]]:
    ensure_src_importable()
    from violawake_sdk.models import MODEL_REGISTRY, get_model_dir  # noqa: PLC0415
    from violawake_sdk import models as model_module  # noqa: PLC0415

    package_managed = set(getattr(model_module, "_PACKAGE_MANAGED_MODELS", set()))

    if args.model:
        if args.model not in MODEL_REGISTRY:
            raise SystemExit(f"ERROR: model '{args.model}' not found in registry")
        spec = MODEL_REGISTRY[args.model]
        return [(args.model, spec, None)]

    entries: list[tuple[str, Any, str | None]] = []
    seen_spec_names: set[str] = set()
    for name, spec in MODEL_REGISTRY.items():
        spec_name = str(getattr(spec, "name", name))
        if spec_name in seen_spec_names:
            entries.append((name, spec, f"alias for {spec_name}"))
            continue
        seen_spec_names.add(spec_name)

        if args.skip_package_managed and (name in package_managed or spec_name in package_managed):
            entries.append((name, spec, "package-managed"))
            continue
        if args.skip_deprecated and is_deprecated(spec):
            entries.append((name, spec, "deprecated"))
            continue
        entries.append((name, spec, None))

    # Touch get_model_dir here so import failures surface before verification begins.
    get_model_dir()
    return entries


def main() -> int:
    args = parse_args()
    ensure_src_importable()
    from violawake_sdk.models import get_model_dir  # noqa: PLC0415

    model_dir = get_model_dir()
    results: list[dict[str, Any]] = []
    print(f"Model directory: {model_dir}")

    for name, spec, skip_reason in iter_registry(args):
        if skip_reason:
            print(f"SKIP {name}: {skip_reason}")
            results.append({
                "model": name,
                "status": "skipped",
                "reason": skip_reason,
            })
            continue
        print(f"VERIFY {name}")
        result = verify_one(name, spec, model_dir)
        print(f"  {result['status'].upper()}")
        for error in result.get("errors", []):
            print(f"  ERROR: {error}")
        for warning in result.get("warnings", []):
            print(f"  WARNING: {warning}")
        results.append(result)

    failed = [result for result in results if result.get("status") == "fail"]
    passed = [result for result in results if result.get("status") == "pass"]
    skipped = [result for result in results if result.get("status") == "skipped"]
    report = {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
        "results": results,
    }
    report_path = Path(args.report)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Results: {len(passed)} passed, {len(failed)} failed, "
        f"{len(skipped)} skipped. Report: {report_path}"
    )

    return 1 if args.ci and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
