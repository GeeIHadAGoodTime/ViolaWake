"""Fetch release model assets from the model registry.

Downloads all non-deprecated, non-package-managed models from MODEL_REGISTRY,
verifies SHA-256 integrity, and saves them to an output directory.
Skips files that already exist with the correct hash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"
MODELS_MODULE_PATH = REPO_ROOT / "src" / "violawake_sdk" / "models.py"

DEPRECATED_MARKER = "DEPRECATED"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch release model assets from the ViolaWake model registry.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where fetched model files should be written (default: models/).",
    )
    return parser.parse_args()


def load_model_registry() -> OrderedDict[str, object]:
    """Load MODEL_REGISTRY and _PACKAGE_MANAGED_MODELS from the SDK module."""
    spec = importlib.util.spec_from_file_location("violawake_release_models", MODELS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model registry from {MODELS_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    package_managed = getattr(module, "_PACKAGE_MANAGED_MODELS", set())
    registry = OrderedDict()
    seen_names: set[str] = set()

    for key, model_spec in module.MODEL_REGISTRY.items():
        name = getattr(model_spec, "name", key)

        # Skip aliases (duplicate name entries)
        if name in seen_names:
            continue
        seen_names.add(name)

        # Skip package-managed models (e.g. oww_backbone)
        if key in package_managed or name in package_managed:
            continue

        # Skip deprecated models
        description = getattr(model_spec, "description", "")
        if DEPRECATED_MARKER in description.upper():
            continue

        registry[name] = model_spec

    return registry


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(name: str, url: str, expected_sha256: str, output_path: Path) -> bool:
    """Download a single model with progress bar and SHA-256 verification.

    Returns True if the file was downloaded (or already existed with correct hash),
    False on failure.
    """
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' is required. Install with: pip install requests")
        return False

    try:
        from tqdm import tqdm
    except ImportError:
        print("ERROR: 'tqdm' is required. Install with: pip install tqdm")
        return False

    # Check if file already exists with correct hash
    if output_path.exists():
        existing_hash = sha256_file(output_path)
        if existing_hash == expected_sha256:
            print(f"  SKIP {name}: already exists with correct hash")
            return True
        print(f"  REDOWNLOAD {name}: hash mismatch (expected {expected_sha256[:12]}..., got {existing_hash[:12]}...)")

    # Reject non-HTTPS URLs
    if not url.startswith("https://"):
        print(f"  ERROR {name}: refusing non-HTTPS URL: {url}")
        return False

    print(f"  DOWNLOAD {name} from {url}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR {name}: download failed: {e}")
        return False

    total_bytes = int(response.headers.get("content-length", 0))

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with (
            open(tmp_path, "wb") as f,
            tqdm(
                total=total_bytes or None,
                unit="B",
                unit_scale=True,
                desc=f"    {name}",
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

        # Verify SHA-256
        actual_hash = sha256_file(tmp_path)
        if actual_hash != expected_sha256:
            tmp_path.unlink(missing_ok=True)
            print(
                f"  ERROR {name}: SHA-256 mismatch! "
                f"Expected {expected_sha256[:16]}..., got {actual_hash[:16]}..."
            )
            return False

        # Atomic rename
        tmp_path.replace(output_path)
        print(f"  OK {name}: verified and saved to {output_path}")
        return True

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        print(f"  ERROR {name}: {e}")
        return False


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print("Loading model registry...")

    registry = load_model_registry()
    print(f"Found {len(registry)} downloadable model(s)\n")

    if not registry:
        print("No models to fetch.")
        return 0

    ok = 0
    failed = 0

    for name, spec in registry.items():
        url = getattr(spec, "url", "")
        sha256 = getattr(spec, "sha256", "")
        filename = Path(url).name

        if not url or not sha256:
            print(f"  SKIP {name}: missing URL or SHA-256")
            failed += 1
            continue

        if "placeholder" in sha256.lower():
            print(f"  SKIP {name}: placeholder SHA-256 hash")
            failed += 1
            continue

        output_path = output_dir / filename

        if download_model(name, url, sha256, output_path):
            ok += 1
        else:
            failed += 1

    print(f"\nDone: {ok} succeeded, {failed} failed out of {len(registry)} model(s)")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
