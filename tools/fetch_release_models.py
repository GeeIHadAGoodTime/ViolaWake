"""Fetch release model assets for GitHub Releases.

Current MVP behavior:
  - Reports the secure artifact-store download actions it would perform.
  - Checks whether MODEL_STORE_TOKEN is available.
  - Falls back to local files in the repository's ``models/`` directory.

The secure model-store implementation is intentionally left as a TODO until
the backing S3/GCS artifact store contract is finalized.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODELS_DIR = REPO_ROOT / "models"
MODELS_MODULE_PATH = REPO_ROOT / "src" / "violawake_sdk" / "models.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch release model assets for a GitHub Release.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version without the leading 'v' (example: 0.1.0).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory where fetched model files should be written.",
    )
    return parser.parse_args()


def load_model_registry() -> OrderedDict[str, object]:
    spec = importlib.util.spec_from_file_location("violawake_release_models", MODELS_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model registry from {MODELS_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    registry = OrderedDict()
    for model_name, model_spec in module.MODEL_REGISTRY.items():
        if getattr(model_spec, "name", None) in registry:
            continue
        registry[model_spec.name] = model_spec
    return registry


def expected_assets() -> list[tuple[str, str]]:
    assets: list[tuple[str, str]] = []
    for model_name, model_spec in load_model_registry().items():
        filename = Path(model_spec.url).name
        assets.append((model_name, filename))
    return assets


def prepare_local_fallback(output_dir: Path) -> int:
    copied = 0
    missing: list[str] = []

    print(f"Falling back to local model directory: {DEFAULT_LOCAL_MODELS_DIR}")
    if not DEFAULT_LOCAL_MODELS_DIR.exists():
        print(
            "ERROR: local fallback directory does not exist. "
            "Create repository-local model assets or implement the artifact-store download."
        )
        return 1

    same_directory = output_dir.resolve() == DEFAULT_LOCAL_MODELS_DIR.resolve()

    for model_name, filename in expected_assets():
        source_path = DEFAULT_LOCAL_MODELS_DIR / filename
        target_path = output_dir / filename

        print(f"Checking local fallback for {model_name}: {filename}")
        if not source_path.exists():
            missing.append(filename)
            continue

        if same_directory:
            print(f"  Using existing file in-place: {target_path}")
            copied += 1
            continue

        shutil.copy2(source_path, target_path)
        print(f"  Copied {source_path} -> {target_path}")
        copied += 1

    if missing:
        print("ERROR: missing local fallback assets:")
        for filename in missing:
            print(f"  - {filename}")
        return 1

    print(f"Prepared {copied} model asset(s) in {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("MODEL_STORE_TOKEN")
    if token:
        print("MODEL_STORE_TOKEN detected.")
    else:
        print("MODEL_STORE_TOKEN is not set.")

    print(f"Preparing release models for v{args.version}")
    for model_name, filename in expected_assets():
        print(
            f"TODO: would fetch {filename} for model '{model_name}' "
            f"from the secure artifact store for release v{args.version}."
        )

    print(
        "TODO: implement artifact-store download support for S3/GCS with MODEL_STORE_TOKEN; "
        "using local fallback for now."
    )
    return prepare_local_fallback(output_dir)


if __name__ == "__main__":
    sys.exit(main())
