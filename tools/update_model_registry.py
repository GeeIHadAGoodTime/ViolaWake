"""Update MODEL_REGISTRY release metadata in ``src/violawake_sdk/models.py``."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_REGISTRY_PATH = REPO_ROOT / "src" / "violawake_sdk" / "models.py"
GITHUB_OWNER = "GeeIHadAGoodTime"
GITHUB_REPO = "ViolaWake"
GITHUB_DOWNLOAD_BASE = (
    f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/download"
)
GITHUB_RELEASE_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/tags"
)


@dataclass(frozen=True)
class AssetInfo:
    """Resolved metadata for a release asset."""

    filename: str
    sha256: str
    size_bytes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update MODEL_REGISTRY URLs, sizes, and SHA-256 hashes for a release.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="Release version without the leading 'v' (example: 0.1.0).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_LOCAL_MODELS_DIR),
        help="Directory containing local release model assets. Default: ./models",
    )
    parser.add_argument(
        "--registry-path",
        default=str(DEFAULT_REGISTRY_PATH),
        help="Path to src/violawake_sdk/models.py. Default: project registry path.",
    )
    return parser.parse_args()


def load_registry_module(registry_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("violawake_release_registry", registry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model registry from {registry_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def find_registry_assignment(module_ast: ast.Module) -> ast.AnnAssign | ast.Assign:
    for node in module_ast.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "MODEL_REGISTRY":
                return node
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_REGISTRY":
                    return node
    raise RuntimeError("Could not find MODEL_REGISTRY assignment in models.py")


def registry_entry_keys(registry_node: ast.AnnAssign | ast.Assign) -> list[str]:
    value = registry_node.value
    if not isinstance(value, ast.Dict):
        raise RuntimeError("MODEL_REGISTRY assignment is not a dict literal")

    keys: list[str] = []
    for key_node in value.keys:
        if key_node is None:
            raise RuntimeError("MODEL_REGISTRY contains an unsupported dict unpack")
        key = ast.literal_eval(key_node)
        if not isinstance(key, str):
            raise RuntimeError("MODEL_REGISTRY contains a non-string key")
        keys.append(key)
    return keys


def registry_specs_in_order(registry_path: Path) -> OrderedDict[str, Any]:
    source = registry_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    assignment = find_registry_assignment(module_ast)
    keys = registry_entry_keys(assignment)

    module = load_registry_module(registry_path)
    ordered_specs: OrderedDict[str, Any] = OrderedDict()
    for key in keys:
        ordered_specs[key] = module.MODEL_REGISTRY[key]
    return ordered_specs


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "violawake-release-tooling",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_release_asset_metadata(version: str) -> dict[str, dict[str, Any]]:
    url = f"{GITHUB_RELEASE_API}/v{version}"
    request = Request(url, headers=github_headers())

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise RuntimeError(
            f"GitHub Release API request failed for v{version}: HTTP {exc.code}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"GitHub Release API request failed for v{version}: {exc.reason}"
        ) from exc

    assets = payload.get("assets", [])
    metadata: dict[str, dict[str, Any]] = {}
    for asset in assets:
        name = asset.get("name")
        if isinstance(name, str):
            metadata[name] = asset
    return metadata


def hash_remote_asset(url: str) -> tuple[str, int]:
    request = Request(url, headers=github_headers())
    digest = hashlib.sha256()
    size_bytes = 0

    try:
        with urlopen(request, timeout=60) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                size_bytes += len(chunk)
    except HTTPError as exc:
        raise RuntimeError(f"Asset download failed: {url} (HTTP {exc.code})") from exc
    except URLError as exc:
        raise RuntimeError(f"Asset download failed: {url} ({exc.reason})") from exc

    return digest.hexdigest(), size_bytes


def resolve_asset_info(version: str, models_dir: Path, specs: OrderedDict[str, Any]) -> dict[str, AssetInfo]:
    asset_info: dict[str, AssetInfo] = {}
    remote_assets: dict[str, dict[str, Any]] | None = None

    for model_name, spec in specs.items():
        filename = Path(spec.url).name
        local_path = models_dir / filename

        if local_path.exists():
            sha256 = hash_file(local_path)
            size_bytes = local_path.stat().st_size
            print(f"Using local asset for {model_name}: {local_path}")
            asset_info[model_name] = AssetInfo(
                filename=filename,
                sha256=sha256,
                size_bytes=size_bytes,
            )
            continue

        if remote_assets is None:
            print(f"Local models not complete in {models_dir}; falling back to GitHub Release assets.")
            remote_assets = fetch_release_asset_metadata(version)

        asset = remote_assets.get(filename)
        if asset is None:
            raise RuntimeError(
                f"Could not find release asset '{filename}' in local models or GitHub Release v{version}"
            )

        browser_download_url = asset.get("browser_download_url")
        if not isinstance(browser_download_url, str):
            raise RuntimeError(f"Release asset '{filename}' is missing browser_download_url")

        print(f"Hashing remote asset for {model_name}: {browser_download_url}")
        sha256, observed_size = hash_remote_asset(browser_download_url)
        size_bytes = int(asset.get("size") or observed_size)
        asset_info[model_name] = AssetInfo(
            filename=filename,
            sha256=sha256,
            size_bytes=size_bytes,
        )

    return asset_info


def quoted(value: str) -> str:
    return json.dumps(value)


def render_registry_block(version: str, specs: OrderedDict[str, Any], assets: dict[str, AssetInfo]) -> str:
    lines = ["MODEL_REGISTRY: dict[str, ModelSpec] = {\n"]

    for model_name, spec in specs.items():
        asset = assets[model_name]
        release_url = f"{GITHUB_DOWNLOAD_BASE}/v{version}/{asset.filename}"
        lines.extend(
            [
                f"    {quoted(model_name)}: ModelSpec(\n",
                f"        name={quoted(spec.name)},\n",
                f"        url={quoted(release_url)},\n",
                f"        sha256={quoted(asset.sha256)},\n",
                f"        size_bytes={asset.size_bytes:_},\n",
                f"        description={quoted(spec.description)},\n",
                f"        version={quoted(version)},\n",
                "    ),\n",
            ]
        )

    lines.append("}\n")
    return "".join(lines)


def replace_registry_block(source: str, registry_node: ast.AnnAssign | ast.Assign, new_block: str) -> str:
    if registry_node.end_lineno is None:
        raise RuntimeError("MODEL_REGISTRY assignment has no end position")

    lines = source.splitlines(keepends=True)
    start_index = registry_node.lineno - 1
    end_index = registry_node.end_lineno
    return "".join(lines[:start_index] + [new_block] + lines[end_index:])


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry_path).resolve()
    models_dir = Path(args.models_dir).resolve()

    source = registry_path.read_text(encoding="utf-8")
    module_ast = ast.parse(source)
    registry_node = find_registry_assignment(module_ast)
    specs = registry_specs_in_order(registry_path)
    assets = resolve_asset_info(args.version, models_dir, specs)
    new_registry_block = render_registry_block(args.version, specs, assets)
    new_source = replace_registry_block(source, registry_node, new_registry_block)

    registry_path.write_text(new_source, encoding="utf-8")
    print(f"Updated MODEL_REGISTRY in {registry_path} for release v{args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
