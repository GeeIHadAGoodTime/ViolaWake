#!/usr/bin/env python3
"""Fetch ViolaWake ONNX model assets from a GitHub Release."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "models"
DEFAULT_REPO = "GeeIHadAGoodTime/ViolaWake"
CHUNK_SIZE = 1024 * 1024


def read_project_version() -> str:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib  # type: ignore[no-redef]

    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["version"])


def normalize_tag(value: str) -> str:
    value = value.strip()
    return value if value.startswith("v") else f"v{value}"


def default_tag() -> str:
    return normalize_tag(read_project_version())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model assets from a GitHub Release")
    parser.add_argument(
        "--tag",
        default=default_tag(),
        help="Release tag to download from (default: pyproject.toml version with v prefix).",
    )
    parser.add_argument(
        "--version",
        help="Compatibility alias used by release.yml; converted to a v-prefixed tag.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="GitHub repository in owner/name form.",
    )
    parser.add_argument(
        "--pattern",
        default="*.onnx",
        help="Release asset glob to download (default: *.onnx).",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where release assets should be written.",
    )
    parser.add_argument(
        "--no-gh",
        action="store_true",
        help="Skip gh CLI even if it is installed and use the GitHub API fallback.",
    )
    args = parser.parse_args()
    if args.version:
        args.tag = normalize_tag(args.version)
    else:
        args.tag = normalize_tag(args.tag)
    return args


def gh_release_download(repo: str, tag: str, pattern: str, output_dir: Path) -> int | None:
    gh = shutil.which("gh")
    if gh is None:
        return None

    command = [
        gh,
        "release",
        "download",
        tag,
        "--repo",
        repo,
        "--pattern",
        pattern,
        "--dir",
        str(output_dir),
        "--clobber",
    ]
    print(f"Running: {' '.join(command)}")
    return subprocess.run(command, check=False).returncode


def github_request(url: str) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "violawake-fetch-release-models",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_asset(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(output_path.parent),
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    tmp_path = Path(tmp_name)
    os.close(fd)

    headers = {"User-Agent": "violawake-fetch-release-models"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=120) as response, tmp_path.open("wb") as out:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def urllib_release_download(repo: str, tag: str, pattern: str, output_dir: Path) -> int:
    encoded_tag = urllib.parse.quote(tag, safe="")
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{encoded_tag}"
    print(f"Fetching release metadata: {release_url}")
    release = github_request(release_url)
    assets = release.get("assets", [])
    matches = [
        asset
        for asset in assets
        if fnmatch.fnmatch(str(asset.get("name", "")), pattern)
    ]

    if not matches:
        print(f"ERROR: no assets matching {pattern!r} found on {repo} {tag}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    for asset in matches:
        name = str(asset["name"])
        url = str(asset["browser_download_url"])
        output_path = output_dir / name
        print(f"Downloading {name} -> {output_path}")
        download_asset(url, output_path)

    print(f"Downloaded {len(matches)} asset(s) to {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    print(f"Release: {args.repo} {args.tag}")
    print(f"Pattern: {args.pattern}")
    print(f"Output:  {output_dir}")

    if not args.no_gh:
        gh_result = gh_release_download(args.repo, args.tag, args.pattern, output_dir)
        if gh_result == 0:
            return 0
        if gh_result is not None:
            print("gh release download failed; falling back to GitHub API.", file=sys.stderr)

    return urllib_release_download(args.repo, args.tag, args.pattern, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
