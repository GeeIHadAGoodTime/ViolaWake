#!/usr/bin/env python3
"""Reject local non-merge commits made from the repository's main checkout."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def resolve_git_path(repo: Path, rev_parse_arg: str) -> Path:
    value = run_git(repo, "rev-parse", "--path-format=absolute", rev_parse_arg)
    return Path(value).resolve()


def is_main_checkout(repo: Path) -> bool:
    git_dir = resolve_git_path(repo, "--git-dir")
    git_common_dir = resolve_git_path(repo, "--git-common-dir")
    return git_dir == git_common_dir


def merge_head_path(repo: Path) -> Path:
    return resolve_git_path(repo, "--git-dir") / "MERGE_HEAD"


def check_repo(repo: Path, ci_mode: bool) -> int:
    repo = Path(run_git(repo, "rev-parse", "--show-toplevel")).resolve()

    if ci_mode:
        print(
            "PASS: CI mode ran the direct-main-commit checker; "
            "local checkout enforcement is handled by the hook."
        )
        return 0

    if merge_head_path(repo).exists():
        print(f"PASS: merge commit allowed from {repo} because MERGE_HEAD is present.")
        return 0

    if is_main_checkout(repo):
        print(
            "FAIL: refusing non-merge commit from the main checkout. "
            "Create a linked worktree off master and commit there."
        )
        return 1

    print(f"PASS: non-merge commit allowed from linked worktree {repo}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        default=".",
        help="repository path to inspect (default: current directory)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="run in CI compatibility mode without local checkout blocking",
    )
    args = parser.parse_args(argv)

    try:
        return check_repo(Path(args.repo).resolve(), args.ci)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
