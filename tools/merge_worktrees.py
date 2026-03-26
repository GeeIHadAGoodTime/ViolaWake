#!/usr/bin/env python3
"""
Merge all agent worktree branches back into the current branch.

Usage:
    python tools/merge_worktrees.py --list          # List worktree branches
    python tools/merge_worktrees.py --merge-all     # Merge all into current
    python tools/merge_worktrees.py --merge BRANCH   # Merge specific branch
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run_git(*args: str) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"git {' '.join(args)} failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def list_worktree_branches() -> list[str]:
    """List all branches created by worktree agents."""
    branches = run_git("branch", "--list").splitlines()
    # Worktree branches typically have patterns like agent-*, worktree-*, etc.
    worktree_branches = []
    for b in branches:
        name = b.strip().lstrip("* ")
        if name and name != "main" and name != "master":
            worktree_branches.append(name)
    return worktree_branches


def merge_branch(branch: str) -> bool:
    """Merge a branch into the current branch."""
    print(f"Merging {branch}...")
    result = subprocess.run(
        ["git", "merge", branch, "--no-edit", "-m", f"Merge {branch} into main"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  CONFLICT merging {branch}:\n{result.stdout}\n{result.stderr}")
        # Try to abort the merge
        subprocess.run(["git", "merge", "--abort"], capture_output=True)
        return False
    print(f"  OK: {branch} merged successfully")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge worktree branches")
    parser.add_argument("--list", action="store_true", help="List worktree branches")
    parser.add_argument("--merge-all", action="store_true", help="Merge all branches")
    parser.add_argument("--merge", metavar="BRANCH", help="Merge specific branch")
    args = parser.parse_args()

    if args.list:
        branches = list_worktree_branches()
        if not branches:
            print("No worktree branches found.")
        else:
            for b in branches:
                print(f"  {b}")
        return

    if args.merge:
        success = merge_branch(args.merge)
        sys.exit(0 if success else 1)

    if args.merge_all:
        branches = list_worktree_branches()
        if not branches:
            print("No branches to merge.")
            return

        print(f"Merging {len(branches)} branches into current:")
        failed = []
        for b in branches:
            if not merge_branch(b):
                failed.append(b)

        if failed:
            print(f"\nFAILED ({len(failed)}):")
            for b in failed:
                print(f"  - {b}")
            sys.exit(1)
        else:
            print(f"\nAll {len(branches)} branches merged successfully.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
