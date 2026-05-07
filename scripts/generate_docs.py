#!/usr/bin/env python3
"""Generate static API reference docs with pdoc."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "api"
MODULES = ["violawake", "violawake_sdk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ViolaWake API docs with pdoc")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where pdoc HTML output should be written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pdoc command without generating files.",
    )
    return parser.parse_args()


def pdoc_available() -> bool:
    try:
        import importlib

        importlib.import_module("pdoc")
        return True
    except ImportError:
        return False


def build_command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pdoc",
        "--output-directory",
        str(output_dir),
        "--docformat",
        "google",
        *MODULES,
    ]


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    command = build_command(output_dir)

    src_path = str(ROOT / "src")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path
    )

    print(f"Command: {' '.join(command)}")
    print(f"Output:  {output_dir}")
    if args.dry_run:
        print(f"pdoc installed: {'yes' if pdoc_available() else 'no'}")
        return 0

    if not pdoc_available():
        print(
            "ERROR: pdoc is not installed. Install docs extras with: pip install -e '.[docs]'",
            file=sys.stderr,
        )
        return 1

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        print(
            "\npdoc exited with errors. Install optional extras if optional modules failed to import: "
            "pip install -e '.[docs,tts,stt]'",
            file=sys.stderr,
        )
        return result.returncode

    html_files = list(output_dir.rglob("*.html"))
    print(f"Generated {len(html_files)} HTML file(s) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
