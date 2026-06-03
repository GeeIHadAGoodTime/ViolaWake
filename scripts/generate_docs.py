#!/usr/bin/env python3
"""Generate static API reference docs with pdoc."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs" / "api"
MODULES = ["violawake", "violawake_sdk"]
PUBLIC_COPY_FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "COMPETITIVE_ANALYSIS.md",
    ROOT / "docs" / "SHOW_HN_DRAFT.md",
    ROOT / "docs" / "index.html",
]
FORBIDDEN_PUBLIC_COPY_PATTERNS = [
    re.compile(r"Reconciled canon", re.IGNORECASE),
    re.compile(r"generated from a single Markdown source of truth", re.IGNORECASE),
    re.compile(r"Corrections published as dated amendments", re.IGNORECASE),
    re.compile(r"Self-Certification(?: Note)?", re.IGNORECASE),
    re.compile(r"Professional legal review is recommended", re.IGNORECASE),
    re.compile(r"real external audit targeted Q3 2026", re.IGNORECASE),
    re.compile(r"Not Offered in This Public Launch", re.IGNORECASE),
    re.compile(r"links? to internal review docs?", re.IGNORECASE),
    re.compile(r"production-tested", re.IGNORECASE),
]
UNSUPPORTED_LATENCY_SNAPSHOT = re.compile(
    r"Measured on .*?\n\n\| Operation \| Latency \(p50\) \| Latency \(p99\) \|",
    re.IGNORECASE | re.DOTALL,
)


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
    parser.add_argument(
        "--check-public-copy",
        action="store_true",
        help="Fail if public copy contains forbidden process language or uncited latency snapshots.",
    )
    parser.add_argument(
        "--check-api-public-surface",
        action="store_true",
        help="Fail if docs/api is missing a symbol exported by violawake_sdk.__all__.",
    )
    parser.add_argument(
        "--public-copy-file",
        action="append",
        default=[],
        help="Additional public-copy file to include in --check-public-copy.",
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


def check_public_copy(extra_files: list[str]) -> int:
    failures: list[str] = []
    files = [*PUBLIC_COPY_FILES, *(Path(path) for path in extra_files)]

    for path in files:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            failures.append(f"{resolved}: missing public-copy file")
            continue
        text = resolved.read_text(encoding="utf-8", errors="replace")
        rel = resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved
        for pattern in FORBIDDEN_PUBLIC_COPY_PATTERNS:
            if pattern.search(text):
                failures.append(f"{rel}: forbidden public-copy pattern: {pattern.pattern}")
        if UNSUPPORTED_LATENCY_SNAPSHOT.search(text):
            failures.append(f"{rel}: fixed latency table lacks a checked-in benchmark result")

    if failures:
        print("Public copy check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"Public copy check passed: {len(files)} file(s)")
    return 0


def check_api_public_surface(api_dir: Path) -> int:
    src_path = str(ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    import violawake_sdk

    docs_file = api_dir / "violawake_sdk.html"
    if not docs_file.exists():
        print(f"ERROR: API docs file missing: {docs_file}", file=sys.stderr)
        return 1

    text = docs_file.read_text(encoding="utf-8", errors="replace")
    # Require the canonical pdoc anchor `id="violawake_sdk.<name>"` rather than
    # any incidental string match. Without this tightening, a symbol listed only
    # in __all__ source text (or any other accidental occurrence) would pass.
    missing = [
        name
        for name in violawake_sdk.__all__
        if name != "__version__"
        and f'id="violawake_sdk.{name}"' not in text
    ]
    if missing:
        print("API public surface check failed:", file=sys.stderr)
        for name in missing:
            print(f"- missing from docs/api: {name}", file=sys.stderr)
        return 1

    print(f"API public surface check passed: {len(violawake_sdk.__all__) - 1} symbol(s)")
    return 0


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    command = build_command(output_dir)

    if args.check_public_copy:
        public_copy_result = check_public_copy(args.public_copy_file)
        if public_copy_result != 0:
            return public_copy_result

    if args.check_api_public_surface:
        api_result = check_api_public_surface(output_dir)
        if api_result != 0:
            return api_result

    if args.check_public_copy or args.check_api_public_surface:
        return 0

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
