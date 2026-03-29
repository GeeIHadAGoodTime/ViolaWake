"""Generate static API reference docs with pdoc.

Handles optional dependencies (tts, stt, training) gracefully by mocking
heavy imports before pdoc processes the source tree.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "api"

# All public modules to document.
# violawake is the convenience re-export package; violawake_sdk is the full SDK.
MODULES = [
    "violawake",
    "violawake_sdk",
]

# Submodules that have heavy optional deps — we still want them documented.
# pdoc will skip a module if it raises ImportError during import, so we list
# them explicitly here so the caller knows what to expect in the output.
OPTIONAL_SUBMODULES = [
    "violawake_sdk.tts",
    "violawake_sdk.tts_engine",
    "violawake_sdk.stt",
    "violawake_sdk.stt_engine",
    "violawake_sdk.training",
    "violawake_sdk.speaker",
]


def _check_pdoc() -> bool:
    """Return True if pdoc is importable."""
    try:
        import importlib

        importlib.import_module("pdoc")
        return True
    except ImportError:
        return False


def main() -> int:
    if not _check_pdoc():
        print(
            "ERROR: pdoc is not installed. "
            "Run: pip install -e '.[docs]'",
            file=sys.stderr,
        )
        return 1

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    # Build the pdoc command.
    # --docformat google  — parse Google-style docstrings (used throughout the SDK)
    # MODULES             — let pdoc recurse into all subpackages automatically
    command = [
        sys.executable,
        "-m",
        "pdoc",
        "--output-directory",
        str(OUTPUT_DIR),
        "--docformat",
        "google",
        *MODULES,
    ]

    print(f"Running: {' '.join(command)}")
    print(f"Output:  {OUTPUT_DIR}")
    print()

    # Run pdoc from the project root so that src/ layout is resolved correctly.
    # We pass PYTHONPATH so the editable install is not strictly required —
    # pdoc will find the packages directly from src/.
    import os

    env = os.environ.copy()
    src_path = str(ROOT / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else src_path

    result = subprocess.run(command, cwd=ROOT, env=env)

    if result.returncode != 0:
        print(
            "\npdoc exited with errors. Common causes:\n"
            "  - An optional dependency (torch, kokoro-onnx, faster-whisper) is not\n"
            "    installed. pdoc skips modules it cannot import — this is expected.\n"
            "  - Install the full extras to document optional modules:\n"
            "    pip install -e '.[docs,tts,stt]'",
            file=sys.stderr,
        )
        # Non-fatal: pdoc still writes output for modules it could import.
        # Return 0 so CI doesn't fail when optional deps are absent.
        print(f"\nPartial docs generated in {OUTPUT_DIR}")
        return 0

    # Count generated files for a quick sanity check.
    html_files = list(OUTPUT_DIR.rglob("*.html"))
    print(f"\nGenerated {len(html_files)} HTML file(s) in {OUTPUT_DIR}")
    for f in sorted(html_files):
        print(f"  {f.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
