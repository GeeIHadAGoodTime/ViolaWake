#!/usr/bin/env python3
"""G5: Dependency audit script — check installed packages for known CVEs.

Usage::

    python tools/audit_deps.py              # Audit all installed deps
    python tools/audit_deps.py --strict     # Exit non-zero on any finding
    python tools/audit_deps.py --json       # Output machine-readable JSON

Requires ``pip-audit`` (pip install pip-audit).
Falls back to ``pip`` vulnerability check if pip-audit is unavailable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def run_pip_audit(strict: bool = False, json_output: bool = False) -> int:
    """Run pip-audit and return exit code."""
    try:
        cmd = [sys.executable, "-m", "pip_audit"]
        if json_output:
            cmd.extend(["--format", "json"])
        else:
            cmd.extend(["--format", "columns"])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0 and strict:
            print("\nSTRICT MODE: Vulnerabilities found — failing.", file=sys.stderr)
            return 1
        return result.returncode

    except FileNotFoundError:
        return -1  # pip-audit not installed


def run_pip_check() -> int:
    """Fallback: run pip check for dependency conflicts."""
    print("pip-audit not installed. Falling back to 'pip check'...")
    print("Install pip-audit for CVE scanning: pip install pip-audit\n")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode


def check_pinned_ranges() -> list[dict[str, str]]:
    """Check that critical dependencies have upper-bound pins in pyproject.toml."""
    from pathlib import Path
    import re

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"WARNING: {pyproject_path} not found", file=sys.stderr)
        return []

    content = pyproject_path.read_text()
    findings: list[dict[str, str]] = []

    # Extract dependencies block
    in_deps = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("dependencies = ["):
            in_deps = True
            continue
        if in_deps and stripped == "]":
            in_deps = False
            continue
        if in_deps and stripped.startswith('"'):
            dep = stripped.strip('",')
            # Check for upper bound (< or <=)
            if "<" not in dep and dep and not dep.startswith("#"):
                pkg_name = re.split(r"[>=<!\[]", dep)[0]
                findings.append({
                    "package": pkg_name,
                    "issue": "No upper-bound version pin",
                    "line": dep,
                })

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="audit-deps",
        description="Audit dependencies for known CVEs and version pin issues.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any finding")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    args = parser.parse_args()

    exit_code = 0

    # Phase 1: Check version pins
    print("=" * 60)
    print("Phase 1: Checking dependency version pins")
    print("=" * 60)
    pin_findings = check_pinned_ranges()
    if pin_findings:
        print(f"\nWARNING: {len(pin_findings)} dependencies missing upper-bound pins:")
        for f in pin_findings:
            print(f"  - {f['package']}: {f['issue']} ({f['line']})")
        if args.strict:
            exit_code = 1
    else:
        print("All critical dependencies have upper-bound version pins.")

    print()

    # Phase 2: CVE scan
    print("=" * 60)
    print("Phase 2: Scanning for known CVEs")
    print("=" * 60)
    audit_code = run_pip_audit(strict=args.strict, json_output=args.json_output)
    if audit_code == -1:
        # pip-audit not available, fallback
        fallback_code = run_pip_check()
        if fallback_code != 0 and args.strict:
            exit_code = 1
    elif audit_code != 0:
        exit_code = audit_code

    print()
    if exit_code == 0:
        print("Dependency audit passed.")
    else:
        print("Dependency audit found issues.", file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
