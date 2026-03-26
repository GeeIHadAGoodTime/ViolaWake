#!/usr/bin/env python3
"""
Run ViolaWake Console E2E tests.

Usage:
    python console/run_e2e.py                    # Run all E2E tests
    python console/run_e2e.py --api-only         # API tests only (no browser)
    python console/run_e2e.py --browser-only     # Browser tests only
    python console/run_e2e.py --install           # Install test dependencies first
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
E2E_DIR = PROJECT_ROOT / "console" / "tests" / "e2e"


def install_deps() -> None:
    """Install E2E test dependencies."""
    deps = [
        "pytest>=8.0",
        "requests>=2.31",
        "numpy>=1.24",
        "playwright>=1.42",
    ]
    print("Installing E2E test dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])

    print("Installing Playwright browsers (Chromium)...")
    subprocess.check_call(
        [sys.executable, "-m", "playwright", "install", "chromium"]
    )
    print("Dependencies installed.")


def run_tests(api_only: bool = False, browser_only: bool = False) -> int:
    """Run the E2E tests."""
    cmd = [sys.executable, "-m", "pytest", str(E2E_DIR), "-v", "--tb=short"]

    if api_only:
        cmd.extend(["-k", "test_api_flow"])
    elif browser_only:
        cmd.extend(["-k", "test_browser_flow"])

    # For browser tests, pass Playwright args for fake audio
    if not api_only:
        cmd.extend([
            "--browser-channel", "chromium",
        ])

    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ViolaWake Console E2E tests")
    parser.add_argument("--api-only", action="store_true", help="API tests only")
    parser.add_argument("--browser-only", action="store_true", help="Browser tests only")
    parser.add_argument("--install", action="store_true", help="Install dependencies first")
    args = parser.parse_args()

    if args.install:
        install_deps()
        return

    sys.exit(run_tests(api_only=args.api_only, browser_only=args.browser_only))


if __name__ == "__main__":
    main()
