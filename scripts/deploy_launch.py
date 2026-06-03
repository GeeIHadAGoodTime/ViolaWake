#!/usr/bin/env python3
"""Deprecated deployment helper.

The old version of this script contained Railway-era launch instructions and
a hard-coded production-looking secret. Production deploys are now documented
in docs/DEPLOYMENT.md and must be run from those steps directly.
"""

from __future__ import annotations

import sys


def main() -> int:
    print(
        "scripts/deploy_launch.py is deprecated. "
        "Use docs/DEPLOYMENT.md for current manual deploy steps.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
