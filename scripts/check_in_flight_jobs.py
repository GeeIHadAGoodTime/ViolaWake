#!/usr/bin/env python3
"""Pre-deploy guard: block backend recreate while training jobs are in flight.

The Console training queue lives in SQLite at /app/data/job_queue.db inside
`wakeword-backend-1`. A backend recreate (`docker compose up -d backend`)
kills any RUNNING job — they are marked PENDING by `_resume_jobs()` on
restart and re-queued, but a single slow progress event during the new
container's warmup can flip them straight to FAILED with
`error_reason=timeout` (Job 51 hit this on 2026-05-07).

This guard counts RUNNING and QUEUED jobs and exits non-zero if any are
in flight, so a deploy script can refuse to recreate the container until
the queue drains. Override with `--force` or `VIOLAWAKE_DEPLOY_FORCE=1`
for emergency hotfixes.

Usage:
    python scripts/check_in_flight_jobs.py
    python scripts/check_in_flight_jobs.py --container wakeword-backend-1
    python scripts/check_in_flight_jobs.py --force          # bypass

Exit codes:
    0  — queue is idle, safe to recreate
    1  — at least one job is RUNNING or PENDING (deploy blocked)
    2  — could not query the container (docker not running, etc.)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from textwrap import dedent


DEFAULT_CONTAINER = "wakeword-backend-1"
DEFAULT_DB_PATH = "/app/data/job_queue.db"


def query_in_flight(container: str, db_path: str) -> dict[str, int]:
    """Run a one-shot SQLite query inside the container."""
    snippet = dedent(
        f"""
        import sqlite3, json
        c = sqlite3.connect({db_path!r})
        rows = list(c.execute('SELECT status, COUNT(*) FROM jobs GROUP BY status'))
        print(json.dumps(dict(rows)))
        """
    ).strip()
    # Exec as the container's app user: the hardened container drops ALL
    # capabilities, so exec'ing as root (the docker exec default) lacks
    # CAP_DAC_OVERRIDE and cannot open the app-owned SQLite file for the
    # journal check — the guard then dies with "attempt to write a readonly
    # database" (exit 2) instead of answering.
    result = subprocess.run(
        ["docker", "exec", "-u", "app", container, "python", "-c", snippet],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"docker exec {container} failed (exit {result.returncode}): "
            f"{result.stderr.strip()[:400]}"
        )
    import json

    return json.loads(result.stdout.strip() or "{}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--container",
        default=DEFAULT_CONTAINER,
        help=f"Backend container name (default: {DEFAULT_CONTAINER}).",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"SQLite path inside the container (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=os.environ.get("VIOLAWAKE_DEPLOY_FORCE") == "1",
        help="Bypass the guard. Equivalent to VIOLAWAKE_DEPLOY_FORCE=1.",
    )
    args = parser.parse_args()

    try:
        counts = query_in_flight(args.container, args.db_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    running = int(counts.get("running", 0))
    pending = int(counts.get("pending", 0))
    in_flight = running + pending

    print(f"job_queue counts: {counts}")

    if in_flight == 0:
        print("OK — no running or pending jobs. Safe to recreate the backend.")
        return 0

    if args.force:
        print(
            f"WARNING — {running} running and {pending} pending jobs in flight, "
            "but --force was passed. Recreating will kill or interrupt them."
        )
        return 0

    print(
        f"BLOCKED — {running} running and {pending} pending jobs in flight.\n"
        "Refusing to recreate the backend. Wait for the queue to drain or "
        "pass --force / set VIOLAWAKE_DEPLOY_FORCE=1 to override.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
