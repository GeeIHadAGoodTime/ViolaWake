"""One-shot repair after the edge-tts outage of 2026-10-14 to 2026-11-06.

Every job in that window failed on our side, yet each one kept the training attempt it
was charged and added a strike, so customers hit their monthly limit and six accounts
were left paused. The fixed code never does this again; this gives back what the old
code took:

  * refunds each failed job's attempt, through the queue's own once-only refund, so it
    lands on the month it was charged and running this twice cannot double-credit;
  * clears the strikes and pauses those failures caused.

Only rows carrying the outage's exact error text are touched. A dry run is the default.

Run inside the backend container, after the fix is live:
    docker exec wakeword-backend-1 python /app/scripts/repair_tts_outage.py
    docker exec wakeword-backend-1 python /app/scripts/repair_tts_outage.py --apply
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import init_db  # noqa: E402
from app.job_queue import JobQueue  # noqa: E402

OUTAGE_ERROR = "Quality gate could not synthesize%"
OUTAGE_START = "2026-10-14"


async def repair(queue: JobQueue, *, apply: bool) -> dict[str, int]:
    """Refund and clear. ``queue`` must not be started: this process dispatches nothing."""
    async with queue._connect() as conn:
        async with conn.execute(
            """
            SELECT id FROM jobs
            WHERE status = 'failed' AND error LIKE ? AND created_at >= ?
              AND usage_refunded = 0 AND usage_period_start IS NOT NULL
            ORDER BY id
            """,
            (OUTAGE_ERROR, OUTAGE_START),
        ) as cursor:
            job_ids = [int(row["id"]) async for row in cursor]
        async with conn.execute(
            """
            SELECT COUNT(*) AS n FROM user_circuit_breakers
            WHERE pause_reason LIKE ?
               OR (paused = 0 AND consecutive_failures > 0 AND last_failure_at >= ?)
            """,
            (OUTAGE_ERROR, OUTAGE_START),
        ) as cursor:
            breakers = int((await cursor.fetchone())["n"])

    refunded = 0
    if apply:
        for job_id in job_ids:
            if await queue._refund_usage_for_job(job_id):
                refunded += 1
        async with queue._connect() as conn:
            await conn.execute(
                """
                UPDATE user_circuit_breakers
                SET consecutive_failures = 0, paused = 0, next_attempt_at = NULL,
                    last_failure_at = NULL, pause_reason = NULL
                WHERE pause_reason LIKE ?
                   OR (paused = 0 AND consecutive_failures > 0 AND last_failure_at >= ?)
                """,
                (OUTAGE_ERROR, OUTAGE_START),
            )
            await conn.commit()
    return {"jobs": len(job_ids), "refunded": refunded, "breakers": breakers}


async def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="write the repair (default: report only)")
    args = parser.parse_args()
    await init_db()
    result = await repair(JobQueue(), apply=args.apply)
    print(("applied: " if args.apply else "dry run: ") + str(result))


if __name__ == "__main__":
    asyncio.run(_main())
