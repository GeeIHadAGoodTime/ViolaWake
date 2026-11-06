"""The outage repair gives back what the old code took, once, and touches nothing else."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.job_queue import JobQueue

OUTAGE = "Quality gate could not synthesize any speech and confusable probe audio -- every edge-tts"
SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "scripts" / "repair_tts_outage.py"


def _script():
    spec = importlib.util.spec_from_file_location("repair_tts_outage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


async def _seed(queue: JobQueue) -> None:
    await queue._initialize_db()
    period = datetime(2026, 11, 1, tzinfo=timezone.utc).isoformat()
    rows = [
        (1, 355, "failed", OUTAGE, "2026-11-01T10:00:00", period),
        (2, 355, "failed", OUTAGE, "2026-11-01T11:00:00", period),
        (3, 356, "failed", "Grade F: model did not pass", "2026-11-01T12:00:00", period),
        (4, 357, "failed", OUTAGE, "2026-09-01T12:00:00", period),
    ]
    async with queue._connect() as conn:
        for job_id, user, status, error, created, charged in rows:
            await conn.execute(
                """
                INSERT INTO jobs (id, user_id, wake_word, status, created_at, error, recording_ids,
                                  epochs, tenant_key, usage_user_id, usage_period_start, usage_refunded)
                VALUES (?, ?, 'arthur', ?, ?, ?, '[]', 80, '', ?, ?, 0)
                """,
                (job_id, user, status, created, error, user, charged),
            )
        await conn.execute(
            """
            INSERT INTO user_circuit_breakers
                (user_id, tenant_key, consecutive_failures, paused, next_attempt_at, last_failure_at, pause_reason)
            VALUES (355, '', 3, 1, NULL, '2026-11-01T13:28:45', ?), (356, '', 3, 1, NULL, '2026-09-01T00:00:00', 'Grade F')
            """,
            (OUTAGE,),
        )
        await conn.commit()


async def test_a_dry_run_changes_nothing(tmp_path, monkeypatch) -> None:
    queue = JobQueue(db_path=tmp_path / "q.db")
    await _seed(queue)
    refund = AsyncMock(return_value=True)
    monkeypatch.setattr(queue, "_refund_usage_for_job", refund)

    result = await _script().repair(queue, apply=False)

    assert result == {"jobs": 2, "refunded": 0, "breakers": 1}
    refund.assert_not_awaited()
    assert (await queue.get_circuit_breaker(355)).paused is True


async def test_apply_refunds_only_outage_jobs_and_clears_only_outage_pauses(tmp_path, monkeypatch) -> None:
    queue = JobQueue(db_path=tmp_path / "q.db")
    await _seed(queue)
    refund = AsyncMock(return_value=True)
    monkeypatch.setattr(queue, "_refund_usage_for_job", refund)

    result = await _script().repair(queue, apply=True)

    assert result["refunded"] == 2
    assert [call.args[0] for call in refund.await_args_list] == [1, 2]
    assert (await queue.get_circuit_breaker(355)).paused is False
    assert (await queue.get_circuit_breaker(356)).paused is True, "a real grade-F pause is not ours to clear"
