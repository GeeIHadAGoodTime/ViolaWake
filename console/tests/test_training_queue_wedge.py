"""Regression tests for training-queue wedge detection (issue #1481).

Ratchet detector for gate ``training-queue-wedge-detection``.

Before the fix, ``health._check_training_queue`` reported ``degraded`` on ANY
non-empty queue and had no notion of job age or breaker-blocked jobs. A single
pending job whose user's circuit breaker was paused (job 85 / user 135) could
therefore never be dispatched by any worker yet pinned the whole backend
``degraded`` / HTTP 503 for hours -- while a real dispatcher wedge produced the
exact same signal, so the signal was useless for paging. These tests fail on
that pre-fix shape and pass on the honest wedge logic: ERROR only on a genuine
dispatcher wedge, OK on ordinary backlog and paused/backoff-blocked jobs.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.job_queue import JobQueue, JobStatus, _serialize_datetime, _utcnow

    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run_with_queue(tmp_path, coro_fn):
    async def _inner():
        q = JobQueue(db_path=tmp_path / "wedge.db", max_concurrent=1, max_pending=10)
        # DB only -- no worker loop, so inserted pending rows stay pending and we
        # can assert the snapshot deterministically.
        await q._initialize_db()
        await coro_fn(q)

    _call(_inner())


async def _insert_pending(q, *, job_id, user_id, age_s, wake_word="w"):
    async with q._connect() as conn:
        await conn.execute(
            "INSERT INTO jobs "
            "(id, user_id, wake_word, status, created_at, recording_ids, epochs, priority) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                user_id,
                wake_word,
                JobStatus.PENDING.value,
                _serialize_datetime(_utcnow() - timedelta(seconds=age_s)),
                "[1, 2, 3, 4, 5]",
                10,
                0,
            ),
        )
        await conn.commit()


async def _set_breaker(q, *, user_id, paused, next_attempt_offset_s=None):
    next_attempt_at = (
        _serialize_datetime(_utcnow() + timedelta(seconds=next_attempt_offset_s))
        if next_attempt_offset_s is not None
        else None
    )
    async with q._connect() as conn:
        await conn.execute(
            "INSERT INTO user_circuit_breakers "
            "(user_id, consecutive_failures, paused, next_attempt_at, last_failure_at, pause_reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                user_id,
                3 if paused else 1,
                1 if paused else 0,
                next_attempt_at,
                _serialize_datetime(_utcnow()),
                "quality gate F" if paused else None,
            ),
        )
        await conn.commit()


# --- runtime_snapshot: the wedge signal -------------------------------------


def test_paused_breaker_pending_is_blocked_not_dispatchable(tmp_path):
    """A pending job under a paused breaker (the job-85 shape) is blocked, so it
    contributes to blocked_pending_jobs and NOT to the dispatchable age."""

    async def _t(q):
        await _insert_pending(q, job_id=85, user_id=135, age_s=100000)
        await _set_breaker(q, user_id=135, paused=True)
        snap = await q.runtime_snapshot()
        assert snap["queue_depth"] == 1
        assert snap["blocked_pending_jobs"] == 1
        assert snap["oldest_pending_age_s"] is not None
        assert snap["oldest_pending_age_s"] > 90000
        # The wedge signal must NOT fire: no worker can pick this up.
        assert snap["oldest_dispatchable_pending_age_s"] is None

    _run_with_queue(tmp_path, _t)


def test_backoff_breaker_pending_is_blocked(tmp_path):
    """A pending job under an active failure backoff is blocked until the
    backoff elapses -- also not a dispatcher wedge."""

    async def _t(q):
        await _insert_pending(q, job_id=1, user_id=7, age_s=5000)
        await _set_breaker(q, user_id=7, paused=False, next_attempt_offset_s=300)
        snap = await q.runtime_snapshot()
        assert snap["blocked_pending_jobs"] == 1
        assert snap["oldest_dispatchable_pending_age_s"] is None

    _run_with_queue(tmp_path, _t)


def test_dispatchable_pending_age_is_tracked(tmp_path):
    """A pending job with no blocking breaker is dispatchable -- its age is the
    signal a wedge is measured against."""

    async def _t(q):
        await _insert_pending(q, job_id=1, user_id=7, age_s=700)
        snap = await q.runtime_snapshot()
        assert snap["blocked_pending_jobs"] == 0
        assert snap["oldest_dispatchable_pending_age_s"] is not None
        assert snap["oldest_dispatchable_pending_age_s"] > 600

    _run_with_queue(tmp_path, _t)


def test_expired_backoff_pending_is_dispatchable(tmp_path):
    """Once the backoff timestamp is in the past, the job is dispatchable again."""

    async def _t(q):
        await _insert_pending(q, job_id=1, user_id=7, age_s=700)
        await _set_breaker(q, user_id=7, paused=False, next_attempt_offset_s=-10)
        snap = await q.runtime_snapshot()
        assert snap["blocked_pending_jobs"] == 0
        assert snap["oldest_dispatchable_pending_age_s"] is not None

    _run_with_queue(tmp_path, _t)


# --- health._check_training_queue: the pageable status ----------------------


def _fake_queue(snapshot):
    class _Q:
        async def runtime_snapshot(self):
            return snapshot

    return _Q()


def _snapshot(
    *,
    queue_depth,
    blocked_pending_jobs=0,
    oldest_pending_age_s=None,
    oldest_dispatchable_pending_age_s=None,
    active_workers=0,
    max_workers=1,
    worker_task_running=True,
):
    return {
        "queue_depth": queue_depth,
        "in_memory_queue_depth": 0,
        "persisted_running_jobs": active_workers,
        "blocked_pending_jobs": blocked_pending_jobs,
        "oldest_pending_age_s": oldest_pending_age_s,
        "oldest_dispatchable_pending_age_s": oldest_dispatchable_pending_age_s,
        "worker_status": {
            "active_workers": active_workers,
            "max_workers": max_workers,
            "available_slots": max(max_workers - active_workers, 0),
            "worker_task_running": worker_task_running,
            "queued_job_ids": [],
            "running_job_ids": list(range(active_workers)),
        },
    }


def _training_queue_status(monkeypatch, snapshot):
    import app.health as health_module

    monkeypatch.setattr(health_module, "get_job_queue", lambda: _fake_queue(snapshot))
    return _call(health_module._check_training_queue())


def test_health_paused_blocked_only_is_ok(monkeypatch):
    """The job-85 shape: one very old pending job, but blocked by a paused
    breaker => health OK (pre-fix: degraded/503 forever)."""
    result = _training_queue_status(
        monkeypatch,
        _snapshot(
            queue_depth=1,
            blocked_pending_jobs=1,
            oldest_pending_age_s=100000.0,
            oldest_dispatchable_pending_age_s=None,
            active_workers=0,
            max_workers=1,
        ),
    )
    assert result["status"] == "ok"
    assert "wedge_reason" not in result


def test_health_normal_backlog_all_slots_busy_is_ok(monkeypatch):
    """A dispatchable job aged past the threshold but every slot is busy is a
    normal backlog, not a wedge (pre-fix: degraded)."""
    result = _training_queue_status(
        monkeypatch,
        _snapshot(
            queue_depth=3,
            oldest_dispatchable_pending_age_s=5000.0,
            active_workers=1,
            max_workers=1,
        ),
    )
    assert result["status"] == "ok"


def test_health_genuine_dispatch_wedge_is_error(monkeypatch):
    """A dispatchable job aged past the threshold WITH a free slot => the
    dispatcher is wedged => ERROR so it pages (pre-fix: degraded, indistinct)."""
    result = _training_queue_status(
        monkeypatch,
        _snapshot(
            queue_depth=1,
            oldest_dispatchable_pending_age_s=5000.0,
            active_workers=0,
            max_workers=1,
        ),
    )
    assert result["status"] == "error"
    assert "wedge_reason" in result


def test_health_dead_worker_is_error(monkeypatch):
    result = _training_queue_status(
        monkeypatch,
        _snapshot(queue_depth=0, worker_task_running=False),
    )
    assert result["status"] == "error"
    assert "wedge_reason" in result


def test_health_empty_queue_is_ok(monkeypatch):
    result = _training_queue_status(monkeypatch, _snapshot(queue_depth=0))
    assert result["status"] == "ok"
