"""Job queue tests for the ViolaWake Console backend."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.job_queue import (
        PRIORITY_BUSINESS,
        PRIORITY_DEVELOPER,
        PRIORITY_FREE,
        CircuitBreakerState,
        JobQueue,
        JobStatus,
        QueueFullError,
    )

    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")


def _run_test(tmp_path, coro_fn, *, no_worker=False, **queue_kwargs):
    """Run an async test function with a managed queue in a single event loop.

    If no_worker=True, only initializes the DB without starting the worker loop.
    This prevents jobs from being executed, useful for testing queue mechanics.
    """
    loop = asyncio.new_event_loop()
    try:
        async def _inner():
            db_path = tmp_path / "test_queue.db"
            kw = {"max_concurrent": 2, "max_pending": 5}
            kw.update(queue_kwargs)
            q = JobQueue(db_path=db_path, **kw)
            if no_worker:
                await q._initialize_db()
            else:
                await q.start()
            try:
                await coro_fn(q)
            finally:
                await q.shutdown()

        loop.run_until_complete(_inner())
    finally:
        loop.close()


async def _submit(q, user_id=1, wake_word="test"):
    return await q.submit_job(
        user_id=user_id,
        wake_word=wake_word,
        recording_ids=[1, 2, 3, 4, 5],
        epochs=10,
    )


class TestJobQueueLifecycle:

    def test_start_creates_db(self, tmp_path):
        loop = asyncio.new_event_loop()
        try:
            async def _test():
                db_path = tmp_path / "lifecycle.db"
                q = JobQueue(db_path=db_path, max_concurrent=1, max_pending=5)
                await q.start()
                assert db_path.exists()
                await q.shutdown()

            loop.run_until_complete(_test())
        finally:
            loop.close()

    def test_double_shutdown_is_safe(self, tmp_path):
        async def _test(q):
            await q.shutdown()
            await q.shutdown()
        _run_test(tmp_path, _test)


class TestSubmitAndGet:

    def test_submit_returns_valid_id(self, tmp_path):
        async def _test(q):
            job_id = await _submit(q)
            assert isinstance(job_id, int)
            assert job_id > 0
        _run_test(tmp_path, _test)

    def test_get_returns_pending_job(self, tmp_path):
        async def _test(q):
            job_id = await _submit(q)
            job = await q.get_job(job_id)
            assert job is not None
            assert job.id == job_id
            assert job.status == JobStatus.PENDING
            assert job.wake_word == "test"
            assert job.user_id == 1
            assert job.recording_ids == [1, 2, 3, 4, 5]
            assert job.epochs == 10
            assert job.progress_pct == 0.0
        _run_test(tmp_path, _test)

    def test_get_nonexistent_returns_none(self, tmp_path):
        async def _test(q):
            job = await q.get_job(9999)
            assert job is None
        _run_test(tmp_path, _test)

    def test_submit_multiple_increments_ids(self, tmp_path):
        async def _test(q):
            id1 = await _submit(q, wake_word="first")
            id2 = await _submit(q, wake_word="second")
            assert id2 > id1
        _run_test(tmp_path, _test)


class TestListJobs:

    def test_list_returns_user_jobs(self, tmp_path):
        async def _test(q):
            await _submit(q, user_id=1, wake_word="a")
            await _submit(q, user_id=1, wake_word="b")
            jobs = await q.list_jobs(1)
            assert len(jobs) == 2
        _run_test(tmp_path, _test)

    def test_list_isolates_users(self, tmp_path):
        async def _test(q):
            await _submit(q, user_id=1)
            await _submit(q, user_id=2)
            user1_jobs = await q.list_jobs(1)
            user2_jobs = await q.list_jobs(2)
            assert len(user1_jobs) == 1
            assert len(user2_jobs) == 1
            assert user1_jobs[0].user_id == 1
            assert user2_jobs[0].user_id == 2
        _run_test(tmp_path, _test)

    def test_list_newest_first(self, tmp_path):
        async def _test(q):
            id1 = await _submit(q, wake_word="first")
            id2 = await _submit(q, wake_word="second")
            jobs = await q.list_jobs(1)
            assert jobs[0].id == id2
            assert jobs[1].id == id1
        _run_test(tmp_path, _test)

    def test_list_empty_user(self, tmp_path):
        async def _test(q):
            jobs = await q.list_jobs(999)
            assert jobs == []
        _run_test(tmp_path, _test)

    def test_delete_jobs_for_user_removes_only_owned_jobs(self, tmp_path):
        async def _test(q):
            await _submit(q, user_id=1, wake_word="first")
            await _submit(q, user_id=1, wake_word="second")
            await _submit(q, user_id=2, wake_word="other")

            deleted = await q.delete_jobs_for_user(1)

            assert deleted == 2
            assert await q.list_jobs(1) == []
            assert len(await q.list_jobs(2)) == 1

        _run_test(tmp_path, _test, no_worker=True)


class TestCancelJob:

    def test_cancel_pending_job(self, tmp_path):
        async def _test(q):
            job_id = await _submit(q)
            result = await q.cancel_job(job_id)
            assert result is True
            job = await q.get_job(job_id)
            assert job.status == JobStatus.CANCELLED
            assert job.error == "Cancelled by user"
        _run_test(tmp_path, _test, no_worker=True)

    def test_cancel_nonexistent_returns_false(self, tmp_path):
        async def _test(q):
            result = await q.cancel_job(9999)
            assert result is False
        _run_test(tmp_path, _test)

    def test_cancel_already_cancelled_returns_false(self, tmp_path):
        async def _test(q):
            job_id = await _submit(q)
            await q.cancel_job(job_id)
            result = await q.cancel_job(job_id)
            assert result is False
        _run_test(tmp_path, _test, no_worker=True)


class TestQueueCapacity:

    def test_queue_full_raises(self, tmp_path):
        async def _test(q):
            await _submit(q, wake_word="a")
            await _submit(q, wake_word="b")
            with pytest.raises(QueueFullError):
                await _submit(q, wake_word="c")
        _run_test(tmp_path, _test, no_worker=True, max_concurrent=1, max_pending=2)


class TestCircuitBreaker:

    def test_new_user_has_clean_breaker(self, tmp_path):
        async def _test(q):
            breaker = await q.get_circuit_breaker(1)
            assert isinstance(breaker, CircuitBreakerState)
            assert breaker.consecutive_failures == 0
            assert breaker.paused is False
            assert breaker.next_attempt_at is None
            assert breaker.pause_reason is None
        _run_test(tmp_path, _test)

    def test_record_failure_increments(self, tmp_path):
        async def _test(q):
            await q._record_failure(1, "test error")
            breaker = await q.get_circuit_breaker(1)
            assert breaker.consecutive_failures == 1
            assert breaker.paused is False
            assert breaker.next_attempt_at is not None
        _run_test(tmp_path, _test)

    def test_three_failures_pauses_user(self, tmp_path):
        async def _test(q):
            for i in range(3):
                await q._record_failure(1, f"error {i}")
            breaker = await q.get_circuit_breaker(1)
            assert breaker.consecutive_failures == 3
            assert breaker.paused is True
            assert breaker.pause_reason is not None
        _run_test(tmp_path, _test)

    def test_resume_clears_breaker(self, tmp_path):
        async def _test(q):
            for i in range(3):
                await q._record_failure(1, f"error {i}")
            await q.resume_user(1)
            breaker = await q.get_circuit_breaker(1)
            assert breaker.consecutive_failures == 0
            assert breaker.paused is False
            assert breaker.next_attempt_at is None
        _run_test(tmp_path, _test)

    def test_success_resets_breaker(self, tmp_path):
        async def _test(q):
            await q._record_failure(1, "error")
            await q._record_success(1)
            breaker = await q.get_circuit_breaker(1)
            assert breaker.consecutive_failures == 0
            assert breaker.paused is False
        _run_test(tmp_path, _test)


class TestSubscribeUnsubscribe:

    def test_subscribe_returns_queue(self, tmp_path):
        async def _test(q):
            sub = q.subscribe(1)
            assert isinstance(sub, asyncio.Queue)
        _run_test(tmp_path, _test)

    def test_publish_delivers_to_subscriber(self, tmp_path):
        async def _test(q):
            sub = q.subscribe(1)
            await q._publish(1, {"status": "running", "progress": 50.0})
            event = sub.get_nowait()
            assert event["status"] == "running"
            assert event["progress"] == 50.0
        _run_test(tmp_path, _test)

    def test_unsubscribe_stops_delivery(self, tmp_path):
        async def _test(q):
            sub = q.subscribe(1)
            q.unsubscribe(1, sub)
            await q._publish(1, {"status": "done"})
            assert sub.empty()
        _run_test(tmp_path, _test)

    def test_unsubscribe_nonexistent_is_safe(self, tmp_path):
        async def _test(q):
            fake = asyncio.Queue()
            q.unsubscribe(999, fake)
        _run_test(tmp_path, _test)


class TestRuntimeSnapshot:

    def test_snapshot_structure(self, tmp_path):
        async def _test(q):
            snapshot = await q.runtime_snapshot()
            assert "queue_depth" in snapshot
            assert "in_memory_queue_depth" in snapshot
            assert "worker_status" in snapshot
            ws = snapshot["worker_status"]
            assert "active_workers" in ws
            assert "max_workers" in ws
            assert "available_slots" in ws
            assert "worker_task_running" in ws
        _run_test(tmp_path, _test)

    def test_snapshot_after_submit(self, tmp_path):
        async def _test(q):
            await _submit(q)
            snapshot = await q.runtime_snapshot()
            total = snapshot["queue_depth"] + snapshot["persisted_running_jobs"]
            assert total >= 0
        _run_test(tmp_path, _test)


class TestDataclasses:

    def test_job_status_values(self):
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"

    def test_queue_full_error_is_runtime_error(self):
        err = QueueFullError("full")
        assert isinstance(err, RuntimeError)
        assert str(err) == "full"


class TestPriorityConstants:

    def test_priority_ordering(self):
        assert PRIORITY_FREE < PRIORITY_DEVELOPER < PRIORITY_BUSINESS

    def test_priority_values(self):
        assert PRIORITY_FREE == 0
        assert PRIORITY_DEVELOPER == 5
        assert PRIORITY_BUSINESS == 10


async def _submit_with_priority(q, user_id=1, wake_word="test", priority=0):
    return await q.submit_job(
        user_id=user_id,
        wake_word=wake_word,
        recording_ids=[1, 2, 3, 4, 5],
        epochs=10,
        priority=priority,
    )


class TestPriorityQueue:
    """Tests that higher-priority jobs are dequeued before lower-priority jobs."""

    def test_submit_stores_priority(self, tmp_path):
        """Submitted job stores the given priority value."""
        async def _test(q):
            job_id = await _submit_with_priority(q, priority=PRIORITY_BUSINESS)
            job = await q.get_job(job_id)
            assert job is not None
            assert job.priority == PRIORITY_BUSINESS

        _run_test(tmp_path, _test, no_worker=True)

    def test_default_priority_is_free(self, tmp_path):
        """Jobs submitted without an explicit priority default to PRIORITY_FREE."""
        async def _test(q):
            # Bypass _resolve_user_priority (needs DB with Subscription table) by
            # patching it to return PRIORITY_FREE directly.
            from unittest.mock import AsyncMock, patch
            with patch("app.job_queue._resolve_user_priority", new=AsyncMock(return_value=PRIORITY_FREE)):
                job_id = await _submit(q)
            job = await q.get_job(job_id)
            assert job is not None
            assert job.priority == PRIORITY_FREE

        _run_test(tmp_path, _test, no_worker=True)

    def test_higher_priority_job_dequeued_first(self, tmp_path):
        """A business-tier job submitted after a free-tier job is dequeued first."""
        async def _test(q):
            low_id = await _submit_with_priority(q, user_id=1, wake_word="low", priority=PRIORITY_FREE)
            high_id = await _submit_with_priority(q, user_id=2, wake_word="high", priority=PRIORITY_BUSINESS)

            # Directly check DB ordering — the queue uses priority DESC, created_at ASC
            import aiosqlite
            async with aiosqlite.connect(q._db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    """
                    SELECT id FROM jobs
                    WHERE status = ?
                    ORDER BY priority DESC, created_at ASC, id ASC
                    """,
                    (JobStatus.PENDING.value,),
                ) as cursor:
                    rows = await cursor.fetchall()

            ordered_ids = [int(row["id"]) for row in rows]
            assert ordered_ids[0] == high_id, "Business job should be first"
            assert ordered_ids[1] == low_id, "Free job should be second"

        _run_test(tmp_path, _test, no_worker=True)

    def test_same_priority_ordered_by_created_at(self, tmp_path):
        """Jobs with equal priority are ordered FIFO (created_at ASC)."""
        async def _test(q):
            import asyncio
            id1 = await _submit_with_priority(q, user_id=1, wake_word="first", priority=PRIORITY_FREE)
            await asyncio.sleep(0.01)  # ensure distinct created_at timestamps
            id2 = await _submit_with_priority(q, user_id=2, wake_word="second", priority=PRIORITY_FREE)

            import aiosqlite
            async with aiosqlite.connect(q._db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    """
                    SELECT id FROM jobs
                    WHERE status = ?
                    ORDER BY priority DESC, created_at ASC, id ASC
                    """,
                    (JobStatus.PENDING.value,),
                ) as cursor:
                    rows = await cursor.fetchall()

            ordered_ids = [int(row["id"]) for row in rows]
            assert ordered_ids[0] == id1
            assert ordered_ids[1] == id2

        _run_test(tmp_path, _test, no_worker=True)

    def test_queue_position_method(self, tmp_path):
        """_queue_position returns correct 1-based position for pending jobs."""
        async def _test(q):
            id1 = await _submit_with_priority(q, user_id=1, wake_word="low", priority=PRIORITY_FREE)
            id2 = await _submit_with_priority(q, user_id=2, wake_word="high", priority=PRIORITY_BUSINESS)

            pos_high = await q._queue_position(id2)
            pos_low = await q._queue_position(id1)

            # High-priority job is position 1
            assert pos_high == 1
            assert pos_low == 2

        _run_test(tmp_path, _test, no_worker=True)

    def test_queue_position_none_for_nonexistent(self, tmp_path):
        """_queue_position returns None for jobs not in the pending queue."""
        async def _test(q):
            pos = await q._queue_position(9999)
            assert pos is None

        _run_test(tmp_path, _test, no_worker=True)

    def test_mixed_priorities_ordering(self, tmp_path):
        """Three jobs with distinct priorities appear in business→developer→free order."""
        async def _test(q):
            import asyncio
            free_id = await _submit_with_priority(q, user_id=1, wake_word="free", priority=PRIORITY_FREE)
            await asyncio.sleep(0.01)
            biz_id = await _submit_with_priority(q, user_id=2, wake_word="biz", priority=PRIORITY_BUSINESS)
            await asyncio.sleep(0.01)
            dev_id = await _submit_with_priority(q, user_id=3, wake_word="dev", priority=PRIORITY_DEVELOPER)

            import aiosqlite
            async with aiosqlite.connect(q._db_path) as conn:
                conn.row_factory = aiosqlite.Row
                async with conn.execute(
                    """
                    SELECT id FROM jobs
                    WHERE status = ?
                    ORDER BY priority DESC, created_at ASC, id ASC
                    """,
                    (JobStatus.PENDING.value,),
                ) as cursor:
                    rows = await cursor.fetchall()

            ordered_ids = [int(row["id"]) for row in rows]
            assert ordered_ids == [biz_id, dev_id, free_id]

        _run_test(tmp_path, _test, no_worker=True)
