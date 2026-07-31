"""One service account, many upstream users: their controls must stay apart.

A privileged service-key caller authenticates as ONE synthetic account and
submits training for its whole user base through it. Before tenant
partitioning, every protective control in the queue was keyed on that single
account id, so each of them was really a control over ALL of that caller's
users at once:

* ``FAILURE_THRESHOLD`` failures from one upstream user paused training for
  every other one, with ``next_attempt_at = NULL`` so nothing released it;
* ``PER_USER_MAX_PENDING`` counted the whole install base's jobs together, so
  one upstream user was refused because of a stranger's queue;
* the in-memory backoff timer was keyed on the account too, so one tenant's
  backoff suppressed every other tenant's refill; and
* job ownership was an account check, so any upstream user could read or
  cancel any other's training job.

Every test here fails on that shape and passes on the partitioned one. They
are deliberately behavioural (submit / fail / dispatch / read back) rather
than assertions about the schema, because the schema is not what strands a
customer -- the dispatcher's decision is.
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.job_queue import (
        FAILURE_THRESHOLD,
        PER_USER_MAX_PENDING,
        JobQueue,
        JobStatus,
        TooManyPendingJobsError,
    )
    from app.tenancy import (
        ACCOUNT_TENANT,
        SERVICE_TENANT_HEADER,
        MissingServiceTenantError,
        QueuePartition,
        normalize_service_tenant_key,
    )

    HAS_BACKEND = True
except ImportError:  # pragma: no cover - mirrors the other console test modules
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")

# One shared service account; two of ITS users. These stand in for two people
# on two different desktops who have never heard of each other.
SERVICE_ACCOUNT_ID = 47
TENANT_A = QueuePartition(user_id=SERVICE_ACCOUNT_ID, tenant_key="a1b2c3d4e5f60718")
TENANT_B = QueuePartition(user_id=SERVICE_ACCOUNT_ID, tenant_key="f0e9d8c7b6a54321")


def _run(tmp_path, coro_fn, *, db_name="tenant_queue.db", **queue_kwargs):
    """Drive an async queue test on its own event loop, DB only (no worker)."""
    loop = asyncio.new_event_loop()
    try:
        async def _inner():
            kw = {"max_concurrent": 2, "max_pending": 20}
            kw.update(queue_kwargs)
            queue = JobQueue(db_path=tmp_path / db_name, **kw)
            await queue._initialize_db()
            try:
                await coro_fn(queue)
            finally:
                await queue.shutdown()

        loop.run_until_complete(_inner())
    finally:
        loop.close()


async def _submit(queue, partition, wake_word="hey viola"):
    return await queue.submit_job(
        partition=partition,
        wake_word=wake_word,
        recording_ids=[1, 2, 3, 4, 5],
        epochs=10,
    )


class TestCircuitBreakerIsolation:

    def test_one_tenants_failures_do_not_pause_another(self, tmp_path):
        async def _test(queue):
            for index in range(FAILURE_THRESHOLD):
                await queue._record_failure(TENANT_A, f"training crashed {index}")

            paused = await queue.get_circuit_breaker(TENANT_A)
            assert paused.paused is True, "the failing tenant should be paused"

            bystander = await queue.get_circuit_breaker(TENANT_B)
            assert bystander.paused is False, (
                "a tenant that never failed is paused -- one user's failures "
                "reached the whole install base"
            )
            assert bystander.consecutive_failures == 0

        _run(tmp_path, _test)

    def test_a_paused_tenant_does_not_strand_another_tenants_job(self, tmp_path):
        """The one that actually costs a customer their wake word."""

        async def _test(queue):
            stranded_candidate = await _submit(queue, TENANT_B)
            for index in range(FAILURE_THRESHOLD):
                await queue._record_failure(TENANT_A, f"training crashed {index}")

            # Submission itself already enqueued the job in memory, so asking
            # the dispatcher now would prove nothing. Clear the in-memory state
            # the way a restart does and make it re-decide from the database --
            # that re-decision is the step that strands work for good.
            queue._queued_job_ids.clear()
            while not queue._queue.empty():
                queue._queue.get_nowait()

            await queue._fill_queue_from_db()

            assert stranded_candidate in queue._queued_job_ids, (
                "a bystander tenant's pending job was skipped by the dispatcher "
                "because a DIFFERENT tenant of the same service account is paused"
            )

        _run(tmp_path, _test)

    def test_backoff_timer_is_per_tenant(self, tmp_path):
        async def _test(queue):
            queue._schedule_retry_fill(TENANT_A, 30.0)
            assert TENANT_A in queue._retry_tasks
            assert TENANT_B not in queue._retry_tasks, (
                "one tenant's backoff timer occupies another tenant's slot, so "
                "the second tenant's refill is suppressed by the first's failure"
            )

            queue._schedule_retry_fill(TENANT_B, 30.0)
            assert len(queue._retry_tasks) == 2

            for task in list(queue._retry_tasks.values()):
                task.cancel()

        _run(tmp_path, _test)

    def test_resume_releases_only_the_asking_tenant(self, tmp_path):
        async def _test(queue):
            for tenant in (TENANT_A, TENANT_B):
                for index in range(FAILURE_THRESHOLD):
                    await queue._record_failure(tenant, f"crash {index}")

            await queue.resume_user(TENANT_A)

            assert (await queue.get_circuit_breaker(TENANT_A)).paused is False
            assert (await queue.get_circuit_breaker(TENANT_B)).paused is True, (
                "resuming one tenant cleared another tenant's pause"
            )

        _run(tmp_path, _test)


class TestPendingCapIsolation:

    def test_one_tenant_cannot_consume_another_tenants_cap(self, tmp_path):
        async def _test(queue):
            for index in range(PER_USER_MAX_PENDING):
                await _submit(queue, TENANT_A, wake_word=f"word {index}")

            with pytest.raises(TooManyPendingJobsError):
                await _submit(queue, TENANT_A, wake_word="one too many")

            # The bystander has submitted nothing at all.
            job_id = await _submit(queue, TENANT_B, wake_word="bystander")
            assert job_id > 0, (
                "a tenant with an empty queue was refused because a stranger "
                "sharing the service account had filled the cap"
            )

        _run(tmp_path, _test)

    def test_listing_is_scoped_to_the_tenant(self, tmp_path):
        async def _test(queue):
            await _submit(queue, TENANT_A, wake_word="mine")
            await _submit(queue, TENANT_B, wake_word="theirs")

            mine = await queue.list_jobs(TENANT_A)
            assert [job.wake_word for job in mine] == ["mine"], (
                "one tenant can see another tenant's training history"
            )

        _run(tmp_path, _test)


class TestRateLimitIsolation:
    """The same defect one layer up: per-USER limits keyed on the account.

    ``TRAINING_SUBMIT_LIMIT`` is 5/hour and ``RECORDING_UPLOAD_LIMIT`` is
    100/15min. Keyed on the shared service account, those are five wake-word
    trainings per hour and one upload budget for the caller's ENTIRE install
    base, so the next user is 429'd because of strangers.
    """

    def test_two_tenants_get_two_rate_limit_buckets(self):
        from app.rate_limit import key_by_user, set_rate_limit_user

        request_a = SimpleNamespace(state=SimpleNamespace(), headers={})
        request_b = SimpleNamespace(state=SimpleNamespace(), headers={})
        set_rate_limit_user(request_a, TENANT_A)
        set_rate_limit_user(request_b, TENANT_B)

        assert key_by_user(request_a) != key_by_user(request_b), (
            "two upstream users of one service account share a rate-limit "
            "bucket, so one user's five trainings spend the other's hour"
        )

    def test_an_ordinary_account_keeps_its_existing_key(self):
        """The bare-id key must not change, or every live limit resets."""
        from app.rate_limit import key_by_user, set_rate_limit_user

        by_id = SimpleNamespace(state=SimpleNamespace(), headers={})
        by_partition = SimpleNamespace(state=SimpleNamespace(), headers={})
        set_rate_limit_user(by_id, 138)
        set_rate_limit_user(by_partition, QueuePartition.for_account(138))

        assert key_by_user(by_id) == key_by_user(by_partition) == "138"


class TestJobOwnership:

    def test_a_tenant_cannot_open_another_tenants_job(self, tmp_path):
        from app.routes.jobs import get_owned_job_or_404

        async def _test(queue):
            other_job_id = await _submit(queue, TENANT_B)

            with pytest.raises(Exception) as caught:  # HTTPException
                from unittest.mock import AsyncMock, patch

                with patch(
                    "app.routes.jobs.init_job_queue",
                    new=AsyncMock(return_value=queue),
                ):
                    await get_owned_job_or_404(other_job_id, TENANT_A)

            assert getattr(caught.value, "status_code", None) == 404, (
                "job ownership is checked against the shared account, so every "
                "upstream user can read every other upstream user's job"
            )

        _run(tmp_path, _test)


class TestPartitionBoundary:

    def test_a_service_call_without_a_tenant_is_refused(self):
        from app.auth import resolve_queue_partition

        service_user = SimpleNamespace(id=SERVICE_ACCOUNT_ID, email="viola-service@viola.internal")
        request = SimpleNamespace(headers={})

        with pytest.raises(Exception) as caught:  # HTTPException
            resolve_queue_partition(request, service_user)
        assert getattr(caught.value, "status_code", None) == 400
        assert SERVICE_TENANT_HEADER in str(getattr(caught.value, "detail", ""))

    def test_a_malformed_tenant_is_refused_not_coerced(self):
        from app.auth import resolve_queue_partition

        service_user = SimpleNamespace(id=SERVICE_ACCOUNT_ID, email="viola-service@viola.internal")
        for bad in ("", "   ", "short", "has spaces in it here", "x" * 65, "UPPER!!@@##$$%%^^"):
            request = SimpleNamespace(headers={SERVICE_TENANT_HEADER: bad})
            with pytest.raises(Exception) as caught:
                resolve_queue_partition(request, service_user)
            assert getattr(caught.value, "status_code", None) == 400, bad

    def test_an_ordinary_account_is_its_own_partition(self):
        from app.auth import resolve_queue_partition

        person = SimpleNamespace(id=138, email="someone@example.com")
        request = SimpleNamespace(headers={SERVICE_TENANT_HEADER: "a1b2c3d4e5f60718"})

        partition = resolve_queue_partition(request, person)
        assert partition == QueuePartition(user_id=138, tenant_key=ACCOUNT_TENANT), (
            "an ordinary account must not be re-partitioned by a header it did "
            "not ask for -- that would split one person's own breaker in two"
        )

    def test_tenant_keys_are_case_folded_not_duplicated(self):
        assert normalize_service_tenant_key("A1B2C3D4E5F60718") == "a1b2c3d4e5f60718"
        with pytest.raises(MissingServiceTenantError):
            normalize_service_tenant_key(None)

    def test_submission_offers_no_bare_account_form(self):
        """``submit_job`` must not accept a bare id for service-path work."""
        import inspect

        signature = inspect.signature(JobQueue.submit_job)
        assert "partition" in signature.parameters
        assert "user_id" not in signature.parameters, (
            "a bare user_id parameter on submission lets a caller enqueue work "
            "whose partition was inferred, which is how the cap became shared"
        )


class TestMigrationFromPreTenantDatabase:

    def test_existing_rows_keep_their_state_and_gain_partitioning(self, tmp_path):
        """An upgrade must not move, resume or pause anybody."""
        db_path = tmp_path / "pre_tenant.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    wake_word TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    error TEXT,
                    progress_pct REAL NOT NULL DEFAULT 0,
                    recording_ids TEXT NOT NULL,
                    epochs INTEGER NOT NULL DEFAULT 80,
                    model_id INTEGER,
                    d_prime REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE user_circuit_breakers (
                    user_id INTEGER PRIMARY KEY,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    paused INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_failure_at TEXT,
                    pause_reason TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO jobs (user_id, wake_word, status, created_at, recording_ids, epochs) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (138, "legacy word", JobStatus.COMPLETED.value, "2026-07-01T00:00:00+00:00", "[1]", 80),
            )
            conn.execute(
                "INSERT INTO user_circuit_breakers "
                "(user_id, consecutive_failures, paused, pause_reason) VALUES (?, ?, ?, ?)",
                (138, 3, 1, "grade F"),
            )
            conn.commit()

        async def _test(queue):
            carried = await queue.get_circuit_breaker(138)
            assert carried.paused is True, "an upgrade silently resumed a paused account"
            assert carried.consecutive_failures == 3
            assert carried.pause_reason == "grade F"

            legacy_jobs = await queue.list_jobs(138)
            assert len(legacy_jobs) == 1, "a legacy job lost its owner in the upgrade"
            assert legacy_jobs[0].tenant_key == ACCOUNT_TENANT

            # And the upgraded database can now tell two tenants apart.
            for index in range(FAILURE_THRESHOLD):
                await queue._record_failure(TENANT_A, f"crash {index}")
            assert (await queue.get_circuit_breaker(TENANT_A)).paused is True
            assert (await queue.get_circuit_breaker(TENANT_B)).paused is False

        _run(tmp_path, _test, db_name="pre_tenant.db")

    def test_migration_is_idempotent(self, tmp_path):
        async def _test(queue):
            await queue._initialize_db()
            await queue._initialize_db()
            assert (await queue.get_circuit_breaker(TENANT_A)).paused is False

        _run(tmp_path, _test, db_name="idempotent.db")
