"""Oracle (#4207): an OUR-side infrastructure fault must not spend a training attempt.

Charge-at-submit is a deliberate premise -- a submission consumes training compute,
so ``routes/jobs.py::submit_training_job`` calls ``record_usage`` immediately after
enqueue. The premise breaks in exactly one place: a job that dies on OUR
infrastructure (a shared-corpus / TTS outage, ``SharedInfrastructureUnavailableError``)
before it has consumed the compute the charge exists to meter, or a job that can never
dispatch at all because the account is already paused. Before this fix that job still
burned one of a free-tier customer's three monthly attempts, with no refund and no
notification -- four real accounts (users 122/130/138/150) were hard-paused and silent.

Every test here is RED on the pre-fix shape (charge-at-submit with no refund seam, no
paused-submit guard, no failure email, no in-product pause channel) and GREEN on the
fix. Together they pin the seven properties the ticket's acceptance oracle names:

1. a job the queue never ran leaves the usage count unchanged -- both the paused-queue
   strand and a ``SharedInfrastructureUnavailableError`` failure;
2. a paused user's submit is refused BEFORE ``record_usage`` fires;
3. a job is credited at most once, however many paths ask;
4. the credit lands on the period the charge was made in, so a 07-31 submit failing
   08-01 cannot mint a bonus August attempt;
5. a genuine per-user failure is STILL charged and still trips the breaker (narrow
   guard, not an amnesty);
6. a failure sends the customer an email (the console had ``send_training_complete``
   and no counterpart);
7. the console can see and clear a pause (the seam the frontend banner calls).
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.database import async_session_factory, init_db
    from app.job_queue import (
        FAILURE_THRESHOLD,
        JobQueue,
        JobStatus,
        _serialize_datetime,
        _utcnow,
    )
    from app.models import UsageRecord, User
    from app.routes import billing as billing_mod
    from app.routes import jobs as jobs_mod
    from app.routes.billing import record_usage
    from app.services.training_service import SharedInfrastructureUnavailableError
    from app.tenancy import QueuePartition

    HAS_BACKEND = True
except ImportError:  # pragma: no cover - environment guard
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")

CORPUS_MESSAGE = (
    "Training is temporarily unavailable on our side: the speech-negatives corpus is "
    "not mounted. Nothing is wrong with your recordings, and this does not count "
    "against your account."
)

JULY = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
AUGUST = datetime(2026, 8, 1, 0, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Async plumbing (one fresh loop per test, mirroring the sibling breaker tests)
# --------------------------------------------------------------------------- #

def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _make_user() -> int:
    await init_db()
    email = f"infra_refund_{time.time_ns()}@example.com"
    async with async_session_factory() as session:
        user = User(email=email, password_hash="x", name="Refund Test", email_verified=True)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return int(user.id)


async def _set_usage(user_id: int, period_start: datetime, count: int) -> None:
    """Seed the usage counter for a specific period directly."""
    async with async_session_factory() as session:
        session.add(
            UsageRecord(
                user_id=user_id,
                action="training_job",
                period_start=period_start,
                count=count,
            )
        )
        await session.commit()


async def _usage_count(user_id: int, period_start: datetime) -> int:
    async with async_session_factory() as session:
        result = await session.execute(
            select(UsageRecord).where(
                UsageRecord.user_id == user_id,
                UsageRecord.action == "training_job",
                UsageRecord.period_start == period_start,
            )
        )
        record = result.scalar_one_or_none()
        return record.count if record else 0


async def _usage_rows(user_id: int) -> list[tuple[datetime, int]]:
    async with async_session_factory() as session:
        result = await session.execute(
            select(UsageRecord.period_start, UsageRecord.count).where(
                UsageRecord.user_id == user_id,
                UsageRecord.action == "training_job",
            )
        )
        return [(row[0], row[1]) for row in result.all()]


# --------------------------------------------------------------------------- #
# Queue harness: drive a REAL _execute_job to the failure handler under test
# --------------------------------------------------------------------------- #

async def _new_queue(tmp_path) -> JobQueue:
    q = JobQueue(db_path=tmp_path / "queue.db", max_concurrent=1, max_pending=10)
    await q._initialize_db()
    return q


async def _insert_pending(q, *, job_id, user_id, wake_word="abigail"):
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
                _serialize_datetime(_utcnow()),
                "[1, 2, 3, 4, 5]",
                10,
                0,
            ),
        )
        await conn.commit()


def _arm_training_to_raise(monkeypatch, q, tmp_path, exc: BaseException):
    """Only upstream inputs are stubbed; the failure handler under test runs for real."""
    from app import job_queue as jq

    recordings = [tmp_path / f"rec{i}.wav" for i in range(5)]
    for rec in recordings:
        rec.write_bytes(b"")

    async def _fake_load(user_id, recording_ids):
        return recordings

    async def _fake_neg(user_id):
        return None

    def _boom(**kwargs):
        raise exc

    monkeypatch.setattr(q, "_load_recording_paths", _fake_load)
    monkeypatch.setattr(q, "_resolve_negatives_dir", _fake_neg)
    monkeypatch.setattr(jq, "run_training_job_sync", _boom)
    tmp_dir = tmp_path / "scratch"
    tmp_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(jq.settings, "tmp_dir", tmp_dir, raising=False)


# =========================================================================== #
# (1) + (4) An infra-fault failure leaves the usage count exactly where it was,
#            and the credit lands on the CHARGED period, not the failure period.
# =========================================================================== #

def test_infra_fault_refunds_the_charged_attempt(tmp_path, monkeypatch) -> None:
    async def _test() -> None:
        user_id = await _make_user()
        # The real charge: a submission this July increments July's counter.
        monkeypatch.setattr(billing_mod, "_current_period_start", lambda: JULY)
        async with async_session_factory() as session:
            await record_usage(session, user_id, action="training_job")
            await session.commit()
        assert await _usage_count(user_id, JULY) == 1

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=1, user_id=user_id)
        await q.mark_usage_charged(1, user_id=user_id, period_start=JULY)

        # The job now fails IN AUGUST on a shared-infrastructure fault. If the
        # refund credited "the current period" it would touch August and leave
        # July charged -- the exact month-boundary bonus-attempt bug.
        monkeypatch.setattr(billing_mod, "_current_period_start", lambda: AUGUST)
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )
        await q._execute_job(1)

        assert await _usage_count(user_id, JULY) == 0, (
            "the infra-fault attempt must be credited back to the period it was charged in"
        )
        rows = await _usage_rows(user_id)
        for period_start, count in rows:
            if period_start.month == 8:
                assert count <= 0, "a month-boundary failure must not mint a bonus August attempt"
        # And the job still records the real failure (refund is not hiding it).
        job = await q.get_job(1)
        assert job is not None and job.status == JobStatus.FAILED

    _call(_test())


# =========================================================================== #
# (3) Credited at most once, however many paths ask -- and never eats a
#     genuine, separate attempt sharing the same period.
# =========================================================================== #

def test_refund_is_idempotent_and_spares_the_legit_attempt(tmp_path, monkeypatch) -> None:
    async def _test() -> None:
        user_id = await _make_user()
        # Two attempts charged in July: one legit + one that will hit an outage.
        await _set_usage(user_id, JULY, 2)

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=7, user_id=user_id)
        await q.mark_usage_charged(7, user_id=user_id, period_start=JULY)
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )
        await q._execute_job(7)
        assert await _usage_count(user_id, JULY) == 1, "one infra fault credits exactly one attempt"

        # Every extra path that asks must be a no-op: without the one-shot guard
        # these would walk the legit attempt down to zero.
        for _ in range(4):
            await q._refund_usage_for_job(7)
        assert await _usage_count(user_id, JULY) == 1, (
            "a job is credited at most once; the customer's genuine attempt survives"
        )

    _call(_test())


# =========================================================================== #
# (5) Control: a genuine per-user failure is STILL charged and still trips the
#     breaker. The guard is narrow, not a blanket amnesty.
# =========================================================================== #

def test_a_genuine_failure_keeps_its_charge_and_trips_the_breaker(tmp_path, monkeypatch) -> None:
    async def _test() -> None:
        user_id = await _make_user()
        await _set_usage(user_id, JULY, 1)

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=9, user_id=user_id)
        await q.mark_usage_charged(9, user_id=user_id, period_start=JULY)
        _arm_training_to_raise(monkeypatch, q, tmp_path, RuntimeError("worker segfaulted"))
        await q._execute_job(9)

        assert await _usage_count(user_id, JULY) == 1, (
            "a genuine per-user failure keeps its charge -- refund is not an amnesty"
        )
        breaker = await q.get_circuit_breaker(QueuePartition(user_id=user_id))
        assert breaker.consecutive_failures == 1, "a real fault still trips the breaker"

    _call(_test())


# =========================================================================== #
# (2) + (1) A paused user's submit is refused BEFORE record_usage fires, so a
#           job that can never dispatch spends no attempt.
# =========================================================================== #

def test_paused_submit_is_refused_before_any_charge(tmp_path, monkeypatch) -> None:
    from app.schemas import JobSubmitRequest
    from fastapi import HTTPException

    async def _test() -> None:
        user_id = await _make_user()

        q = await _new_queue(tmp_path)
        partition = QueuePartition(user_id=user_id)
        # Drive the breaker to a real lockout (three genuine failures).
        for _ in range(FAILURE_THRESHOLD):
            await q._record_failure(partition, "worker segfaulted")
        assert (await q.get_circuit_breaker(partition)).paused is True

        # Stub only the upstream validation; the paused guard under test is real.
        async def _fake_validate(body, current_user, db):
            return ("abigail", [1, 2, 3, 4, 5], 10)

        charge_calls: list[int] = []

        async def _spy_record_usage(db, uid, action="training_job"):
            charge_calls.append(uid)

        monkeypatch.setattr(jobs_mod, "validate_training_request", _fake_validate)
        monkeypatch.setattr(jobs_mod, "record_usage", _spy_record_usage)

        async def _fake_init_queue():
            return q

        monkeypatch.setattr(jobs_mod, "init_job_queue", _fake_init_queue)

        user = User(id=user_id, email="x@example.com", password_hash="x", name="x")
        body = JobSubmitRequest(wake_word="abigail", recording_ids=[1, 2, 3, 4, 5], epochs=10)

        async with async_session_factory() as session:
            with pytest.raises(HTTPException) as excinfo:
                await jobs_mod.submit_training_job(body, user, session, partition)

        assert excinfo.value.status_code == 409
        assert charge_calls == [], "a paused user's submit must not reach record_usage"
        assert await _usage_count(user_id, _utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)) == 0

    _call(_test())


# =========================================================================== #
# (6) A failure sends the customer an email -- the missing counterpart to
#     send_training_complete.
# =========================================================================== #

def test_a_failure_emails_the_customer(tmp_path, monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from app import job_queue as jq

    async def _test() -> None:
        user_id = await _make_user()
        async with async_session_factory() as session:
            user = await session.get(User, user_id)
            recipient = user.email

        sent = AsyncMock(return_value=True)

        class _FakeEmail:
            enabled = True

            async def send_training_failed(self, **kwargs):
                return await sent(**kwargs)

        monkeypatch.setattr(jq, "get_email_service", lambda: _FakeEmail(), raising=False)
        # get_email_service is imported lazily inside _send_failure_email via
        # `from app.email_service import get_email_service`, so patch the source.
        import app.email_service as email_mod

        monkeypatch.setattr(email_mod, "get_email_service", lambda: _FakeEmail())

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=3, user_id=user_id, wake_word="hey_viola")
        await q.mark_usage_charged(3, user_id=user_id, period_start=JULY)
        _arm_training_to_raise(monkeypatch, q, tmp_path, RuntimeError("worker crashed"))
        await q._execute_job(3)

        assert sent.await_count == 1, "a failed training run must email the customer"
        kwargs = sent.await_args.kwargs
        assert kwargs["to"] == recipient
        assert kwargs["model_name"] == "hey_viola"
        assert kwargs["charged"] is True, "a genuine failure email says the attempt was charged"

    _call(_test())


def test_infra_fault_email_says_the_attempt_was_not_charged(tmp_path, monkeypatch) -> None:
    from unittest.mock import AsyncMock

    async def _test() -> None:
        user_id = await _make_user()
        sent = AsyncMock(return_value=True)

        class _FakeEmail:
            enabled = True

            async def send_training_failed(self, **kwargs):
                return await sent(**kwargs)

        import app.email_service as email_mod

        monkeypatch.setattr(email_mod, "get_email_service", lambda: _FakeEmail())

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=4, user_id=user_id)
        await q.mark_usage_charged(4, user_id=user_id, period_start=JULY)
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )
        await q._execute_job(4)

        assert sent.await_count == 1
        assert sent.await_args.kwargs["charged"] is False, (
            "an our-side fault email must not imply the customer lost an attempt"
        )

    _call(_test())


# =========================================================================== #
# (7) The console can SEE and CLEAR a pause -- the seam the frontend banner and
#     resume button call.
# =========================================================================== #

def test_a_pause_is_visible_and_clearable(tmp_path) -> None:
    async def _test() -> None:
        q = await _new_queue(tmp_path)
        partition = QueuePartition(user_id=4242)
        for _ in range(FAILURE_THRESHOLD):
            await q._record_failure(partition, "worker segfaulted")

        breaker = await q.get_circuit_breaker(partition)
        assert breaker.paused is True, "a lockout must be visible to the console"
        assert breaker.consecutive_failures == FAILURE_THRESHOLD

        await q.resume_user(partition)
        cleared = await q.get_circuit_breaker(partition)
        assert cleared.paused is False, "resume must clear the pause the frontend button hits"
        assert cleared.consecutive_failures == 0

    _call(_test())


def test_the_breaker_and_resume_routes_the_frontend_calls_are_registered() -> None:
    """The api.ts banner calls GET /api/jobs/circuit-breaker/state and POST /api/jobs/resume."""
    from app.routes.jobs import router

    paths = {(route.path, tuple(sorted(route.methods))) for route in router.routes}
    assert ("/api/jobs/circuit-breaker/state", ("GET",)) in {
        (p, tuple(m for m in methods if m != "HEAD")) for p, methods in paths
    }
    resume = [route for route in router.routes if route.path == "/api/jobs/resume"]
    assert resume and "POST" in resume[0].methods
