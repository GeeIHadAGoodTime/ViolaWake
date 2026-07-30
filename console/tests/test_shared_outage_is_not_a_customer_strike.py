"""Ratchet (#2611 / C-302): our missing corpus must not spend a customer's strikes.

``training_service`` raised a bare ``RuntimeError`` when the speech-negatives corpus
was not mounted. Bare, so the #28 guard correctly let it through to
``_record_failure`` -- it IS a real fault, unlike a grade-F verdict. But the circuit
breaker is PER USER, so a fault that is inherently SHARED gets charged to whoever
happened to submit during it, independently, three at a time, until each account sits
at ``consecutive_failures=3, paused=1, next_attempt_at=NULL``. Only ``resume_user``
clears that, and ``POST /api/jobs/resume`` has no frontend caller, so the customer
cannot get out from inside the product.

Measured on ``wakeword-backend-1``'s ``/app/data/job_queue.db``: 9 of the 57
real-customer jobs (users 117, 119, 120, 121, 122, 124; jobs 62-70, 2026-07-04 to
07-10) failed with ``No speech negatives available``, and it is half of how user 122
reached a lockout (jobs 68 and 69 corpus, job 114 grade-F).

The fix keeps every bit of the pacing and none of the blame: same backoff, same
self-healing retry fill, ``consecutive_failures`` untouched, ``paused`` never set.

These tests are RED on the pre-fix shape (a bare ``RuntimeError`` walking the breaker
to a lockout) and GREEN on the fix, and they prove a genuine per-user fault still
trips the breaker so the guard is not a blanket amnesty.
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
    from app.job_queue import FAILURE_THRESHOLD, JobQueue, JobStatus, _serialize_datetime, _utcnow

    HAS_BACKEND = True
except ImportError:  # pragma: no cover - environment guard
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")

CORPUS_MESSAGE = (
    "Training is temporarily unavailable on our side: the speech-negatives corpus is "
    "not mounted. Nothing is wrong with your recordings, and this does not count "
    "against your account."
)


class SharedInfrastructureUnavailableError(RuntimeError):
    """Stands in for the real ``training_service`` class.

    Deliberately a look-alike, matching how the #28 tests treat
    ``ModelQualityGateError``: the production matcher works by class name across the
    MRO so the console backend never imports the heavy SDK, and using a look-alike
    locks that decoupling in place. An ``isinstance`` "simplification" reds this.
    """


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run_with_queue(tmp_path, coro_fn):
    async def _inner():
        q = JobQueue(db_path=tmp_path / "breaker.db", max_concurrent=1, max_pending=10)
        await q._initialize_db()
        await coro_fn(q)

    _call(_inner())


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


async def _expire_backoff(q, user_id):
    """Let the backoff window elapse, as the 300s simply go by in production."""
    async with q._connect() as conn:
        await conn.execute(
            "UPDATE user_circuit_breakers SET next_attempt_at = ? WHERE user_id = ?",
            (_serialize_datetime(_utcnow() - timedelta(seconds=1)), user_id),
        )
        await conn.commit()


def _arm_training_to_raise(monkeypatch, q, tmp_path, exc: BaseException):
    """Drive a real ``_execute_job`` to reach training and fail with ``exc``.

    Only the upstream inputs are stubbed (recording resolution, negatives dir,
    scratch dir). The failure handler under test runs for real.
    """
    from app import job_queue as jq

    recordings = [tmp_path / f"rec{i}.wav" for i in range(5)]
    for rec in recordings:
        rec.write_bytes(b"")

    async def _fake_load_recording_paths(user_id, recording_ids):
        return recordings

    async def _fake_resolve_negatives_dir(user_id):
        return None

    def _boom(**kwargs):
        raise exc

    monkeypatch.setattr(q, "_load_recording_paths", _fake_load_recording_paths)
    monkeypatch.setattr(q, "_resolve_negatives_dir", _fake_resolve_negatives_dir)
    monkeypatch.setattr(jq, "run_training_job_sync", _boom)

    tmp_dir = tmp_path / "scratch"
    tmp_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(jq.settings, "tmp_dir", tmp_dir, raising=False)


def test_a_shared_outage_never_locks_a_customer_out(tmp_path, monkeypatch) -> None:
    """The headline: FAILURE_THRESHOLD corpus outages in a row leave the account open.

    Pre-fix this is the exact walk to a lockout that put 122/130/138/150 in the
    production table, so this reds on the old bare-RuntimeError shape.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )

        # One more than the threshold: even sustained outages must not accumulate.
        for job_id in range(1, FAILURE_THRESHOLD + 2):
            await _insert_pending(q, job_id=job_id, user_id=300)
            await q._execute_job(job_id)
            await _expire_backoff(q, 300)

        breaker = await q.get_circuit_breaker(300)
        assert breaker.consecutive_failures == 0, (
            "a shared outage must not spend the customer's strike budget"
        )
        assert breaker.paused is False, "our outage must never lock a customer's account"
        assert breaker.pause_reason is None

    _run_with_queue(tmp_path, _test)


def test_the_job_still_fails_and_still_says_why(tmp_path, monkeypatch) -> None:
    """Not charging a strike is not hiding the failure.

    The job is still FAILED with the real reason, so the customer is told what
    happened and operators still see the fault. Only the blame is dropped.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )
        await _insert_pending(q, job_id=1, user_id=301)
        await q._execute_job(1)

        job = await q.get_job(1)
        assert job is not None
        assert job.status == JobStatus.FAILED
        assert "corpus is not mounted" in (job.error or "")
        assert "does not count against your account" in (job.error or "")

    _run_with_queue(tmp_path, _test)


def test_back_pressure_is_still_applied(tmp_path, monkeypatch) -> None:
    """Pacing survives: the queue does not hammer a dependency that is down.

    This is what separates the fix from simply exempting the fault. ``next_attempt_at``
    still moves into the future on each outage, so the user is throttled exactly as
    before -- they just never accumulate toward a lock.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )
        await _insert_pending(q, job_id=1, user_id=302)
        await q._execute_job(1)

        breaker = await q.get_circuit_breaker(302)
        assert breaker.next_attempt_at is not None, "a shared fault must still back off"
        assert breaker.next_attempt_at > _utcnow(), "backoff must be in the future"
        assert breaker.last_failure_at is not None, "the fault must still be recorded"
        assert breaker.consecutive_failures == 0

    _run_with_queue(tmp_path, _test)


def test_a_genuine_per_user_fault_still_trips_the_breaker(tmp_path, monkeypatch) -> None:
    """Control: the guard is narrow, not a blanket amnesty on RuntimeError.

    A plain fault still walks to the lockout, which is what the breaker is for. This
    passes both pre- and post-fix by design -- it proves the test file is not simply
    failing everything.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(monkeypatch, q, tmp_path, RuntimeError("worker segfaulted"))

        for job_id in range(1, FAILURE_THRESHOLD + 1):
            await _insert_pending(q, job_id=job_id, user_id=303)
            await q._execute_job(job_id)
            await _expire_backoff(q, 303)

        breaker = await q.get_circuit_breaker(303)
        assert breaker.consecutive_failures == FAILURE_THRESHOLD
        assert breaker.paused is True

    _run_with_queue(tmp_path, _test)


def test_an_existing_lockout_is_not_silently_cleared(tmp_path, monkeypatch) -> None:
    """A shared outage must not become an accidental un-pause either.

    ``_record_transient_fault`` only moves the timing columns, so a customer already
    paused for their own three real failures stays paused. Releasing them is a
    deliberate ``resume_user`` decision, not a side effect of our corpus dropping out.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(monkeypatch, q, tmp_path, RuntimeError("worker segfaulted"))
        for job_id in range(1, FAILURE_THRESHOLD + 1):
            await _insert_pending(q, job_id=job_id, user_id=304)
            await q._execute_job(job_id)
            await _expire_backoff(q, 304)
        assert (await q.get_circuit_breaker(304)).paused is True

        # Now the corpus drops out for the same already-paused user.
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, SharedInfrastructureUnavailableError(CORPUS_MESSAGE)
        )
        await _expire_backoff(q, 304)
        await _insert_pending(q, job_id=99, user_id=304)
        await q._execute_job(99)

        breaker = await q.get_circuit_breaker(304)
        assert breaker.paused is True, "a shared fault must not un-pause a real lockout"
        assert breaker.consecutive_failures == FAILURE_THRESHOLD

    _run_with_queue(tmp_path, _test)


def test_the_matcher_is_by_class_name_across_the_mro() -> None:
    """Locks the decoupling and the narrowness of the new category."""
    from app import job_queue as jq

    class Subclassed(SharedInfrastructureUnavailableError):
        pass

    assert jq._is_shared_infrastructure_fault(SharedInfrastructureUnavailableError("x"))
    assert jq._is_shared_infrastructure_fault(Subclassed("x"))
    assert not jq._is_shared_infrastructure_fault(RuntimeError("x"))
    assert not jq._is_shared_infrastructure_fault(ValueError("x"))
    # And the two categories stay distinct -- a shared fault is NOT an expected
    # verdict, because unlike a grade-F it is a real fault worth alarming on.
    assert not jq._is_expected_training_outcome(SharedInfrastructureUnavailableError("x"))


def test_the_real_corpus_check_raises_the_shared_fault_type() -> None:
    """Wiring: the production raise site uses the new type, not a bare RuntimeError.

    Without this the guard above is dead code -- which is precisely the pre-fix state.
    """
    import inspect

    from app.services import training_service

    assert issubclass(training_service.SharedInfrastructureUnavailableError, RuntimeError)
    src = inspect.getsource(training_service)
    marker = "No speech negatives available"
    assert marker not in src, "the bare-RuntimeError corpus message is retired"
    assert "raise SharedInfrastructureUnavailableError(" in src
    assert "corpus is not mounted" in src
