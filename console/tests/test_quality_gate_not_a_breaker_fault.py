"""Ratchet (#1775 / #2066): a failed quality gate must not trip the user's circuit breaker.

Training ends a grade-F job by raising ``ModelQualityGateError``. That is an
EXPECTED outcome -- ``app.monitoring.classify_exception`` already buckets it as
EXPECTED, precisely so it does not page ops -- and the gate's own user-facing
message tells the customer:

    "Wake-word training varies run to run, so the quickest fix is to train again
     with the same recordings."

Pre-fix, ``JobQueue._execute_job``'s blanket ``except Exception`` fed that
expected verdict into ``_record_failure`` with no filtering, and
``FAILURE_THRESHOLD = 3`` consecutive failures pause the user's queue with
``paused=1, next_attempt_at=NULL``. Only ``resume_user`` clears that state, and
``POST /api/jobs/resume`` has no frontend caller. So the product told the user to
retry and the retries locked their training queue.

This is not hypothetical. On ``wakeword-backend-1``'s ``/app/data/job_queue.db``,
read 2026-07-26, four users are in exactly that state -- 122, 130, 138 and 150 --
each with ``consecutive_failures=3, paused=1, next_attempt_at=NULL`` and a
``pause_reason`` that is verbatim the grade-F quality-gate message.

These tests are RED on the pre-fix shape and GREEN on the fix, and they prove a
genuine systemic fault still trips the breaker.
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


class ModelQualityGateError(RuntimeError):
    """Stands in for ``violawake_sdk.tools.train.ModelQualityGateError``.

    Deliberately NOT the real class. The production matcher works by class name
    across the MRO so the console backend never has to import the heavy SDK, and
    using a look-alike here locks that decoupling in place: if someone
    "simplifies" the guard into an ``isinstance`` against the real SDK class,
    these tests go red.
    """


GRADE_F_MESSAGE = (
    "Your wake word didn't pass the quality check, so it wasn't saved. On no-wake "
    "audio (silence, everyday speech, or similar-sounding words) the model scored "
    "at or above the 0.80 detection threshold, which means it would trigger on the "
    "wrong sound. Wake-word training varies run to run, so the quickest fix is to "
    "train again with the same recordings."
)


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _run_with_queue(tmp_path, coro_fn):
    async def _inner():
        q = JobQueue(db_path=tmp_path / "breaker.db", max_concurrent=1, max_pending=10)
        # DB only -- no worker loop, so we drive _execute_job ourselves and the
        # assertions stay deterministic.
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
    """Let the FAILURE_BACKOFF_SECONDS window elapse.

    After a counted failure the breaker sets ``next_attempt_at = now + 300s`` and
    ``_execute_job`` returns early until it passes. In production those 300s just
    go by -- user 150's three failures were 24 and 19 minutes apart. Rather than
    sleep, move the clock back so the next attempt is dispatchable. This is a
    no-op once the fix is in, because an expected verdict writes no breaker row.
    """
    async with q._connect() as conn:
        await conn.execute(
            "UPDATE user_circuit_breakers SET next_attempt_at = ? WHERE user_id = ?",
            (_serialize_datetime(_utcnow() - timedelta(seconds=1)), user_id),
        )
        await conn.commit()


def _arm_training_to_raise(monkeypatch, q, tmp_path, exc: BaseException):
    """Make a real ``_execute_job`` run reach training and fail with ``exc``.

    Everything stubbed here is upstream of the failure we are testing: recording
    resolution, the negatives corpus, and the scratch directory. The failure
    handler under test -- the blanket ``except Exception`` -- is NOT stubbed; it
    runs for real, which is the entire point.
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


def test_a_grade_f_verdict_does_not_pause_the_users_queue(tmp_path, monkeypatch) -> None:
    """The load-bearing ratchet, driven through the real ``_execute_job``.

    Reproduces users 122/130/138/150: FAILURE_THRESHOLD consecutive grade-F
    verdicts, nothing else wrong. Pre-fix this ends with paused=1 and
    next_attempt_at=None -- the state those four accounts are stuck in.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, ModelQualityGateError(GRADE_F_MESSAGE)
        )

        for job_id in range(1, FAILURE_THRESHOLD + 1):
            await _insert_pending(q, job_id=job_id, user_id=135)
            await q._execute_job(job_id)
            await _expire_backoff(q, 135)

        # The jobs must still be reported as failed and the reason still shown:
        # this fix drops the breaker fault, not the user's feedback.
        for job_id in range(1, FAILURE_THRESHOLD + 1):
            job = await q.get_job(job_id)
            assert job is not None
            assert job.status is JobStatus.FAILED, "the job must still be marked failed"
            assert job.error == GRADE_F_MESSAGE, "the user must still be told why"

        breaker = await q.get_circuit_breaker(135)
        assert breaker.consecutive_failures == 0, (
            "a grade-F quality verdict was counted toward the circuit breaker"
        )
        assert breaker.paused is False, (
            "the user's training queue was paused because their model failed the "
            "quality gate the number of times our own error message told them to retry"
        )
        assert breaker.next_attempt_at is None

    _run_with_queue(tmp_path, _test)


def test_the_next_job_is_still_dispatchable_after_three_grade_f_verdicts(
    tmp_path, monkeypatch
) -> None:
    """The customer-visible consequence, asserted directly.

    Pre-fix, ``_execute_job`` returns early on a paused breaker without changing
    the job's status, so job 4 stays PENDING forever. This is the shape of the
    #2066 strand.
    """

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, ModelQualityGateError(GRADE_F_MESSAGE)
        )

        for job_id in range(1, FAILURE_THRESHOLD + 1):
            await _insert_pending(q, job_id=job_id, user_id=150)
            await q._execute_job(job_id)
            await _expire_backoff(q, 150)

        await _insert_pending(q, job_id=99, user_id=150)
        await q._execute_job(99)

        job = await q.get_job(99)
        assert job is not None
        assert job.status is not JobStatus.PENDING, (
            "the job after three grade-F verdicts was never dispatched -- the user "
            "sees it queued forever because their breaker is paused"
        )

    _run_with_queue(tmp_path, _test)


def test_a_real_systemic_failure_still_trips_the_breaker(tmp_path, monkeypatch) -> None:
    """The fix must not disarm the breaker: a genuine fault still pauses."""

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(monkeypatch, q, tmp_path, RuntimeError("worker exploded"))

        for job_id in range(1, FAILURE_THRESHOLD + 1):
            await _insert_pending(q, job_id=job_id, user_id=200)
            await q._execute_job(job_id)
            await _expire_backoff(q, 200)

        breaker = await q.get_circuit_breaker(200)
        assert breaker.consecutive_failures == FAILURE_THRESHOLD
        assert breaker.paused is True, "a real systemic fault must still pause the queue"

    _run_with_queue(tmp_path, _test)


def test_a_quality_gate_outage_is_not_an_expected_outcome(tmp_path, monkeypatch) -> None:
    """The mirror image: if the gate cannot build its own test material (a TTS
    outage retiring the voice it synthesizes negatives with, CL-20260717-b117),
    that is OUR infrastructure failing, not a verdict about the user's model. It
    must still trip the breaker -- burning the user's remaining quota against a
    broken dependency is exactly what the breaker is for."""

    class QualityGateUnavailableError(RuntimeError):
        pass

    async def _test(q: JobQueue) -> None:
        _arm_training_to_raise(
            monkeypatch, q, tmp_path, QualityGateUnavailableError("edge-tts voice retired")
        )

        for job_id in range(1, FAILURE_THRESHOLD + 1):
            await _insert_pending(q, job_id=job_id, user_id=201)
            await q._execute_job(job_id)
            await _expire_backoff(q, 201)

        breaker = await q.get_circuit_breaker(201)
        assert breaker.paused is True

    _run_with_queue(tmp_path, _test)


def test_the_matcher_is_by_class_name_across_the_mro() -> None:
    """Locks the decoupling: the backend must not need the heavy SDK import, and
    a subclass of the gate error must still be recognised."""
    from app import job_queue as jq

    class Subclassed(ModelQualityGateError):
        pass

    assert jq._is_expected_training_outcome(ModelQualityGateError("x"))
    assert jq._is_expected_training_outcome(Subclassed("x"))
    assert not jq._is_expected_training_outcome(RuntimeError("x"))
    assert not jq._is_expected_training_outcome(ValueError("x"))
