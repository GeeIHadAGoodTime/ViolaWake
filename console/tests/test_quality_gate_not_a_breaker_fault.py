"""Ratchet (#1775): a failed quality gate must not trip the user's circuit breaker.

The training quality gate ends a job by raising ModelQualityGateError. That is an
EXPECTED outcome -- app.monitoring already classifies it as such precisely so it
does not page ops -- and the gate's own user-facing message tells the customer:

    "Wake-word training varies run to run, so the quickest fix is to train again
     with the same recordings."

Pre-fix, job_queue's blanket `except Exception` fed that expected outcome into
`_record_failure`, and FAILURE_THRESHOLD=3 consecutive failures pauses the user's
queue with next_attempt_at=None -- which strands every pending job until someone
manually calls resume_user (CL-20260717-9bc3). So the product instructed the user
to retry, and the retries locked their account's training queue.

These tests RED on the pre-fix shape (breaker counts a grade-F verdict) and GREEN
on the fix, while proving a genuine systemic fault STILL trips the breaker.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app.job_queue import FAILURE_THRESHOLD, JobQueue

    HAS_BACKEND = True
except ImportError:
    HAS_BACKEND = False

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")


class ModelQualityGateError(RuntimeError):
    """Stands in for violawake_sdk.tools.train.ModelQualityGateError.

    Deliberately NOT the real class: the production matcher is by class name
    across the MRO so the backend never has to import the heavy SDK, and this
    test locks that decoupling in place too.
    """


def _run(tmp_path, coro_fn):
    loop = asyncio.new_event_loop()
    try:

        async def _inner():
            q = JobQueue(db_path=tmp_path / "breaker.db", max_concurrent=1, max_pending=10)
            await q._initialize_db()
            try:
                await coro_fn(q)
            finally:
                await q.shutdown()

        loop.run_until_complete(_inner())
    finally:
        loop.close()


def _failure_recorder(exc: BaseException):
    """Replays the production decision: does this exception count as a fault?"""
    from app import job_queue as jq

    return not jq._is_expected_training_outcome(exc)


def test_quality_gate_failure_does_not_count_as_a_breaker_fault(tmp_path) -> None:
    """The load-bearing ratchet."""
    from app import job_queue as jq

    assert hasattr(jq, "_is_expected_training_outcome"), (
        "pre-fix: job_queue has no notion of an expected training outcome, so a "
        "grade-F verdict is counted as a systemic fault"
    )

    async def _test(q: JobQueue) -> None:
        for _ in range(FAILURE_THRESHOLD + 2):
            exc = ModelQualityGateError("Your wake word didn't pass the quality check")
            if _failure_recorder(exc):
                await q._record_failure(1, str(exc))

        breaker = await q.get_circuit_breaker(1)
        assert breaker.consecutive_failures == 0, (
            "a grade-F quality verdict was counted toward the circuit breaker"
        )
        assert breaker.paused is False, (
            "the user's training queue was paused because their model failed the "
            "quality gate the number of times the error message told them to retry"
        )

    _run(tmp_path, _test)


def test_a_real_systemic_failure_still_trips_the_breaker(tmp_path) -> None:
    """The fix must not disarm the breaker: a genuine fault still pauses."""

    async def _test(q: JobQueue) -> None:
        for i in range(FAILURE_THRESHOLD):
            exc = RuntimeError(f"worker exploded {i}")
            if _failure_recorder(exc):
                await q._record_failure(1, str(exc))

        breaker = await q.get_circuit_breaker(1)
        assert breaker.consecutive_failures == FAILURE_THRESHOLD
        assert breaker.paused is True, "a real systemic fault must still pause the queue"

    _run(tmp_path, _test)


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


def test_the_guard_is_actually_wired_into_the_job_failure_handler() -> None:
    """Without this, every other test here can stay green while the fix is dead.

    The tests above exercise the classification helper. That helper is only
    load-bearing if `_execute_job`'s failure handler -- the blanket
    `except Exception` that catches the gate error -- consults it before calling
    `_record_failure`. Reds if the guard is removed from the call site while the
    helper survives, which is exactly how this regression would come back.
    """
    import inspect

    from app import job_queue as jq

    source = inspect.getsource(jq.JobQueue._execute_job)
    assert "_record_failure" in source, "handler no longer records failures at all"
    assert "_is_expected_training_outcome" in source, (
        "_execute_job calls _record_failure without consulting "
        "_is_expected_training_outcome, so an expected grade-F verdict is once "
        "again counted toward the user's circuit breaker"
    )


def test_a_tts_outage_is_not_treated_as_an_expected_outcome() -> None:
    """QualityGateUnavailableError (#1775) is OUR infrastructure failing, so it
    is NOT expected: it must still count toward the breaker, which correctly
    stops burning the queue on a broken TTS dependency."""
    from app import job_queue as jq

    class QualityGateUnavailableError(RuntimeError):
        pass

    assert not jq._is_expected_training_outcome(QualityGateUnavailableError("tts down"))
