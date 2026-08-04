"""Oracle (GeeIHadAGoodTime/Viola#4617): a queued job's recordings must outlive its siblings.

Training job 148 failed in production on 2026-08-03 with
``RuntimeError: No valid recordings found for training job 148``. The ground truth
behind that line, read off the live queue DB and the live Postgres:

* jobs 147 and 148 belong to user 179, wake word "olive", and carry the SAME ten
  recording ids ``[2074..2083]``, submitted 765ms apart -- a duplicate submit;
* both were accepted, because ``validate_training_request`` proved all ten existed,
  were owned, undeleted and matched the wake word, and both were charged at submit;
* job 147 completed at ``11:54:16.711511``;
* all ten recordings were soft-deleted at ``11:54:16.725336`` -- 14ms later, by
  ``retention.mark_recordings_for_deletion`` via ``_schedule_recording_cleanup``;
* job 148 started at ``11:54:16.977731`` and failed 9ms after that, with
  ``usage_refunded = 0``.

``mark_recordings_for_deletion`` was the ONLY deletion path in the module without the
``_get_active_recording_ids()`` guard. Both hard-delete sweeps
(``cleanup_soft_deleted_recordings``, ``cleanup_expired_recordings``) already had it,
and they are the two that can least afford to need it: they run hours later on an age
cutoff. The one that fires milliseconds after a job completes -- the only one that can
realistically overtake a sibling still in the queue -- went without.

``_load_recording_paths`` filters on ``deleted_at IS NULL``, so a soft delete is
indistinguishable from a hard delete to a queued job, and the resulting failure was
classified as a genuine per-user fault: the attempt stayed spent and the strike counted
toward the ``FAILURE_THRESHOLD`` lockout that only ``resume_user`` clears.

Two things are proven here, and the second is why the first is not the whole fix:

1. our own retention never removes a recording another job still needs (the cause);
2. if a job's inputs vanish anyway -- the owner deleting a recording from the console
   while the job waits, a storage purge, a future sweep written without the guard --
   the failure is honest about what happened and does not spend a strike (the class).

Anti-regression is a first-class case here: the naive version of fix (1) is to stop
soft-deleting on completion, which would silently disable the privacy retention the
FAQ promises. ``test_retention_still_fires_when_no_other_job_holds_the_recordings``
and ``test_deferred_recordings_are_deleted_when_the_last_job_finishes`` exist to fail
on that shape.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from app import quota as billing_mod
    from app.database import async_session_factory, init_db
    from app.job_queue import JobQueue, JobStatus, _serialize_datetime, _utcnow
    from app.models import Recording, UsageRecord, User
    from app.quota import record_usage
    from app.services.training_service import (
        SharedInfrastructureUnavailableError,
        TrainingArtifact,
    )
    from app.tenancy import QueuePartition

    HAS_BACKEND = True
except ImportError:  # pragma: no cover - environment guard
    HAS_BACKEND = False

# The symbols the fix introduces are imported SEPARATELY and deliberately without a
# skip. Folding them into the probe above would mean a tree that lost the fix reports
# "backend not installed" and skips the whole oracle green -- a regression test that
# disarms itself the moment it matters is worse than none. Here their absence is a
# NameError inside each test that needs them, which is a failure.
FIX_IMPORT_ERROR: str | None = None
if HAS_BACKEND:
    try:
        from app.job_queue import MIN_RECORDINGS_PER_JOB
        from app.services.training_service import RecordingsUnavailableError
    except ImportError as exc:  # pragma: no cover - asserted below
        FIX_IMPORT_ERROR = str(exc)

pytestmark = pytest.mark.skipif(not HAS_BACKEND, reason="backend not installed")

JULY = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_the_fix_is_present_at_all() -> None:
    """Fails loudly on a tree where the fix was reverted, instead of skipping."""
    assert FIX_IMPORT_ERROR is None, (
        "the recordings-outlive-their-siblings fix is missing from this tree: "
        f"{FIX_IMPORT_ERROR}"
    )


def _call(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# Real rows: a user, ten real recordings, and a queue holding two real jobs
# --------------------------------------------------------------------------- #

async def _make_user() -> int:
    await init_db()
    email = f"strand_4617_{time.time_ns()}@example.com"
    async with async_session_factory() as session:
        user = User(email=email, password_hash="x", name="Strand Test", email_verified=True)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return int(user.id)


async def _make_recordings(user_id: int, count: int, wake_word: str = "olive") -> list[int]:
    """Insert `count` real, undeleted recordings and return their ids."""
    ids: list[int] = []
    async with async_session_factory() as session:
        for index in range(count):
            recording = Recording(
                user_id=user_id,
                wake_word=wake_word,
                filename=f"rec_{index}.wav",
                file_path=f"recordings/{user_id}/{wake_word}/rec_{index}.wav",
                duration_s=1.5,
                sample_rate=16000,
                size_bytes=48000,
            )
            session.add(recording)
        await session.commit()

        result = await session.execute(
            select(Recording.id).where(
                Recording.user_id == user_id,
                Recording.deleted_at.is_(None),
            )
        )
        ids = [int(row[0]) for row in result.all()]
    return ids


async def _live_recording_ids(user_id: int) -> set[int]:
    """Ids the product still considers usable -- exactly what dispatch can see."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(Recording.id).where(
                Recording.user_id == user_id,
                Recording.deleted_at.is_(None),
            )
        )
        return {int(row[0]) for row in result.all()}


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


async def _new_queue(tmp_path) -> JobQueue:
    q = JobQueue(db_path=tmp_path / "queue.db", max_concurrent=1, max_pending=10)
    await q._initialize_db()
    return q


async def _insert_pending(q, *, job_id, user_id, recording_ids, wake_word="olive") -> None:
    import json as _json

    async with q._connect() as conn:
        await conn.execute(
            "INSERT INTO jobs "
            "(id, user_id, wake_word, status, created_at, recording_ids, epochs, priority, tenant_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                user_id,
                wake_word,
                JobStatus.PENDING.value,
                _serialize_datetime(_utcnow()),
                _json.dumps(recording_ids),
                10,
                0,
                "",
            ),
        )
        await conn.commit()


def _arm_training_to_succeed(monkeypatch, q, tmp_path):
    """Stub ONLY the trainer and the object store.

    `_load_recording_paths`, `_schedule_recording_cleanup` and
    `retention.mark_recordings_for_deletion` all run for real -- they are the code
    under test, and the whole defect lives in the ordering between them.
    """
    from app import job_queue as jq

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"onnx")

    def _train(**kwargs):
        return TrainingArtifact(
            local_path=model_file,
            config_json='{"ok": true}',
            config_bytes=b'{"ok": true}',
            d_prime=4.2,
            size_bytes=4,
        )

    async def _fake_neg(user_id):
        return None

    monkeypatch.setattr(jq, "run_training_job_sync", _train)
    monkeypatch.setattr(jq, "get_storage", lambda: MagicMock())
    monkeypatch.setattr(q, "_resolve_negatives_dir", _fake_neg)
    tmp_dir = tmp_path / "scratch"
    tmp_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(jq.settings, "tmp_dir", tmp_dir, raising=False)
    monkeypatch.setattr(jq.settings, "post_training_retention_hours", 24, raising=False)


# =========================================================================== #
# (1) The cause, reproduced end to end: jobs 147 and 148.
# =========================================================================== #

def test_a_sibling_job_still_has_its_recordings_after_its_twin_completes(
    tmp_path, monkeypatch
) -> None:
    """The live 147/148 shape: two jobs, one recording set, both must be able to run."""

    async def _test() -> None:
        user_id = await _make_user()
        recording_ids = await _make_recordings(user_id, 10)
        assert len(recording_ids) == 10

        q = await _new_queue(tmp_path)
        # Both jobs accepted and charged at submit, exactly as production did.
        await _insert_pending(q, job_id=147, user_id=user_id, recording_ids=recording_ids)
        await _insert_pending(q, job_id=148, user_id=user_id, recording_ids=recording_ids)
        _arm_training_to_succeed(monkeypatch, q, tmp_path)

        await q._execute_job(147)

        first = await q.get_job(147)
        assert first is not None and first.status == JobStatus.COMPLETED

        # THE ASSERTION. Pre-fix, all ten were soft-deleted 14ms after 147
        # completed and this set was empty.
        survivors = await _live_recording_ids(user_id)
        assert survivors == set(recording_ids), (
            "job 148 is still PENDING on these exact recordings; completing its twin "
            f"must not delete them (survivors={sorted(survivors)})"
        )

        # And the sibling actually runs, rather than dying on missing inputs.
        await q._execute_job(148)
        second = await q.get_job(148)
        assert second is not None and second.status == JobStatus.COMPLETED, (
            f"the sibling job must train, not fail with {second.error if second else None!r}"
        )

    _call(_test())


def test_deferred_recordings_are_deleted_when_the_last_job_finishes(
    tmp_path, monkeypatch
) -> None:
    """Retention is DEFERRED by the guard, never skipped.

    The failure this guards against is a fix that protects the sibling by simply not
    deleting, which would quietly break the privacy promise that recordings go away
    after training.
    """

    async def _test() -> None:
        user_id = await _make_user()
        recording_ids = await _make_recordings(user_id, 10)

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=1, user_id=user_id, recording_ids=recording_ids)
        await _insert_pending(q, job_id=2, user_id=user_id, recording_ids=recording_ids)
        _arm_training_to_succeed(monkeypatch, q, tmp_path)

        await q._execute_job(1)
        assert await _live_recording_ids(user_id) == set(recording_ids)

        # Job 2 was the last holder. Once it finishes, nothing is holding the
        # reference and the recordings must go.
        await q._execute_job(2)
        assert await _live_recording_ids(user_id) == set(), (
            "once no job references them, post-training deletion must still happen"
        )

    _call(_test())


def test_retention_still_fires_when_no_other_job_holds_the_recordings(
    tmp_path, monkeypatch
) -> None:
    """The ordinary single-job case is unchanged: complete, then delete."""

    async def _test() -> None:
        user_id = await _make_user()
        recording_ids = await _make_recordings(user_id, 10)

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=1, user_id=user_id, recording_ids=recording_ids)
        _arm_training_to_succeed(monkeypatch, q, tmp_path)

        await q._execute_job(1)

        assert await _live_recording_ids(user_id) == set(), (
            "a completed job with no sibling must still have its recordings deleted"
        )

    _call(_test())


def test_a_different_users_pending_job_does_not_defer_our_deletion(
    tmp_path, monkeypatch
) -> None:
    """The guard keys on recording id, so an unrelated queued job must not block us."""

    async def _test() -> None:
        owner_id = await _make_user()
        owner_recordings = await _make_recordings(owner_id, 10)
        other_id = await _make_user()
        other_recordings = await _make_recordings(other_id, 10)

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=1, user_id=owner_id, recording_ids=owner_recordings)
        # Unrelated account, still queued, different recordings entirely.
        await _insert_pending(q, job_id=2, user_id=other_id, recording_ids=other_recordings)
        _arm_training_to_succeed(monkeypatch, q, tmp_path)

        await q._execute_job(1)

        assert await _live_recording_ids(owner_id) == set(), (
            "another account's queued job must not defer this account's retention"
        )
        assert await _live_recording_ids(other_id) == set(other_recordings), (
            "and it must keep its own recordings"
        )

    _call(_test())


# =========================================================================== #
# (2) The class: inputs that vanish anyway are honest and unpunished.
# =========================================================================== #

def _arm_training_with_missing_recordings(monkeypatch, q, tmp_path, survivors: int):
    """Leave `_load_recording_paths` real but make the DB hold only `survivors` rows."""
    from app import job_queue as jq

    async def _fake_neg(user_id):
        return None

    def _should_not_run(**kwargs):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("training must not start without its recordings")

    monkeypatch.setattr(jq, "run_training_job_sync", _should_not_run)
    monkeypatch.setattr(jq, "get_storage", lambda: MagicMock())
    monkeypatch.setattr(q, "_resolve_negatives_dir", _fake_neg)
    tmp_dir = tmp_path / "scratch"
    tmp_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(jq.settings, "tmp_dir", tmp_dir, raising=False)


async def _soft_delete(recording_ids: list[int]) -> None:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Recording).where(Recording.id.in_(recording_ids))
        )
        for recording in result.scalars().all():
            recording.deleted_at = _utcnow()
        await session.commit()


def test_a_job_whose_recordings_vanished_is_not_charged_and_not_struck(
    tmp_path, monkeypatch
) -> None:
    """Submit proved the inputs existed, so losing them later is never the user's strike."""

    async def _test() -> None:
        user_id = await _make_user()
        recording_ids = await _make_recordings(user_id, 10)

        monkeypatch.setattr(billing_mod, "_current_period_start", lambda: JULY)
        async with async_session_factory() as session:
            await record_usage(session, user_id, action="training_job")
            await session.commit()
        assert await _usage_count(user_id, JULY) == 1

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=148, user_id=user_id, recording_ids=recording_ids)
        await q.mark_usage_charged(148, user_id=user_id, period_start=JULY)
        _arm_training_with_missing_recordings(monkeypatch, q, tmp_path, survivors=0)

        # Something outside this job removed the inputs after it was accepted.
        await _soft_delete(recording_ids)

        await q._execute_job(148)

        job = await q.get_job(148)
        assert job is not None and job.status == JobStatus.FAILED

        assert await _usage_count(user_id, JULY) == 0, (
            "a job that never trained because its inputs were gone must be credited back"
        )
        breaker = await q.get_circuit_breaker(QueuePartition(user_id=user_id))
        assert breaker.consecutive_failures == 0, (
            "losing the inputs is not a verdict on the customer's model, so it must not "
            "walk them toward the FAILURE_THRESHOLD lockout"
        )
        assert not breaker.paused

    _call(_test())


def test_the_failure_message_reports_the_real_counts(tmp_path, monkeypatch) -> None:
    """"No valid recordings found" was wrong twice: not about validity, and not zero."""

    async def _test() -> None:
        user_id = await _make_user()
        recording_ids = await _make_recordings(user_id, 10)

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=5, user_id=user_id, recording_ids=recording_ids)
        _arm_training_with_missing_recordings(monkeypatch, q, tmp_path, survivors=4)

        # Four survive: below the floor, but emphatically not "no recordings".
        await _soft_delete(recording_ids[4:])

        await q._execute_job(5)

        job = await q.get_job(5)
        assert job is not None and job.status == JobStatus.FAILED
        assert job.error is not None
        assert "4 of 10" in job.error, (
            f"the customer must be told what actually happened, got {job.error!r}"
        )
        assert "No valid recordings found" not in job.error

    _call(_test())


def test_the_vanished_inputs_error_inherits_the_no_strike_classification() -> None:
    """The classification is structural, not a second registry that can drift."""
    from app.job_queue import _is_shared_infrastructure_fault

    exc = RecordingsUnavailableError("gone")
    assert isinstance(exc, SharedInfrastructureUnavailableError)
    assert _is_shared_infrastructure_fault(exc), (
        "inheriting must be sufficient to earn the no-strike / refund treatment"
    )


def test_a_genuine_training_failure_is_still_charged_and_still_struck(
    tmp_path, monkeypatch
) -> None:
    """Control: the new class must not have widened into a blanket amnesty."""

    async def _test() -> None:
        from app import job_queue as jq

        user_id = await _make_user()
        recording_ids = await _make_recordings(user_id, 10)

        monkeypatch.setattr(billing_mod, "_current_period_start", lambda: JULY)
        async with async_session_factory() as session:
            await record_usage(session, user_id, action="training_job")
            await session.commit()

        q = await _new_queue(tmp_path)
        await _insert_pending(q, job_id=9, user_id=user_id, recording_ids=recording_ids)
        await q.mark_usage_charged(9, user_id=user_id, period_start=JULY)
        _arm_training_to_succeed(monkeypatch, q, tmp_path)

        def _boom(**kwargs):
            raise RuntimeError("worker segfaulted")

        monkeypatch.setattr(jq, "run_training_job_sync", _boom)

        await q._execute_job(9)

        assert await _usage_count(user_id, JULY) == 1, (
            "a genuine per-user failure keeps its charge"
        )
        breaker = await q.get_circuit_breaker(QueuePartition(user_id=user_id))
        assert breaker.consecutive_failures == 1, "and still trips the breaker"

    _call(_test())


# =========================================================================== #
# Unit level: the guard itself, in the shape the sibling sweeps already use.
# =========================================================================== #

class TestMarkRecordingsForDeletionGuard:
    """Mirrors `test_retention.py`'s existing `_get_active_recording_ids` patching."""

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    @staticmethod
    def _session_with(recordings):
        result = MagicMock()
        result.scalars.return_value.all.return_value = recordings
        session = AsyncMock()
        session.execute = AsyncMock(return_value=result)
        session.commit = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        return session

    @staticmethod
    def _recording(rid: int):
        r = MagicMock()
        r.id = rid
        r.deleted_at = None
        return r

    def test_recording_held_by_an_active_job_is_not_marked(self):
        from app.retention import mark_recordings_for_deletion

        held = self._recording(2074)
        free = self._recording(2084)
        session = self._session_with([held, free])

        with (
            patch("app.retention.async_session_factory", return_value=session),
            patch(
                "app.retention._get_active_recording_ids",
                new=AsyncMock(return_value={2074}),
            ),
        ):
            marked = self._run(mark_recordings_for_deletion([2074, 2084]))

        assert marked == 1
        assert held.deleted_at is None, "a recording an active job still needs must survive"
        assert free.deleted_at is not None

    def test_nothing_is_marked_when_every_recording_is_held(self):
        from app.retention import mark_recordings_for_deletion

        recordings = [self._recording(rid) for rid in range(2074, 2084)]
        session = self._session_with(recordings)

        with (
            patch("app.retention.async_session_factory", return_value=session),
            patch(
                "app.retention._get_active_recording_ids",
                new=AsyncMock(return_value=set(range(2074, 2084))),
            ),
        ):
            marked = self._run(mark_recordings_for_deletion(list(range(2074, 2084))))

        assert marked == 0
        assert all(r.deleted_at is None for r in recordings)
        session.commit.assert_not_called()

    def test_all_are_marked_when_no_job_holds_them(self):
        from app.retention import mark_recordings_for_deletion

        recordings = [self._recording(rid) for rid in range(2074, 2084)]
        session = self._session_with(recordings)

        with (
            patch("app.retention.async_session_factory", return_value=session),
            patch("app.retention._get_active_recording_ids", new=AsyncMock(return_value=set())),
        ):
            marked = self._run(mark_recordings_for_deletion(list(range(2074, 2084))))

        assert marked == 10
        assert all(r.deleted_at is not None for r in recordings)
        session.commit.assert_called_once()


def test_submit_and_dispatch_share_one_floor() -> None:
    """The two `< 5` checks must read the same constant or they can drift apart."""
    from app.routes import jobs as jobs_mod

    assert jobs_mod.MIN_RECORDINGS_PER_JOB is MIN_RECORDINGS_PER_JOB
    assert MIN_RECORDINGS_PER_JOB == 5
