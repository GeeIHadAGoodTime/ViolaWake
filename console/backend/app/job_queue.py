"""Persistent async training job queue with circuit breaker protection."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.models import Recording, TrainedModel, User
from app.monitoring import log_exception
from app.services.training_service import TrainingCancelledError, run_training_job_sync
from app.storage import build_companion_config_identifier, build_model_key, get_storage

logger = logging.getLogger("violawake.jobs")

QUEUE_MAX_SIZE = 50
PER_USER_MAX_PENDING = 3
MAX_SUBSCRIBERS_PER_JOB = 5
FAILURE_THRESHOLD = 3
FAILURE_BACKOFF_SECONDS = 300
ACCOUNT_DELETE_CANCEL_TIMEOUT_SECONDS = 30.0

# Training outcomes that are EXPECTED and are the user's to retry, so they must
# not count toward the circuit breaker (#1775 / #2066). The breaker exists to
# stop the queue burning jobs on a systemically broken worker; a model that did
# not clear the deployment quality gate is not that -- `app.monitoring`'s
# `classify_exception` already buckets ModelQualityGateError as EXPECTED for
# exactly this reason, but the breaker never consulted that judgement.
#
# Counting it was actively harmful. The gate's own user-facing message says
# "wake-word training varies run to run, so the quickest fix is to train again
# with the same recordings", and FAILURE_THRESHOLD consecutive failures then
# pause the user's queue with next_attempt_at=None -- a state only `resume_user`
# clears (CL-20260717-9bc3), which has no frontend caller. The product was
# routing customers into an account-level lockout by following its own advice.
#
# Matched by class name across the MRO so the backend stays decoupled from the
# heavy violawake_sdk import, the same technique `app.monitoring` uses.
_EXPECTED_TRAINING_OUTCOMES = frozenset({"ModelQualityGateError"})


def _is_expected_training_outcome(exc: BaseException) -> bool:
    """True if `exc` is an expected training verdict rather than a systemic fault."""
    return any(base.__name__ in _EXPECTED_TRAINING_OUTCOMES for base in type(exc).__mro__)


# A SHARED-infrastructure fault is a third category, and the breaker being per-user
# is what makes it one. It is not a verdict about the customer's model (so unlike a
# grade-F it is a real fault worth backing off on), and it is not that customer's
# fault either (so unlike a broken worker it must not accumulate against their
# account). The corpus not being mounted is the archetype: it hits every customer who
# submits during the outage, and under a single per-user counter one operational gap
# spends everybody's strike budget three at a time until each account locks with
# next_attempt_at=NULL -- a state only resume_user clears, and resume_user has no
# frontend caller, so there is no way out from inside the product. 9 of 57
# real-customer jobs were this class, and it is half of how user 122 got locked
# (GeeIHadAGoodTime/Viola#2611, ledger C-302).
#
# Back-pressure is preserved and the strike is not charged: see
# JobQueue._record_transient_fault.
_SHARED_INFRASTRUCTURE_FAULTS = frozenset({"SharedInfrastructureUnavailableError"})


def _is_shared_infrastructure_fault(exc: BaseException) -> bool:
    """True if `exc` is our own missing prerequisite rather than this user's failure."""
    return any(base.__name__ in _SHARED_INFRASTRUCTURE_FAULTS for base in type(exc).__mro__)


# Not spending a strike (above) was only half the debt. The OTHER per-customer
# currency is the monthly training quota, and it is charged at submit time on the
# premise that "every submission burns real training compute" -- a premise that is
# simply false for a job we never ran. A free-tier customer gets three attempts a
# month, so one corpus outage could permanently cost a third of their month with no
# refund path in the product at all (GeeIHadAGoodTime/Viola#4207, ledger C-337 /
# C-212). Whenever a job ends without having consumed the compute the charge pays
# for, the attempt goes back and the customer is told so in the same sentence.
ATTEMPT_CREDITED_NOTE = (
    "This attempt was credited back and does not count toward your monthly "
    "training limit."
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _deserialize_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


class JobStatus(str, Enum):
    """Persisted job states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Priority values assigned by subscription tier.
PRIORITY_FREE = 0
PRIORITY_DEVELOPER = 10
PRIORITY_BUSINESS = 20
PRIORITY_ENTERPRISE = 30


@dataclass(slots=True)
class Job:
    """Persisted training job metadata."""

    id: int
    user_id: int
    wake_word: str
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    progress_pct: float = 0.0
    recording_ids: list[int] = field(default_factory=list)
    epochs: int = 80
    model_id: int | None = None
    d_prime: float | None = None
    priority: int = PRIORITY_FREE


@dataclass(slots=True)
class CircuitBreakerState:
    """Per-user failure tracking."""

    user_id: int
    consecutive_failures: int = 0
    paused: bool = False
    next_attempt_at: datetime | None = None
    last_failure_at: datetime | None = None
    pause_reason: str | None = None


_TIER_PRIORITY: dict[str, int] = {
    "free": PRIORITY_FREE,
    "developer": PRIORITY_DEVELOPER,
    "business": PRIORITY_BUSINESS,
    "enterprise": PRIORITY_ENTERPRISE,
}


async def _resolve_user_priority(user_id: int) -> int:
    """Return the queue priority for a user based on their subscription tier."""
    from app.models import Subscription

    async with async_session_factory() as session:
        result = await session.execute(
            select(Subscription.tier).where(Subscription.user_id == user_id)
        )
        row = result.first()
        tier = row[0] if row else "free"

    return _TIER_PRIORITY.get(str(tier), PRIORITY_FREE)


class QueueFullError(RuntimeError):
    """Raised when the persistent queue is at capacity."""


class TooManyPendingJobsError(RuntimeError):
    """Raised when a user already has too many active jobs."""


class TooManySubscribersError(RuntimeError):
    """Raised when a job already has too many SSE subscribers."""


class UserQueuePausedError(RuntimeError):
    """Raised when a paused user submits a job the queue could never dispatch.

    ``_fill_queue_from_db`` and ``_execute_job`` both skip a paused user, so before
    this existed a submission from a paused account was accepted, charged a training
    attempt, answered "Queued for training.", and then sat PENDING forever. The
    customer saw a queued job and lost an attempt for it; nothing ever ran. Refusing
    the submit is the honest form of the same decision, and it happens before
    ``record_usage`` so nothing is spent.
    """


class JobQueue:
    """Persistent async training job queue."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        max_concurrent: int = 2,
        max_pending: int = QUEUE_MAX_SIZE,
    ) -> None:
        self._db_path = db_path or (settings.data_dir / "job_queue.db")
        self._queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_pending)
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._subscribers: dict[int, list[asyncio.Queue[dict[str, Any]]]] = {}
        self._queued_job_ids: set[int] = set()
        self._running_job_ids: set[int] = set()
        self._cancel_events: dict[int, threading.Event] = {}
        self._inflight_tasks: set[asyncio.Task[None]] = set()
        self._retry_tasks: dict[int, asyncio.Task[None]] = {}
        self._submission_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._refill_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Initialize persistence and start the dispatcher loop."""
        await self._initialize_db()
        await self._resume_jobs()
        # Reconcile jobs stranded by a pause that happened before this process (or
        # before this code) existed, so a restart is when they stop lying instead of
        # one more restart they survive.
        await self._abandon_stranded_pending_jobs()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="job-queue-worker")
        await self._fill_queue_from_db()
        logger.info("Job queue started with max_concurrent=%s", settings.max_concurrent_jobs)

    async def shutdown(self) -> None:
        """Stop the dispatcher loop and cancel outstanding retry timers."""
        self._closed = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        for cancel_event in list(self._cancel_events.values()):
            cancel_event.set()

        if self._inflight_tasks:
            done, pending = await asyncio.wait(list(self._inflight_tasks), timeout=15)
            for task in pending:
                task.cancel()
            for task in done:
                with suppress(asyncio.CancelledError):
                    await task

        for task in list(self._retry_tasks.values()):
            task.cancel()
        for task in list(self._retry_tasks.values()):
            with suppress(asyncio.CancelledError):
                await task
        self._retry_tasks.clear()

    async def submit_job(
        self,
        *,
        user_id: int,
        wake_word: str,
        recording_ids: list[int],
        epochs: int,
        priority: int | None = None,
    ) -> int:
        """Persist a new training job and enqueue it when capacity allows.

        When *priority* is not supplied it is resolved automatically from the
        user's subscription tier (free=0, developer=5, business=10).
        """
        # Refuse before anything is persisted or charged: a paused user's job is
        # skipped by both the dispatcher and the executor, so accepting it would
        # spend a training attempt on a run that can never start.
        breaker = await self.get_circuit_breaker(user_id)
        if breaker.paused:
            raise UserQueuePausedError(
                "Training is paused on your account after "
                f"{breaker.consecutive_failures} failed runs in a row, so this job "
                "was not queued and no training attempt was used. Resume training "
                "from your dashboard to try again."
            )

        if await self._pending_count() >= self._queue.maxsize:
            raise QueueFullError("Training queue is full. Please try again later.")

        if priority is None:
            priority = await _resolve_user_priority(user_id)

        created_at = _utcnow()
        payload = json.dumps(recording_ids)

        async with self._submission_lock:
            async with self._connect() as conn:
                async with conn.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM jobs
                    WHERE user_id = ? AND status IN (?, ?)
                    """,
                    (
                        user_id,
                        JobStatus.PENDING.value,
                        JobStatus.RUNNING.value,
                    ),
                ) as cursor:
                    row = await cursor.fetchone()
                active_job_count = int(row["count"]) if row is not None else 0
                if active_job_count >= PER_USER_MAX_PENDING:
                    raise TooManyPendingJobsError(
                        "Too many pending jobs. Wait for current jobs to complete."
                    )

                cursor = await conn.execute(
                    """
                    INSERT INTO jobs (
                        user_id,
                        wake_word,
                        status,
                        created_at,
                        started_at,
                        completed_at,
                        error,
                        progress_pct,
                        recording_ids,
                        epochs,
                        model_id,
                        d_prime,
                        priority
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        wake_word,
                        JobStatus.PENDING.value,
                        _serialize_datetime(created_at),
                        None,
                        None,
                        None,
                        0.0,
                        payload,
                        epochs,
                        None,
                        None,
                        priority,
                    ),
                )
                await conn.commit()
                job_id = int(cursor.lastrowid)

        logger.info(
            "Queued training job %s for user %s (priority=%s)",
            job_id,
            user_id,
            priority,
        )
        await self._fill_queue_from_db()
        # Publish an initial PENDING event so SSE subscribers immediately see
        # their queue position after submission.
        queue_position = await self._queue_position(job_id)
        await self._publish(
            job_id,
            {
                "status": JobStatus.PENDING.value,
                "progress": 0.0,
                "epoch": 0,
                "total_epochs": epochs,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "message": "Queued for training.",
                "error": None,
                "d_prime": None,
                "model_id": None,
                "queue_position": queue_position,
            },
        )
        return job_id

    async def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending or running job."""
        job = await self.get_job(job_id)
        if job is None:
            return False
        if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}:
            return False

        if job.status is JobStatus.RUNNING:
            async with self._state_lock:
                cancel_event = self._cancel_events.get(job_id)
            if cancel_event is not None:
                cancel_event.set()
            await self._update_job(job_id, cancel_requested=True)
            logger.info("Cancellation requested for running job %s", job_id)
            return True

        completed_at = _utcnow()
        await self._update_job(
            job_id,
            status=JobStatus.CANCELLED,
            completed_at=completed_at,
            error="Cancelled by user",
            cancel_requested=False,
        )
        await self._publish(
            job_id,
            {
                "status": JobStatus.CANCELLED.value,
                "progress": job.progress_pct,
                "epoch": 0,
                "total_epochs": job.epochs,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "message": "Training cancelled.",
                "error": "Cancelled by user",
                "d_prime": job.d_prime,
                "model_id": job.model_id,
                "queue_position": None,
            },
        )
        await self._fill_queue_from_db()
        return True

    async def get_job(self, job_id: int) -> Job | None:
        """Return a persisted job by ID."""
        async with self._connect() as conn, conn.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    async def list_jobs(self, user_id: int) -> list[Job]:
        """List persisted jobs for a user, newest first."""
        async with self._connect() as conn, conn.execute(
            "SELECT * FROM jobs WHERE user_id = ? ORDER BY created_at DESC, id DESC",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def delete_jobs_for_user(self, user_id: int) -> int:
        """Cancel and delete all persisted jobs for a user."""
        async with self._connect() as conn, conn.execute(
            "SELECT id, status FROM jobs WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            async with self._connect() as conn:
                await conn.execute(
                    "DELETE FROM user_circuit_breakers WHERE user_id = ?",
                    (user_id,),
                )
                await conn.commit()
            return 0

        job_ids = [int(row["id"]) for row in rows]
        running_job_ids = [
            int(row["id"])
            for row in rows
            if str(row["status"]) == JobStatus.RUNNING.value
        ]

        async with self._state_lock:
            for job_id in running_job_ids:
                cancel_event = self._cancel_events.get(job_id)
                if cancel_event is not None:
                    cancel_event.set()
            self._queued_job_ids.difference_update(job_ids)
            for job_id in job_ids:
                self._subscribers.pop(job_id, None)

        deadline = asyncio.get_running_loop().time() + ACCOUNT_DELETE_CANCEL_TIMEOUT_SECONDS
        while running_job_ids:
            async with self._state_lock:
                running_job_ids = [
                    job_id for job_id in running_job_ids if job_id in self._running_job_ids
                ]
            if not running_job_ids:
                break
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning(
                    "Timed out waiting for user %s jobs to stop during account deletion: %s",
                    user_id,
                    running_job_ids,
                )
                break
            await asyncio.sleep(0.1)

        async with self._connect() as conn:
            await conn.execute("DELETE FROM jobs WHERE user_id = ?", (user_id,))
            await conn.execute("DELETE FROM user_circuit_breakers WHERE user_id = ?", (user_id,))
            await conn.commit()

        logger.info("Deleted %s queued jobs for user %s", len(job_ids), user_id)
        await self._fill_queue_from_db()
        return len(job_ids)

    async def resume_user(self, user_id: int) -> None:
        """Clear the circuit breaker pause for a user and resume queued work."""
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, 0, 0, NULL, NULL, NULL)
                ON CONFLICT(user_id) DO UPDATE SET
                    consecutive_failures = 0,
                    paused = 0,
                    next_attempt_at = NULL,
                    last_failure_at = NULL,
                    pause_reason = NULL
                """,
                (user_id,),
            )
            await conn.commit()

        retry_task = self._retry_tasks.pop(user_id, None)
        if retry_task is not None:
            retry_task.cancel()
        logger.info("Resumed job queue for user %s", user_id)
        await self._fill_queue_from_db()

    async def get_circuit_breaker(self, user_id: int) -> CircuitBreakerState:
        """Return the circuit breaker state for a user."""
        async with self._connect() as conn:
            return await self._get_circuit_breaker_with_conn(conn, user_id)

    async def runtime_snapshot(self) -> dict[str, Any]:
        """Return queue depth and worker state for health checks.

        Beyond raw counts this exposes the *wedge* signal the health check
        keys on (issue #1481): the age of the oldest pending job that is
        actually **dispatchable** -- i.e. not blocked by a paused or
        backing-off circuit breaker. A job blocked by a paused breaker can
        never be dispatched until the user resumes, so counting it as a stuck
        queue (the pre-fix behaviour) flagged the whole backend unhealthy
        forever with no worker able to touch it. ``oldest_dispatchable_pending_age_s``
        rises only when the dispatcher genuinely fails to pick up runnable work.
        """
        now = _utcnow()
        async with self._connect() as conn:
            async with conn.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status = ?",
                (JobStatus.RUNNING.value,),
            ) as cursor:
                running_row = await cursor.fetchone()
            async with conn.execute(
                """
                SELECT j.created_at AS created_at, cb.paused AS paused,
                       cb.next_attempt_at AS next_attempt_at
                FROM jobs j
                LEFT JOIN user_circuit_breakers cb ON cb.user_id = j.user_id
                WHERE j.status = ?
                """,
                (JobStatus.PENDING.value,),
            ) as cursor:
                pending_rows = await cursor.fetchall()

        persisted_running_count = int(running_row["count"]) if running_row is not None else 0

        pending_count = len(pending_rows)
        oldest_pending_age_s: float | None = None
        oldest_dispatchable_pending_age_s: float | None = None
        blocked_pending_count = 0
        for row in pending_rows:
            created = _deserialize_datetime(row["created_at"]) or now
            age = max(0.0, (now - created).total_seconds())
            if oldest_pending_age_s is None or age > oldest_pending_age_s:
                oldest_pending_age_s = age
            # Mirror _fill_queue_from_db's dispatch-skip logic exactly.
            next_attempt_at = _deserialize_datetime(row["next_attempt_at"])
            blocked = bool(row["paused"]) or (
                next_attempt_at is not None and next_attempt_at > now
            )
            if blocked:
                blocked_pending_count += 1
                continue
            if (
                oldest_dispatchable_pending_age_s is None
                or age > oldest_dispatchable_pending_age_s
            ):
                oldest_dispatchable_pending_age_s = age

        async with self._state_lock:
            queued_job_ids = sorted(self._queued_job_ids)
            running_job_ids = sorted(self._running_job_ids)

        worker_task_running = self._worker_task is not None and not self._worker_task.done()
        active_workers = len(running_job_ids)
        max_workers = settings.max_concurrent_jobs
        return {
            "queue_depth": pending_count,
            "in_memory_queue_depth": self._queue.qsize(),
            "persisted_running_jobs": persisted_running_count,
            "blocked_pending_jobs": blocked_pending_count,
            "oldest_pending_age_s": oldest_pending_age_s,
            "oldest_dispatchable_pending_age_s": oldest_dispatchable_pending_age_s,
            "worker_status": {
                "active_workers": active_workers,
                "max_workers": max_workers,
                "available_slots": max(max_workers - active_workers, 0),
                "worker_task_running": worker_task_running,
                "queued_job_ids": queued_job_ids,
                "running_job_ids": running_job_ids,
            },
        }

    def subscribe(self, job_id: int) -> asyncio.Queue[dict[str, Any]]:
        """Subscribe to SSE-style job updates."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        listeners = self._subscribers.setdefault(job_id, [])
        if len(listeners) >= MAX_SUBSCRIBERS_PER_JOB:
            raise TooManySubscribersError("Too many subscribers for this job.")
        listeners.append(queue)
        return queue

    def unsubscribe(self, job_id: int, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribe from SSE-style job updates."""
        listeners = self._subscribers.get(job_id)
        if listeners is None:
            return
        try:
            listeners.remove(queue)
        except ValueError:
            return
        if not listeners:
            self._subscribers.pop(job_id, None)

    async def _initialize_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self._connect() as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA foreign_keys=ON")
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
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
                    d_prime REAL,
                    priority INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_circuit_breakers (
                    user_id INTEGER PRIMARY KEY,
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    paused INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_failure_at TEXT,
                    pause_reason TEXT
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_user_created ON jobs(user_id, created_at DESC)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at ASC)"
            )

            # Migration: add priority column to existing databases that predate
            # this feature.  Must run BEFORE the priority index creation below.
            async with conn.execute("PRAGMA table_info(jobs)") as cursor:
                columns = {row["name"] async for row in cursor}
            if "priority" not in columns:
                await conn.execute(
                    "ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
                )
                logger.info("Migrated jobs table: added priority column")
            if "cancel_requested" not in columns:
                await conn.execute(
                    "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
                )
                logger.info("Migrated jobs table: added cancel_requested column")
            if "usage_refunded" not in columns:
                # The credit claim lives on the job row, not in the caller, so a
                # job can be credited at most once no matter how many code paths
                # decide it deserves one. Defaulting existing rows to 0 is right:
                # nothing has ever been credited before this column existed.
                await conn.execute(
                    "ALTER TABLE jobs ADD COLUMN usage_refunded INTEGER NOT NULL DEFAULT 0"
                )
                logger.info("Migrated jobs table: added usage_refunded column")

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_priority_created ON jobs(status, priority DESC, created_at ASC)"
            )

            await conn.commit()

    async def _resume_jobs(self) -> None:
        running_user_ids: set[int] = set()
        now = _utcnow()
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT DISTINCT user_id
                FROM jobs
                WHERE status = ? AND COALESCE(cancel_requested, 0) = 0
                """,
                (JobStatus.RUNNING.value,),
            ) as cursor:
                rows = await cursor.fetchall()
                running_user_ids = {int(row["user_id"]) for row in rows}

            await conn.execute(
                """
                UPDATE jobs
                SET
                    status = ?,
                    completed_at = ?,
                    error = ?,
                    cancel_requested = 0
                WHERE status = ? AND COALESCE(cancel_requested, 0) = 1
                """,
                (
                    JobStatus.CANCELLED.value,
                    _serialize_datetime(now),
                    "Cancelled by user",
                    JobStatus.RUNNING.value,
                ),
            )
            await conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = NULL, error = NULL, cancel_requested = 0
                WHERE status IN (?, ?) AND COALESCE(cancel_requested, 0) = 0
                """,
                (
                    JobStatus.PENDING.value,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
            await conn.commit()

            async with conn.execute(
                """
                SELECT user_id, next_attempt_at, paused
                FROM user_circuit_breakers
                WHERE next_attempt_at IS NOT NULL
                """
            ) as cursor:
                breaker_rows = await cursor.fetchall()

        for user_id in running_user_ids:
            logger.info("Resumed interrupted training jobs for user %s", user_id)

        for row in breaker_rows:
            if bool(row["paused"]):
                continue
            next_attempt_at = _deserialize_datetime(row["next_attempt_at"])
            if next_attempt_at is None:
                continue
            delay = max(0.0, (next_attempt_at - now).total_seconds())
            self._schedule_retry_fill(int(row["user_id"]), delay)

    async def _worker_loop(self) -> None:
        while not self._closed:
            job_id = await self._queue.get()
            async with self._state_lock:
                self._queued_job_ids.discard(job_id)
            await self._semaphore.acquire()
            task = asyncio.create_task(self._execute_job(job_id), name=f"job-{job_id}")
            self._inflight_tasks.add(task)

            def _on_done(completed: asyncio.Task[None]) -> None:
                self._semaphore.release()
                self._inflight_tasks.discard(completed)

            task.add_done_callback(_on_done)

    async def _execute_job(self, job_id: int) -> None:
        async with self._state_lock:
            self._running_job_ids.add(job_id)

        output_dir: Path | None = None
        try:
            job = await self.get_job(job_id)
            if job is None:
                return
            if job.status is not JobStatus.PENDING:
                return

            breaker = await self.get_circuit_breaker(job.user_id)
            now = _utcnow()
            if breaker.paused:
                # A job already PENDING when the account paused (a sibling of the
                # run that tripped the breaker) is stranded: nothing dispatches it
                # and nothing ever will. Returning here left it PENDING forever --
                # the customer's console said "Queued" indefinitely while the
                # attempt stayed spent. End it honestly and give the attempt back.
                await self._abandon_before_running(
                    job,
                    "Training did not start: your training queue is paused after "
                    f"{breaker.consecutive_failures} failed runs in a row. Resume "
                    "training from your dashboard to try again.",
                )
                return
            if breaker.next_attempt_at is not None and breaker.next_attempt_at > now:
                delay = (breaker.next_attempt_at - now).total_seconds()
                self._schedule_retry_fill(job.user_id, delay)
                logger.info(
                    "Delaying job %s for user %s due to failure backoff (%ss)",
                    job_id,
                    job.user_id,
                    round(delay, 2),
                )
                return

            cancel_event = threading.Event()
            async with self._state_lock:
                self._cancel_events[job_id] = cancel_event

            await self._update_job(
                job_id,
                status=JobStatus.RUNNING,
                started_at=now,
                error=None,
            )
            await self._publish(
                job_id,
                {
                    "status": JobStatus.RUNNING.value,
                    "progress": job.progress_pct,
                    "epoch": 0,
                    "total_epochs": job.epochs,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "message": "Training started.",
                    "error": None,
                    "d_prime": job.d_prime,
                    "model_id": job.model_id,
                    "queue_position": None,
                },
            )

            recording_paths = await self._load_recording_paths(job.user_id, job.recording_ids)
            if len(recording_paths) < 5:
                raise RuntimeError(f"No valid recordings found for training job {job_id}")

            # Resolve negatives corpus for paid tiers
            negatives_dir = await self._resolve_negatives_dir(job.user_id)

            output_dir = Path(tempfile.mkdtemp(prefix=f"violawake_job_{job.id}_", dir=str(settings.tmp_dir)))
            output_path = output_dir / f"{job.wake_word}_{job.id}_{int(now.timestamp())}.onnx"

            loop = asyncio.get_running_loop()

            def _on_progress(event: dict[str, Any]) -> None:
                future = asyncio.run_coroutine_threadsafe(
                    self._handle_progress_event(job_id, job.epochs, event),
                    loop,
                )
                # 60s tolerates transient DB / SSE stalls that previously
                # killed otherwise-healthy training jobs (Job 51 on
                # 2026-05-07 went straight to status=failed/timeout after a
                # backend restart because a single progress write stalled
                # >10s while the new container warmed up). The training
                # itself is still bounded by `settings.training_timeout`
                # (default 900s) below, so this only affects per-event
                # back-pressure, not total job duration.
                try:
                    future.result(timeout=60)
                except concurrent.futures.TimeoutError:
                    logger.warning(
                        "Progress event for job %s took >60s; dropping event but keeping job alive",
                        job_id,
                    )
                    future.cancel()

            artifact = await asyncio.to_thread(
                run_training_job_sync,
                job_id=job.id,
                wake_word=job.wake_word,
                recording_identifiers=recording_paths,
                output_path=output_path,
                epochs=job.epochs,
                timeout_seconds=settings.training_timeout,
                progress_callback=_on_progress,
                is_cancelled=cancel_event.is_set,
                negatives_dir=negatives_dir,
            )

            storage = get_storage()
            model_key = build_model_key(job.user_id, artifact.local_path.name)
            storage.upload(
                model_key,
                artifact.local_path.read_bytes(),
                "application/octet-stream",
            )
            if artifact.config_bytes is not None:
                storage.upload(
                    build_companion_config_identifier(model_key),
                    artifact.config_bytes,
                    "application/json",
                )

            model_id = await self._create_model_record(
                user_id=job.user_id,
                wake_word=job.wake_word,
                file_path=model_key,
                config_json=artifact.config_json,
                d_prime=artifact.d_prime,
                size_bytes=artifact.size_bytes,
            )

            completed_at = _utcnow()
            await self._update_job(
                job_id,
                status=JobStatus.COMPLETED,
                progress_pct=100.0,
                completed_at=completed_at,
                error=None,
                model_id=model_id,
                d_prime=artifact.d_prime,
                cancel_requested=False,
            )
            await self._record_success(job.user_id)

            # Schedule post-training recording deletion (privacy: recordings
            # are deleted after training per the privacy FAQ).
            await self._schedule_recording_cleanup(job.recording_ids)

            await self._publish(
                job_id,
                {
                    "status": JobStatus.COMPLETED.value,
                    "progress": 100.0,
                    "epoch": job.epochs,
                    "total_epochs": job.epochs,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "message": "Training complete.",
                    "error": None,
                    "d_prime": artifact.d_prime,
                    "model_id": model_id,
                    "queue_position": None,
                },
            )
            logger.info("Training job %s completed for user %s", job_id, job.user_id)

            # Best-effort training-complete email notification.
            try:
                from app.email_service import get_email_service

                email_svc = get_email_service()
                if email_svc.enabled:
                    async with async_session_factory() as session:
                        user = await session.get(User, job.user_id)
                    if user is not None:
                        download_url = f"/models/{model_id}/download"
                        await email_svc.send_training_complete(
                            to=user.email,
                            model_name=job.wake_word,
                            download_url=download_url,
                        )
            except Exception as email_exc:
                log_exception(logger, email_exc, message="Training-complete email failed", source="email")

        except TrainingCancelledError as exc:
            current_job = await self.get_job(job_id)
            completed_at = _utcnow()
            progress_pct = current_job.progress_pct if current_job is not None else 0.0
            await self._update_job(
                job_id,
                status=JobStatus.CANCELLED,
                completed_at=completed_at,
                error=str(exc),
                cancel_requested=False,
            )
            await self._publish(
                job_id,
                {
                    "status": JobStatus.CANCELLED.value,
                    "progress": progress_pct,
                    "epoch": 0,
                    "total_epochs": current_job.epochs if current_job is not None else 0,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "message": "Training cancelled.",
                    "error": str(exc),
                    "d_prime": current_job.d_prime if current_job is not None else None,
                    "model_id": current_job.model_id if current_job is not None else None,
                    "queue_position": None,
                },
            )
            logger.info("Training job %s cancelled", job_id)
        except Exception as exc:
            current_job = await self.get_job(job_id)
            completed_at = _utcnow()
            user_id = current_job.user_id if current_job is not None else None
            credited = False
            if user_id is not None and not _is_expected_training_outcome(exc):
                if _is_shared_infrastructure_fault(exc):
                    # Our missing prerequisite: pace the queue, charge no strike,
                    # and give the attempt back. The strike exemption alone still
                    # left the customer paying for our outage in the only currency
                    # the free tier meters -- three runs a month.
                    await self._record_transient_fault(user_id, str(exc))
                    if current_job is not None:
                        credited = await self._credit_training_attempt(current_job)
                else:
                    await self._record_failure(user_id, str(exc))

            # Built after the classification so the customer is never told their
            # attempt came back when it did not.
            error_text = f"{exc} {ATTEMPT_CREDITED_NOTE}" if credited else str(exc)
            await self._update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=completed_at,
                error=error_text,
                cancel_requested=False,
            )
            await self._publish(
                job_id,
                {
                    "status": JobStatus.FAILED.value,
                    "progress": current_job.progress_pct if current_job is not None else 0.0,
                    "epoch": 0,
                    "total_epochs": current_job.epochs if current_job is not None else 0,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "message": "Training failed.",
                    "error": error_text,
                    "d_prime": current_job.d_prime if current_job is not None else None,
                    "model_id": current_job.model_id if current_job is not None else None,
                    "queue_position": None,
                },
            )
            log_exception(
                logger,
                exc,
                message="Training job failed",
                source="job_queue",
                extra={"job_id": job_id},
            )
            if current_job is not None:
                await self._notify_training_failed(
                    current_job, error_text, credited=credited
                )
        finally:
            async with self._state_lock:
                self._running_job_ids.discard(job_id)
                self._cancel_events.pop(job_id, None)
            if output_dir is not None and output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            await self._fill_queue_from_db()

    async def _credit_training_attempt(self, job: Job) -> bool:
        """Give one training attempt back for a job that produced no model.

        The claim is taken on the job row first (``usage_refunded`` 0 -> 1 in a
        single guarded UPDATE), so a job is credited at most once however many code
        paths think it deserves one. Only then does the billing counter move, and it
        moves in the period the charge was actually made in (derived from
        ``job.created_at``), so a job submitted on the last day of a month cannot
        turn a stale charge into a bonus attempt in the next one.

        Returns True only when a counter really moved, so no caller can tell a
        customer their attempt came back when it did not. On any failure the claim is
        released rather than swallowed -- a lost credit is a customer silently out an
        attempt, which is the entire bug this exists to close.
        """
        async with self._connect() as conn:
            cursor = await conn.execute(
                "UPDATE jobs SET usage_refunded = 1 "
                "WHERE id = ? AND COALESCE(usage_refunded, 0) = 0",
                (job.id,),
            )
            await conn.commit()
            if cursor.rowcount == 0:
                return False

        credited = False
        try:
            from app.routes.billing import period_start_for, refund_usage

            async with async_session_factory() as session:
                credited = await refund_usage(
                    session,
                    job.user_id,
                    action="training_job",
                    period_start=period_start_for(job.created_at),
                )
                await session.commit()
        except Exception as exc:
            log_exception(
                logger,
                exc,
                message="Training attempt credit failed",
                source="job_queue",
                extra={"job_id": job.id},
            )

        if not credited:
            # Either the credit raised, or there was nothing to reverse. Release the
            # claim so a retry (or an operator) can still issue it.
            async with self._connect() as conn:
                await conn.execute(
                    "UPDATE jobs SET usage_refunded = 0 WHERE id = ?",
                    (job.id,),
                )
                await conn.commit()
            return False

        return True

    async def _abandon_stranded_pending_jobs(self, user_id: int | None = None) -> int:
        """End every PENDING job that belongs to a paused user. Return the count.

        A pause strands the rest of that user's queue: ``_fill_queue_from_db`` skips
        a paused user, so their other PENDING jobs are never dispatched and never
        will be. They stayed PENDING forever, charged, showing "Queued" in the
        console (this is the shape that stranded job 85 in the #1481 wedge).

        Called at the causal moment (``_record_failure`` when it pauses the account,
        scoped to that user) and once at boot with no scope, which reconciles
        anything stranded by a restart, a deploy, or a pause that predates this code.

        The boot pass deliberately does not email. Those strands are weeks old, the
        customer stopped waiting long ago, and the human contact for the already
        affected accounts is a founder-approved decision (GeeIHadAGoodTime/Viola#2066),
        not something a container restart should fire off on its own. The ledger and
        the job state are still corrected, which is what the customer is owed here.
        """
        sql = (
            "SELECT j.id FROM jobs j "
            "JOIN user_circuit_breakers b ON b.user_id = j.user_id "
            "WHERE j.status = ? AND b.paused = 1"
        )
        params: list[Any] = [JobStatus.PENDING.value]
        if user_id is not None:
            sql += " AND j.user_id = ?"
            params.append(user_id)

        async with self._connect() as conn, conn.execute(sql, tuple(params)) as cursor:
            rows = await cursor.fetchall()

        abandoned = 0
        for row in rows:
            job = await self.get_job(int(row["id"]))
            if job is None or job.status is not JobStatus.PENDING:
                continue
            await self._abandon_before_running(
                job,
                "Training did not start: your training queue was paused before this "
                "run could begin. Resume training from your dashboard to try again.",
                notify=user_id is not None,
            )
            abandoned += 1

        if abandoned:
            logger.warning(
                "Ended %s stranded pending job(s) for paused user(s) %s",
                abandoned,
                user_id if user_id is not None else "(boot reconciliation)",
            )
        return abandoned

    async def _abandon_before_running(
        self,
        job: Job,
        reason: str,
        *,
        notify: bool = True,
    ) -> None:
        """End a job the queue never dispatched, and give the attempt back.

        A job that never started consumed none of the training compute the
        submit-time charge pays for, so leaving it PENDING was the worst of both
        outcomes: the customer stayed out an attempt AND the console kept saying
        "Queued" indefinitely, with no failure, no email and no end state.
        """
        credited = await self._credit_training_attempt(job)
        message = f"{reason} {ATTEMPT_CREDITED_NOTE}" if credited else reason

        await self._update_job(
            job.id,
            status=JobStatus.FAILED,
            completed_at=_utcnow(),
            error=message,
            cancel_requested=False,
        )
        await self._publish(
            job.id,
            {
                "status": JobStatus.FAILED.value,
                "progress": job.progress_pct,
                "epoch": 0,
                "total_epochs": job.epochs,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "message": "Training did not start.",
                "error": message,
                "d_prime": job.d_prime,
                "model_id": job.model_id,
                "queue_position": None,
            },
        )
        logger.warning(
            "Job %s for user %s never ran (%s); attempt credited=%s",
            job.id,
            job.user_id,
            reason,
            credited,
        )
        if notify:
            await self._notify_training_failed(job, message, credited=credited)

    async def _notify_training_failed(
        self,
        job: Job,
        reason: str,
        *,
        credited: bool,
    ) -> None:
        """Best-effort "your training didn't finish" email.

        ``send_training_complete`` had no counterpart, so a run that succeeded
        emailed the customer and a run that failed told them nothing outside a live
        SSE stream -- which only reaches a browser tab that happens to still be open
        on the progress page. Close the tab, or fail overnight, and the product said
        nothing at all (ledger C-050). Mirrors the completion email's best-effort
        shape: a mail failure must never change the job's outcome.
        """
        try:
            from app.email_service import get_email_service

            email_svc = get_email_service()
            if not email_svc.enabled:
                return

            async with async_session_factory() as session:
                user = await session.get(User, job.user_id)
            if user is None:
                return

            breaker = await self.get_circuit_breaker(job.user_id)
            await email_svc.send_training_failed(
                to=user.email,
                wake_word=job.wake_word,
                reason=reason,
                attempt_credited=credited,
                queue_paused=breaker.paused,
            )
        except Exception as email_exc:
            log_exception(
                logger,
                email_exc,
                message="Training-failed email failed",
                source="email",
            )

    async def _pending_count(self) -> int:
        async with self._connect() as conn, conn.execute(
            "SELECT COUNT(*) AS count FROM jobs WHERE status = ?",
            (JobStatus.PENDING.value,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["count"]) if row is not None else 0

    async def _fill_queue_from_db(self) -> None:
        async with self._refill_lock:
            free_slots = self._queue.maxsize - self._queue.qsize()
            if free_slots <= 0:
                return

            now = _utcnow()
            async with self._connect() as conn:
                async with conn.execute(
                    """
                    SELECT id, user_id
                    FROM jobs
                    WHERE status = ?
                    ORDER BY priority DESC, created_at ASC, id ASC
                    """,
                    (JobStatus.PENDING.value,),
                ) as cursor:
                    rows = await cursor.fetchall()

                for row in rows:
                    if free_slots <= 0:
                        break
                    job_id = int(row["id"])
                    user_id = int(row["user_id"])
                    async with self._state_lock:
                        if job_id in self._queued_job_ids or job_id in self._running_job_ids:
                            continue

                    breaker = await self._get_circuit_breaker_with_conn(conn, user_id)
                    if breaker.paused:
                        continue
                    if breaker.next_attempt_at is not None and breaker.next_attempt_at > now:
                        delay = (breaker.next_attempt_at - now).total_seconds()
                        self._schedule_retry_fill(user_id, delay)
                        continue

                    try:
                        self._queue.put_nowait(job_id)
                    except asyncio.QueueFull:
                        break

                    async with self._state_lock:
                        self._queued_job_ids.add(job_id)
                    free_slots -= 1

    async def _queue_position(self, job_id: int) -> int | None:
        """Return the 1-based queue position for a pending job, or None if not pending."""
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT id
                FROM jobs
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC, id ASC
                """,
                (JobStatus.PENDING.value,),
            ) as cursor:
                rows = await cursor.fetchall()

        for position, row in enumerate(rows, start=1):
            if int(row["id"]) == job_id:
                return position
        return None

    async def _handle_progress_event(
        self,
        job_id: int,
        epochs: int,
        event: dict[str, Any],
    ) -> None:
        progress = float(event.get("progress", 0.0))
        await self._update_job(job_id, progress_pct=progress)
        await self._publish(
            job_id,
            {
                "status": str(event.get("status", JobStatus.RUNNING.value)),
                "progress": progress,
                "epoch": int(event.get("epoch", 0)),
                "total_epochs": int(event.get("total_epochs", epochs)),
                "train_loss": float(event.get("train_loss", 0.0)),
                "val_loss": float(event.get("val_loss", 0.0)),
                "message": str(event.get("message", "")),
                "error": event.get("error"),
                "d_prime": event.get("d_prime"),
                "model_id": event.get("model_id"),
                "queue_position": None,  # running jobs have no queue position
            },
        )

    async def _publish(self, job_id: int, event: dict[str, Any]) -> None:
        queues = list(self._subscribers.get(job_id, []))
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("Dropping event for job %s because subscriber queue is full", job_id)

    async def _update_job(
        self,
        job_id: int,
        *,
        status: JobStatus | None = None,
        progress_pct: float | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error: str | None = None,
        model_id: int | None = None,
        d_prime: float | None = None,
        cancel_requested: bool | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[Any] = []

        if status is not None:
            assignments.append("status = ?")
            values.append(status.value)
        if progress_pct is not None:
            assignments.append("progress_pct = ?")
            values.append(progress_pct)
        if started_at is not None:
            assignments.append("started_at = ?")
            values.append(_serialize_datetime(started_at))
        if completed_at is not None:
            assignments.append("completed_at = ?")
            values.append(_serialize_datetime(completed_at))
        if error is not None or status in {JobStatus.RUNNING, JobStatus.COMPLETED, JobStatus.CANCELLED}:
            assignments.append("error = ?")
            values.append(error)
        if model_id is not None:
            assignments.append("model_id = ?")
            values.append(model_id)
        if d_prime is not None:
            assignments.append("d_prime = ?")
            values.append(d_prime)
        if cancel_requested is not None:
            assignments.append("cancel_requested = ?")
            values.append(1 if cancel_requested else 0)

        if not assignments:
            return

        values.append(job_id)
        async with self._connect() as conn:
            await conn.execute(
                f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            await conn.commit()

    async def _get_circuit_breaker_with_conn(
        self,
        conn: aiosqlite.Connection,
        user_id: int,
    ) -> CircuitBreakerState:
        async with conn.execute(
            """
            SELECT
                user_id,
                consecutive_failures,
                paused,
                next_attempt_at,
                last_failure_at,
                pause_reason
            FROM user_circuit_breakers
            WHERE user_id = ?
            """,
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return CircuitBreakerState(user_id=user_id)

        return CircuitBreakerState(
            user_id=int(row["user_id"]),
            consecutive_failures=int(row["consecutive_failures"]),
            paused=bool(row["paused"]),
            next_attempt_at=_deserialize_datetime(row["next_attempt_at"]),
            last_failure_at=_deserialize_datetime(row["last_failure_at"]),
            pause_reason=row["pause_reason"],
        )

    async def _record_success(self, user_id: int) -> None:
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, 0, 0, NULL, NULL, NULL)
                ON CONFLICT(user_id) DO UPDATE SET
                    consecutive_failures = 0,
                    paused = 0,
                    next_attempt_at = NULL,
                    last_failure_at = NULL,
                    pause_reason = NULL
                """,
                (user_id,),
            )
            await conn.commit()

        retry_task = self._retry_tasks.pop(user_id, None)
        if retry_task is not None:
            retry_task.cancel()

    async def _record_failure(self, user_id: int, error: str) -> None:
        breaker = await self.get_circuit_breaker(user_id)
        consecutive_failures = breaker.consecutive_failures + 1
        paused = consecutive_failures >= FAILURE_THRESHOLD
        next_attempt_at = None if paused else _utcnow() + timedelta(seconds=FAILURE_BACKOFF_SECONDS)
        pause_reason = error if paused else None

        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    consecutive_failures = excluded.consecutive_failures,
                    paused = excluded.paused,
                    next_attempt_at = excluded.next_attempt_at,
                    last_failure_at = excluded.last_failure_at,
                    pause_reason = excluded.pause_reason
                """,
                (
                    user_id,
                    consecutive_failures,
                    1 if paused else 0,
                    _serialize_datetime(next_attempt_at),
                    _serialize_datetime(_utcnow()),
                    pause_reason,
                ),
            )
            await conn.commit()

        if paused:
            retry_task = self._retry_tasks.pop(user_id, None)
            if retry_task is not None:
                retry_task.cancel()
            logger.warning(
                "Paused job queue for user %s after %s consecutive failures",
                user_id,
                consecutive_failures,
            )
            # The pause just stranded the rest of this user's queue. End those jobs
            # now instead of leaving them PENDING forever with the attempts spent.
            await self._abandon_stranded_pending_jobs(user_id)
            return

        self._schedule_retry_fill(user_id, FAILURE_BACKOFF_SECONDS)

    async def _record_transient_fault(self, user_id: int, error: str) -> None:
        """Back the user off WITHOUT spending a strike (shared-infrastructure fault).

        Everything `_record_failure` does about pacing, and nothing it does about
        blame: `next_attempt_at` moves out by the same backoff so the queue does not
        hammer a dependency that is down, and the same retry fill is scheduled so the
        job resumes by itself once it recovers -- but `consecutive_failures` is left
        exactly where it was, so our outage can never walk a customer to the
        FAILURE_THRESHOLD lockout that only `resume_user` clears.

        Deliberately never sets `paused`. A shared outage affects everyone, so the
        instrument for it is an operator alarm (the fault is still logged and still
        classified by `app.monitoring`), not N per-customer account locks.
        """
        breaker = await self.get_circuit_breaker(user_id)
        next_attempt_at = _utcnow() + timedelta(seconds=FAILURE_BACKOFF_SECONDS)

        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    next_attempt_at = excluded.next_attempt_at,
                    last_failure_at = excluded.last_failure_at
                """,
                (
                    user_id,
                    breaker.consecutive_failures,
                    1 if breaker.paused else 0,
                    _serialize_datetime(next_attempt_at),
                    _serialize_datetime(_utcnow()),
                    breaker.pause_reason,
                ),
            )
            await conn.commit()

        logger.warning(
            "Backed off user %s for %ss on a shared-infrastructure fault "
            "(strike NOT charged, consecutive_failures stays %s): %s",
            user_id,
            FAILURE_BACKOFF_SECONDS,
            breaker.consecutive_failures,
            error,
        )
        self._schedule_retry_fill(user_id, FAILURE_BACKOFF_SECONDS)

    def _schedule_retry_fill(self, user_id: int, delay_seconds: float) -> None:
        existing = self._retry_tasks.get(user_id)
        if existing is not None and not existing.done():
            return

        async def _delayed_fill() -> None:
            try:
                await asyncio.sleep(max(0.0, delay_seconds))
                await self._fill_queue_from_db()
            except asyncio.CancelledError:
                raise
            finally:
                self._retry_tasks.pop(user_id, None)

        self._retry_tasks[user_id] = asyncio.create_task(
            _delayed_fill(),
            name=f"user-{user_id}-queue-retry",
        )

    async def _resolve_negatives_dir(self, user_id: int) -> Path | None:
        """Return curated negatives corpus path for paid-tier users, None for free."""
        corpus_path = settings.negatives_corpus_dir
        if not corpus_path:
            return None

        corpus = Path(corpus_path)
        if not corpus.is_dir():
            logger.warning("Negatives corpus dir configured but missing: %s", corpus_path)
            return None

        # Check user's subscription tier
        from app.models import Subscription

        async with async_session_factory() as session:
            result = await session.execute(
                select(Subscription.tier).where(Subscription.user_id == user_id)
            )
            row = result.first()
            tier = row[0] if row else "free"

        if tier == "free":
            return None

        logger.info("Using curated negatives corpus for user %s (tier=%s)", user_id, tier)
        return corpus

    async def _schedule_recording_cleanup(self, recording_ids: list[int]) -> None:
        """Soft-delete recordings after training completes.

        The actual storage file purge happens later via the periodic
        retention cleanup loop (``cleanup_soft_deleted_recordings``).
        """
        if settings.post_training_retention_hours <= 0:
            return

        try:
            from app.retention import mark_recordings_for_deletion
            await mark_recordings_for_deletion(recording_ids)
        except Exception as exc:
            # Non-fatal: recordings will still be cleaned up by the
            # age-based retention policy even if this fails.
            logger.warning(
                "Failed to mark recordings for post-training deletion: %s",
                exc,
            )

    async def _load_recording_paths(self, user_id: int, recording_ids: list[int]) -> list[str]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(Recording.file_path)
                .where(
                    Recording.id.in_(recording_ids),
                    Recording.user_id == user_id,
                    Recording.deleted_at.is_(None),
                )
            )
            return [str(row[0]) for row in result.all()]

    async def _create_model_record(
        self,
        *,
        user_id: int,
        wake_word: str,
        file_path: str,
        config_json: str | None,
        d_prime: float | None,
        size_bytes: int,
    ) -> int:
        async with async_session_factory() as session:
            model = TrainedModel(
                user_id=user_id,
                wake_word=wake_word,
                file_path=str(file_path),
                config_json=config_json,
                d_prime=d_prime,
                size_bytes=size_bytes,
            )
            session.add(model)
            await session.flush()
            await session.commit()
            return int(model.id)

    def _row_to_job(self, row: aiosqlite.Row) -> Job:
        # priority column was added via migration; guard against missing column
        # in case _row_to_job is called from a test that does not run _initialize_db.
        try:
            priority = int(row["priority"])
        except (IndexError, KeyError):
            priority = PRIORITY_FREE
        return Job(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            wake_word=str(row["wake_word"]),
            status=JobStatus(str(row["status"])),
            created_at=_deserialize_datetime(row["created_at"]) or _utcnow(),
            started_at=_deserialize_datetime(row["started_at"]),
            completed_at=_deserialize_datetime(row["completed_at"]),
            error=row["error"],
            progress_pct=float(row["progress_pct"]),
            recording_ids=[int(value) for value in json.loads(row["recording_ids"])],
            epochs=int(row["epochs"]),
            model_id=int(row["model_id"]) if row["model_id"] is not None else None,
            d_prime=float(row["d_prime"]) if row["d_prime"] is not None else None,
            priority=priority,
        )

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[aiosqlite.Connection]:
        connection = await aiosqlite.connect(self._db_path, timeout=30)
        connection.row_factory = aiosqlite.Row
        try:
            yield connection
        finally:
            await connection.close()


_job_queue: JobQueue | None = None


async def init_job_queue() -> JobQueue:
    """Initialize the process-wide job queue singleton."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(max_concurrent=settings.max_concurrent_jobs)
        await _job_queue.start()
    return _job_queue


def get_job_queue() -> JobQueue:
    """Return the initialized process-wide job queue singleton."""
    if _job_queue is None:
        raise RuntimeError("Job queue has not been initialized")
    return _job_queue


async def shutdown_job_queue() -> None:
    """Shutdown the process-wide job queue singleton."""
    global _job_queue
    if _job_queue is None:
        return
    await _job_queue.shutdown()
    _job_queue = None
