"""Persistent async training job queue with circuit breaker protection.

Every protective control here -- the failure circuit breaker, the pending-job
cap, the failure backoff timer -- is scoped to a :class:`QueuePartition`, not
to a bare account id. For an ordinary account the two are the same thing. They
are NOT the same thing for the privileged service-key path, where one account
submits work for a whole upstream user base; see ``app.tenancy`` for why
keying these controls on the account alone makes every one of them global over
that caller's users.
"""

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
from app.tenancy import ACCOUNT_TENANT, QueuePartition

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
    tenant_key: str = ACCOUNT_TENANT

    @property
    def partition(self) -> QueuePartition:
        """The scope this job's protective controls apply to.

        Carried on the row itself so the dispatcher, the resume-on-boot pass
        and the failure handler all read the SAME answer -- there is no second
        place that could disagree about whose breaker a job trips.
        """
        return QueuePartition(user_id=self.user_id, tenant_key=self.tenant_key)


@dataclass(slots=True)
class CircuitBreakerState:
    """Failure tracking for one queue partition."""

    user_id: int
    consecutive_failures: int = 0
    paused: bool = False
    next_attempt_at: datetime | None = None
    last_failure_at: datetime | None = None
    pause_reason: str | None = None
    tenant_key: str = ACCOUNT_TENANT


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
        # Keyed by partition, not by account: a backoff timer keyed on the
        # account would let one upstream tenant's failure suppress the refill
        # every other tenant of that account is waiting on.
        self._retry_tasks: dict[QueuePartition, asyncio.Task[None]] = {}
        self._submission_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._refill_lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Initialize persistence and start the dispatcher loop."""
        await self._initialize_db()
        await self._resume_jobs()
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
        partition: QueuePartition,
        wake_word: str,
        recording_ids: list[int],
        epochs: int,
        priority: int | None = None,
    ) -> int:
        """Persist a new training job and enqueue it when capacity allows.

        Takes a :class:`QueuePartition` rather than a bare account id, and
        offers no bare-id form: the pending-job cap below is counted over
        whatever this argument says, so a caller that could pass an account id
        for service-path work would be re-creating the shared cap. Submission
        is the one place a job's partition is chosen, and it is recorded on the
        row so every later control decision reads it back instead of guessing.

        When *priority* is not supplied it is resolved automatically from the
        account's subscription tier (free=0, developer=5, business=10).
        """
        if await self._pending_count() >= self._queue.maxsize:
            raise QueueFullError("Training queue is full. Please try again later.")

        user_id = partition.user_id
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
                    WHERE user_id = ? AND tenant_key = ? AND status IN (?, ?)
                    """,
                    (
                        user_id,
                        partition.tenant_key,
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
                        priority,
                        tenant_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        partition.tenant_key,
                    ),
                )
                await conn.commit()
                job_id = int(cursor.lastrowid)

        logger.info(
            "Queued training job %s for %s (priority=%s)",
            job_id,
            partition,
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

    async def list_jobs(self, partition: QueuePartition | int) -> list[Job]:
        """List persisted jobs for one partition, newest first.

        Scoped to the partition, not the account: on the service path an
        account-wide listing would hand every upstream tenant the whole
        install base's job history.
        """
        scope = QueuePartition.coerce(partition)
        async with self._connect() as conn, conn.execute(
            """
            SELECT * FROM jobs
            WHERE user_id = ? AND tenant_key = ?
            ORDER BY created_at DESC, id DESC
            """,
            (scope.user_id, scope.tenant_key),
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

    async def resume_user(self, partition: QueuePartition | int) -> None:
        """Clear one partition's breaker pause and resume its queued work.

        Clearing is partition-scoped in both directions: it releases exactly
        the tenant that asked, and it cannot release (or be blocked by) any
        other tenant sharing the same account.
        """
        scope = QueuePartition.coerce(partition)
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    tenant_key,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, ?, 0, 0, NULL, NULL, NULL)
                ON CONFLICT(user_id, tenant_key) DO UPDATE SET
                    consecutive_failures = 0,
                    paused = 0,
                    next_attempt_at = NULL,
                    last_failure_at = NULL,
                    pause_reason = NULL
                """,
                (scope.user_id, scope.tenant_key),
            )
            await conn.commit()

        retry_task = self._retry_tasks.pop(scope, None)
        if retry_task is not None:
            retry_task.cancel()
        logger.info("Resumed job queue for %s", scope)
        await self._fill_queue_from_db()

    async def get_circuit_breaker(self, partition: QueuePartition | int) -> CircuitBreakerState:
        """Return the circuit breaker state for one partition."""
        scope = QueuePartition.coerce(partition)
        async with self._connect() as conn:
            return await self._get_circuit_breaker_with_conn(conn, scope)

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
                    priority INTEGER NOT NULL DEFAULT 0,
                    tenant_key TEXT NOT NULL DEFAULT ''
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_circuit_breakers (
                    user_id INTEGER NOT NULL,
                    tenant_key TEXT NOT NULL DEFAULT '',
                    consecutive_failures INTEGER NOT NULL DEFAULT 0,
                    paused INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at TEXT,
                    last_failure_at TEXT,
                    pause_reason TEXT,
                    PRIMARY KEY (user_id, tenant_key)
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
            if "tenant_key" not in columns:
                # Existing rows belong to their own account's partition, which
                # is exactly what the empty key means -- no row changes owner.
                await conn.execute(
                    "ALTER TABLE jobs ADD COLUMN tenant_key TEXT NOT NULL DEFAULT ''"
                )
                logger.info("Migrated jobs table: added tenant_key column")

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_priority_created ON jobs(status, priority DESC, created_at ASC)"
            )
            # Backs the pending-job cap, which counts over the full partition.
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_partition_status "
                "ON jobs(user_id, tenant_key, status)"
            )

            await self._migrate_breakers_to_partitions(conn)

            await conn.commit()

    async def _migrate_breakers_to_partitions(self, conn: aiosqlite.Connection) -> None:
        """Widen an existing breaker table's key from account to partition.

        SQLite cannot alter a PRIMARY KEY in place, so a pre-tenant database
        needs a table rebuild. Every carried row keeps its state verbatim
        under the empty tenant key, so an ordinary account's breaker -- paused
        or not -- is exactly where it was; this migration widens what CAN be
        distinguished, it does not resume or pause anybody.
        """
        async with conn.execute("PRAGMA table_info(user_circuit_breakers)") as cursor:
            breaker_columns = {row["name"] async for row in cursor}
        if not breaker_columns or "tenant_key" in breaker_columns:
            return

        await conn.execute(
            "ALTER TABLE user_circuit_breakers RENAME TO user_circuit_breakers_pre_tenant"
        )
        await conn.execute(
            """
            CREATE TABLE user_circuit_breakers (
                user_id INTEGER NOT NULL,
                tenant_key TEXT NOT NULL DEFAULT '',
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                paused INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT,
                last_failure_at TEXT,
                pause_reason TEXT,
                PRIMARY KEY (user_id, tenant_key)
            )
            """
        )
        await conn.execute(
            """
            INSERT INTO user_circuit_breakers (
                user_id,
                tenant_key,
                consecutive_failures,
                paused,
                next_attempt_at,
                last_failure_at,
                pause_reason
            )
            SELECT
                user_id,
                '',
                consecutive_failures,
                paused,
                next_attempt_at,
                last_failure_at,
                pause_reason
            FROM user_circuit_breakers_pre_tenant
            """
        )
        await conn.execute("DROP TABLE user_circuit_breakers_pre_tenant")
        logger.info("Migrated user_circuit_breakers: key widened to (user_id, tenant_key)")

    async def _resume_jobs(self) -> None:
        running_partitions: set[QueuePartition] = set()
        now = _utcnow()
        async with self._connect() as conn:
            async with conn.execute(
                """
                SELECT DISTINCT user_id, tenant_key
                FROM jobs
                WHERE status = ? AND COALESCE(cancel_requested, 0) = 0
                """,
                (JobStatus.RUNNING.value,),
            ) as cursor:
                rows = await cursor.fetchall()
                running_partitions = {
                    QueuePartition(
                        user_id=int(row["user_id"]),
                        tenant_key=str(row["tenant_key"] or ACCOUNT_TENANT),
                    )
                    for row in rows
                }

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
                SELECT user_id, tenant_key, next_attempt_at, paused
                FROM user_circuit_breakers
                WHERE next_attempt_at IS NOT NULL
                """
            ) as cursor:
                breaker_rows = await cursor.fetchall()

        for partition in running_partitions:
            logger.info("Resumed interrupted training jobs for %s", partition)

        for row in breaker_rows:
            if bool(row["paused"]):
                continue
            next_attempt_at = _deserialize_datetime(row["next_attempt_at"])
            if next_attempt_at is None:
                continue
            delay = max(0.0, (next_attempt_at - now).total_seconds())
            self._schedule_retry_fill(
                QueuePartition(
                    user_id=int(row["user_id"]),
                    tenant_key=str(row["tenant_key"] or ACCOUNT_TENANT),
                ),
                delay,
            )

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

            partition = job.partition
            breaker = await self.get_circuit_breaker(partition)
            now = _utcnow()
            if breaker.paused:
                logger.warning("Skipping job %s because %s queue is paused", job_id, partition)
                return
            if breaker.next_attempt_at is not None and breaker.next_attempt_at > now:
                delay = (breaker.next_attempt_at - now).total_seconds()
                self._schedule_retry_fill(partition, delay)
                logger.info(
                    "Delaying job %s for %s due to failure backoff (%ss)",
                    job_id,
                    partition,
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
            await self._record_success(partition)

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
            await self._update_job(
                job_id,
                status=JobStatus.FAILED,
                completed_at=completed_at,
                error=str(exc),
                cancel_requested=False,
            )
            failed_partition = current_job.partition if current_job is not None else None
            if failed_partition is not None and not _is_expected_training_outcome(exc):
                if _is_shared_infrastructure_fault(exc):
                    # Our missing prerequisite: pace the queue, charge no strike.
                    await self._record_transient_fault(failed_partition, str(exc))
                else:
                    await self._record_failure(failed_partition, str(exc))
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
                    "error": str(exc),
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
        finally:
            async with self._state_lock:
                self._running_job_ids.discard(job_id)
                self._cancel_events.pop(job_id, None)
            if output_dir is not None and output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            await self._fill_queue_from_db()

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
                    SELECT id, user_id, tenant_key
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
                    # The row's own partition, so a paused tenant only ever
                    # holds back its own jobs -- the pre-tenant dispatcher read
                    # user_id here and skipped every job of the shared account.
                    partition = QueuePartition(
                        user_id=int(row["user_id"]),
                        tenant_key=str(row["tenant_key"] or ACCOUNT_TENANT),
                    )
                    async with self._state_lock:
                        if job_id in self._queued_job_ids or job_id in self._running_job_ids:
                            continue

                    breaker = await self._get_circuit_breaker_with_conn(conn, partition)
                    if breaker.paused:
                        continue
                    if breaker.next_attempt_at is not None and breaker.next_attempt_at > now:
                        delay = (breaker.next_attempt_at - now).total_seconds()
                        self._schedule_retry_fill(partition, delay)
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
        partition: QueuePartition,
    ) -> CircuitBreakerState:
        async with conn.execute(
            """
            SELECT
                user_id,
                tenant_key,
                consecutive_failures,
                paused,
                next_attempt_at,
                last_failure_at,
                pause_reason
            FROM user_circuit_breakers
            WHERE user_id = ? AND tenant_key = ?
            """,
            (partition.user_id, partition.tenant_key),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return CircuitBreakerState(
                user_id=partition.user_id,
                tenant_key=partition.tenant_key,
            )

        return CircuitBreakerState(
            user_id=int(row["user_id"]),
            tenant_key=str(row["tenant_key"]),
            consecutive_failures=int(row["consecutive_failures"]),
            paused=bool(row["paused"]),
            next_attempt_at=_deserialize_datetime(row["next_attempt_at"]),
            last_failure_at=_deserialize_datetime(row["last_failure_at"]),
            pause_reason=row["pause_reason"],
        )

    async def _record_success(self, partition: QueuePartition) -> None:
        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    tenant_key,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, ?, 0, 0, NULL, NULL, NULL)
                ON CONFLICT(user_id, tenant_key) DO UPDATE SET
                    consecutive_failures = 0,
                    paused = 0,
                    next_attempt_at = NULL,
                    last_failure_at = NULL,
                    pause_reason = NULL
                """,
                (partition.user_id, partition.tenant_key),
            )
            await conn.commit()

        retry_task = self._retry_tasks.pop(partition, None)
        if retry_task is not None:
            retry_task.cancel()

    async def _record_failure(self, partition: QueuePartition, error: str) -> None:
        breaker = await self.get_circuit_breaker(partition)
        consecutive_failures = breaker.consecutive_failures + 1
        paused = consecutive_failures >= FAILURE_THRESHOLD
        next_attempt_at = None if paused else _utcnow() + timedelta(seconds=FAILURE_BACKOFF_SECONDS)
        pause_reason = error if paused else None

        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    tenant_key,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, tenant_key) DO UPDATE SET
                    consecutive_failures = excluded.consecutive_failures,
                    paused = excluded.paused,
                    next_attempt_at = excluded.next_attempt_at,
                    last_failure_at = excluded.last_failure_at,
                    pause_reason = excluded.pause_reason
                """,
                (
                    partition.user_id,
                    partition.tenant_key,
                    consecutive_failures,
                    1 if paused else 0,
                    _serialize_datetime(next_attempt_at),
                    _serialize_datetime(_utcnow()),
                    pause_reason,
                ),
            )
            await conn.commit()

        if paused:
            retry_task = self._retry_tasks.pop(partition, None)
            if retry_task is not None:
                retry_task.cancel()
            logger.warning(
                "Paused job queue for %s after %s consecutive failures",
                partition,
                consecutive_failures,
            )
            return

        self._schedule_retry_fill(partition, FAILURE_BACKOFF_SECONDS)

    async def _record_transient_fault(self, partition: QueuePartition, error: str) -> None:
        """Back the partition off WITHOUT spending a strike (shared-infrastructure fault).

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
        breaker = await self.get_circuit_breaker(partition)
        next_attempt_at = _utcnow() + timedelta(seconds=FAILURE_BACKOFF_SECONDS)

        async with self._connect() as conn:
            await conn.execute(
                """
                INSERT INTO user_circuit_breakers (
                    user_id,
                    tenant_key,
                    consecutive_failures,
                    paused,
                    next_attempt_at,
                    last_failure_at,
                    pause_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, tenant_key) DO UPDATE SET
                    next_attempt_at = excluded.next_attempt_at,
                    last_failure_at = excluded.last_failure_at
                """,
                (
                    partition.user_id,
                    partition.tenant_key,
                    breaker.consecutive_failures,
                    1 if breaker.paused else 0,
                    _serialize_datetime(next_attempt_at),
                    _serialize_datetime(_utcnow()),
                    breaker.pause_reason,
                ),
            )
            await conn.commit()

        logger.warning(
            "Backed off %s for %ss on a shared-infrastructure fault "
            "(strike NOT charged, consecutive_failures stays %s): %s",
            partition,
            FAILURE_BACKOFF_SECONDS,
            breaker.consecutive_failures,
            error,
        )
        self._schedule_retry_fill(partition, FAILURE_BACKOFF_SECONDS)

    def _schedule_retry_fill(self, partition: QueuePartition, delay_seconds: float) -> None:
        existing = self._retry_tasks.get(partition)
        if existing is not None and not existing.done():
            return

        async def _delayed_fill() -> None:
            try:
                await asyncio.sleep(max(0.0, delay_seconds))
                await self._fill_queue_from_db()
            except asyncio.CancelledError:
                raise
            finally:
                self._retry_tasks.pop(partition, None)

        self._retry_tasks[partition] = asyncio.create_task(
            _delayed_fill(),
            name=f"queue-retry-{partition.user_id}-{partition.tenant_key or 'account'}",
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
        # Same guard for tenant_key, added by the same kind of migration. A
        # row that predates it belongs to the account's own partition.
        try:
            tenant_key = str(row["tenant_key"] or ACCOUNT_TENANT)
        except (IndexError, KeyError):
            tenant_key = ACCOUNT_TENANT
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
            tenant_key=tenant_key,
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
