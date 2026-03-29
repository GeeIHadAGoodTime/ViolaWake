"""Persistent async training job queue with circuit breaker protection."""

from __future__ import annotations

import asyncio
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
from app.models import Recording, TrainedModel
from app.monitoring import log_exception
from app.services.training_service import TrainingCancelledError, run_training_job_sync
from app.storage import build_companion_config_identifier, build_model_key, get_storage

logger = logging.getLogger("violawake.jobs")

QUEUE_MAX_SIZE = 50
FAILURE_THRESHOLD = 3
FAILURE_BACKOFF_SECONDS = 300
ACCOUNT_DELETE_CANCEL_TIMEOUT_SECONDS = 30.0


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
        if await self._pending_count() >= self._queue.maxsize:
            raise QueueFullError("Training queue is full. Please try again later.")

        if priority is None:
            priority = await _resolve_user_priority(user_id)

        created_at = _utcnow()
        payload = json.dumps(recording_ids)

        async with self._connect() as conn:
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
            logger.info("Cancellation requested for running job %s", job_id)
            return True

        completed_at = _utcnow()
        await self._update_job(
            job_id,
            status=JobStatus.CANCELLED,
            completed_at=completed_at,
            error="Cancelled by user",
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
        """Return queue depth and worker state for health checks."""
        async with self._connect() as conn:
            async with conn.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status = ?",
                (JobStatus.PENDING.value,),
            ) as cursor:
                pending_row = await cursor.fetchone()
            async with conn.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE status = ?",
                (JobStatus.RUNNING.value,),
            ) as cursor:
                running_row = await cursor.fetchone()

        pending_count = int(pending_row["count"]) if pending_row is not None else 0
        persisted_running_count = int(running_row["count"]) if running_row is not None else 0
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

            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_priority_created ON jobs(status, priority DESC, created_at ASC)"
            )

            await conn.commit()

    async def _resume_jobs(self) -> None:
        running_user_ids: set[int] = set()
        async with self._connect() as conn:
            async with conn.execute(
                "SELECT DISTINCT user_id FROM jobs WHERE status = ?",
                (JobStatus.RUNNING.value,),
            ) as cursor:
                rows = await cursor.fetchall()
                running_user_ids = {int(row["user_id"]) for row in rows}

            await conn.execute(
                """
                UPDATE jobs
                SET status = ?, started_at = NULL, error = NULL
                WHERE status IN (?, ?)
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

        now = _utcnow()
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
                logger.warning("Skipping job %s because user %s queue is paused", job_id, job.user_id)
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

            output_dir = Path(tempfile.mkdtemp(prefix=f"violawake_job_{job.id}_"))
            output_path = output_dir / f"{job.wake_word}_{job.id}_{int(now.timestamp())}.onnx"

            loop = asyncio.get_running_loop()

            def _on_progress(event: dict[str, Any]) -> None:
                future = asyncio.run_coroutine_threadsafe(
                    self._handle_progress_event(job_id, job.epochs, event),
                    loop,
                )
                future.result(timeout=10)

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
        except TrainingCancelledError as exc:
            current_job = await self.get_job(job_id)
            completed_at = _utcnow()
            progress_pct = current_job.progress_pct if current_job is not None else 0.0
            await self._update_job(
                job_id,
                status=JobStatus.CANCELLED,
                completed_at=completed_at,
                error=str(exc),
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
            )
            user_id = current_job.user_id if current_job is not None else None
            if user_id is not None:
                await self._record_failure(user_id, str(exc))
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
            return

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
