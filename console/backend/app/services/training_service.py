"""Training pipeline helpers used by the async job queue."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select, update

from app.config import settings
from app.database import async_session_factory
from app.models import Recording, TrainedModel, TrainingJob
from app.monitoring import log_exception
from app.storage import build_companion_config_identifier, build_model_key, get_storage

logger = logging.getLogger("violawake.training")

_training_semaphore = threading.Semaphore(settings.max_concurrent_jobs)
_event_queues: dict[int, list[asyncio.Queue[dict[str, Any]]]] = {}
_queue_lock = threading.Lock()
_active_job_ids: set[int] = set()
_active_jobs_lock = threading.Lock()


class TrainingCancelledError(RuntimeError):
    """Raised when a running training job is cancelled."""


@dataclass(slots=True)
class TrainingArtifact:
    """Artifacts produced by a completed training run."""

    local_path: Path
    config_json: str | None
    config_bytes: bytes | None
    d_prime: float | None
    size_bytes: int


def subscribe(job_id: int) -> asyncio.Queue[dict[str, Any]]:
    """Subscribe to training events for a job."""
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    with _queue_lock:
        _event_queues.setdefault(job_id, []).append(queue)
    return queue


def unsubscribe(job_id: int, queue: asyncio.Queue[dict[str, Any]]) -> None:
    """Remove a training event subscriber."""
    with _queue_lock:
        queues = _event_queues.get(job_id)
        if queues is None:
            return
        try:
            queues.remove(queue)
        except ValueError:
            return
        if not queues:
            _event_queues.pop(job_id, None)


def get_training_worker_snapshot() -> dict[str, Any]:
    """Return a snapshot of current worker utilization."""
    with _active_jobs_lock:
        active_job_ids = sorted(_active_job_ids)

    active_workers = len(active_job_ids)
    max_workers = settings.max_concurrent_jobs
    return {
        "active_workers": active_workers,
        "max_workers": max_workers,
        "available_slots": max(max_workers - active_workers, 0),
        "active_job_ids": active_job_ids,
    }


def run_training_job_sync(
    *,
    job_id: int,
    wake_word: str,
    recording_identifiers: list[str],
    output_path: Path,
    epochs: int,
    timeout_seconds: int,
    progress_callback: Callable[[dict[str, Any]], None],
    is_cancelled: Callable[[], bool],
    negatives_dir: Path | None = None,
) -> TrainingArtifact:
    """Run the ViolaWake SDK training pipeline synchronously."""
    positives_dir: Path | None = None
    storage = get_storage()

    def _ensure_not_cancelled() -> None:
        if is_cancelled():
            raise TrainingCancelledError("Training cancelled by user")

    try:
        _ensure_not_cancelled()
        progress_callback({
            "status": "running",
            "progress": 0.0,
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": "Preparing training data...",
            "error": None,
        })

        positives_dir = Path(tempfile.mkdtemp(prefix="violawake_train_"))
        for index, recording_identifier in enumerate(recording_identifiers):
            _ensure_not_cancelled()
            if not storage.exists(recording_identifier):
                logger.warning("Recording %s was missing for training job %s", recording_identifier, job_id)
                continue

            dst = positives_dir / f"sample_{index:04d}.wav"
            dst.write_bytes(storage.download(recording_identifier))

        wav_count = len(list(positives_dir.glob("*.wav")))
        if wav_count < 5:
            raise RuntimeError("Only %s valid WAV files found. Need at least 5." % wav_count)

        progress_callback({
            "status": "running",
            "progress": 5.0,
            "epoch": 0,
            "total_epochs": epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": "Loaded %s recordings. Starting training..." % wav_count,
            "error": None,
        })

        from violawake_sdk.tools.train import _train_mlp_on_oww

        started_at = time.monotonic()

        def _on_epoch(info: dict[str, Any]) -> None:
            _ensure_not_cancelled()
            elapsed = time.monotonic() - started_at
            if elapsed > timeout_seconds:
                raise RuntimeError(
                    "Training job timed out after %ss (%s minutes)"
                    % (timeout_seconds, timeout_seconds // 60)
                )

            epoch = int(info.get("epoch", 0))
            total_epochs = int(info.get("total_epochs", epochs)) or epochs
            train_loss = float(info.get("train_loss", 0.0))
            val_loss = float(info.get("val_loss", 0.0))
            progress = min(10.0 + 85.0 * (epoch / total_epochs), 95.0)

            progress_callback({
                "status": "running",
                "progress": round(progress, 2),
                "epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "message": "Epoch %s/%s - loss: %.4f" % (epoch, total_epochs, train_loss),
                "error": None,
            })

        _train_mlp_on_oww(
            positives_dir=positives_dir,
            output_path=output_path,
            epochs=epochs,
            augment=True,
            eval_dir=None,
            negatives_dir=negatives_dir,
            verbose=True,
            progress_callback=_on_epoch,
        )

        _ensure_not_cancelled()
        if not output_path.exists():
            raise RuntimeError("Training completed but no model file was produced")

        config_path = output_path.with_suffix(".config.json")
        config_json: str | None = None
        config_bytes: bytes | None = None
        d_prime_value: float | None = None
        if config_path.exists():
            config_bytes = config_path.read_bytes()
            config_data = json.loads(config_bytes.decode("utf-8"))
            config_json = json.dumps(config_data)
            if isinstance(config_data, dict):
                raw_d_prime = config_data.get("d_prime")
                if isinstance(raw_d_prime, (int, float)):
                    d_prime_value = float(raw_d_prime)

        return TrainingArtifact(
            local_path=output_path,
            config_json=config_json,
            config_bytes=config_bytes,
            d_prime=d_prime_value,
            size_bytes=output_path.stat().st_size,
        )
    except TrainingCancelledError:
        logger.info("Training job %s cancelled", job_id)
        raise
    except Exception as exc:
        log_exception(
            logger,
            exc,
            message="Training job failed",
            source="training",
            extra={"job_id": job_id, "wake_word": wake_word},
        )
        raise
    finally:
        if positives_dir is not None and positives_dir.exists():
            shutil.rmtree(positives_dir, ignore_errors=True)


async def start_training_job(
    job_id: int,
    user_id: int,
    wake_word: str,
    recording_ids: list[int],
    epochs: int,
) -> None:
    """Launch a training job in a background thread."""
    async with async_session_factory() as session:
        result = await session.execute(
            select(Recording.file_path).where(
                Recording.id.in_(recording_ids),
                Recording.user_id == user_id,
            )
        )
        recording_identifiers = [row[0] for row in result.all()]

    if not recording_identifiers:
        await _update_job_status(
            job_id,
            status="failed",
            error="No valid recordings found",
            completed=True,
        )
        return

    loop = asyncio.get_running_loop()
    worker = threading.Thread(
        target=_run_training_worker,
        kwargs={
            "job_id": job_id,
            "user_id": user_id,
            "wake_word": wake_word,
            "recording_identifiers": recording_identifiers,
            "epochs": epochs,
            "loop": loop,
        },
        daemon=True,
        name=f"training-job-{job_id}",
    )
    worker.start()


async def _update_job_status(
    job_id: int,
    *,
    status: str | None = None,
    progress: float | None = None,
    d_prime: float | None = None,
    model_id: int | None = None,
    error: str | None = None,
    completed: bool = False,
) -> None:
    """Persist training job state changes."""
    values: dict[str, Any] = {}
    if status is not None:
        values["status"] = status
    if progress is not None:
        values["progress"] = progress
    if d_prime is not None:
        values["d_prime"] = d_prime
    if model_id is not None:
        values["model_id"] = model_id
    if error is not None:
        values["error"] = error
    if completed:
        values["completed_at"] = datetime.now(timezone.utc)

    if not values:
        return

    async with async_session_factory() as session:
        await session.execute(update(TrainingJob).where(TrainingJob.id == job_id).values(**values))
        await session.commit()


def _publish(job_id: int, event: dict[str, Any]) -> None:
    """Send an event to all live SSE subscribers for a training job."""
    with _queue_lock:
        queues = list(_event_queues.get(job_id, []))
    for queue in queues:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Dropped full training event queue for job %s", job_id)


def _run_training_worker(
    *,
    job_id: int,
    user_id: int,
    wake_word: str,
    recording_identifiers: list[str],
    epochs: int,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Execute one training job in a background thread."""
    output_dir: Path | None = None

    def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=30)

    def _emit(event: dict[str, Any]) -> None:
        _publish(job_id, event)
        try:
            _run_async(_update_job_status(
                job_id,
                status=str(event.get("status", "running")),
                progress=float(event.get("progress", 0.0)),
                error=event.get("error"),
            ))
        except Exception:
            logger.exception("Failed to persist progress for training job %s", job_id)

    acquired = _training_semaphore.acquire(timeout=5)
    if not acquired:
        error_message = "Server is at maximum training capacity. Please try again later."
        log_exception(
            logger,
            TimeoutError(error_message),
            message="Training job rejected: semaphore not available",
            source="training",
            extra={"job_id": job_id, "user_id": user_id, "wake_word": wake_word},
            include_traceback=False,
        )
        try:
            _run_async(_update_job_status(
                job_id,
                status="failed",
                progress=0.0,
                error=error_message,
                completed=True,
            ))
        finally:
            _publish(job_id, {
                "status": "failed",
                "progress": 0.0,
                "epoch": 0,
                "train_loss": 0.0,
                "val_loss": 0.0,
                "message": error_message,
                "error": error_message,
            })
        return

    try:
        with _active_jobs_lock:
            _active_job_ids.add(job_id)

        output_dir = Path(tempfile.mkdtemp(prefix="violawake_model_"))
        output_filename = "%s_%s.onnx" % (wake_word, int(time.time()))
        output_path = output_dir / output_filename

        # Resolve negatives corpus for paid-tier users
        neg_dir: Path | None = None
        corpus_path = settings.negatives_corpus_dir
        if corpus_path:
            neg_candidate = Path(corpus_path)
            if neg_candidate.is_dir():
                try:
                    from app.models import Subscription

                    import asyncio as _aio

                    async def _get_tier() -> str:
                        async with async_session_factory() as _s:
                            _r = await _s.execute(
                                select(Subscription.tier).where(Subscription.user_id == user_id)
                            )
                            _row = _r.first()
                            return _row[0] if _row else "free"

                    tier = _run_async(_get_tier())
                    if tier != "free":
                        neg_dir = neg_candidate
                        logger.info("Using curated negatives for user %s (tier=%s)", user_id, tier)
                except Exception:
                    logger.exception("Failed to resolve negatives corpus tier")

        artifact = run_training_job_sync(
            job_id=job_id,
            wake_word=wake_word,
            recording_identifiers=recording_identifiers,
            output_path=output_path,
            epochs=epochs,
            timeout_seconds=settings.training_timeout,
            progress_callback=_emit,
            is_cancelled=lambda: False,
            negatives_dir=neg_dir,
        )

        storage = get_storage()
        model_key = build_model_key(user_id, output_filename)
        storage.upload(model_key, artifact.local_path.read_bytes(), "application/octet-stream")
        if artifact.config_bytes is not None:
            storage.upload(
                build_companion_config_identifier(model_key),
                artifact.config_bytes,
                "application/json",
            )

        model_id = _run_async(_create_model_record(
            user_id=user_id,
            wake_word=wake_word,
            model_key=model_key,
            config_json=artifact.config_json,
            d_prime=artifact.d_prime,
            size_bytes=artifact.size_bytes,
        ))

        _run_async(_update_job_status(
            job_id,
            status="completed",
            progress=100.0,
            d_prime=artifact.d_prime,
            model_id=model_id,
            error=None,
            completed=True,
        ))
        _publish(job_id, {
            "status": "completed",
            "progress": 100.0,
            "epoch": epochs,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": "Training complete! Model saved (%s bytes)." % artifact.size_bytes,
            "error": None,
        })
    except TrainingCancelledError as exc:
        message = str(exc)
        _run_async(_update_job_status(
            job_id,
            status="failed",
            progress=0.0,
            error=message,
            completed=True,
        ))
        _publish(job_id, {
            "status": "failed",
            "progress": 0.0,
            "epoch": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": message,
            "error": message,
        })
    except Exception as exc:
        message = str(exc)
        _run_async(_update_job_status(
            job_id,
            status="failed",
            progress=0.0,
            error=message,
            completed=True,
        ))
        _publish(job_id, {
            "status": "failed",
            "progress": 0.0,
            "epoch": 0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "message": "Training failed: %s" % message,
            "error": message,
        })
    finally:
        with _active_jobs_lock:
            _active_job_ids.discard(job_id)
        _training_semaphore.release()
        if output_dir is not None and output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)


async def _create_model_record(
    *,
    user_id: int,
    wake_word: str,
    model_key: str,
    config_json: str | None,
    d_prime: float | None,
    size_bytes: int,
) -> int:
    """Persist a trained model row and return its ID."""
    async with async_session_factory() as session:
        model = TrainedModel(
            user_id=user_id,
            wake_word=wake_word,
            file_path=model_key,
            config_json=config_json,
            d_prime=d_prime,
            size_bytes=size_bytes,
        )
        session.add(model)
        await session.flush()
        model_id = model.id
        await session.commit()
        return model_id
