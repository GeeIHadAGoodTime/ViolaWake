"""Automatic retention cleanup for recordings and trained models."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.models import Recording, TrainedModel
from app.storage import get_storage

logger = logging.getLogger("violawake.retention")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


async def _get_active_recording_ids() -> set[int]:
    """Return recording IDs referenced by active (pending or running) jobs.

    Recordings tied to an active training job must never be deleted while
    that job is still running.  We look up the live job queue rather than
    querying the SQLAlchemy session so the check is always accurate even
    when the queue's aiosqlite DB diverges from the main app database.
    """
    try:
        from app.job_queue import JobStatus, get_job_queue
        queue = get_job_queue()
    except RuntimeError:
        # Queue not yet initialised (e.g. during testing without a running app)
        return set()

    import aiosqlite

    active_ids: set[int] = set()

    async with aiosqlite.connect(queue._db_path, timeout=10) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT recording_ids FROM jobs WHERE status IN (?, ?)",
            (JobStatus.PENDING.value, JobStatus.RUNNING.value),
        ) as cursor:
            rows = await cursor.fetchall()

    for row in rows:
        try:
            ids = json.loads(row["recording_ids"])
            active_ids.update(int(rid) for rid in ids)
        except Exception:
            pass

    return active_ids


async def mark_recordings_for_deletion(recording_ids: list[int]) -> int:
    """Soft-delete recordings by setting their deleted_at timestamp.

    Called after a training job completes successfully. The actual storage
    file deletion happens later via ``cleanup_soft_deleted_recordings``.

    Returns the number of recordings marked.
    """
    if not recording_ids:
        return 0

    now = _utcnow()
    marked = 0

    async with async_session_factory() as session:
        result = await session.execute(
            select(Recording).where(
                Recording.id.in_(recording_ids),
                Recording.deleted_at.is_(None),
            )
        )
        recordings = result.scalars().all()

        for recording in recordings:
            recording.deleted_at = now
            marked += 1

        if marked:
            await session.commit()

    if marked:
        logger.info(
            "Marked %s recording(s) for post-training deletion (will be purged after %s hours)",
            marked,
            settings.post_training_retention_hours,
        )

    return marked


async def cleanup_soft_deleted_recordings() -> int:
    """Purge storage files for recordings that were soft-deleted past the retention period.

    Recordings are soft-deleted (deleted_at set) when training completes.
    This function removes the actual storage files and hard-deletes the DB
    rows once ``post_training_retention_hours`` has elapsed since the
    soft-delete timestamp.

    Returns the number of recordings purged.  Does nothing when
    ``post_training_retention_hours`` is 0.
    """
    retention_hours = settings.post_training_retention_hours
    if retention_hours <= 0:
        return 0

    cutoff = _utcnow() - timedelta(hours=retention_hours)
    logger.info(
        "Running post-training recording cleanup: purging recordings soft-deleted before %s",
        cutoff.isoformat(),
    )

    # Don't purge recordings that are still needed by active jobs
    active_recording_ids = await _get_active_recording_ids()

    storage = get_storage()
    purged_count = 0

    async with async_session_factory() as session:
        result = await session.execute(
            select(Recording).where(
                Recording.deleted_at.is_not(None),
                Recording.deleted_at < cutoff,
            )
        )
        expired = result.scalars().all()

        for recording in expired:
            if recording.id in active_recording_ids:
                logger.debug(
                    "Skipping soft-deleted recording %s (reused by an active training job)",
                    recording.id,
                )
                continue

            try:
                storage.delete(recording.file_path)
            except Exception:
                logger.warning(
                    "Failed to delete storage object for soft-deleted recording %s at %s",
                    recording.id,
                    recording.file_path,
                )

            await session.delete(recording)
            purged_count += 1

        if purged_count:
            await session.commit()

    logger.info("Post-training recording cleanup complete: purged %s recording(s)", purged_count)
    return purged_count


async def cleanup_expired_recordings() -> int:
    """Delete recordings older than ``RECORDING_RETENTION_DAYS`` from storage and DB.

    Returns the number of recordings deleted.  Does nothing when
    ``recording_retention_days`` is 0.  Recordings linked to active (pending
    or running) training jobs are never deleted.  Already soft-deleted
    recordings are skipped (handled by ``cleanup_soft_deleted_recordings``).
    """
    retention_days = settings.recording_retention_days
    if retention_days <= 0:
        logger.info("Recording retention cleanup is disabled (recording_retention_days=0)")
        return 0

    cutoff = _utcnow() - timedelta(days=retention_days)
    logger.info(
        "Running recording retention cleanup: removing recordings created before %s",
        cutoff.isoformat(),
    )

    active_recording_ids = await _get_active_recording_ids()

    storage = get_storage()
    deleted_count = 0

    async with async_session_factory() as session:
        result = await session.execute(
            select(Recording).where(
                Recording.created_at < cutoff,
                Recording.deleted_at.is_(None),
            )
        )
        expired = result.scalars().all()

        for recording in expired:
            if recording.id in active_recording_ids:
                logger.debug(
                    "Skipping recording %s (used by an active training job)",
                    recording.id,
                )
                continue

            try:
                storage.delete(recording.file_path)
            except Exception:
                logger.warning(
                    "Failed to delete storage object for recording %s at %s",
                    recording.id,
                    recording.file_path,
                )

            await session.delete(recording)
            deleted_count += 1

        if deleted_count:
            await session.commit()

    logger.info("Recording retention cleanup complete: deleted %s recording(s)", deleted_count)
    return deleted_count


async def cleanup_expired_models() -> int:
    """Delete trained models older than ``MODEL_RETENTION_DAYS`` from storage and DB.

    Returns the number of models deleted.  Does nothing when
    ``model_retention_days`` is 0.
    """
    retention_days = settings.model_retention_days
    if retention_days <= 0:
        logger.info("Model retention cleanup is disabled (model_retention_days=0)")
        return 0

    cutoff = _utcnow() - timedelta(days=retention_days)
    logger.info(
        "Running model retention cleanup: removing models created before %s",
        cutoff.isoformat(),
    )

    storage = get_storage()
    deleted_count = 0

    async with async_session_factory() as session:
        result = await session.execute(
            select(TrainedModel).where(TrainedModel.created_at < cutoff)
        )
        expired = result.scalars().all()

        for model in expired:
            try:
                storage.delete(model.file_path)
            except Exception:
                logger.warning(
                    "Failed to delete storage object for model %s at %s",
                    model.id,
                    model.file_path,
                )

            # Also attempt to delete the companion config artifact.
            try:
                from app.storage import build_companion_config_identifier
                config_key = build_companion_config_identifier(model.file_path)
                storage.delete(config_key)
            except Exception:
                pass

            await session.delete(model)
            deleted_count += 1

        if deleted_count:
            await session.commit()

    logger.info("Model retention cleanup complete: deleted %s model(s)", deleted_count)
    return deleted_count
