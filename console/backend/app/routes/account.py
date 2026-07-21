"""Account data export and deletion routes."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.config import settings
from app.database import get_db
from app.job_queue import get_job_queue
from app.models import Recording, TrainedModel, User
from app.routes import auth as auth_routes
from app.schemas import DeleteAccountRequest, MessageResponse
from app.storage import get_storage

router = APIRouter(prefix="/api/account", tags=["account"])


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_json(zf: zipfile.ZipFile, path: str, payload: Any) -> None:
    zf.writestr(
        path,
        json.dumps(payload, indent=2, default=_json_default, sort_keys=True),
    )


def _artifact_name(prefix: str, item_id: int, storage_key: str) -> str:
    filename = PurePosixPath(storage_key.replace("\\", "/")).name or f"{item_id}.bin"
    return f"{prefix}/{item_id}_{filename}"


def _audit_log_entries(user_id: int) -> list[dict[str, Any]]:
    candidates = [
        settings.upload_volume_path / "logs" / "uploads.jsonl",
        settings.data_dir / "logs" / "uploads.jsonl",
    ]
    entries: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("user_id") == user_id:
                entries.append(payload)
        break
    return entries


async def _queue_jobs_for_user(user_id: int) -> list[dict[str, Any]]:
    try:
        queue = get_job_queue()
    except RuntimeError:
        return []
    jobs = await queue.list_jobs(user_id)
    return [
        {
            "id": job.id,
            "wake_word": job.wake_word,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error": job.error,
            "progress_pct": job.progress_pct,
            "recording_ids": job.recording_ids,
            "epochs": job.epochs,
            "model_id": job.model_id,
            "d_prime": job.d_prime,
            "priority": job.priority,
        }
        for job in jobs
    ]


@router.get("/export")
async def export_account(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Export account, recording, model, job, and audit data as a ZIP file."""
    storage = get_storage()
    buffer = io.BytesIO()

    recording_rows = (
        await db.execute(select(Recording).where(Recording.user_id == current_user.id))
    ).scalars().all()
    model_rows = (
        await db.execute(select(TrainedModel).where(TrainedModel.user_id == current_user.id))
    ).scalars().all()

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_json(
            zf,
            "user.json",
            {
                "id": current_user.id,
                "email": current_user.email,
                "name": current_user.name,
                "email_verified": current_user.email_verified,
                "created_at": current_user.created_at,
                "deleted_at": current_user.deleted_at,
                "scheduled_hard_delete_at": current_user.scheduled_hard_delete_at,
            },
        )

        recording_manifest = []
        for recording in recording_rows:
            artifact_path = _artifact_name("recordings", recording.id, recording.file_path)
            available = False
            try:
                zf.writestr(artifact_path, storage.download(recording.file_path))
                available = True
            except Exception:
                pass
            recording_manifest.append(
                {
                    "id": recording.id,
                    "wake_word": recording.wake_word,
                    "filename": recording.filename,
                    "display_name": recording.display_name,
                    "file_path": recording.file_path,
                    "duration_s": recording.duration_s,
                    "sample_rate": recording.sample_rate,
                    "size_bytes": recording.size_bytes,
                    "uploaded_at": recording.uploaded_at,
                    "created_at": recording.created_at,
                    "deleted_at": recording.deleted_at,
                    "export_path": artifact_path if available else None,
                    "available": available,
                }
            )
        _write_json(zf, "recordings/manifest.json", recording_manifest)

        model_manifest = []
        for model in model_rows:
            artifact_path = _artifact_name("models", model.id, model.file_path)
            available = False
            try:
                zf.writestr(artifact_path, storage.download(model.file_path))
                available = True
            except Exception:
                pass
            model_manifest.append(
                {
                    "id": model.id,
                    "wake_word": model.wake_word,
                    "file_path": model.file_path,
                    "config_json": model.config_json,
                    "d_prime": model.d_prime,
                    "size_bytes": model.size_bytes,
                    "created_at": model.created_at,
                    "deleted_at": model.deleted_at,
                    "export_path": artifact_path if available else None,
                    "available": available,
                }
            )
        _write_json(zf, "models/manifest.json", model_manifest)

        _write_json(
            zf,
            "jobs.json",
            {
                # Training-job history is persisted only in the async job queue's
                # SQLite store; ``queue_jobs`` is the complete, authoritative export
                # (see issue #1438 — the Postgres ``training_jobs`` table was never
                # written at runtime and has been retired).
                "queue_jobs": await _queue_jobs_for_user(current_user.id),
            },
        )
        _write_json(zf, "audit_log.json", _audit_log_entries(current_user.id))

    filename = f"violawake-account-{current_user.id}-export.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("", response_model=MessageResponse)
async def delete_account(
    body: DeleteAccountRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Delete the current account through the GDPR soft-delete flow."""
    return await auth_routes.delete_account(body, current_user, db)
