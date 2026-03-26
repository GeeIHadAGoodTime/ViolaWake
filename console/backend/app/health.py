"""Production health checks for ViolaWake Console."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.database import async_session_factory
from app.job_queue import get_job_queue
from app.monitoring import (
    APP_VERSION,
    ERROR_TRACKER,
    HEALTH_STATUS_DEGRADED,
    HEALTH_STATUS_ERROR,
    HEALTH_STATUS_OK,
    get_uptime_seconds,
)

router = APIRouter(prefix="/api/health", tags=["health"])


def _combine_statuses(*statuses: str) -> str:
    if any(component_status == HEALTH_STATUS_ERROR for component_status in statuses):
        return HEALTH_STATUS_ERROR
    if any(component_status == HEALTH_STATUS_DEGRADED for component_status in statuses):
        return HEALTH_STATUS_DEGRADED
    return HEALTH_STATUS_OK


def _check_directory(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_dir = path.is_dir()
    writable = False
    error: str | None = None

    if exists and is_dir:
        try:
            with tempfile.NamedTemporaryFile(dir=path, prefix=".health-", delete=True):
                writable = True
        except OSError as exc:
            error = str(exc)
    else:
        error = "Directory is missing" if not exists else "Path is not a directory"

    return {
        "path": str(path),
        "exists": exists,
        "writable": writable,
        "status": HEALTH_STATUS_OK if exists and is_dir and writable else HEALTH_STATUS_ERROR,
        "error": error,
    }


async def _check_database() -> dict[str, Any]:
    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        return {
            "status": HEALTH_STATUS_ERROR,
            "connected": False,
            "target": settings.database_log_target,
            "error": str(exc),
        }

    return {
        "status": HEALTH_STATUS_OK,
        "connected": True,
        "target": settings.database_log_target,
    }


async def _check_training_queue() -> dict[str, Any]:
    try:
        queue_snapshot = await get_job_queue().runtime_snapshot()
    except RuntimeError as exc:
        return {
            "status": HEALTH_STATUS_ERROR,
            "queue_depth": 0,
            "worker_status": {
                "active_workers": 0,
                "max_workers": settings.max_concurrent_jobs,
                "available_slots": settings.max_concurrent_jobs,
                "worker_task_running": False,
            },
            "error": str(exc),
        }

    queue_depth = int(queue_snapshot["queue_depth"])
    worker_status = dict(queue_snapshot["worker_status"])
    worker_status["persisted_running_jobs"] = int(queue_snapshot["persisted_running_jobs"])

    component_status = HEALTH_STATUS_OK
    if not worker_status["worker_task_running"] or worker_status["active_workers"] > worker_status["max_workers"]:
        component_status = HEALTH_STATUS_ERROR
    elif queue_depth > 0:
        component_status = HEALTH_STATUS_DEGRADED

    return {
        "status": component_status,
        "queue_depth": queue_depth,
        "worker_status": worker_status,
    }


def _check_storage() -> dict[str, Any]:
    upload_dir = _check_directory(settings.upload_dir)
    models_dir = _check_directory(settings.models_dir)
    component_status = _combine_statuses(upload_dir["status"], models_dir["status"])
    return {
        "status": component_status,
        "upload_dir": upload_dir,
        "models_dir": models_dir,
    }


def _check_billing() -> dict[str, Any]:
    configured = bool(settings.stripe_secret_key)
    return {
        "status": HEALTH_STATUS_OK if configured else HEALTH_STATUS_DEGRADED,
        "configured": configured,
    }


async def build_health_payload(app: Any) -> dict[str, Any]:
    database = await _check_database()
    training_queue = await _check_training_queue()
    storage = _check_storage()
    billing = _check_billing()

    startup_complete = bool(getattr(app.state, "startup_complete", False))
    ready = startup_complete and database["status"] == HEALTH_STATUS_OK

    status_value = _combine_statuses(
        database["status"],
        training_queue["status"],
        storage["status"],
        billing["status"],
    )
    if not ready:
        status_value = HEALTH_STATUS_ERROR

    return {
        "status": status_value,
        "uptime_s": get_uptime_seconds(app),
        "ready": ready,
        "version": APP_VERSION,
        "startup_complete": startup_complete,
        "components": {
            "database": database,
            "training_queue": training_queue,
            "storage": storage,
            "billing": billing,
        },
        "recent_errors": ERROR_TRACKER.snapshot(),
    }


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": payload["status"],
        "uptime_s": payload["uptime_s"],
        "ready": payload["ready"],
        "version": payload["version"],
    }


@router.get("")
async def health(request: Request) -> dict[str, Any]:
    payload = await build_health_payload(request.app)
    return _summary_from_payload(payload)


@router.get("/live")
async def live(request: Request) -> dict[str, Any]:
    return {
        "status": HEALTH_STATUS_OK,
        "uptime_s": get_uptime_seconds(request.app),
        "ready": bool(getattr(request.app.state, "startup_complete", False)),
        "version": APP_VERSION,
    }


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    payload = await build_health_payload(request.app)
    return JSONResponse(
        status_code=status.HTTP_200_OK if payload["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=_summary_from_payload(payload),
    )


@router.get("/details")
async def health_details(request: Request) -> dict[str, Any]:
    return await build_health_payload(request.app)
