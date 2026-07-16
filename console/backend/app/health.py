"""Production health checks for ViolaWake Console."""

from __future__ import annotations

import secrets
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
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
from app.storage import get_storage

router = APIRouter(prefix="/api/health", tags=["health"])


async def require_admin_health_details(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """Hide the detailed health endpoint unless the configured admin token matches."""
    admin_token = getattr(settings, "admin_token", "")
    if not admin_token or x_admin_token is None or not secrets.compare_digest(x_admin_token, admin_token):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")


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

    blocked_pending_jobs = int(queue_snapshot.get("blocked_pending_jobs", 0))
    oldest_pending_age_s = queue_snapshot.get("oldest_pending_age_s")
    oldest_dispatchable_pending_age_s = queue_snapshot.get("oldest_dispatchable_pending_age_s")

    # Honest wedge detection (issue #1481). A non-empty queue is NORMAL: work is
    # dispatched near-instantly whenever a worker slot frees up, so a backlog
    # while every slot is busy, and jobs blocked only by a paused/backoff
    # circuit breaker (which no worker can pick up until the user resumes), are
    # both healthy. The queue is genuinely WEDGED -- and must page rather than
    # sit silent for hours -- only when the dispatcher fails to pick up runnable
    # work: the worker task died, more jobs are running than slots allow, or a
    # *dispatchable* pending job has waited past the wedge threshold while a slot
    # is free. The pre-fix "any queue_depth > 0 => degraded" rule both
    # false-alarmed on ordinary backlog and pinned the backend unhealthy forever
    # on a single paused-breaker job, making the signal useless for paging.
    wedge_threshold_s = settings.training_queue_wedge_threshold_seconds
    component_status = HEALTH_STATUS_OK
    wedge_reason: str | None = None
    if not worker_status["worker_task_running"]:
        component_status = HEALTH_STATUS_ERROR
        wedge_reason = "worker task not running"
    elif worker_status["active_workers"] > worker_status["max_workers"]:
        component_status = HEALTH_STATUS_ERROR
        wedge_reason = "more running jobs than worker slots"
    elif (
        worker_status["available_slots"] > 0
        and oldest_dispatchable_pending_age_s is not None
        and oldest_dispatchable_pending_age_s > wedge_threshold_s
    ):
        component_status = HEALTH_STATUS_ERROR
        wedge_reason = (
            f"dispatchable job pending {oldest_dispatchable_pending_age_s:.0f}s "
            f"(> {wedge_threshold_s}s) with {worker_status['available_slots']} free slot(s)"
        )

    result: dict[str, Any] = {
        "status": component_status,
        "queue_depth": queue_depth,
        "blocked_pending_jobs": blocked_pending_jobs,
        "oldest_pending_age_s": oldest_pending_age_s,
        "oldest_dispatchable_pending_age_s": oldest_dispatchable_pending_age_s,
        "wedge_threshold_s": wedge_threshold_s,
        "worker_status": worker_status,
    }
    if wedge_reason is not None:
        result["wedge_reason"] = wedge_reason
    return result


def _check_storage() -> dict[str, Any]:
    upload_dir = _check_directory(settings.upload_dir)
    models_dir = _check_directory(settings.models_dir)
    backend: dict[str, Any] = {
        "status": HEALTH_STATUS_OK,
        "roundtrip": True,
        "error": None,
    }
    health_key = "recordings/_health/check/storage-health.txt"
    try:
        storage = get_storage()
        storage.upload(health_key, b"ok", "text/plain")
        if storage.download(health_key) != b"ok":
            backend["status"] = HEALTH_STATUS_ERROR
            backend["roundtrip"] = False
            backend["error"] = "Storage roundtrip returned unexpected bytes"
        storage.delete(health_key)
    except Exception as exc:
        backend["status"] = HEALTH_STATUS_ERROR
        backend["roundtrip"] = False
        backend["error"] = str(exc)

    component_status = _combine_statuses(upload_dir["status"], models_dir["status"], backend["status"])
    return {
        "status": component_status,
        "upload_dir": upload_dir,
        "models_dir": models_dir,
        "backend": backend,
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


def _failed_checks_from_payload(payload: dict[str, Any]) -> list[str]:
    failed: list[str] = []

    if not payload["ready"]:
        failed.append("startup")

    def walk(path: str, value: Any) -> None:
        if not isinstance(value, dict):
            return
        component_status = value.get("status")
        if component_status in {HEALTH_STATUS_DEGRADED, HEALTH_STATUS_ERROR}:
            failed.append(path)
        for key, nested in value.items():
            if isinstance(nested, dict):
                walk(f"{path}.{key}", nested)

    for name, component in payload["components"].items():
        walk(name, component)

    return sorted(set(failed))


def _summary_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "status": payload["status"],
        "uptime_s": payload["uptime_s"],
        "ready": payload["ready"],
        "version": payload["version"],
    }
    failed_checks = _failed_checks_from_payload(payload)
    if failed_checks:
        summary["failed_checks"] = failed_checks
    return summary


def _health_http_status(payload: dict[str, Any]) -> int:
    if payload["ready"] and payload["status"] == HEALTH_STATUS_OK:
        return status.HTTP_200_OK
    return status.HTTP_503_SERVICE_UNAVAILABLE


@router.get("")
async def health(request: Request) -> JSONResponse:
    payload = await build_health_payload(request.app)
    return JSONResponse(
        status_code=_health_http_status(payload),
        content=_summary_from_payload(payload),
    )


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
        status_code=_health_http_status(payload),
        content=_summary_from_payload(payload),
    )


@router.get("/details")
async def health_details(
    request: Request,
    _: None = Depends(require_admin_health_details),
) -> dict[str, Any]:
    return await build_health_payload(request.app)
