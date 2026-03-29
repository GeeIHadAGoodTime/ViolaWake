"""FastAPI application for ViolaWake Console backend."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.database import init_db
from app.health import router as health_router
from app.job_queue import init_job_queue, shutdown_job_queue
from app.middleware import (
    ErrorHandlingMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    configure_logging,
    init_sentry,
    register_exception_handlers,
)
from app.monitoring import APP_VERSION, init_monitoring_state, log_exception, mark_startup_complete
from app.rate_limit import limiter
from app.routes import auth, billing, files, jobs, models, recordings, teams, training

configure_logging()
logger = logging.getLogger("violawake.console")
init_sentry()

_RETENTION_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


async def _retention_loop() -> None:
    """Run retention cleanup every 24 hours in the background."""
    from app.retention import (
        cleanup_expired_models,
        cleanup_expired_recordings,
        cleanup_soft_deleted_recordings,
    )

    while True:
        try:
            await cleanup_soft_deleted_recordings()
            await cleanup_expired_recordings()
            await cleanup_expired_models()
        except Exception as exc:
            log_exception(logger, exc, message="Retention cleanup cycle failed", source="retention")

        await asyncio.sleep(_RETENTION_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialize DB and mark the app ready."""
    init_monitoring_state(app)
    logger.info(
        "%s",
        "Starting ViolaWake Console backend",
        extra={"event_data": {"source": "startup", "database_target": settings.database_log_target}},
    )

    retention_task: asyncio.Task[None] | None = None
    try:
        await init_db()
        await init_job_queue()

        # Eagerly initialize the email service so the "Resend not configured"
        # warning is logged at startup rather than on the first registration.
        from app.email_service import get_email_service
        email_svc = get_email_service()
        if not email_svc.enabled:
            logger.warning(
                "Email service is disabled (VIOLAWAKE_RESEND_API_KEY not set). "
                "Users will be auto-verified on registration."
            )

        retention_task = asyncio.create_task(_retention_loop(), name="retention-cleanup")

        mark_startup_complete(app)
        logger.info(
            "%s",
            "ViolaWake Console backend ready",
            extra={"event_data": {"source": "startup", "version": APP_VERSION}},
        )
        yield
    except Exception as exc:
        log_exception(logger, exc, message="Application startup failed", source="startup")
        raise
    finally:
        if retention_task is not None:
            retention_task.cancel()
            with suppress(asyncio.CancelledError):
                await retention_task
        await shutdown_job_queue()
        logger.info("%s", "Shutting down ViolaWake Console backend", extra={"event_data": {"source": "shutdown"}})


app = FastAPI(
    title="ViolaWake Console",
    description="Backend API for ViolaWake wake word training console",
    version=APP_VERSION,
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
init_monitoring_state(app)
register_exception_handlers(app)

app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.effective_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

app.include_router(health_router)
app.include_router(auth.router)
app.include_router(recordings.router)
app.include_router(jobs.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(billing.router)
app.include_router(files.router)
app.include_router(teams.router)


# ---------------------------------------------------------------------------
# Admin endpoint: POST /api/admin/cleanup
# ---------------------------------------------------------------------------
# Guarded by VIOLAWAKE_ADMIN_TOKEN.  When the env var is not set (development
# default), the endpoint is disabled and returns 404 so it is never reachable
# without explicit configuration.
# ---------------------------------------------------------------------------

from fastapi import APIRouter, Depends, Header, HTTPException, status  # noqa: E402

_admin_router = APIRouter(prefix="/api/admin", tags=["admin"])


async def _require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    """Dependency that verifies the X-Admin-Token header against config."""
    admin_token: str = getattr(settings, "admin_token", "")
    if not admin_token:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if x_admin_token != admin_token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


@_admin_router.post("/cleanup")
async def trigger_cleanup(
    _: None = Depends(_require_admin),
) -> dict[str, int]:
    """Manually trigger retention cleanup for recordings and models.

    Requires ``VIOLAWAKE_ADMIN_TOKEN`` to be set in the environment.
    Protected by the ``X-Admin-Token`` request header.
    """
    from app.retention import (
        cleanup_expired_models,
        cleanup_expired_recordings,
        cleanup_soft_deleted_recordings,
    )

    soft_deleted_purged = await cleanup_soft_deleted_recordings()
    recordings_deleted = await cleanup_expired_recordings()
    models_deleted = await cleanup_expired_models()
    logger.info(
        "Admin cleanup triggered: %s soft-deleted recording(s) purged, %s recording(s) and %s model(s) deleted",
        soft_deleted_purged,
        recordings_deleted,
        models_deleted,
    )
    return {
        "soft_deleted_recordings_purged": soft_deleted_purged,
        "recordings_deleted": recordings_deleted,
        "models_deleted": models_deleted,
    }


app.include_router(_admin_router)
