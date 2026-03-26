"""FastAPI application for ViolaWake Console backend."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
from app.routes import auth, billing, files, jobs, models, recordings, training

configure_logging()
logger = logging.getLogger("violawake.console")
init_sentry()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialize DB and mark the app ready."""
    init_monitoring_state(app)
    logger.info(
        "%s",
        "Starting ViolaWake Console backend",
        extra={"event_data": {"source": "startup", "database_target": settings.database_log_target}},
    )

    try:
        await init_db()
        await init_job_queue()

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
        await shutdown_job_queue()
        logger.info("%s", "Shutting down ViolaWake Console backend", extra={"event_data": {"source": "shutdown"}})


app = FastAPI(
    title="ViolaWake Console",
    description="Backend API for ViolaWake wake word training console",
    version=APP_VERSION,
    lifespan=lifespan,
)
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
