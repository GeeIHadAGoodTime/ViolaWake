"""Shared monitoring primitives for health checks and error tracking."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter, deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


EXPECTED_ERROR = "expected"
UNEXPECTED_ERROR = "unexpected"
HEALTH_STATUS_OK = "ok"
HEALTH_STATUS_DEGRADED = "degraded"
HEALTH_STATUS_ERROR = "error"


@dataclass(frozen=True)
class ErrorClassification:
    """Normalized classification for application errors."""

    kind: str
    reason: str
    log_level: int


class ErrorTracker:
    """Track a bounded history of recent application errors in memory."""

    def __init__(self, max_events: int = 200) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = Lock()

    def record(
        self,
        classification: ErrorClassification,
        *,
        source: str,
        error_type: str,
        error_message: str,
    ) -> None:
        event = {
            "timestamp": time.time(),
            "kind": classification.kind,
            "reason": classification.reason,
            "source": source,
            "error_type": error_type,
            "error_message": error_message,
        }
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            events = list(self._events)

        by_kind = Counter(event["kind"] for event in events)
        by_reason = Counter(event["reason"] for event in events)

        return {
            "count": len(events),
            "expected": by_kind.get(EXPECTED_ERROR, 0),
            "unexpected": by_kind.get(UNEXPECTED_ERROR, 0),
            "by_reason": dict(sorted(by_reason.items())),
        }


def _load_project_version() -> str:
    pyproject_path = settings.base_dir.parent.parent / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as pyproject_file:
            payload = tomllib.load(pyproject_file)
    except (OSError, tomllib.TOMLDecodeError):
        return "0.0.0"

    project = payload.get("project", {})
    version = project.get("version")
    if isinstance(version, str) and version.strip():
        return version.strip()
    return "0.0.0"


APP_VERSION = _load_project_version()
ERROR_TRACKER = ErrorTracker()


def classify_exception(exc: Exception) -> ErrorClassification:
    """Classify exceptions into expected vs unexpected buckets."""
    if isinstance(exc, RequestValidationError):
        return ErrorClassification(EXPECTED_ERROR, "user_input", logging.INFO)

    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        detail = str(getattr(exc, "detail", "")).lower()
        if exc.status_code == 429:
            return ErrorClassification(EXPECTED_ERROR, "rate_limit", logging.INFO)
        if exc.status_code in (408, 504):
            return ErrorClassification(EXPECTED_ERROR, "timeout", logging.INFO)
        if exc.status_code == 503 and (
            "queue is full" in detail or "maximum training capacity" in detail
        ):
            return ErrorClassification(EXPECTED_ERROR, "rate_limit", logging.INFO)
        if 400 <= exc.status_code < 500:
            return ErrorClassification(EXPECTED_ERROR, "user_input", logging.INFO)
        if exc.status_code == 503:
            return ErrorClassification(UNEXPECTED_ERROR, "config", logging.WARNING)
        return ErrorClassification(UNEXPECTED_ERROR, "bug", logging.ERROR)

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorClassification(EXPECTED_ERROR, "timeout", logging.INFO)

    if isinstance(exc, ValueError):
        return ErrorClassification(UNEXPECTED_ERROR, "data", logging.WARNING)

    if isinstance(exc, OSError):
        return ErrorClassification(UNEXPECTED_ERROR, "config", logging.ERROR)

    return ErrorClassification(UNEXPECTED_ERROR, "bug", logging.ERROR)


def log_exception(
    logger: logging.Logger,
    exc: Exception,
    *,
    message: str,
    source: str,
    extra: dict[str, Any] | None = None,
    include_traceback: bool | None = None,
) -> ErrorClassification:
    """Classify, track, and log an exception using structured fields."""
    classification = classify_exception(exc)
    ERROR_TRACKER.record(
        classification,
        source=source,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )

    event_data: dict[str, Any] = {
        "source": source,
        "error_kind": classification.kind,
        "error_reason": classification.reason,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    if extra:
        event_data.update(extra)

    should_include_traceback = include_traceback
    if should_include_traceback is None:
        should_include_traceback = classification.kind == UNEXPECTED_ERROR

    logger.log(
        classification.log_level,
        "%s",
        message,
        extra={"event_data": event_data},
        exc_info=(type(exc), exc, exc.__traceback__) if should_include_traceback else False,
    )
    return classification


def init_monitoring_state(app: Any) -> None:
    """Initialize per-process monitoring state on the FastAPI app."""
    if not hasattr(app.state, "started_at_monotonic"):
        app.state.started_at_monotonic = time.monotonic()
    if not hasattr(app.state, "startup_complete"):
        app.state.startup_complete = False


def mark_startup_complete(app: Any) -> None:
    """Mark application startup as complete."""
    app.state.startup_complete = True


def get_uptime_seconds(app: Any) -> float:
    """Return process uptime as seconds since app initialization."""
    started_at = getattr(app.state, "started_at_monotonic", time.monotonic())
    return round(max(time.monotonic() - started_at, 0.0), 3)


def is_health_request_path(path: str) -> bool:
    """Return True when a request path targets health endpoints."""
    return path == "/api/health" or path.startswith("/api/health/")


def route_template_from_request(request: Any) -> str:
    """Return a normalized route template when available."""
    route = request.scope.get("route")
    if isinstance(route, APIRoute):
        return route.path
    return request.url.path
