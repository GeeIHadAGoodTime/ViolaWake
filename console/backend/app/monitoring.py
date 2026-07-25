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
    # True when this classification must still produce a durable GlitchTip/Sentry
    # dashboard signal even though log_level is deliberately kept below the
    # Sentry LoggingIntegration's default event_level=ERROR (so it does NOT page
    # ops via automatic log capture). Without this, a WARNING-or-below
    # classification is invisible in GlitchTip: the LoggingIntegration only
    # auto-captures an *event* (dashboard-visible issue) at/above event_level; a
    # WARNING becomes a breadcrumb only, attached to nothing (#1482). See
    # _emit_dashboard_signal.
    dashboard_signal: bool = False


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

    # A model that fails the deployment quality gate (grade F) is an EXPECTED
    # outcome for weak/insufficient user recordings -- the training job is
    # correctly marked failed and the user is told, but it is NOT a code bug and
    # must not page ops via Sentry. Matched by class name across the MRO so this
    # stays decoupled from the (heavy) violawake_sdk import; the SDK raises
    # violawake_sdk.tools.train.ModelQualityGateError (GlitchTip violawake issue
    # 28, 2026-07-12). WARNING is below the Sentry LoggingIntegration ERROR
    # event_level, so it becomes a breadcrumb, not a captured event.
    if any(base.__name__ == "ModelQualityGateError" for base in type(exc).__mro__):
        return ErrorClassification(
            EXPECTED_ERROR, "model_quality", logging.WARNING, dashboard_signal=True
        )

    # The mirror image of the above (#1775): the quality gate could not build
    # enough of its OWN negative test material (TTS outage / voice retired
    # server-side) and refused to grade rather than blaming the user. That is
    # OUR infrastructure failing, so unlike a grade-F verdict it SHOULD page.
    if any(base.__name__ == "QualityGateUnavailableError" for base in type(exc).__mro__):
        return ErrorClassification(
            UNEXPECTED_ERROR, "tts_unavailable", logging.ERROR, dashboard_signal=True
        )

    if isinstance(exc, ValueError):
        return ErrorClassification(UNEXPECTED_ERROR, "data", logging.WARNING)

    if isinstance(exc, OSError):
        return ErrorClassification(UNEXPECTED_ERROR, "config", logging.ERROR)

    return ErrorClassification(UNEXPECTED_ERROR, "bug", logging.ERROR)


def _emit_dashboard_signal(
    classification: ErrorClassification,
    *,
    source: str,
    error_type: str,
    error_message: str,
    extra: dict[str, Any] | None,
) -> None:
    """Explicitly capture a Sentry/GlitchTip event for a classification that is
    intentionally logged below the Sentry LoggingIntegration's default
    event_level (ERROR) so it does not page ops (see classify_exception's
    model_quality branch, GlitchTip violawake issue 28).

    ``sentry_sdk.capture_message`` is a direct, explicit API call -- unlike the
    LoggingIntegration's automatic capture from stdlib ``logging`` calls (which
    is filtered by ``event_level``), an explicit capture ALWAYS creates a
    dashboard-visible event when Sentry is initialized, regardless of the
    ``level=`` passed. That keeps a durable GlitchTip signal (an issue whose
    event count keeps moving) for outcomes we deliberately keep at WARNING so
    they don't page -- the signal PR#5 silently removed (#1482). A stable
    fingerprint groups every occurrence into one long-lived issue so its event
    count is the at-a-glance block-rate indicator (mirrors the old issue 28
    role), instead of minting a fresh issue per message.

    Never raises: a broken/unconfigured Sentry must not break the caller's
    exception-handling path.
    """
    try:
        import sentry_sdk
    except ImportError:
        return

    try:
        if not sentry_sdk.is_initialized():
            return

        with sentry_sdk.push_scope() as scope:
            scope.set_tag("source", source)
            scope.set_tag("error_reason", classification.reason)
            scope.set_tag("error_type", error_type)
            scope.fingerprint = ["dashboard-signal", classification.reason, error_type]
            if extra:
                for key, value in extra.items():
                    scope.set_extra(key, value)
            sentry_sdk.capture_message(
                f"[{classification.reason}] {source}: {error_message}",
                level=logging.getLevelName(classification.log_level).lower(),
            )
    except Exception:  # noqa: BLE001 - a dashboard-signal failure must not break the caller
        logging.getLogger("violawake.console").exception(
            "Sentry dashboard-signal capture raised",
            extra={"event_data": {"source": "sentry", "error_reason": classification.reason}},
        )


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

    if classification.dashboard_signal:
        _emit_dashboard_signal(
            classification,
            source=source,
            error_type=type(exc).__name__,
            error_message=str(exc),
            extra=extra,
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
