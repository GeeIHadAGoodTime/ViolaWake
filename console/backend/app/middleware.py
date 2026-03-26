"""Structured logging, exception handling, and Sentry integration."""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, Request
from fastapi.exception_handlers import (
    http_exception_handler as fastapi_http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.config import settings
from app.monitoring import (
    APP_VERSION,
    is_health_request_path,
    log_exception,
    route_template_from_request,
)

logger = logging.getLogger("violawake.console")
request_logger = logging.getLogger("violawake.request")
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_SENSITIVE_FIELD_MARKERS = ("api_key", "authorization", "password", "secret", "token")


class RequestContextFilter(logging.Filter):
    """Attach request-scoped context to all log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("-")
        return True


class JsonFormatter(logging.Formatter):
    """Emit logs as JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }

        event_data = getattr(record, "event_data", None)
        if isinstance(event_data, dict):
            payload.update(event_data)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    """Configure root logging once with a JSON formatter."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = JsonFormatter()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(RequestContextFilter())
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
            if not any(isinstance(existing_filter, RequestContextFilter) for existing_filter in handler.filters):
                handler.addFilter(RequestContextFilter())

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers.clear()
    uvicorn_access_logger.propagate = False


def _scrub_sensitive_data(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            lowered_key = str(key).lower()
            if any(marker in lowered_key for marker in _SENSITIVE_FIELD_MARKERS):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = _scrub_sensitive_data(nested_value)
        return sanitized
    if isinstance(value, list):
        return [_scrub_sensitive_data(item) for item in value]
    return value


def init_sentry() -> None:
    """Initialize Sentry when a DSN is configured."""
    if not settings.sentry_dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
    except ImportError:
        logger.warning(
            "%s",
            "Sentry DSN configured but sentry-sdk is not installed",
            extra={"event_data": {"source": "sentry", "release": APP_VERSION}},
        )
        return

    def _before_send(event: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
        del hint
        return _scrub_sensitive_data(event)

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.env,
        release=APP_VERSION,
        integrations=[FastApiIntegration()],
        send_default_pii=False,
        before_send=_before_send,
    )

    with sentry_sdk.configure_scope() as scope:
        scope.set_tag("environment", settings.env)
        scope.set_tag("release", APP_VERSION)

    logger.info(
        "%s",
        "Sentry initialized",
        extra={"event_data": {"source": "sentry", "environment": settings.env, "release": APP_VERSION}},
    )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return a clean 500 response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        try:
            return await call_next(request)
        except (HTTPException, StarletteHTTPException):
            raise
        except Exception as exc:
            request_id = getattr(request.state, "request_id", request_id_var.get("-"))
            log_exception(
                logger,
                exc,
                message="Unhandled request exception",
                source="request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "route": route_template_from_request(request),
                    "request_id": request_id,
                },
            )
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "request_id": request_id},
            )
            response.headers["X-Request-ID"] = request_id
            return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Add request IDs and structured request logging."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = uuid4().hex
        request.state.request_id = request_id
        request.state.request_started_at = datetime.now(timezone.utc)
        request_id_token = request_id_var.set(request_id)

        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(request_id_token)

        response.headers["X-Request-ID"] = request_id

        if not is_health_request_path(request.url.path):
            duration_ms = round(
                (datetime.now(timezone.utc) - request.state.request_started_at).total_seconds() * 1000,
                3,
            )
            request_logger.info(
                "%s",
                "request.complete",
                extra={
                    "event_data": {
                        "method": request.method,
                        "path": request.url.path,
                        "route": route_template_from_request(request),
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                        "request_id": request_id,
                    }
                },
            )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def register_exception_handlers(app: Any) -> None:
    """Register handlers that log expected request errors with classification."""

    async def _http_exception_handler(
        request: Request,
        exc: HTTPException | StarletteHTTPException,
    ) -> Response:
        log_exception(
            logger,
            exc,
            message="Handled request exception",
            source="request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "route": route_template_from_request(request),
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", request_id_var.get("-")),
            },
            include_traceback=False,
        )
        response = await fastapi_http_exception_handler(request, exc)
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", request_id_var.get("-"))
        return response

    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
        log_exception(
            logger,
            exc,
            message="Request validation failed",
            source="request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "route": route_template_from_request(request),
                "status_code": 422,
                "request_id": getattr(request.state, "request_id", request_id_var.get("-")),
            },
            include_traceback=False,
        )
        response = await request_validation_exception_handler(request, exc)
        response.headers["X-Request-ID"] = getattr(request.state, "request_id", request_id_var.get("-"))
        return response
