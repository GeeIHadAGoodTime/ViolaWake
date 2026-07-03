"""Public, unauthenticated routes for violawake.com.

Currently one endpoint: anonymous bug reports. No login required — any site
visitor (a prospect reading /compare/picovoice, not just a signed-in
customer) can flag a problem. Honeypot-guarded and rate-limited to resist
spam without a CAPTCHA.

Reports are forwarded to Sentry, which this backend already initializes in
``app.middleware.init_sentry`` whenever ``VIOLAWAKE_SENTRY_DSN`` is set — so
a report becomes an ordinary Sentry issue, and the founder sees it wherever
they already watch backend errors. There is no separate inbox to check.

If Sentry is unavailable at request time (not installed, not configured, or
the capture call itself fails) the endpoint returns 503 rather than a false
"received" 200 — a report must never look sent when it silently went
nowhere.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.rate_limit import BUG_REPORT_LIMIT, get_public_client_ip, limiter
from app.schemas import BugReportRequest, BugReportResponse

logger = logging.getLogger("violawake.public")

router = APIRouter(prefix="/api/public", tags=["public"])


def _capture_bug_report(*, message: str, page: str | None, client_ip: str) -> str | None:
    """Send a bug report to Sentry as a message event.

    Returns the Sentry event id, or ``None`` if Sentry isn't installed, isn't
    initialized (no DSN configured), or the capture call itself raised.
    """
    try:
        import sentry_sdk
    except ImportError:
        logger.error("sentry-sdk not installed; cannot capture web bug report")
        return None

    if not sentry_sdk.is_initialized():
        logger.error(
            "Sentry not initialized (no VIOLAWAKE_SENTRY_DSN); cannot capture web bug report"
        )
        return None

    try:
        with sentry_sdk.push_scope() as scope:
            scope.set_tag("source", "public_bug_report")
            scope.set_tag("page", page or "unknown")
            scope.set_extra("client_ip", client_ip)
            event_id = sentry_sdk.capture_message(f"[Bug report] {message}", level="info")
    except Exception:  # noqa: BLE001 - capture failures must not crash the request
        logger.exception("Sentry capture raised while submitting web bug report")
        return None

    return event_id


@router.post("/bug-report", response_model=BugReportResponse)
@limiter.limit(BUG_REPORT_LIMIT)
async def submit_bug_report(request: Request, body: BugReportRequest) -> BugReportResponse:
    """Accept an anonymous bug report from any violawake.com page.

    No auth required. Rate-limited at 3 req/IP/min (see BUG_REPORT_LIMIT) and
    honeypot-guarded: bots that fill the hidden ``website`` field get a
    silent success response so they can't tell the report was discarded.

    The parameter is named ``request`` (the Starlette ``Request``) rather
    than ``http_request`` deliberately: slowapi's ``@limiter.limit`` looks up
    the request object by that exact parameter name.
    """
    client_ip = get_public_client_ip(request)

    if body.website:
        logger.debug("Honeypot triggered on /api/public/bug-report from %s", client_ip)
        return BugReportResponse()

    safe_message = body.message.strip()
    safe_page = body.page.strip()[:200] or None

    event_id = _capture_bug_report(message=safe_message, page=safe_page, client_ip=client_ip)

    if event_id is None:
        logger.error(
            "Web bug report could not be captured (Sentry unavailable) from ip=%s page=%s",
            client_ip,
            safe_page or "unknown",
        )
        raise HTTPException(
            status_code=503, detail="Bug report could not be submitted. Please try again."
        )

    logger.info(
        "Web bug report captured sentry_event_id=%s ip=%s page=%s",
        event_id,
        client_ip,
        safe_page or "unknown",
    )
    return BugReportResponse()
