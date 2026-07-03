"""Rate limiting via slowapi for console API routes."""

from __future__ import annotations

import math
import time

from fastapi import HTTPException, Request, Response, status
from limits import parse
from slowapi import Limiter
from slowapi.util import get_remote_address


def _client_ip_key(request: Request) -> str:
    """Return the real end-user IP for rate-limit keying.

    Behind Cloudflare (Pages + Tunnel), ``request.client.host`` is the
    Cloudflare edge IP, which collapses all end users into a small pool of
    addresses and effectively makes per-IP limits global. Cloudflare passes
    the real client IP in ``CF-Connecting-IP``; that header is trustworthy
    here because the backend is reachable ONLY through the Cloudflare
    Tunnel (no public origin port). Fall back to remote_addr for local/dev
    runs that bypass Cloudflare.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip()
    return get_remote_address(request)


# ---------------------------------------------------------------------------
# Core limiter instance -- keyed by real end-user IP (CF-Connecting-IP)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=_client_ip_key)

# ---------------------------------------------------------------------------
# Limit strings (centralised so routes and tests can reference them)
# ---------------------------------------------------------------------------
LOGIN_LIMIT = "5/minute"
REGISTER_LIMIT = "100/hour"
VERIFY_EMAIL_LIMIT = "20/5 minutes"
FORGOT_PASSWORD_LIMIT = "5/5 minutes"
RESET_PASSWORD_LIMIT = "10/5 minutes"
CHANGE_PASSWORD_LIMIT = "5/minute"
RECORDING_UPLOAD_LIMIT = "100/15 minutes"
TRAINING_SUBMIT_LIMIT = "5/hour"
CHECKOUT_LIMIT = "10/hour"
PORTAL_LIMIT = "10/hour"
TEAM_INVITE_LIMIT = "20/hour"
# Public bug-report form: no login, so keep it strict — enough for a genuine
# reporter to file a couple of tickets, tight enough to resist bulk spam
# without needing a CAPTCHA.
BUG_REPORT_LIMIT = "3/minute"


def get_public_client_ip(request: Request) -> str:
    """Public wrapper around the CF-aware IP resolver, for logging call sites
    outside the rate limiter itself (e.g. structured logs on a captured
    report) that want the same identity the limiter used."""
    return _client_ip_key(request)


# ---------------------------------------------------------------------------
# Custom key function for user-scoped limits
# ---------------------------------------------------------------------------
def key_by_user(request: Request) -> str:
    """Return the authenticated user's ID as the rate-limit key.

    Reads ``request.state.rate_limit_user_id`` which must be set by a
    FastAPI dependency that resolves *before* the route function is called.
    Falls back to client IP if the attribute is missing.
    """
    user_id: int | None = getattr(request.state, "rate_limit_user_id", None)
    if user_id is not None:
        return str(user_id)
    return get_remote_address(request)


def set_rate_limit_user(request: Request, user_id: int) -> None:
    """Stash the user id on the request so ``key_by_user`` can read it."""
    request.state.rate_limit_user_id = user_id


def consume_rate_limit(
    request: Request,
    *,
    limit_value: str,
    key: str,
    scope: str,
    cost: int = 1,
    response: Response | None = None,
) -> None:
    """Consume a rate-limit budget using the shared slowapi storage.

    slowapi's decorator supports a static/callable cost, but route bodies are
    not available to that callable. Bulk upload needs to count each file in a
    multipart request, so this helper spends the same limiter budget after the
    route has parsed the upload list.
    """
    if not limiter.enabled or cost <= 0:
        return

    limit = parse(limit_value)
    identifiers = [key, scope]
    key_prefix = getattr(limiter, "_key_prefix", None)
    if key_prefix:
        identifiers.insert(0, key_prefix)

    allowed = limiter._limiter.hit(limit, *identifiers, cost=cost)  # noqa: SLF001
    stats = limiter._limiter.get_window_stats(limit, *identifiers)  # noqa: SLF001
    retry_after = max(1, math.ceil(stats.reset_time - time.time()))
    rate_headers = {
        "X-RateLimit-Limit": str(limit.amount),
        "X-RateLimit-Remaining": str(max(0, stats.remaining)),
        "X-RateLimit-Reset": str(math.ceil(stats.reset_time)),
    }

    if response is not None:
        for name, value in rate_headers.items():
            response.headers[name] = value

    if not allowed:
        headers = {
            "Retry-After": str(retry_after),
            **rate_headers,
        }
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Upload rate limit exceeded. Please wait before uploading more recordings.",
            headers=headers,
        )


# ---------------------------------------------------------------------------
# Backwards-compat helpers (used by tests / other modules)
# ---------------------------------------------------------------------------
def reset_rate_limits() -> None:
    """Clear all in-memory rate-limit state.  Used by tests."""
    storage = getattr(getattr(limiter, "_limiter", None), "storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
