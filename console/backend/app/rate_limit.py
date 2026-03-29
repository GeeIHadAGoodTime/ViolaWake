"""Rate limiting via slowapi for console API routes."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

# ---------------------------------------------------------------------------
# Core limiter instance -- keyed by client IP by default
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

# ---------------------------------------------------------------------------
# Limit strings (centralised so routes and tests can reference them)
# ---------------------------------------------------------------------------
LOGIN_LIMIT = "5/minute"
REGISTER_LIMIT = "100/hour"
VERIFY_EMAIL_LIMIT = "20/5 minutes"
FORGOT_PASSWORD_LIMIT = "5/5 minutes"
RESET_PASSWORD_LIMIT = "10/5 minutes"
RECORDING_UPLOAD_LIMIT = "50/hour"
TRAINING_SUBMIT_LIMIT = "5/hour"


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


# ---------------------------------------------------------------------------
# Backwards-compat helpers (used by tests / other modules)
# ---------------------------------------------------------------------------
def reset_rate_limits() -> None:
    """Clear all in-memory rate-limit state.  Used by tests."""
    storage = getattr(getattr(limiter, "_limiter", None), "storage", None)
    if storage is not None and hasattr(storage, "reset"):
        storage.reset()
