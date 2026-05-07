"""Shared pytest setup for console backend tests.

Direct route-call tests bypass FastAPI's middleware and pass lightweight
request doubles instead of `starlette.requests.Request`. slowapi's
`@limiter.limit(...)` decorator validates the request type, so we disable
the limiter for the entire test suite. Real rate-limit behavior is covered
by integration tests against a running ASGI app, not these unit tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.rate_limit import limiter

limiter.enabled = False
