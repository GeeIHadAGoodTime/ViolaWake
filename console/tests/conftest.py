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

import pytest

from app.rate_limit import limiter
from app.services import speech_readiness

limiter.enabled = False


@pytest.fixture(autouse=True)
def _speech_is_up_unless_a_test_says_otherwise(request, monkeypatch):
    """Submitting a job asks whether speech works, which is a real network call."""
    if "test_training_speech_outage" in request.node.nodeid:
        return
    healthy = speech_readiness.SpeechReadiness(True, "edge_tts", float("inf"))
    monkeypatch.setattr(speech_readiness, "_cached", healthy)
