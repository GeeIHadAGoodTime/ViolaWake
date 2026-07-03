"""Tests for the anonymous public bug-report endpoint.

Covers: honeypot silent-discard, request validation, rate limiting, and the
"never silently drop a report" contract (Sentry unavailable -> 503, not a
false-success 200).

Note: `conftest.py` disables the shared slowapi limiter for the whole
`console/tests/` suite (direct route-call tests elsewhere depend on that).
The rate-limit test here re-enables it for the duration of that one test and
restores the module default afterward, so it doesn't leak into other tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

ROUTE = "/api/public/bug-report"


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_bug_report_rate_limit():
    """Isolate this file's rate-limit budget from other test modules."""
    from app.rate_limit import reset_rate_limits

    reset_rate_limits()
    yield
    reset_rate_limits()


@pytest.fixture
def sentry_initialized(monkeypatch):
    """Simulate a configured, working Sentry sink and capture what it was sent."""
    import sentry_sdk

    captured: dict[str, object] = {}

    class _FakeScope:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def set_tag(self, key, value):
            captured.setdefault("tags", {})[key] = value

        def set_extra(self, key, value):
            captured.setdefault("extras", {})[key] = value

    def _fake_push_scope():
        return _FakeScope()

    def _fake_capture_message(message, level="info"):
        captured["message"] = message
        captured["level"] = level
        return "fake-event-id-123"

    monkeypatch.setattr(sentry_sdk, "is_initialized", lambda: True)
    monkeypatch.setattr(sentry_sdk, "push_scope", _fake_push_scope)
    monkeypatch.setattr(sentry_sdk, "capture_message", _fake_capture_message)
    return captured


def test_bug_report_success_reaches_sentry(client, sentry_initialized) -> None:
    response = client.post(
        ROUTE,
        json={"message": "Clicking Download did nothing on Safari.", "page": "pricing"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["result"] == "received"
    assert "Clicking Download did nothing on Safari." in sentry_initialized["message"]
    assert sentry_initialized["tags"]["page"] == "pricing"


def test_bug_report_honeypot_silently_discards(client, sentry_initialized) -> None:
    response = client.post(
        ROUTE,
        json={
            "message": "This is definitely a bot filling every field it can find.",
            "page": "pricing",
            "website": "http://spam.example",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["result"] == "received"
    # Bot gets an identical-looking success, but nothing was actually sent.
    assert "message" not in sentry_initialized


def test_bug_report_rejects_short_message(client) -> None:
    response = client.post(ROUTE, json={"message": "too short"})

    assert response.status_code == 422


def test_bug_report_rejects_oversized_message(client) -> None:
    response = client.post(ROUTE, json={"message": "x" * 2001})

    assert response.status_code == 422


def test_bug_report_missing_message_field(client) -> None:
    response = client.post(ROUTE, json={"page": "pricing"})

    assert response.status_code == 422


def test_bug_report_fails_loudly_when_sentry_unavailable(client, monkeypatch) -> None:
    """A report must never look "sent" when the sink can't take it."""
    import sentry_sdk

    monkeypatch.setattr(sentry_sdk, "is_initialized", lambda: False)

    response = client.post(
        ROUTE,
        json={"message": "Nobody should see a fake success for this one."},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Bug report could not be submitted. Please try again."


def test_bug_report_rate_limited_after_three_per_minute(client, sentry_initialized) -> None:
    from app.rate_limit import limiter

    previously_enabled = limiter.enabled
    limiter.enabled = True
    try:
        payload = {"message": "Same bug reported several times in a row here."}
        statuses = [client.post(ROUTE, json=payload).status_code for _ in range(4)]
    finally:
        limiter.enabled = previously_enabled

    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429
