"""Security header middleware contract tests."""

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


@pytest.fixture(scope="module")
def client():
    from app.main import app

    return TestClient(app)


def test_security_headers_are_present(client) -> None:
    response = client.get("/api/health/live")

    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert response.headers["X-XSS-Protection"] == "1; mode=block"


def test_hsts_header_is_present_in_production(client, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(type(settings), "is_production", property(lambda self: True))

    response = client.get("/api/health/live")

    assert response.headers["Strict-Transport-Security"] == (
        "max-age=63072000; includeSubDomains; preload"
    )
