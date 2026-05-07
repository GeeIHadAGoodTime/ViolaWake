"""MaxBodySizeMiddleware upload boundary tests."""

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


def test_upload_declared_body_over_15mb_returns_413(client) -> None:
    from app.middleware import MAX_BODY_SIZE

    response = client.post(
        "/api/recordings/upload",
        content=b"x" * (MAX_BODY_SIZE + 1),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body too large"


def test_upload_declared_body_under_15mb_reaches_route_or_auth(client) -> None:
    from app.middleware import MAX_BODY_SIZE

    response = client.post(
        "/api/recordings/upload",
        content=b"x" * (MAX_BODY_SIZE - 1),
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code in (401, 422)
    assert response.status_code != 413
