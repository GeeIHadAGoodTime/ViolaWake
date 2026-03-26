"""Tests for health endpoints, monitoring helpers, and middleware headers."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

try:
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client for the backend app."""
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    try:
        from app.main import app
    except ImportError as exc:
        pytest.skip(f"Backend not yet built: {exc}")

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def healthy_runtime(client, monkeypatch):
    """Patch health dependencies so endpoint assertions stay deterministic."""
    from app import health as health_module

    async def fake_check_database() -> dict[str, object]:
        return {
            "status": "ok",
            "connected": True,
            "target": "test-db",
        }

    async def fake_check_training_queue() -> dict[str, object]:
        return {
            "status": "ok",
            "queue_depth": 0,
            "worker_status": {
                "active_workers": 0,
                "max_workers": 2,
                "available_slots": 2,
                "worker_task_running": True,
                "persisted_running_jobs": 0,
            },
        }

    def fake_check_storage() -> dict[str, object]:
        return {
            "status": "ok",
            "upload_dir": {
                "path": "/tmp/uploads",
                "exists": True,
                "writable": True,
                "status": "ok",
                "error": None,
            },
            "models_dir": {
                "path": "/tmp/models",
                "exists": True,
                "writable": True,
                "status": "ok",
                "error": None,
            },
        }

    def fake_check_billing() -> dict[str, object]:
        return {
            "status": "ok",
            "configured": True,
        }

    monkeypatch.setattr(health_module, "_check_database", fake_check_database)
    monkeypatch.setattr(health_module, "_check_training_queue", fake_check_training_queue)
    monkeypatch.setattr(health_module, "_check_storage", fake_check_storage)
    monkeypatch.setattr(health_module, "_check_billing", fake_check_billing)
    monkeypatch.setattr(client.app.state, "startup_complete", True, raising=False)


def test_health_returns_ok_status(client, healthy_runtime) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_live_returns_ok_status(client) -> None:
    response = client.get("/api/health/live")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_ready_returns_200_when_database_is_available(client, healthy_runtime) -> None:
    response = client.get("/api/health/ready")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert data["ready"] is True


def test_health_details_returns_component_breakdown(client, healthy_runtime) -> None:
    response = client.get("/api/health/details")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "components" in data
    assert "database" in data["components"]
    assert "storage" in data["components"]
    assert "training_queue" in data["components"]
    assert "billing" in data["components"]
    assert data["components"]["database"]["connected"] is True
    assert data["components"]["storage"]["status"] == "ok"


def test_error_tracker_record_and_snapshot_counts() -> None:
    from app.monitoring import ErrorClassification, ErrorTracker

    tracker = ErrorTracker()

    tracker.record(
        ErrorClassification(kind="expected", reason="user_input", log_level=logging.INFO),
        source="request",
        error_type="HTTPException",
        error_message="Bad request",
    )
    tracker.record(
        ErrorClassification(kind="unexpected", reason="bug", log_level=logging.ERROR),
        source="job_queue",
        error_type="RuntimeError",
        error_message="Boom",
    )
    tracker.record(
        ErrorClassification(kind="unexpected", reason="bug", log_level=logging.ERROR),
        source="startup",
        error_type="Exception",
        error_message="Still boom",
    )

    snapshot = tracker.snapshot()

    assert snapshot["count"] == 3
    assert snapshot["expected"] == 1
    assert snapshot["unexpected"] == 2
    assert snapshot["by_reason"] == {
        "bug": 2,
        "user_input": 1,
    }


def test_classify_exception_expected_vs_unexpected() -> None:
    from app.monitoring import classify_exception

    expected = classify_exception(HTTPException(status_code=400, detail="bad request"))
    unexpected = classify_exception(Exception("unexpected"))

    assert expected.kind == "expected"
    assert expected.reason == "user_input"
    assert unexpected.kind == "unexpected"
    assert unexpected.reason == "bug"


@pytest.mark.parametrize(
    "path",
    [
        "/api/health",
        "/api/health/live",
        "/api/health/ready",
        "/api/health/details",
    ],
)
def test_health_responses_include_request_id_header(client, healthy_runtime, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "path",
    [
        "/api/health",
        "/api/health/live",
        "/api/health/ready",
        "/api/health/details",
    ],
)
def test_health_responses_include_nosniff_header(client, healthy_runtime, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
