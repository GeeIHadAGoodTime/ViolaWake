"""Training-quota tests for the ViolaWake Console backend.

The Console is a free service; the monthly training limit protects shared
capacity. These tests cover the quota path that used to live in the billing
suite (payments were removed 2026-07-31).
"""

from __future__ import annotations

import asyncio
import io
import math
import struct
import sys
import sqlite3
import time
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def make_wav_bytes(duration: float = 1.0, sr: int = 16000) -> bytes:
    """Generate a valid mono WAV file with a 440Hz sine tone."""
    frame_count = int(duration * sr)
    samples = []
    for i in range(frame_count):
        value = int(16384 * math.sin(2 * math.pi * 440 * i / sr))
        samples.append(struct.pack("<h", value))
    pcm = b"".join(samples)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm)

    buf.seek(0)
    return buf.read()


def register_user(client: TestClient) -> dict[str, object]:
    """Register a user and return auth headers plus user metadata."""
    email = f"quota_{time.time_ns()}@example.com"
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "TestPass123!", "name": "Quota Test"},
    )
    assert response.status_code in (200, 201), response.text

    from app.config import settings

    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
        assert row is not None, "Registered user not found"
        user_id = int(row[0])
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))
        conn.commit()

    login_response = client.post(
        "/api/auth/login",
        json={"email": email, "password": "TestPass123!"},
    )
    assert login_response.status_code == 200, login_response.text
    token = login_response.json()["token"]

    return {
        "email": email,
        "user_id": user_id,
        "headers": {"Authorization": f"Bearer {token}"},
    }


def upload_recordings(
    client: TestClient,
    auth_headers: dict[str, str],
    wake_word: str,
    count: int = 5,
) -> list[int]:
    """Upload enough recordings for a training request."""
    recording_ids: list[int] = []
    for index in range(count):
        response = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": (f"sample_{index}.wav", make_wav_bytes(), "audio/wav")},
            data={"wake_word": wake_word},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        recording_ids.append(payload.get("id") or payload.get("recording_id"))
    return recording_ids


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    try:
        from app.main import app
        from app.database import init_db
    except ImportError as exc:
        pytest.skip(f"Backend not yet built: {exc}")

    asyncio.run(init_db())
    return TestClient(app)


@pytest.fixture
def auth_user(client) -> dict[str, object]:
    """Create an authenticated user for a test."""
    return register_user(client)


@pytest.fixture
def mock_training_queue():
    """Patch the queue dependency so training requests only exercise quota logic."""
    queue = SimpleNamespace()
    job_ids = iter(range(1, 1000))
    queue.submit_job = AsyncMock(side_effect=lambda **_: next(job_ids))
    # submit_training_job consults the breaker (refuse a paused submit before
    # charging, #4207) and stamps the charged period on the job for a later
    # refund; the stub queue must answer both like a healthy, unpaused queue.
    queue.get_circuit_breaker = AsyncMock(return_value=SimpleNamespace(paused=False))
    queue.mark_usage_charged = AsyncMock(return_value=None)

    with patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)):
        yield queue


class TestTrainingQuota:

    def test_free_user_can_start_three_training_jobs(
        self,
        client,
        auth_user,
        mock_training_queue,
    ) -> None:
        recording_ids = upload_recordings(client, auth_user["headers"], "quota-free-ok")

        for _ in range(3):
            response = client.post(
                "/api/training/start",
                headers=auth_user["headers"],
                json={"wake_word": "quota-free-ok", "recording_ids": recording_ids, "epochs": 5},
            )
            assert response.status_code == 202, response.text

        assert mock_training_queue.submit_job.await_count == 3

    def test_free_user_gets_403_on_fourth_training_job_attempt(
        self,
        client,
        auth_user,
        mock_training_queue,
    ) -> None:
        recording_ids = upload_recordings(client, auth_user["headers"], "quota-free-limit")

        for _ in range(3):
            response = client.post(
                "/api/training/start",
                headers=auth_user["headers"],
                json={"wake_word": "quota-free-limit", "recording_ids": recording_ids, "epochs": 5},
            )
            assert response.status_code == 202, response.text

        response = client.post(
            "/api/training/start",
            headers=auth_user["headers"],
            json={"wake_word": "quota-free-limit", "recording_ids": recording_ids, "epochs": 5},
        )

        assert response.status_code == 403, response.text
        detail = response.json()["detail"]
        assert "Monthly training limit reached" in detail
        # The free-service message must not push a paid upgrade.
        assert "Upgrade" not in detail
        assert "X-Upgrade-URL" not in response.headers
        assert mock_training_queue.submit_job.await_count == 3

    def test_quota_403_mentions_monthly_reset(
        self,
        client,
        auth_user,
        mock_training_queue,
    ) -> None:
        recording_ids = upload_recordings(client, auth_user["headers"], "quota-reset-msg")

        for _ in range(3):
            client.post(
                "/api/training/start",
                headers=auth_user["headers"],
                json={"wake_word": "quota-reset-msg", "recording_ids": recording_ids, "epochs": 5},
            )
        response = client.post(
            "/api/training/start",
            headers=auth_user["headers"],
            json={"wake_word": "quota-reset-msg", "recording_ids": recording_ids, "epochs": 5},
        )
        assert response.status_code == 403, response.text
        assert "resets" in response.json()["detail"]
