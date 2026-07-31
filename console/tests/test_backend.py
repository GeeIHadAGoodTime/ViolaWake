"""
Backend unit tests for the ViolaWake Console API.

Tests all API endpoints using FastAPI's TestClient (no real server needed).
This is the fast inner loop — run these before E2E tests.

Usage:
    pytest console/tests/test_backend.py -v
"""
from __future__ import annotations

import io
import json
import sqlite3
import uuid
import wave
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from scipy.io import wavfile

# These tests need the backend to be importable
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import sys as _sys

_BACKEND_DIR = str(Path(__file__).resolve().parents[1] / "backend")
if _BACKEND_DIR not in _sys.path:
    _sys.path.insert(0, _BACKEND_DIR)

try:
    from app.tenancy import SERVICE_TENANT_HEADER
except ImportError:  # pragma: no cover - mirrors the HAS_FASTAPI guard above
    SERVICE_TENANT_HEADER = "X-Viola-Tenant"

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    import sys
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    try:
        from app.main import app
        with TestClient(app) as test_client:
            yield test_client
    except ImportError as e:
        pytest.skip(f"Backend not yet built: {e}")


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """Clear rate limits before each test to prevent cross-test interference."""
    import sys
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    try:
        from app.auth import reset_download_tokens
        from app.routes.auth import reset_rate_limits
        reset_rate_limits()
        reset_download_tokens()
    except ImportError:
        pass


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    """Register and verify a test user, then return auth headers."""
    import time
    email = f"test_{time.time_ns()}@example.com"
    password = "TestPass123!"
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "Test User"},
    )
    if resp.status_code not in (200, 201):
        pytest.fail(f"Registration failed: {resp.text}")

    import sys
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app.config import settings

    with sqlite3.connect(settings.db_path) as conn:
        updated = conn.execute(
            "UPDATE users SET email_verified = 1 WHERE email = ?",
            (email,),
        ).rowcount
        conn.commit()
    if updated != 1:
        raise AssertionError("Registered user not found")

    login_resp = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    if login_resp.status_code != 200:
        pytest.fail(f"Login failed: {login_resp.text}")
    token = login_resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def unverified_auth_headers(client) -> dict[str, str]:
    """Register a test user without verifying their email."""
    import time

    email = f"unverified_{time.time_ns()}@example.com"
    password = "TestPass123!"
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": "Unverified User"},
    )
    if resp.status_code not in (200, 201):
        pytest.fail(f"Registration failed: {resp.text}")

    import sys
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app.config import settings

    with sqlite3.connect(settings.db_path) as conn:
        updated = conn.execute(
            "UPDATE users SET email_verified = 0 WHERE email = ?",
            (email,),
        ).rowcount
        conn.commit()
    if updated != 1:
        raise AssertionError("Registered user not found")

    login_resp = client.post(
        "/api/auth/login",
        json={"email": email, "password": password},
    )
    if login_resp.status_code != 200:
        pytest.fail(f"Login failed: {login_resp.text}")
    return {"Authorization": f"Bearer {login_resp.json()['token']}"}


@pytest.fixture
def fake_email_service(monkeypatch):
    """Capture outbound auth emails without calling Resend."""
    import sys

    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    class FakeEmailService:
        def __init__(self) -> None:
            self.enabled = True
            self.verification_emails: list[dict[str, str]] = []
            self.password_reset_emails: list[dict[str, str]] = []
            self.welcome_emails: list[dict[str, str]] = []
            self.existing_account_notices: list[dict[str, str]] = []

        async def send_verification_email(self, to: str, token: str, name: str) -> bool:
            self.verification_emails.append({"to": to, "token": token, "name": name})
            return True

        async def send_password_reset(self, to: str, token: str, name: str) -> bool:
            self.password_reset_emails.append({"to": to, "token": token, "name": name})
            return True

        async def send_welcome(self, to: str, name: str) -> bool:
            self.welcome_emails.append({"to": to, "name": name})
            return True

        async def send_existing_account_notice(self, to: str, name: str) -> bool:
            self.existing_account_notices.append({"to": to, "name": name})
            return True

        async def send_training_complete(
            self, to: str, model_name: str, download_url: str,
        ) -> bool:
            return True

        async def send_quota_warning(self, to: str, used: int, limit: int, tier: str) -> bool:
            return True

        async def send_account_deleted(
            self,
            to: str,
            name: str,
            scheduled_hard_delete_at: str,
        ) -> bool:
            return True

    service = FakeEmailService()

    from app.routes import auth as auth_routes

    monkeypatch.setattr(auth_routes, "get_email_service", lambda: service)
    return service


def make_wav_bytes(duration: float = 1.5, sr: int = 16000) -> bytes:
    """Generate a valid WAV file as bytes."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t) * 0.5 + rng.normal(0, 0.05, len(t))
    pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

    buf.seek(0)
    return buf.read()


def make_float32_wav_bytes(duration: float = 1.5, sr: int = 16000) -> bytes:
    """Generate a float32 WAV file as bytes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (0.7 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = io.BytesIO()
    wavfile.write(buf, sr, signal)
    buf.seek(0)
    return buf.read()


def _seed_account_artifacts(user_id: int) -> dict[str, str]:
    import sys

    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app.config import settings
    from app.storage import build_companion_config_identifier, build_model_key, get_storage

    storage = get_storage()
    model_key = build_model_key(user_id, "delete-test.onnx")
    storage.upload(model_key, b"fake model bytes", "application/octet-stream")
    storage.upload(
        build_companion_config_identifier(model_key),
        b'{"d_prime": 1.23}',
        "application/json",
    )

    with sqlite3.connect(settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO trained_models (user_id, wake_word, file_path, config_json, d_prime, size_bytes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "delete-account",
                model_key,
                '{"d_prime": 1.23}',
                1.23,
                len(b"fake model bytes"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        # Training-job state is persisted only by the async job queue's SQLite
        # store (table ``jobs``), never in a Postgres ``training_jobs`` table —
        # that table was never written at runtime and has been retired (#1438).
        conn.execute(
            """
            INSERT INTO subscriptions (user_id, stripe_customer_id, stripe_subscription_id, tier, status, current_period_end, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                None,
                None,
                "developer",
                "active",
                None,
                datetime.now(timezone.utc).isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.execute(
            """
            INSERT INTO usage_records (user_id, action, period_start, count, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                user_id,
                "training_job",
                datetime.now(timezone.utc).isoformat(),
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()

    return {"model_key": model_key}


def _account_state(user_id: int) -> dict[str, int]:
    import sys

    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app.config import settings

    with sqlite3.connect(settings.db_path) as conn:
        return {
            "users": int(conn.execute("SELECT COUNT(*) FROM users WHERE id = ?", (user_id,)).fetchone()[0]),
            "recordings": int(conn.execute("SELECT COUNT(*) FROM recordings WHERE user_id = ?", (user_id,)).fetchone()[0]),
            "deleted_recordings": int(conn.execute("SELECT COUNT(*) FROM recordings WHERE user_id = ? AND deleted_at IS NOT NULL", (user_id,)).fetchone()[0]),
            "models": int(conn.execute("SELECT COUNT(*) FROM trained_models WHERE user_id = ?", (user_id,)).fetchone()[0]),
            "deleted_models": int(conn.execute("SELECT COUNT(*) FROM trained_models WHERE user_id = ? AND deleted_at IS NOT NULL", (user_id,)).fetchone()[0]),
            "subscriptions": int(conn.execute("SELECT COUNT(*) FROM subscriptions WHERE user_id = ?", (user_id,)).fetchone()[0]),
            "usage_records": int(conn.execute("SELECT COUNT(*) FROM usage_records WHERE user_id = ?", (user_id,)).fetchone()[0]),
        }


# ── Auth Tests ───────────────────────────────────────────────────────────────

class TestAuth:

    def test_register(self, client) -> None:
        import time
        email = f"register_{time.time_ns()}@example.com"
        resp = client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "TestPass123!",
                "name": "Register Test",
            },
        )
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert "token" in data
        assert data["user"]["email"] == email

    def test_register_sends_verification_email(self, client, fake_email_service) -> None:
        import time

        email = f"verify_{time.time_ns()}@example.com"
        resp = client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "TestPass123!",
                "name": "Verify Test",
            },
        )
        assert resp.status_code in (200, 201)
        assert len(fake_email_service.verification_emails) == 1
        assert fake_email_service.verification_emails[0]["to"] == email

    def test_register_duplicate(self, client, fake_email_service) -> None:
        import time
        email = f"dup_{time.time_ns()}@example.com"
        client.post(
            "/api/auth/register",
            json={"email": email, "password": "TestPass123!", "name": "First"},
        )
        resp = client.post(
            "/api/auth/register",
            json={"email": email, "password": "TestPass123!", "name": "Second"},
        )
        assert resp.status_code == 201
        assert resp.json()["token"] == ""
        assert len(fake_email_service.existing_account_notices) == 1

    def test_login(self, client, fake_email_service) -> None:
        import time
        email = f"login_{time.time_ns()}@example.com"
        client.post(
            "/api/auth/register",
            json={"email": email, "password": "TestPass123!", "name": "Login"},
        )
        resp = client.post(
            "/api/auth/login",
            json={"email": email, "password": "TestPass123!"},
        )
        assert resp.status_code == 200
        assert "token" in resp.json()
        assert resp.json()["user"]["email_verified"] is False

    def test_login_wrong_password(self, client) -> None:
        import time
        email = f"wrongpw_{time.time_ns()}@example.com"
        client.post(
            "/api/auth/register",
            json={"email": email, "password": "Correct123!", "name": "Wrong PW"},
        )
        resp = client.post(
            "/api/auth/login",
            json={"email": email, "password": "WrongPassword!"},
        )
        assert resp.status_code in (400, 401)

    def test_me_authenticated(self, client, auth_headers) -> None:
        resp = client.get("/api/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        assert "email" in resp.json()
        assert resp.json()["email_verified"] is True

    def test_me_no_auth(self, client) -> None:
        resp = client.get("/api/auth/me")
        assert resp.status_code in (401, 403, 422)

    def test_me_bad_token(self, client) -> None:
        resp = client.get(
            "/api/auth/me",
            headers={"Authorization": "Bearer invalid_token"},
        )
        assert resp.status_code in (401, 403)

    def test_verify_email(self, client, fake_email_service) -> None:
        import time

        email = f"verify_flow_{time.time_ns()}@example.com"
        register_resp = client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "TestPass123!",
                "name": "Verify Flow",
            },
        )
        assert register_resp.status_code in (200, 201)

        token = fake_email_service.verification_emails[0]["token"]
        verify_resp = client.post("/api/auth/verify-email", json={"token": token})
        assert verify_resp.status_code == 200
        assert verify_resp.json()["message"] == "Email verified successfully"
        assert len(fake_email_service.welcome_emails) == 1
        assert fake_email_service.welcome_emails[0]["to"] == email

    def test_forgot_password_and_reset_password(self, client, fake_email_service) -> None:
        import time

        email = f"reset_{time.time_ns()}@example.com"
        register_resp = client.post(
            "/api/auth/register",
            json={
                "email": email,
                "password": "OriginalPass123!",
                "name": "Reset Flow",
            },
        )
        assert register_resp.status_code in (200, 201)

        forgot_resp = client.post("/api/auth/forgot-password", json={"email": email})
        assert forgot_resp.status_code == 200
        assert len(fake_email_service.password_reset_emails) == 1

        reset_token = fake_email_service.password_reset_emails[0]["token"]
        reset_resp = client.post(
            "/api/auth/reset-password",
            json={"token": reset_token, "password": "NewPass123!"},
        )
        assert reset_resp.status_code == 200
        assert reset_resp.json()["message"] == "Password reset successfully"

        login_resp = client.post(
            "/api/auth/login",
            json={"email": email, "password": "NewPass123!"},
        )
        assert login_resp.status_code == 200

    def test_forgot_password_unknown_email_returns_generic_success(
        self, client, fake_email_service,
    ) -> None:
        resp = client.post("/api/auth/forgot-password", json={"email": "missing@example.com"})
        assert resp.status_code == 200
        assert resp.json()["message"].startswith("If an account exists")
        assert fake_email_service.password_reset_emails == []

    def test_delete_account_removes_user_data_and_storage(self, client, auth_headers) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import decode_token
        from app.storage import build_companion_config_identifier, get_storage

        wav_data = make_wav_bytes()
        upload_resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "delete-account"},
        )
        assert upload_resp.status_code == 200, upload_resp.text

        token = auth_headers["Authorization"].removeprefix("Bearer ").strip()
        user_id = decode_token(token)
        artifacts = _seed_account_artifacts(user_id)
        queue = SimpleNamespace(delete_jobs_for_user=AsyncMock(return_value=1))

        with patch("app.routes.auth.init_job_queue", new=AsyncMock(return_value=queue)):
            response = client.request(
                "DELETE",
                "/api/auth/account",
                headers=auth_headers,
                json={"password": "TestPass123!"},
            )

        assert response.status_code == 200, response.text
        assert response.json()["message"] == "Account deleted. Data is scheduled for permanent deletion in 30 days."
        queue.delete_jobs_for_user.assert_awaited_once_with(user_id)

        state = _account_state(user_id)
        assert state == {
            "users": 1,
            "recordings": 1,
            "deleted_recordings": 1,
            "models": 1,
            "deleted_models": 1,
            "subscriptions": 1,
            "usage_records": 1,
        }

        storage = get_storage()
        assert storage.exists(artifacts["model_key"]) is True
        assert storage.exists(build_companion_config_identifier(artifacts["model_key"])) is True

        from app.config import settings

        with sqlite3.connect(settings.db_path) as conn:
            user_row = conn.execute(
                "SELECT email, name, deleted_at, scheduled_hard_delete_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            sub_row = conn.execute(
                "SELECT tier, status, stripe_customer_id, stripe_subscription_id FROM subscriptions WHERE user_id = ?",
                (user_id,),
            ).fetchone()

        assert user_row is not None
        assert user_row[0] == f"deleted-user-{user_id}@deleted.violawake.local"
        assert user_row[1] == "Deleted user"
        assert user_row[2] is not None
        assert user_row[3] is not None
        assert sub_row == ("free", "canceled", None, None)

        me_response = client.get("/api/auth/me", headers=auth_headers)
        assert me_response.status_code == 401

    def test_account_export_includes_profile_artifacts_and_manifests(
        self, client, auth_headers,
    ) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import decode_token

        wav_data = make_wav_bytes()
        upload_resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "export-account"},
        )
        assert upload_resp.status_code == 200, upload_resp.text

        user_id = decode_token(auth_headers["Authorization"].removeprefix("Bearer ").strip())
        _seed_account_artifacts(user_id)

        response = client.get("/api/account/export", headers=auth_headers)

        assert response.status_code == 200, response.text
        assert response.headers["content-type"] == "application/zip"

        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            names = set(zf.namelist())
            assert "user.json" in names
            assert "recordings/manifest.json" in names
            assert "models/manifest.json" in names
            assert "jobs.json" in names
            user_payload = json.loads(zf.read("user.json"))
            recordings = json.loads(zf.read("recordings/manifest.json"))
            models = json.loads(zf.read("models/manifest.json"))

        assert user_payload["id"] == user_id
        assert len(recordings) >= 1
        assert any(item["available"] for item in recordings)
        assert len(models) >= 1
        assert any(item["available"] for item in models)

    def test_unverified_user_blocked_from_recording_training_and_billing(
        self, client, unverified_auth_headers,
    ) -> None:
        wav_data = make_wav_bytes()

        recording_resp = client.post(
            "/api/recordings/upload",
            headers=unverified_auth_headers,
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "blocked"},
        )
        assert recording_resp.status_code == 403
        assert recording_resp.json()["detail"] == "Verify your email to upload recordings."

        training_resp = client.post(
            "/api/training/start",
            headers=unverified_auth_headers,
            json={"wake_word": "blocked", "recording_ids": [1], "epochs": 5},
        )
        assert training_resp.status_code == 403

        billing_resp = client.get("/api/billing/usage", headers=unverified_auth_headers)
        assert billing_resp.status_code == 403


# ── Recording Tests ──────────────────────────────────────────────────────────

class TestRecordings:

    def test_upload_wav(self, client, auth_headers) -> None:
        wav_data = make_wav_bytes()
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "testword"},
        )
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        data = resp.json()
        assert "id" in data or "recording_id" in data

    def test_upload_no_auth(self, client) -> None:
        wav_data = make_wav_bytes()
        resp = client.post(
            "/api/recordings/upload",
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "testword"},
        )
        assert resp.status_code in (401, 403, 422)

    def test_list_recordings(self, client, auth_headers) -> None:
        # Upload one first
        wav_data = make_wav_bytes()
        client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "listtest"},
        )

        resp = client.get(
            "/api/recordings",
            headers=auth_headers,
            params={"wake_word": "listtest"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_upload_10_recordings(self, client, auth_headers) -> None:
        ids = []
        for i in range(10):
            wav_data = make_wav_bytes(duration=1.5 + i * 0.01)
            resp = client.post(
                "/api/recordings/upload",
                headers=auth_headers,
                files={"file": (f"sample_{i:02d}.wav", wav_data, "audio/wav")},
                data={"wake_word": "batchtest"},
            )
            assert resp.status_code == 200
            data = resp.json()
            ids.append(data.get("id") or data.get("recording_id"))
        assert len(ids) == 10

    @pytest.mark.skip(reason="slowapi limiter disabled globally in tests/conftest.py; re-enable per test before asserting headers")
    def test_upload_rate_limit_returns_headers_and_429(self, client, auth_headers, monkeypatch) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.routes import recordings as recordings_routes

        monkeypatch.setattr(recordings_routes, "UPLOAD_RATE_LIMIT", 2)

        first = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("first.wav", make_wav_bytes(), "audio/wav")},
            data={"wake_word": "ratelimit-upload"},
        )
        second = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("second.wav", make_wav_bytes(), "audio/wav")},
            data={"wake_word": "ratelimit-upload"},
        )
        third = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("third.wav", make_wav_bytes(), "audio/wav")},
            data={"wake_word": "ratelimit-upload"},
        )

        assert first.status_code == 200, first.text
        assert first.headers["X-RateLimit-Remaining"] == "1"
        assert "X-RateLimit-Reset" in first.headers
        assert second.status_code == 200, second.text
        assert second.headers["X-RateLimit-Remaining"] == "0"
        assert third.status_code == 429
        assert third.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in third.headers


# ── Training Tests ───────────────────────────────────────────────────────────

class TestTraining:

    def _upload_samples(self, client, auth_headers, wake_word: str, n: int = 10) -> list[int]:
        """Helper: upload N samples and return recording IDs."""
        ids = []
        for i in range(n):
            wav_data = make_wav_bytes(duration=1.5)
            resp = client.post(
                "/api/recordings/upload",
                headers=auth_headers,
                files={"file": (f"s_{i}.wav", wav_data, "audio/wav")},
                data={"wake_word": wake_word},
            )
            assert resp.status_code == 200
            data = resp.json()
            ids.append(data.get("id") or data.get("recording_id"))
        return ids

    def test_start_training(self, client, auth_headers) -> None:
        ids = self._upload_samples(client, auth_headers, "traintest")
        queue = SimpleNamespace(
            submit_job=AsyncMock(return_value=101),
            get_circuit_breaker=AsyncMock(return_value=SimpleNamespace(paused=False)),
            mark_usage_charged=AsyncMock(return_value=None),
        )
        with patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)):
            resp = client.post(
                "/api/training/start",
                headers=auth_headers,
                json={"wake_word": "traintest", "recording_ids": ids, "epochs": 5},
            )
        assert resp.status_code in (200, 202), f"Start failed: {resp.text}"
        data = resp.json()
        assert "job_id" in data
        assert data["status"] in ("queued", "running")

    def test_training_status(self, client, auth_headers) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import decode_token
        from app.job_queue import Job, JobStatus

        ids = self._upload_samples(client, auth_headers, "statustest")
        token = auth_headers["Authorization"].removeprefix("Bearer ").strip()
        job = Job(
            id=102,
            user_id=decode_token(token),
            wake_word="statustest",
            status=JobStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            recording_ids=ids,
            epochs=5,
        )
        queue = SimpleNamespace(
            submit_job=AsyncMock(return_value=job.id),
            get_job=AsyncMock(return_value=job),
            get_circuit_breaker=AsyncMock(return_value=SimpleNamespace(paused=False)),
            mark_usage_charged=AsyncMock(return_value=None),
        )
        with patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)):
            resp = client.post(
                "/api/training/start",
                headers=auth_headers,
                json={"wake_word": "statustest", "recording_ids": ids, "epochs": 5},
            )
            job_id = resp.json()["job_id"]

            resp = client.get(
                f"/api/training/status/{job_id}",
                headers=auth_headers,
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["status"] in ("queued", "running", "completed", "failed")

    def test_cancel_job_accepts_service_key(self, client, monkeypatch) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import reset_service_user_cache
        from app.config import settings

        reset_service_user_cache()
        monkeypatch.setattr(settings, "service_key", "svc-test-key")

        queue = SimpleNamespace(cancel_job=AsyncMock(return_value=True))
        with (
            patch("app.routes.jobs.get_owned_job_or_404", new=AsyncMock(return_value=SimpleNamespace(id=54))),
            patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)),
        ):
            resp = client.delete(
                "/api/jobs/54",
                headers={
                    "Authorization": "Bearer svc-test-key",
                    # The service key alone names an account shared by the
                    # caller's whole user base; the tenant header is what says
                    # WHICH of its users this call belongs to.
                    SERVICE_TENANT_HEADER: "a1b2c3d4e5f60718",
                },
            )

        assert resp.status_code == 200, resp.text
        assert resp.json()["message"] == "Training job cancellation requested"
        queue.cancel_job.assert_awaited_once_with(54)

    def test_service_key_without_tenant_is_refused(self, client, monkeypatch) -> None:
        """A service-key call that names no tenant must not reach the queue.

        Without this refusal the call lands in the shared partition, where one
        upstream user's failures pause every other upstream user's training
        and one upstream user's queue consumes the pending cap of the rest.
        """
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import reset_service_user_cache
        from app.config import settings

        reset_service_user_cache()
        monkeypatch.setattr(settings, "service_key", "svc-test-key")

        queue = SimpleNamespace(cancel_job=AsyncMock(return_value=True))
        with (
            patch("app.routes.jobs.get_owned_job_or_404", new=AsyncMock(return_value=SimpleNamespace(id=54))),
            patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)),
        ):
            resp = client.delete(
                "/api/jobs/54",
                headers={"Authorization": "Bearer svc-test-key"},
            )

        assert resp.status_code == 400, resp.text
        assert SERVICE_TENANT_HEADER in resp.json()["detail"]
        queue.cancel_job.assert_not_awaited()

    def test_start_no_auth(self, client) -> None:
        resp = client.post(
            "/api/training/start",
            json={"wake_word": "x", "recording_ids": [1]},
        )
        assert resp.status_code in (401, 403, 422)

    @pytest.mark.skip(reason="slowapi limiter disabled globally in tests/conftest.py; re-enable per test before asserting headers")
    def test_training_rate_limit_returns_headers_and_429(
        self,
        client,
        auth_headers,
        monkeypatch,
    ) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.routes import jobs as jobs_routes

        monkeypatch.setattr(jobs_routes, "TRAINING_SUBMISSION_RATE_LIMIT", 2)

        queue = SimpleNamespace()
        queue.submit_job = AsyncMock(side_effect=[101, 102, 103])
        recording_ids = self._upload_samples(client, auth_headers, "ratelimit-train", n=5)

        with patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)):
            first = client.post(
                "/api/training/start",
                headers=auth_headers,
                json={"wake_word": "ratelimit-train", "recording_ids": recording_ids, "epochs": 5},
            )
            second = client.post(
                "/api/training/start",
                headers=auth_headers,
                json={"wake_word": "ratelimit-train", "recording_ids": recording_ids, "epochs": 5},
            )
            third = client.post(
                "/api/training/start",
                headers=auth_headers,
                json={"wake_word": "ratelimit-train", "recording_ids": recording_ids, "epochs": 5},
            )

        assert first.status_code == 202, first.text
        assert first.headers["X-RateLimit-Remaining"] == "1"
        assert second.status_code == 202, second.text
        assert second.headers["X-RateLimit-Remaining"] == "0"
        assert third.status_code == 429
        assert third.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in third.headers


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestModels:

    def test_list_models_empty(self, client, auth_headers) -> None:
        resp = client.get("/api/models", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_models_no_auth(self, client) -> None:
        resp = client.get("/api/models")
        assert resp.status_code in (401, 403, 422)

    def test_model_download_uses_one_time_download_token(self, client, auth_headers) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import decode_token
        from app.config import settings
        from app.storage import build_model_key, get_storage

        token = auth_headers["Authorization"].removeprefix("Bearer ").strip()
        user_id = decode_token(token)
        model_bytes = b"\x08\x09violawake-test-model"
        model_key = build_model_key(user_id, "token_test.onnx")
        get_storage().upload(model_key, model_bytes, "application/octet-stream")

        with sqlite3.connect(settings.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO trained_models (user_id, wake_word, file_path, config_json, d_prime, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    "token-test",
                    model_key,
                    None,
                    None,
                    len(model_bytes),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            model_id = int(cursor.lastrowid)

        token_resp = client.post(
            "/api/auth/download-token",
            headers=auth_headers,
            json={"action": "model_download", "resource_id": model_id},
        )
        assert token_resp.status_code == 200, token_resp.text
        download_token = token_resp.json()["token"]

        first_download = client.get(f"/api/models/{model_id}/download", params={"token": download_token})
        assert first_download.status_code == 200
        assert first_download.content == model_bytes

        second_download = client.get(f"/api/models/{model_id}/download", params={"token": download_token})
        assert second_download.status_code == 401

    def test_model_config_accepts_service_key(self, client, monkeypatch) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.auth import reset_service_user_cache
        from app.config import settings
        from app.storage import build_model_key

        reset_service_user_cache()
        monkeypatch.setattr(settings, "service_key", "service-key-123")
        monkeypatch.setattr(settings, "service_user_email", "viola-service@viola.internal")

        probe = client.get("/api/models/999999/config", headers={"Authorization": "Bearer service-key-123"})
        assert probe.status_code == 404

        with sqlite3.connect(settings.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (settings.service_user_email,),
            ).fetchone()
            assert row is not None
            service_user_id = int(row[0])
            cursor = conn.execute(
                """
                INSERT INTO trained_models (user_id, wake_word, file_path, config_json, d_prime, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    service_user_id,
                    "service-key-test",
                    build_model_key(service_user_id, "service-key-test.onnx"),
                    json.dumps({"architecture": "temporal_cnn", "quality_grade": "A"}),
                    12.4,
                    12345,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()
            model_id = int(cursor.lastrowid)

        response = client.get(
            f"/api/models/{model_id}/config",
            headers={"Authorization": "Bearer service-key-123"},
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["architecture"] == "temporal_cnn"
        assert payload["quality_grade"] == "A"
        assert payload["d_prime"] == 12.4


# ── Upload Validation Edge Cases ────────────────────────────────────────────

class TestUploadEdgeCases:

    def test_upload_too_short_wav(self, client, auth_headers) -> None:
        """WAV shorter than 0.5s should be rejected with 400."""
        wav_data = make_wav_bytes(duration=0.1)  # 100ms -- well under 0.5s minimum
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("short.wav", wav_data, "audio/wav")},
            data={"wake_word": "edgetest"},
        )
        assert resp.status_code == 400
        assert "too short" in resp.json()["detail"].lower()

    def test_upload_non_wav_file(self, client, auth_headers) -> None:
        """A non-WAV/FLAC file (random bytes) should be rejected with 415."""
        fake_data = b"this is not a wav file at all, just plain text garbage"
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("fake.wav", fake_data, "audio/wav")},
            data={"wake_word": "edgetest"},
        )
        assert resp.status_code == 415

    def test_upload_empty_wake_word(self, client, auth_headers) -> None:
        """Empty wake_word should be rejected with 400 or 422."""
        wav_data = make_wav_bytes()
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("sample.wav", wav_data, "audio/wav")},
            data={"wake_word": "   "},  # whitespace-only
        )
        assert resp.status_code in (400, 422)

    def test_upload_float32_wav_is_normalized_to_int16(self, client, auth_headers) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.config import settings
        from app.storage import get_storage

        wav_data = make_float32_wav_bytes()
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("float.wav", wav_data, "audio/wav")},
            data={"wake_word": "floatnorm"},
        )
        assert resp.status_code == 200, resp.text
        recording_id = resp.json()["recording_id"]

        with sqlite3.connect(settings.db_path) as conn:
            row = conn.execute(
                "SELECT file_path FROM recordings WHERE id = ?",
                (recording_id,),
            ).fetchone()
        assert row is not None

        stored_bytes = get_storage().download(row[0])
        sr, stored_data = wavfile.read(io.BytesIO(stored_bytes))
        assert sr == 16000
        assert stored_data.dtype == np.int16

    def test_upload_uses_uuid_filename_and_records_storage_metadata(
        self, client, auth_headers,
    ) -> None:
        import sys

        backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from app.config import settings

        wav_data = make_wav_bytes()
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("claimed-name.wav", wav_data, "audio/wav")},
            data={"wake_word": "metadata"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        stored_filename = data["filename"]
        assert stored_filename.endswith(".wav")
        uuid.UUID(stored_filename.removesuffix(".wav"))

        with sqlite3.connect(settings.db_path) as conn:
            row = conn.execute(
                "SELECT filename, display_name, size_bytes, uploaded_at FROM recordings WHERE id = ?",
                (data["recording_id"],),
            ).fetchone()

        assert row is not None
        assert row[0] == stored_filename
        assert row[1] == "claimed-name.wav"
        assert row[2] >= len(wav_data)
        assert row[3] is not None
