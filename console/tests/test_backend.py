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
import wave
from pathlib import Path

import numpy as np
import pytest

# These tests need the backend to be importable
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

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
        return TestClient(app)
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
        from app.routes.auth import reset_rate_limits
        reset_rate_limits()
    except ImportError:
        pass


@pytest.fixture
def auth_headers(client) -> dict[str, str]:
    """Register a test user and return auth headers."""
    import time
    email = f"test_{time.time_ns()}@example.com"
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "password": "TestPass123!", "name": "Test User"},
    )
    if resp.status_code not in (200, 201):
        pytest.fail(f"Registration failed: {resp.text}")
    token = resp.json()["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def fake_email_service(monkeypatch):
    """Capture outbound auth emails without calling Resend."""
    import sys

    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    class FakeEmailService:
        def __init__(self) -> None:
            self.verification_emails: list[dict[str, str]] = []
            self.password_reset_emails: list[dict[str, str]] = []
            self.welcome_emails: list[dict[str, str]] = []

        async def send_verification_email(self, to: str, token: str, name: str) -> bool:
            self.verification_emails.append({"to": to, "token": token, "name": name})
            return True

        async def send_password_reset(self, to: str, token: str, name: str) -> bool:
            self.password_reset_emails.append({"to": to, "token": token, "name": name})
            return True

        async def send_welcome(self, to: str, name: str) -> bool:
            self.welcome_emails.append({"to": to, "name": name})
            return True

        async def send_training_complete(
            self, to: str, model_name: str, download_url: str,
        ) -> bool:
            return True

        async def send_quota_warning(self, to: str, used: int, limit: int, tier: str) -> bool:
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

    def test_register_duplicate(self, client) -> None:
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
        assert resp.status_code in (400, 409)

    def test_login(self, client) -> None:
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
        ids = self._upload_samples(client, auth_headers, "statustest")
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

    def test_start_no_auth(self, client) -> None:
        resp = client.post(
            "/api/training/start",
            json={"wake_word": "x", "recording_ids": [1]},
        )
        assert resp.status_code in (401, 403, 422)


# ── Model Tests ──────────────────────────────────────────────────────────────

class TestModels:

    def test_list_models_empty(self, client, auth_headers) -> None:
        resp = client.get("/api/models", headers=auth_headers)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_models_no_auth(self, client) -> None:
        resp = client.get("/api/models")
        assert resp.status_code in (401, 403, 422)


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
        """A non-WAV file (random bytes) should be rejected with 400."""
        fake_data = b"this is not a wav file at all, just plain text garbage"
        resp = client.post(
            "/api/recordings/upload",
            headers=auth_headers,
            files={"file": ("fake.wav", fake_data, "audio/wav")},
            data={"wake_word": "edgetest"},
        )
        assert resp.status_code == 400

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
