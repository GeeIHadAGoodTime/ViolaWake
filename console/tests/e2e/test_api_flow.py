"""
E2E API tests — verify the full backend flow without a browser.

Tests the complete lifecycle:
  register → login → upload 10 recordings → start training → poll status → download model

This runs against the real backend (started by conftest fixtures).
No mocks — this is the real deal.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    BACKEND_URL,
    TEST_USER_EMAIL,
    TEST_USER_NAME,
    TEST_USER_PASSWORD,
    mark_email_verified,
)

pytestmark = pytest.mark.e2e


class TestAuthFlow:
    """Test registration and login."""

    def test_register(self, backend_server: str) -> None:
        email = f"register_{time.time_ns()}@violawake.dev"
        resp = requests.post(
            f"{backend_server}/api/auth/register",
            json={
                "email": email,
                "password": TEST_USER_PASSWORD,
                "name": TEST_USER_NAME,
            },
            timeout=10,
        )
        assert resp.status_code in (200, 201), f"Register failed: {resp.text}"
        data = resp.json()
        assert "token" in data
        assert data["user"]["email"] == email

    def test_login(self, backend_server: str) -> None:
        email = f"login_{time.time_ns()}@violawake.dev"
        # Register first
        requests.post(
            f"{backend_server}/api/auth/register",
            json={
                "email": email,
                "password": TEST_USER_PASSWORD,
                "name": "Login Test",
            },
            timeout=10,
        )

        resp = requests.post(
            f"{backend_server}/api/auth/login",
            json={"email": email, "password": TEST_USER_PASSWORD},
            timeout=10,
        )
        assert resp.status_code == 200, f"Login failed: {resp.text}"
        data = resp.json()
        assert "token" in data

    def test_me_endpoint(self, backend_server: str) -> None:
        email = f"me_{time.time_ns()}@violawake.dev"
        # Register + get token
        resp = requests.post(
            f"{backend_server}/api/auth/register",
            json={
                "email": email,
                "password": TEST_USER_PASSWORD,
                "name": "Me Test",
            },
            timeout=10,
        )
        token = resp.json()["token"]

        # Hit /me
        resp = requests.get(
            f"{backend_server}/api/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["email"] == email

    def test_bad_login(self, backend_server: str) -> None:
        resp = requests.post(
            f"{backend_server}/api/auth/login",
            json={"email": "nonexistent@test.com", "password": "wrong"},
            timeout=10,
        )
        assert resp.status_code in (401, 400, 422)

    def test_no_auth_header(self, backend_server: str) -> None:
        resp = requests.get(f"{backend_server}/api/auth/me", timeout=10)
        assert resp.status_code in (401, 403, 422)


class TestRecordingFlow:
    """Test uploading voice recordings."""

    @pytest.fixture
    def auth_token(self, backend_server: str) -> str:
        email = f"recording_{time.time_ns()}@violawake.dev"
        resp = requests.post(
            f"{backend_server}/api/auth/register",
            json={
                "email": email,
                "password": TEST_USER_PASSWORD,
                "name": "Recording Test",
            },
            timeout=10,
        )
        assert resp.status_code in (200, 201), f"Register failed: {resp.text}"
        mark_email_verified(email)
        return resp.json()["token"]

    def test_upload_single_recording(
        self,
        backend_server: str,
        auth_token: str,
        fake_wake_word_wav: Path,
    ) -> None:
        with open(fake_wake_word_wav, "rb") as f:
            resp = requests.post(
                f"{backend_server}/api/recordings/upload",
                headers={"Authorization": f"Bearer {auth_token}"},
                files={"file": ("sample_01.wav", f, "audio/wav")},
                data={"wake_word": "testword"},
                timeout=10,
            )
        assert resp.status_code == 200, f"Upload failed: {resp.text}"
        data = resp.json()
        assert "recording_id" in data or "id" in data
        assert data.get("wake_word", data.get("wake_word")) == "testword"

    def test_upload_10_recordings(
        self,
        backend_server: str,
        auth_token: str,
        fake_wake_word_wavs: list[Path],
    ) -> None:
        recording_ids = []
        for wav_path in fake_wake_word_wavs:
            with open(wav_path, "rb") as f:
                resp = requests.post(
                    f"{backend_server}/api/recordings/upload",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    files={"file": (wav_path.name, f, "audio/wav")},
                    data={"wake_word": "testword"},
                    timeout=10,
                )
            assert resp.status_code == 200, f"Upload {wav_path.name} failed: {resp.text}"
            data = resp.json()
            rid = data.get("recording_id") or data.get("id")
            recording_ids.append(rid)

        assert len(recording_ids) == 10

    def test_list_recordings(
        self,
        backend_server: str,
        auth_token: str,
        fake_wake_word_wav: Path,
    ) -> None:
        # Upload one first
        with open(fake_wake_word_wav, "rb") as f:
            requests.post(
                f"{backend_server}/api/recordings/upload",
                headers={"Authorization": f"Bearer {auth_token}"},
                files={"file": ("sample.wav", f, "audio/wav")},
                data={"wake_word": "listtest"},
                timeout=10,
            )

        resp = requests.get(
            f"{backend_server}/api/recordings",
            headers={"Authorization": f"Bearer {auth_token}"},
            params={"wake_word": "listtest"},
            timeout=10,
        )
        assert resp.status_code == 200
        recordings = resp.json()
        assert isinstance(recordings, list)
        assert len(recordings) >= 1


class TestFullTrainingFlow:
    """Test the complete training pipeline: upload → train → download."""

    @pytest.fixture
    def auth_token(self, backend_server: str) -> str:
        email = f"training_{time.time_ns()}@violawake.dev"
        resp = requests.post(
            f"{backend_server}/api/auth/register",
            json={
                "email": email,
                "password": TEST_USER_PASSWORD,
                "name": "Training Test",
            },
            timeout=10,
        )
        assert resp.status_code in (200, 201), f"Register failed: {resp.text}"
        mark_email_verified(email)
        return resp.json()["token"]

    def test_full_flow(
        self,
        backend_server: str,
        auth_token: str,
        fake_wake_word_wavs: list[Path],
    ) -> None:
        """The golden path: upload 10 samples → train → get model."""
        # Step 1: Upload 10 recordings
        recording_ids = []
        for wav_path in fake_wake_word_wavs:
            with open(wav_path, "rb") as f:
                resp = requests.post(
                    f"{backend_server}/api/recordings/upload",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    files={"file": (wav_path.name, f, "audio/wav")},
                    data={"wake_word": "goldentest"},
                    timeout=10,
                )
            assert resp.status_code == 200
            rid = resp.json().get("recording_id") or resp.json().get("id")
            recording_ids.append(rid)

        assert len(recording_ids) == 10

        # Step 2: Start training
        resp = requests.post(
            f"{backend_server}/api/training/start",
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "wake_word": "goldentest",
                "recording_ids": recording_ids,
                "epochs": 5,  # Few epochs for fast test
            },
            timeout=10,
        )
        assert resp.status_code in (200, 202), f"Training start failed: {resp.text}"
        job_id = resp.json()["job_id"]

        # Step 3: Poll training status until complete (max 5 min)
        deadline = time.monotonic() + 300
        final_status = None
        while time.monotonic() < deadline:
            resp = requests.get(
                f"{backend_server}/api/training/status/{job_id}",
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=10,
            )
            assert resp.status_code == 200
            status_data = resp.json()
            final_status = status_data["status"]

            if final_status in ("completed", "failed"):
                break
            time.sleep(2)

        assert final_status == "completed", f"Training did not complete: {status_data}"
        model_id = status_data.get("model_id")
        assert model_id is not None, "No model_id in completed job"

        # Step 4: Download the trained model
        resp = requests.get(
            f"{backend_server}/api/models/{model_id}/download",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=30,
        )
        assert resp.status_code == 200
        assert len(resp.content) > 0
        # ONNX files start with specific magic bytes (protobuf)
        assert resp.content[:2] == b"\x08\x09" or len(resp.content) > 1000

        # Step 5: Get model config
        resp = requests.get(
            f"{backend_server}/api/models/{model_id}/config",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=10,
        )
        assert resp.status_code == 200
        config = resp.json()
        assert "d_prime" in config or "architecture" in config

        # Step 6: List models
        resp = requests.get(
            f"{backend_server}/api/models",
            headers={"Authorization": f"Bearer {auth_token}"},
            timeout=10,
        )
        assert resp.status_code == 200
        models = resp.json()
        assert any(
            m.get("wake_word") == "goldentest" or m.get("id") == model_id
            for m in models
        )
