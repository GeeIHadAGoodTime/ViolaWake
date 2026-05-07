from __future__ import annotations

import sqlite3
import sys
import time
import types
from datetime import datetime, timezone

from fastapi.testclient import TestClient

# The change-password tests do not exercise outbound email, but app import loads
# the email module. Keep this focused test independent from optional Resend deps.
sys.modules.setdefault(
    "resend",
    types.SimpleNamespace(
        api_key="",
        Emails=types.SimpleNamespace(send=lambda _params: None),
    ),
)

from app.auth import create_access_token, hash_password, verify_password
from app.config import settings
from app.main import app


def _create_user(email: str, password: str = "CurrentPass123!") -> int:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (email, password_hash, name, email_verified, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                email,
                hash_password(password),
                "Change Password",
                1,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _password_hash_for(email: str) -> str:
    with sqlite3.connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_change_password_updates_password() -> None:
    email = f"change_password_{time.time_ns()}@example.com"
    with TestClient(app) as client:
        user_id = _create_user(email)
        token = create_access_token(user_id)

        response = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "CurrentPass123!",
                "new_password": "NewPass123!",
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["message"] == "Password changed successfully"
        assert verify_password("NewPass123!", _password_hash_for(email))


def test_change_password_rejects_wrong_current_password() -> None:
    email = f"change_password_wrong_{time.time_ns()}@example.com"
    with TestClient(app) as client:
        user_id = _create_user(email)
        token = create_access_token(user_id)

        response = client.post(
            "/api/auth/change-password",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "current_password": "WrongPass123!",
                "new_password": "NewPass123!",
            },
        )

        assert response.status_code == 401
        assert verify_password("CurrentPass123!", _password_hash_for(email))
