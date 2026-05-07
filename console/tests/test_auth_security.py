"""Focused auth security regression tests."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.auth import create_password_reset_token, reset_download_tokens
from app.models import User
from app.routes import auth as auth_routes
from app.schemas import DeleteAccountRequest, LoginRequest, ResetPasswordRequest
from tests.test_auth_email_routes import FakeSession


@pytest.fixture(autouse=True)
def clear_auth_state() -> None:
    auth_routes.reset_rate_limits()
    reset_download_tokens()


@pytest.fixture
def fake_db() -> FakeSession:
    return FakeSession()


@pytest.fixture
def fake_request():
    return SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))


async def _seed_user(
    db: FakeSession,
    *,
    email: str | None = None,
    password_hash: str = "stored-password-hash",
) -> User:
    user = User(
        email=email or f"security_{time.time_ns()}@example.com",
        password_hash=password_hash,
        name="Security Test",
        email_verified=True,
        failed_login_count=0,
        locked_until=None,
    )
    db.add(user)
    await db.flush()
    return user


@pytest.mark.asyncio
async def test_unknown_user_login_runs_dummy_bcrypt_check(
    fake_db: FakeSession,
    fake_request,
    monkeypatch,
) -> None:
    calls: list[tuple[bytes, bytes]] = []

    def fake_checkpw(password: bytes, hashed: bytes) -> bool:
        calls.append((password, hashed))
        return False

    monkeypatch.setattr(auth_routes.bcrypt, "checkpw", fake_checkpw)

    with pytest.raises(HTTPException) as exc_info:
        await auth_routes.login(
            fake_request,
            LoginRequest(email="missing@example.com", password="WrongPass123!"),
            fake_db,
        )

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid email or password"
    assert len(calls) == 1
    assert calls[0][0] == b"dummy"


@pytest.mark.asyncio
async def test_wrong_passwords_increment_and_lock_account(
    fake_db: FakeSession,
    fake_request,
    monkeypatch,
) -> None:
    user = await _seed_user(fake_db)
    monkeypatch.setattr(auth_routes, "verify_password", lambda plain, hashed: False)

    for attempt in range(1, auth_routes._LOGIN_LOCKOUT_THRESHOLD):
        with pytest.raises(HTTPException) as exc_info:
            await auth_routes.login(
                fake_request,
                LoginRequest(email=user.email, password="WrongPass123!"),
                fake_db,
            )
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert user.failed_login_count == attempt
        assert user.locked_until is None

    with pytest.raises(HTTPException) as exc_info:
        await auth_routes.login(
            fake_request,
            LoginRequest(email=user.email, password="WrongPass123!"),
            fake_db,
        )

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert user.failed_login_count == 0
    assert user.locked_until is not None
    assert user.locked_until > datetime.now(timezone.utc)

    with pytest.raises(HTTPException) as locked_exc:
        await auth_routes.login(
            fake_request,
            LoginRequest(email=user.email, password="WrongPass123!"),
            fake_db,
        )

    assert locked_exc.value.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert "Account temporarily locked" in locked_exc.value.detail


@pytest.mark.asyncio
async def test_successful_login_resets_failed_count_and_lockout(
    fake_db: FakeSession,
    fake_request,
    monkeypatch,
) -> None:
    user = await _seed_user(fake_db)
    user.failed_login_count = 4
    user.locked_until = datetime.now(timezone.utc) - timedelta(minutes=1)
    monkeypatch.setattr(auth_routes, "verify_password", lambda plain, hashed: plain == "RightPass123!")

    response = await auth_routes.login(
        fake_request,
        LoginRequest(email=user.email, password="RightPass123!"),
        fake_db,
    )

    assert response.token
    assert response.user.email == user.email
    assert user.failed_login_count == 0
    assert user.locked_until is None


@pytest.mark.asyncio
async def test_password_reset_token_is_single_use(
    fake_db: FakeSession,
    fake_request,
    monkeypatch,
) -> None:
    user = await _seed_user(fake_db)
    token = create_password_reset_token(user.id)
    monkeypatch.setattr(auth_routes, "hash_password", lambda password: f"hashed:{password}")

    response = await auth_routes.reset_password(
        fake_request,
        ResetPasswordRequest(token=token, password="NewPass123!"),
        fake_db,
    )

    assert response.message == "Password reset successfully"
    assert user.password_hash == "hashed:NewPass123!"

    with pytest.raises(HTTPException) as exc_info:
        await auth_routes.reset_password(
            fake_request,
            ResetPasswordRequest(token=token, password="AnotherPass123!"),
            fake_db,
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "already used" in exc_info.value.detail


def test_delete_account_requires_password_field(fake_db: FakeSession) -> None:
    user = User(
        id=1,
        email="delete-validation@example.com",
        password_hash="stored-password-hash",
        name="Delete Validation",
        email_verified=True,
        failed_login_count=0,
        locked_until=None,
    )
    app = FastAPI()

    async def get_current_user_override() -> User:
        return user

    async def get_db_override() -> FakeSession:
        return fake_db

    app.dependency_overrides[auth_routes.get_current_user] = get_current_user_override
    app.dependency_overrides[auth_routes.get_db] = get_db_override
    app.include_router(auth_routes.router)

    response = TestClient(app).request(
        "DELETE",
        "/api/auth/account",
        json={"current_password": "RightPass123!"},
    )

    assert response.status_code == 422
    assert any(error["loc"][-1] == "password" for error in response.json()["detail"])


@pytest.mark.asyncio
async def test_delete_account_wrong_password_returns_401(
    fake_db: FakeSession,
    monkeypatch,
) -> None:
    user = await _seed_user(fake_db)
    monkeypatch.setattr(auth_routes, "verify_password", lambda plain, hashed: False)

    with pytest.raises(HTTPException) as exc_info:
        await auth_routes.delete_account(
            DeleteAccountRequest(password="WrongPass123!"),
            user,
            fake_db,
        )

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Incorrect password"
