"""Focused auth route tests for verification and password reset flows."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.models import User
from app.auth import reset_download_tokens
from app.routes import auth as auth_routes
from app.schemas import (
    ForgotPasswordRequest,
    LoginRequest,
    RegisterRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
)


class FakeResult:
    def __init__(self, value: User | None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> User | None:
        return self._value


class FakeSession:
    def __init__(self) -> None:
        self.users_by_id: dict[int, User] = {}
        self.users_by_email: dict[str, User] = {}
        self._next_id = 1
        self._pending: User | None = None

    async def execute(self, statement) -> FakeResult:
        clause = statement.whereclause
        column = clause.left.name
        value = clause.right.value
        if column == "email":
            return FakeResult(self.users_by_email.get(value))
        if column == "id":
            return FakeResult(self.users_by_id.get(value))
        raise AssertionError(f"Unexpected query column: {column}")

    def add(self, user: User) -> None:
        if getattr(user, "email_verified", None) is None:
            user.email_verified = False
        self._pending = user

    async def flush(self) -> None:
        if self._pending is None:
            return
        if getattr(self._pending, "id", None) is None:
            self._pending.id = self._next_id
            self._next_id += 1
        self.users_by_id[self._pending.id] = self._pending
        self.users_by_email[self._pending.email] = self._pending
        self._pending = None


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


@pytest.fixture(autouse=True)
def clear_rate_limits() -> None:
    auth_routes.reset_rate_limits()
    reset_download_tokens()


@pytest.fixture
def fake_db() -> FakeSession:
    return FakeSession()


@pytest.fixture
def fake_request():
    return SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))


@pytest.fixture
def fake_email_service(monkeypatch) -> FakeEmailService:
    service = FakeEmailService()
    monkeypatch.setattr(auth_routes, "get_email_service", lambda: service)
    return service


@pytest.mark.asyncio
async def test_register_sends_verification_email(
    fake_db: FakeSession,
    fake_request,
    fake_email_service: FakeEmailService,
) -> None:
    email = f"register_{time.time_ns()}@example.com"

    response = await auth_routes.register(
        RegisterRequest(email=email, password="TestPass123!", name="Register Test"),
        fake_request,
        fake_db,
    )

    assert response.user.email == email
    assert response.user.email_verified is False
    assert fake_db.users_by_email[email].email_verified is False
    assert len(fake_email_service.verification_emails) == 1
    assert fake_email_service.verification_emails[0]["to"] == email


@pytest.mark.asyncio
async def test_verify_email_marks_user_verified_and_sends_welcome(
    fake_db: FakeSession,
    fake_request,
    fake_email_service: FakeEmailService,
) -> None:
    email = f"verify_{time.time_ns()}@example.com"
    await auth_routes.register(
        RegisterRequest(email=email, password="TestPass123!", name="Verify Test"),
        fake_request,
        fake_db,
    )

    verification_token = fake_email_service.verification_emails[0]["token"]
    response = await auth_routes.verify_email(
        VerifyEmailRequest(token=verification_token),
        fake_request,
        fake_db,
    )

    assert response.message == "Email verified successfully"
    assert fake_db.users_by_email[email].email_verified is True
    assert len(fake_email_service.welcome_emails) == 1
    assert fake_email_service.welcome_emails[0]["to"] == email


@pytest.mark.asyncio
async def test_forgot_password_and_reset_password_flow(
    fake_db: FakeSession,
    fake_request,
    fake_email_service: FakeEmailService,
) -> None:
    email = f"reset_{time.time_ns()}@example.com"
    await auth_routes.register(
        RegisterRequest(email=email, password="OriginalPass123!", name="Reset Test"),
        fake_request,
        fake_db,
    )

    forgot_response = await auth_routes.forgot_password(
        ForgotPasswordRequest(email=email),
        fake_request,
        fake_db,
    )
    reset_token = fake_email_service.password_reset_emails[0]["token"]
    reset_response = await auth_routes.reset_password(
        ResetPasswordRequest(token=reset_token, password="NewPass123!"),
        fake_request,
        fake_db,
    )
    login_response = await auth_routes.login(
        LoginRequest(email=email, password="NewPass123!"),
        fake_request,
        fake_db,
    )

    assert forgot_response.message.startswith("If an account exists")
    assert reset_response.message == "Password reset successfully"
    assert len(fake_email_service.password_reset_emails) == 1
    assert login_response.user.email == email
    assert login_response.user.email_verified is False


def test_client_ip_ignores_x_forwarded_for_when_no_trusted_proxy(monkeypatch, fake_request) -> None:
    fake_request.headers = {"x-forwarded-for": "203.0.113.10, 198.51.100.20"}
    monkeypatch.setattr(auth_routes.settings, "trusted_proxy_count", 0)
    assert auth_routes._client_ip(fake_request) == "127.0.0.1"


def test_client_ip_uses_nth_from_right_x_forwarded_for(monkeypatch, fake_request) -> None:
    fake_request.headers = {"x-forwarded-for": "198.51.100.10, 198.51.100.20, 198.51.100.30"}
    monkeypatch.setattr(auth_routes.settings, "trusted_proxy_count", 2)
    assert auth_routes._client_ip(fake_request) == "198.51.100.20"
