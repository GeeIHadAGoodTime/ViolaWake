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
from app.email_service import EmailService
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

    async def commit(self) -> None:
        await self.flush()

    async def rollback(self) -> None:
        self._pending = None


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
        fake_request,
        RegisterRequest(email=email, password="TestPass123!", name="Register Test"),
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
        fake_request,
        RegisterRequest(email=email, password="TestPass123!", name="Verify Test"),
        fake_db,
    )

    verification_token = fake_email_service.verification_emails[0]["token"]
    response = await auth_routes.verify_email(
        fake_request,
        VerifyEmailRequest(token=verification_token),
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
        fake_request,
        RegisterRequest(email=email, password="OriginalPass123!", name="Reset Test"),
        fake_db,
    )

    forgot_response = await auth_routes.forgot_password(
        fake_request,
        ForgotPasswordRequest(email=email),
        fake_db,
    )
    reset_token = fake_email_service.password_reset_emails[0]["token"]
    reset_response = await auth_routes.reset_password(
        fake_request,
        ResetPasswordRequest(token=reset_token, password="NewPass123!"),
        fake_db,
    )
    login_response = await auth_routes.login(
        fake_request,
        LoginRequest(email=email, password="NewPass123!"),
        fake_db,
    )

    assert forgot_response.message.startswith("If an account exists")
    assert reset_response.message == "Password reset successfully"
    assert len(fake_email_service.password_reset_emails) == 1
    assert login_response.user.email == email
    assert login_response.user.email_verified is False


# ---------------------------------------------------------------------------
# Gate: verification-email-server-side-link
#
# Regression guard for the production incident where every real signup after
# the frontend was moved behind a CDN redirect stayed unverified: the
# verification email linked to the client-rendered SPA route
# (`{console}/verify-email?token=`), which the CDN 308-redirected away from,
# so the SPA page never mounted and the verify API was never called. The fix
# links the email to a server-side GET endpoint that verifies on click. These
# tests fail on the old shape (SPA link + no GET handler) and pass on the fix.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verification_email_links_to_backend_get_endpoint(monkeypatch) -> None:
    """The verification email must link to the backend GET verify endpoint,
    not the client-rendered SPA route that a CDN redirect can strip."""
    captured: dict[str, str] = {}

    async def _capture(self, to: str, subject: str, html: str) -> bool:  # noqa: ANN001
        captured["html"] = html
        return True

    monkeypatch.setattr(EmailService, "_send_email", _capture)
    svc = EmailService(
        api_key="test-key",
        console_base_url="https://violawake.com",
        api_base_url="https://api.violawake.com",
    )

    await svc.send_verification_email(to="user@example.com", token="TOK123", name="User")

    html = captured["html"]
    # New shape: server-side backend endpoint.
    assert "https://api.violawake.com/api/auth/verify-email?token=TOK123" in html
    # Old shape (the bug): a bare SPA route on the console domain.
    assert 'href="https://violawake.com/verify-email' not in html


@pytest.mark.asyncio
async def test_get_verify_email_link_marks_verified_and_redirects(
    fake_db: FakeSession,
    fake_request,
    fake_email_service: FakeEmailService,
) -> None:
    """Clicking the emailed GET link verifies the account server-side and
    redirects the browser back to the console signed-in."""
    email = f"getverify_{time.time_ns()}@example.com"
    await auth_routes.register(
        fake_request,
        RegisterRequest(email=email, password="TestPass123!", name="Get Verify"),
        fake_db,
    )
    token = fake_email_service.verification_emails[0]["token"]

    response = await auth_routes.verify_email_link(fake_request, token, fake_db)

    assert response.status_code == 303
    assert "verified=1" in response.headers["location"]
    assert fake_db.users_by_email[email].email_verified is True
    assert len(fake_email_service.welcome_emails) == 1


@pytest.mark.asyncio
async def test_get_verify_email_link_bad_token_redirects_to_error(
    fake_db: FakeSession,
    fake_request,
) -> None:
    """A bad/expired token must redirect to login with an error flag rather
    than surfacing a raw JSON 400 to a human clicking an email link."""
    response = await auth_routes.verify_email_link(fake_request, "not-a-valid-token", fake_db)

    assert response.status_code == 303
    assert "verify_error=1" in response.headers["location"]


def test_client_ip_ignores_x_forwarded_for_when_no_trusted_proxy(monkeypatch, fake_request) -> None:
    fake_request.headers = {"x-forwarded-for": "203.0.113.10, 198.51.100.20"}
    monkeypatch.setattr(auth_routes.settings, "trusted_proxy_count", 0)
    assert auth_routes._client_ip(fake_request) == "127.0.0.1"


def test_client_ip_uses_nth_from_right_x_forwarded_for(monkeypatch, fake_request) -> None:
    fake_request.headers = {"x-forwarded-for": "198.51.100.10, 198.51.100.20, 198.51.100.30"}
    monkeypatch.setattr(auth_routes.settings, "trusted_proxy_count", 2)
    assert auth_routes._client_ip(fake_request) == "198.51.100.20"
