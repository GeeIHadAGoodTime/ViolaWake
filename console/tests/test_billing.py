"""Billing and Stripe webhook tests for the ViolaWake Console backend."""

from __future__ import annotations

import asyncio
import io
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

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
    """Generate a valid mono WAV file with a 440Hz sine tone.

    Previous implementation wrote all-zero PCM data which silently passed
    validation (before the energy check existed) and polluted training data.
    """
    import math
    import struct as _struct

    frame_count = int(duration * sr)
    samples = []
    for i in range(frame_count):
        # 440Hz sine at ~50% amplitude (16384 out of 32767)
        value = int(16384 * math.sin(2 * math.pi * 440 * i / sr))
        samples.append(_struct.pack("<h", value))
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
    email = f"billing_{time.time_ns()}@example.com"
    response = client.post(
        "/api/auth/register",
        json={"email": email, "password": "TestPass123!", "name": "Billing Test"},
    )
    assert response.status_code in (200, 201), response.text
    data = response.json()

    import sys

    if BACKEND_DIR not in sys.path:
        sys.path.insert(0, BACKEND_DIR)

    from app.auth import create_email_verification_token

    verify_response = client.post(
        "/api/auth/verify-email",
        json={"token": create_email_verification_token(data["user"]["id"])},
    )
    assert verify_response.status_code == 200, verify_response.text

    return {
        "email": email,
        "user_id": data["user"]["id"],
        "headers": {"Authorization": f"Bearer {data['token']}"},
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


async def seed_subscription(
    *,
    user_id: int,
    tier: str = "developer",
    status: str = "active",
    stripe_customer_id: str | None = "cus_test_123",
    stripe_subscription_id: str | None = "sub_test_123",
) -> None:
    """Insert a subscription row directly for webhook setup."""
    from app.database import async_session_factory
    from app.models import Subscription

    async with async_session_factory() as session:
        session.add(
            Subscription(
                user_id=user_id,
                tier=tier,
                status=status,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
                current_period_end=datetime.now(timezone.utc),
            )
        )
        await session.commit()


async def get_subscription_row(user_id: int) -> dict[str, object] | None:
    """Fetch the subscription row for assertions that are not exposed via the API."""
    from app.database import async_session_factory
    from app.models import Subscription

    async with async_session_factory() as session:
        result = await session.execute(
            select(Subscription).where(Subscription.user_id == user_id)
        )
        sub = result.scalar_one_or_none()
        if sub is None:
            return None
        return {
            "tier": sub.tier,
            "status": sub.status,
            "stripe_customer_id": sub.stripe_customer_id,
            "stripe_subscription_id": sub.stripe_subscription_id,
            "current_period_end": sub.current_period_end,
        }


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI test client."""
    try:
        from app.main import app
    except ImportError as exc:
        pytest.skip(f"Backend not yet built: {exc}")

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_user(client) -> dict[str, object]:
    """Create an authenticated user for a test."""
    return register_user(client)


@pytest.fixture
def billing_settings(monkeypatch):
    """Enable billing and provide deterministic Stripe settings."""
    from app.config import settings

    monkeypatch.setattr(settings, "stripe_secret_key", "sk_test_123")
    monkeypatch.setattr(settings, "stripe_webhook_secret", "whsec_test_123")
    monkeypatch.setattr(settings, "stripe_price_developer", "price_developer_test")
    monkeypatch.setattr(settings, "stripe_price_business", "price_business_test")
    monkeypatch.setattr(settings, "console_base_url", "http://localhost:5173")
    return settings


@pytest.fixture
def mock_training_queue():
    """Patch the queue dependency so training requests only exercise quota logic."""
    queue = SimpleNamespace()
    job_ids = iter(range(1, 1000))
    queue.submit_job = AsyncMock(side_effect=lambda **_: next(job_ids))

    with patch("app.routes.jobs.init_job_queue", new=AsyncMock(return_value=queue)):
        yield queue


def make_stripe_mock() -> MagicMock:
    """Build a Stripe mock with the members billing routes use."""

    class FakeSignatureVerificationError(Exception):
        pass

    stripe = MagicMock()
    stripe.error.SignatureVerificationError = FakeSignatureVerificationError
    return stripe


class TestBillingRoutes:

    def test_get_subscription_returns_free_tier_for_new_user(self, client, auth_user) -> None:
        response = client.get("/api/billing/subscription", headers=auth_user["headers"])

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["tier"] == "free"
        assert data["status"] == "active"
        assert data["usage"]["models_used"] == 0
        assert data["usage"]["models_limit"] == 3

    def test_get_usage_returns_zero_for_new_user(self, client, auth_user) -> None:
        response = client.get("/api/billing/usage", headers=auth_user["headers"])

        assert response.status_code == 200, response.text
        data = response.json()
        assert data["models_used"] == 0
        assert data["models_limit"] == 3

    def test_free_tier_user_can_start_three_training_jobs(
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

        usage_response = client.get("/api/billing/usage", headers=auth_user["headers"])
        assert usage_response.status_code == 200, usage_response.text
        assert usage_response.json()["models_used"] == 3
        assert mock_training_queue.submit_job.await_count == 3

    def test_free_tier_user_gets_403_on_fourth_training_job_attempt(
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
        assert "Monthly training limit reached" in response.json()["detail"]
        assert response.headers["X-Upgrade-URL"] == "/pricing"
        assert mock_training_queue.submit_job.await_count == 3

    def test_checkout_returns_checkout_url(self, client, auth_user, billing_settings) -> None:
        stripe = make_stripe_mock()
        stripe.Customer.create.return_value = SimpleNamespace(id="cus_checkout_123")
        stripe.checkout.Session.create.return_value = SimpleNamespace(
            id="cs_checkout_123",
            url="https://stripe.test/checkout/cs_checkout_123",
        )

        with patch("app.routes.billing._get_stripe", return_value=stripe):
            response = client.post(
                "/api/billing/checkout",
                headers=auth_user["headers"],
                json={"tier": "developer"},
            )

        assert response.status_code == 200, response.text
        assert response.json()["checkout_url"] == "https://stripe.test/checkout/cs_checkout_123"
        stripe.checkout.Session.create.assert_called_once()

    def test_checkout_without_auth_requires_auth(self, client, billing_settings) -> None:
        response = client.post("/api/billing/checkout", json={"tier": "developer"})

        assert response.status_code == 403, response.text
        assert response.json()["detail"] == "Not authenticated"


class TestBillingWebhooks:

    def test_webhook_checkout_session_completed_updates_subscription_tier(
        self,
        client,
        auth_user,
        billing_settings,
    ) -> None:
        unique_suffix = time.time_ns()
        customer_id = f"cus_checkout_complete_{unique_suffix}"
        subscription_id = f"sub_checkout_complete_{unique_suffix}"
        period_end = int(datetime(2030, 1, 1, tzinfo=timezone.utc).timestamp())
        event = {
            "id": "evt_checkout_complete",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "id": "cs_completed_123",
                    "customer": customer_id,
                    "subscription": subscription_id,
                    "metadata": {
                        "violawake_user_id": str(auth_user["user_id"]),
                        "tier": "developer",
                    },
                }
            },
        }
        stripe = make_stripe_mock()
        stripe.Webhook.construct_event.return_value = event
        stripe.Subscription.retrieve.return_value = SimpleNamespace(current_period_end=period_end)

        with patch("app.routes.billing._get_stripe", return_value=stripe):
            response = client.post(
                "/api/billing/webhook",
                content=b'{"test": true}',
                headers={"stripe-signature": "sig_test_123"},
            )

        assert response.status_code == 200, response.text
        assert response.json() == {"status": "ok"}

        subscription_response = client.get(
            "/api/billing/subscription",
            headers=auth_user["headers"],
        )
        assert subscription_response.status_code == 200, subscription_response.text
        data = subscription_response.json()
        assert data["tier"] == "developer"
        assert data["status"] == "active"
        assert data["current_period_end"] is not None

        row = asyncio.run(get_subscription_row(auth_user["user_id"]))
        assert row is not None
        assert row["stripe_customer_id"] == customer_id
        assert row["stripe_subscription_id"] == subscription_id

    def test_webhook_subscription_deleted_downgrades_to_free(
        self,
        client,
        auth_user,
        billing_settings,
    ) -> None:
        unique_suffix = time.time_ns()
        customer_id = f"cus_delete_{unique_suffix}"
        subscription_id = f"sub_delete_{unique_suffix}"
        asyncio.run(
            seed_subscription(
                user_id=auth_user["user_id"],
                tier="developer",
                status="active",
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
            )
        )

        event = {
            "id": "evt_subscription_deleted",
            "type": "customer.subscription.deleted",
            "data": {"object": {"id": subscription_id}},
        }
        stripe = make_stripe_mock()
        stripe.Webhook.construct_event.return_value = event

        with patch("app.routes.billing._get_stripe", return_value=stripe):
            response = client.post(
                "/api/billing/webhook",
                content=b'{"test": true}',
                headers={"stripe-signature": "sig_test_123"},
            )

        assert response.status_code == 200, response.text

        subscription_response = client.get(
            "/api/billing/subscription",
            headers=auth_user["headers"],
        )
        assert subscription_response.status_code == 200, subscription_response.text
        data = subscription_response.json()
        assert data["tier"] == "free"
        assert data["status"] == "canceled"

        row = asyncio.run(get_subscription_row(auth_user["user_id"]))
        assert row is not None
        assert row["stripe_subscription_id"] is None
        assert row["current_period_end"] is None

    def test_invalid_webhook_signature_returns_400(
        self,
        client,
        billing_settings,
    ) -> None:
        stripe = make_stripe_mock()
        stripe.Webhook.construct_event.side_effect = stripe.error.SignatureVerificationError(
            "invalid signature"
        )

        with patch("app.routes.billing._get_stripe", return_value=stripe):
            response = client.post(
                "/api/billing/webhook",
                content=b'{"test": true}',
                headers={"stripe-signature": "sig_invalid"},
            )

        assert response.status_code == 400, response.text
        assert response.json()["detail"] == "Invalid webhook signature."
