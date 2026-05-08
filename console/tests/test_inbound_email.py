"""Tests for inbound support email auto-replies."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

pytestmark = pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")


@pytest.fixture(scope="module")
def client():
    backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


class FakeEmailService:
    def __init__(self) -> None:
        self.autoreplies: list[dict[str, str]] = []

    async def send_support_autoreply(self, to: str, ticket_reference: str) -> bool:
        self.autoreplies.append({"to": to, "ticket_reference": ticket_reference})
        return True


def test_inbound_email_autoreply_dedupes_sender_for_24h(client, monkeypatch) -> None:
    from app.config import settings
    from app.routes import inbound_email

    monkeypatch.setattr(settings, "email_inbound_webhook_secret", "test-secret")
    monkeypatch.setattr(settings, "support_autoreply_window_hours", 24)

    fake_email = FakeEmailService()
    monkeypatch.setattr(inbound_email, "get_email_service", lambda: fake_email)

    sender = f"support_{time.time_ns()}@example.com"
    payload = {
        "from": f"Support Sender <{sender}>",
        "to": "hello@violawake.com",
        "subject": "Launch question",
    }
    headers = {"X-ViolaWake-Email-Secret": "test-secret"}

    first = client.post("/api/email/inbound", headers=headers, json=payload)
    second = client.post("/api/email/inbound", headers=headers, json=payload)

    assert first.status_code == 200, first.text
    assert first.json()["status"] == "sent"
    assert first.json()["sent"] is True
    assert first.json()["ticket_reference"].startswith("VW-")

    assert second.status_code == 200, second.text
    assert second.json()["status"] == "deduped"
    assert second.json()["sent"] is False
    assert second.json()["ticket_reference"] == first.json()["ticket_reference"]
    assert fake_email.autoreplies == [
        {"to": sender, "ticket_reference": first.json()["ticket_reference"]}
    ]


def test_inbound_email_requires_shared_secret(client, monkeypatch) -> None:
    from app.config import settings

    monkeypatch.setattr(settings, "email_inbound_webhook_secret", "test-secret")

    response = client.post(
        "/api/email/inbound",
        headers={"X-ViolaWake-Email-Secret": "wrong"},
        json={"from": "sender@example.com"},
    )

    assert response.status_code == 403
