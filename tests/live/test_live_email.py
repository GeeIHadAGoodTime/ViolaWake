"""Resend/email verification probes for the live deployment."""

from __future__ import annotations

import os
import re
import time
import asyncio
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from conftest import AuthUser, auth_headers, json_or_text, register_live_user


pytestmark = pytest.mark.live


def _extract_verification_token(text: str) -> str | None:
    match = re.search(r"https?://[^\s\"'<>]+/verify-email\?token=([^\s\"'<>]+)", text)
    if match:
        return match.group(1).rstrip(").,;")
    for url_match in re.finditer(r"https?://[^\s\"'<>]+", text):
        parsed = urlparse(url_match.group(0))
        if parsed.path.endswith("/verify-email"):
            token = parse_qs(parsed.query).get("token", [None])[0]
            if token:
                return token
    return None


async def _wait_mailosaur(email: str, timeout_s: int = 60) -> str | None:
    api_key = os.getenv("VIOLAWAKE_LIVE_MAILOSAUR_KEY")
    server_id = os.getenv("VIOLAWAKE_LIVE_MAILOSAUR_SERVER_ID")
    if not api_key or not server_id:
        return None

    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=15.0, auth=(api_key, "")) as client:
        while time.monotonic() < deadline:
            response = await client.post(
                "https://mailosaur.com/api/messages/search",
                params={"server": server_id},
                json={"sentTo": email},
            )
            if response.status_code == 200:
                items = response.json().get("items", [])
                if items:
                    message_id = items[0]["id"]
                    message = await client.get(f"https://mailosaur.com/api/messages/{message_id}")
                    if message.status_code == 200:
                        body = message.text
                        token = _extract_verification_token(body)
                        if token:
                            return token
            await asyncio.sleep(5)
    return None


async def _wait_webhook(timeout_s: int = 60) -> str | None:
    webhook_url = os.getenv("VIOLAWAKE_LIVE_WEBHOOK_URL")
    if not webhook_url:
        return None

    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=15.0) as client:
        while time.monotonic() < deadline:
            response = await client.get(webhook_url)
            if response.status_code == 200:
                token = _extract_verification_token(response.text)
                if token:
                    return token
            await asyncio.sleep(5)
    return None


async def _verify_token_and_assert_me(
    http_session: httpx.AsyncClient,
    token: str,
    user: AuthUser,
) -> None:
    verify = await http_session.post("/api/auth/verify-email", json={"token": token})
    assert verify.status_code == 200, json_or_text(verify)
    me = await http_session.get("/api/auth/me", headers=auth_headers(user.token))
    assert me.status_code == 200, json_or_text(me)
    assert me.json()["email_verified"] is True


@pytest.mark.smoke
async def test_resend_verification_delivery_or_auto_verify(
    http_session: httpx.AsyncClient,
    email_factory,
    record_property: pytest.RecordProperty,
) -> None:
    mailosaur_server = os.getenv("VIOLAWAKE_LIVE_MAILOSAUR_SERVER_ID")
    if mailosaur_server and os.getenv("VIOLAWAKE_LIVE_MAILOSAUR_KEY"):
        email = f"violawake-{int(time.time())}@{mailosaur_server}.mailosaur.net"
    else:
        email = os.getenv("VIOLAWAKE_LIVE_EMAIL") or email_factory("email")

    user = await register_live_user(
        http_session,
        email,
        name="Live Email Probe",
        require_token=False,
    )
    record_property("email", email)
    record_property("email_verified_on_register", user.is_verified)

    if user.is_verified:
        record_property("resend_outcome", "auto_verified_no_email_required")
        return

    if not user.token:
        record_property("resend_outcome", "unverified_no_token")
        pytest.skip("Live registration returned no token and no auto-verification; email verification cannot complete.")

    if os.getenv("VIOLAWAKE_LIVE_MAILOSAUR_KEY") and mailosaur_server:
        token = await _wait_mailosaur(email)
        record_property("resend_outcome", "mailosaur_timeout" if token is None else "mailosaur_received")
        assert token is not None, "No verification email arrived in Mailosaur within 60s."
        await _verify_token_and_assert_me(http_session, token, user)
        return

    if os.getenv("VIOLAWAKE_LIVE_WEBHOOK_URL"):
        token = await _wait_webhook()
        record_property("resend_outcome", "webhook_timeout" if token is None else "webhook_received")
        assert token is not None, "No verification email/link arrived at webhook within 60s."
        await _verify_token_and_assert_me(http_session, token, user)
        return

    pytest.skip(
        "Registered user remained unverified and no inbox is configured. "
        "Set VIOLAWAKE_LIVE_MAILOSAUR_KEY plus VIOLAWAKE_LIVE_MAILOSAUR_SERVER_ID, "
        "or VIOLAWAKE_LIVE_WEBHOOK_URL."
    )
