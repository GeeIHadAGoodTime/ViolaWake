"""Stripe end-to-end checkout probes.

These tests are intentionally skipped unless the deployment is known to use
Stripe test mode. They enter Stripe's documented 4242 card into Checkout.
"""

from __future__ import annotations

import os
import re
import time

import httpx
import pytest

from conftest import TEST_PASSWORD, auth_headers, join_url

try:
    from playwright.sync_api import Page, expect
except ImportError:  # pragma: no cover
    Page = object  # type: ignore[assignment,misc]
    expect = None  # type: ignore[assignment]


pytestmark = pytest.mark.live


def _register_sync(api_base_url: str, email: str) -> str:
    response = httpx.post(
        join_url(api_base_url, "/api/auth/register"),
        json={"email": email, "password": TEST_PASSWORD, "name": "Live Stripe Probe"},
        timeout=20.0,
    )
    assert response.status_code in (200, 201), response.text
    return response.json()["token"]


def test_stripe_checkout_test_card_updates_subscription(
    api_base_url: str,
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    if os.getenv("VIOLAWAKE_STRIPE_TEST_MODE") != "1":
        pytest.skip("Set VIOLAWAKE_STRIPE_TEST_MODE=1 only when live Stripe keys are test-mode keys.")

    email = email_factory("stripe")
    token = _register_sync(api_base_url, email)

    live_page.goto(site_url)
    live_page.evaluate("token => localStorage.setItem('token', token)", token)
    live_page.goto(join_url(site_url, "/pricing"), wait_until="networkidle")
    live_page.locator(".pricing-card").filter(has_text="Developer").locator(
        "button",
        has_text="Get Started",
    ).click()
    live_page.wait_for_url(re.compile(r"https://checkout\.stripe\.com/.*"), timeout=20_000)

    live_page.locator('input[name="cardNumber"], input[placeholder*="1234"]').first.fill("4242424242424242")
    live_page.locator('input[name="cardExpiry"], input[placeholder*="MM"]').first.fill("1234")
    live_page.locator('input[name="cardCvc"], input[placeholder*="CVC"]').first.fill("123")
    postal = live_page.locator('input[name="billingPostalCode"], input[placeholder*="ZIP"]').first
    if postal.count() > 0:
        postal.fill("12345")
    live_page.get_by_role("button", name=re.compile("subscribe|start|pay", re.I)).click()
    live_page.wait_for_url(re.compile(r".*/billing.*|.*/dashboard.*"), timeout=60_000)

    deadline = time.monotonic() + 30
    last_body: object = None
    with httpx.Client(base_url=api_base_url, timeout=20.0) as client:
        while time.monotonic() < deadline:
            status_response = client.get("/api/billing/status", headers=auth_headers(token))
            if status_response.status_code == 404:
                status_response = client.get("/api/billing/subscription", headers=auth_headers(token))
            last_body = status_response.text
            if status_response.status_code == 200:
                body = status_response.json()
                if body.get("tier") in {"developer", "business"} and body.get("status") == "active":
                    return
            time.sleep(3)

    raise AssertionError(f"Subscription did not update within 30s: {last_body}")
