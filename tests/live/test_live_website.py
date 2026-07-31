"""Playwright probes against the deployed ViolaWake frontend."""

from __future__ import annotations

import re
import time

import httpx
import pytest

from conftest import TEST_PASSWORD, join_url, register_live_user

try:
    from playwright.sync_api import Page, expect
except ImportError:  # pragma: no cover - handled by fixture skip
    Page = object  # type: ignore[assignment,misc]
    expect = None  # type: ignore[assignment]


pytestmark = pytest.mark.live


def _register_sync(api_base_url: str, email: str, *, name: str = "Live Browser User") -> str:
    response = httpx.post(
        join_url(api_base_url, "/api/auth/register"),
        json={"email": email, "password": TEST_PASSWORD, "name": name},
        timeout=20.0,
    )
    assert response.status_code in (200, 201), response.text
    return response.json()["token"]


def _fill_register(page: Page, *, name: str, email: str, password: str = TEST_PASSWORD) -> None:
    page.fill("#name", name)
    page.fill("#email", email)
    page.fill("#password", password)


def _assert_no_fatal_browser_errors(page: Page) -> None:
    assert getattr(page, "live_console_errors", []) == []
    assert getattr(page, "live_request_failures", []) == []
    assert getattr(page, "live_5xx_responses", []) == []


@pytest.mark.smoke
def test_landing_renders_core_claims(site_url: str, live_page: Page) -> None:
    live_page.goto(site_url, wait_until="networkidle")
    expect(live_page.get_by_role("heading", name=re.compile("Custom Wake Words", re.I))).to_be_visible()
    expect(live_page.get_by_text("How we compare")).to_be_visible()
    expect(live_page.locator("table.comparison-table")).to_be_visible()
    expect(live_page.get_by_text("Free / $29 / $99")).to_be_visible()
    expect(live_page.get_by_text("102KB", exact=True).first).to_be_visible()
    expect(live_page.get_by_text("10").first).to_be_visible()


def test_landing_marketing_metrics_match_claims(site_url: str, live_page: Page) -> None:
    live_page.goto(site_url, wait_until="networkidle")
    body = live_page.locator("body").inner_text()
    for claim in ("0.8%", "8.58", "<5ms", "102KB", "10"):
        assert claim in body


@pytest.mark.parametrize("path,heading", [("/privacy", "Privacy"), ("/terms", "Terms")])
def test_legal_pages_render(site_url: str, live_page: Page, path: str, heading: str) -> None:
    live_page.goto(join_url(site_url, path), wait_until="networkidle")
    expect(live_page.get_by_role("heading", name=re.compile(heading, re.I)).first).to_be_visible()


def test_register_valid_user_reaches_dashboard(
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    email = email_factory("browser-register")
    live_page.goto(join_url(site_url, "/register"))
    _fill_register(live_page, name="Browser Register", email=email)
    live_page.click('button[type="submit"]')
    live_page.wait_for_url(re.compile(r".*/dashboard"), timeout=15_000)
    expect(live_page.get_by_text("Your Wake Word Models")).to_be_visible()


def test_register_duplicate_email_shows_error(
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    email = email_factory("browser-duplicate")
    live_page.goto(join_url(site_url, "/register"))
    _fill_register(live_page, name="First Duplicate", email=email)
    live_page.click('button[type="submit"]')
    live_page.wait_for_url(re.compile(r".*/dashboard"), timeout=15_000)

    live_page.evaluate("localStorage.clear()")
    live_page.goto(join_url(site_url, "/register"))
    _fill_register(live_page, name="Second Duplicate", email=email)
    live_page.click('button[type="submit"]')
    expect(live_page.locator(".auth-error")).to_be_visible(timeout=10_000)
    expect(live_page.locator(".auth-error")).to_contain_text(re.compile("registered", re.I))


def test_register_invalid_email_validation(site_url: str, live_page: Page) -> None:
    live_page.goto(join_url(site_url, "/register"))
    _fill_register(live_page, name="Invalid Email", email="not-an-email")
    live_page.locator("#email").blur()
    assert live_page.locator("#email").evaluate("el => el.checkValidity()") is False
    expect(live_page.locator('button[type="submit"]')).to_be_disabled()


def test_register_weak_password_validation(site_url: str, live_page: Page) -> None:
    live_page.goto(join_url(site_url, "/register"))
    _fill_register(live_page, name="Weak Password", email="weak-password@example.com", password="short")
    live_page.locator("#password").blur()
    expect(live_page.get_by_text(re.compile("characters needed|minimum 8", re.I))).to_be_visible()
    expect(live_page.locator('button[type="submit"]')).to_be_disabled()


def test_xss_payload_in_name_does_not_execute(
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    dialogs: list[str] = []
    live_page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
    email = email_factory("browser-xss")
    payload = '<img src=x onerror=alert("xss")>'

    live_page.goto(join_url(site_url, "/register"))
    _fill_register(live_page, name=payload, email=email)
    live_page.click('button[type="submit"]')
    live_page.wait_for_url(re.compile(r".*/dashboard"), timeout=15_000)
    live_page.wait_for_timeout(1000)

    assert dialogs == []
    assert live_page.locator('img[src="x"]').count() == 0


def test_login_valid_user_reaches_dashboard(
    api_base_url: str,
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    email = email_factory("browser-login")
    _register_sync(api_base_url, email, name="Browser Login")

    live_page.goto(join_url(site_url, "/login"))
    live_page.fill("#email", email)
    live_page.fill("#password", TEST_PASSWORD)
    live_page.click('button[type="submit"]')
    live_page.wait_for_url(re.compile(r".*/dashboard"), timeout=15_000)
    expect(live_page.get_by_text("Your Wake Word Models")).to_be_visible()


def test_login_wrong_password_shows_error(site_url: str, live_page: Page) -> None:
    live_page.goto(join_url(site_url, "/login"))
    live_page.fill("#email", f"wrong-{time.time_ns()}@example.com")
    live_page.fill("#password", "WrongPass123!")
    live_page.click('button[type="submit"]')
    expect(live_page.locator(".auth-error")).to_be_visible(timeout=10_000)


@pytest.mark.rate_limit
def test_login_account_lockout_or_rate_limit(
    api_base_url: str,
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    if not re.match("1|true|yes", __import__("os").getenv("VIOLAWAKE_LIVE_RATE_LIMIT", ""), re.I):
        pytest.skip("Set VIOLAWAKE_LIVE_RATE_LIMIT=1 to burn live login rate-limit budget.")

    email = email_factory("browser-lockout")
    _register_sync(api_base_url, email, name="Browser Lockout")
    live_page.goto(join_url(site_url, "/login"))
    for _ in range(6):
        live_page.fill("#email", email)
        live_page.fill("#password", "WrongPass123!")
        live_page.click('button[type="submit"]')
        live_page.wait_for_timeout(750)

    text = live_page.locator("body").inner_text().lower()
    assert "too many" in text or "rate" in text or "lock" in text or "429" in text


def test_forgot_password_renders_and_submits(
    site_url: str,
    live_page: Page,
    email_factory,
) -> None:
    live_page.goto(join_url(site_url, "/forgot-password"))
    expect(live_page.get_by_role("heading", name=re.compile("Forgot password", re.I))).to_be_visible()
    live_page.fill("#email", email_factory("forgot"))
    live_page.click('button[type="submit"]')
    expect(live_page.get_by_text(re.compile("reset link", re.I))).to_be_visible(timeout=10_000)


def test_cookie_consent_banner_accepts(site_url: str, live_page: Page) -> None:
    live_page.goto(site_url, wait_until="networkidle")
    accept = live_page.get_by_role("button", name=re.compile("Accept", re.I))
    if accept.count() == 0:
        pytest.skip("Cookie banner was already accepted in this browser context.")
    expect(accept).to_be_visible()
    accept.click()
    expect(accept).not_to_be_visible(timeout=5000)


def test_bogus_route_has_404_ui(site_url: str, live_page: Page) -> None:
    live_page.goto(join_url(site_url, f"/bogus-live-route-{time.time_ns()}"), wait_until="networkidle")
    body_text = live_page.locator("body").inner_text().lower()
    assert "404" in body_text or "not found" in body_text
    assert len(body_text.strip()) > 0


def test_mobile_landing_readable(site_url: str, browser_context: object) -> None:
    page = browser_context.new_page()
    page.set_viewport_size({"width": 375, "height": 667})
    page.goto(site_url, wait_until="networkidle")
    expect(page.get_by_role("heading", name=re.compile("Custom Wake Words", re.I))).to_be_visible()
    metrics = page.evaluate(
        "() => ({width: document.documentElement.scrollWidth, client: document.documentElement.clientWidth})"
    )
    assert metrics["width"] <= metrics["client"] + 12
    page.close()


def test_no_console_or_network_errors_on_public_pages(site_url: str, live_page: Page) -> None:
    for path in ("/", "/privacy", "/terms"):
        live_page.goto(join_url(site_url, path), wait_until="networkidle")
    _assert_no_fatal_browser_errors(live_page)
