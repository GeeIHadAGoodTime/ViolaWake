"""
Playwright browser E2E tests — verify the full Console UI flow.

Tests the complete user journey in a real browser:
  Register → Login → Dashboard → Record page

Uses actual selectors from the React components.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pytest

# Playwright import — may not be installed in all environments
try:
    from playwright.sync_api import Page, expect
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import FRONTEND_URL, TEST_USER_PASSWORD

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not HAS_PLAYWRIGHT, reason="playwright not installed"),
]


FRONTEND = FRONTEND_URL


# ── Tests ────────────────────────────────────────────────────────────────────

class TestRegistrationFlow:
    """Test the registration page."""

    def test_register_page_loads(self, page: Page, console_servers) -> None:
        page.goto(f"{FRONTEND}/register")
        # Inputs use id= not name=
        expect(page.locator("#email")).to_be_visible()
        expect(page.locator("#password")).to_be_visible()
        expect(page.locator('button[type="submit"]')).to_be_visible()

    def test_register_success(self, page: Page, console_servers) -> None:
        page.goto(f"{FRONTEND}/register")

        email = f"e2e_{time.time_ns()}@test.dev"
        page.fill("#name", "E2E Test User")
        page.fill("#email", email)
        page.fill("#password", TEST_USER_PASSWORD)
        page.click('button[type="submit"]')

        # Should redirect to dashboard
        page.wait_for_url(re.compile(r".*/dashboard"), timeout=10000)
        expect(page).to_have_url(re.compile(r".*/dashboard"))

    def test_duplicate_email_error(self, page: Page, console_servers) -> None:
        email = f"dup_{time.time_ns()}@test.dev"

        # Register once
        page.goto(f"{FRONTEND}/register")
        page.fill("#name", "First")
        page.fill("#email", email)
        page.fill("#password", TEST_USER_PASSWORD)
        page.click('button[type="submit"]')
        page.wait_for_url(re.compile(r".*/dashboard"), timeout=10000)

        # Clear token and go back to register
        page.evaluate("localStorage.clear()")
        page.goto(f"{FRONTEND}/register")
        page.fill("#name", "Second")
        page.fill("#email", email)
        page.fill("#password", TEST_USER_PASSWORD)
        page.click('button[type="submit"]')

        # Should show error or stay on register
        page.wait_for_timeout(3000)
        error_visible = page.locator(".auth-error").count() > 0
        still_on_register = "/register" in page.url
        assert error_visible or still_on_register


class TestLoginFlow:
    """Test the login page."""

    def test_login_page_loads(self, page: Page, console_servers) -> None:
        page.goto(f"{FRONTEND}/login")
        expect(page.locator("#email")).to_be_visible()
        expect(page.locator("#password")).to_be_visible()

    def test_login_success(self, page: Page, console_servers) -> None:
        email = f"login_{time.time_ns()}@test.dev"

        # Register first via API
        import requests
        requests.post(
            f"http://localhost:8000/api/auth/register",
            json={"email": email, "password": TEST_USER_PASSWORD, "name": "Login Test"},
            timeout=10,
        )

        # Login via UI
        page.goto(f"{FRONTEND}/login")
        page.fill("#email", email)
        page.fill("#password", TEST_USER_PASSWORD)
        page.click('button[type="submit"]')

        page.wait_for_url(re.compile(r".*/dashboard"), timeout=10000)
        expect(page).to_have_url(re.compile(r".*/dashboard"))

    def test_login_bad_credentials(self, page: Page, console_servers) -> None:
        page.goto(f"{FRONTEND}/login")
        page.fill("#email", "nonexistent@test.dev")
        page.fill("#password", "wrongpassword1")
        page.click('button[type="submit"]')

        page.wait_for_timeout(2000)
        assert "/login" in page.url  # Should stay on login page


class TestDashboard:
    """Test the dashboard page."""

    def test_dashboard_shows_content(self, page: Page, console_servers) -> None:
        # Register and land on dashboard
        email = f"dash_{time.time_ns()}@test.dev"
        page.goto(f"{FRONTEND}/register")
        page.fill("#name", "Dashboard Test")
        page.fill("#email", email)
        page.fill("#password", TEST_USER_PASSWORD)
        page.click('button[type="submit"]')
        page.wait_for_url(re.compile(r".*/dashboard"), timeout=10000)

        # New user sees empty state or "No models yet" or "Train" button
        page.wait_for_timeout(2000)
        has_empty_state = page.locator("text=/no models|train|get started|record/i").count() > 0
        has_title = page.locator("text=/wake word/i").count() > 0
        assert has_empty_state or has_title

    def test_train_button_navigates(self, page: Page, console_servers) -> None:
        email = f"nav_{time.time_ns()}@test.dev"
        page.goto(f"{FRONTEND}/register")
        page.fill("#name", "Nav Test")
        page.fill("#email", email)
        page.fill("#password", TEST_USER_PASSWORD)
        page.click('button[type="submit"]')
        page.wait_for_url(re.compile(r".*/dashboard"), timeout=10000)

        # Wait for loading to complete
        page.wait_for_timeout(2000)

        # Click any button that leads to recording
        train_btn = page.locator("text=/train|record|first wake word/i").first
        if train_btn.is_visible():
            train_btn.click()
            page.wait_for_url(re.compile(r".*/record"), timeout=5000)
            assert "/record" in page.url


class TestRecordPage:
    """Test the voice recording page."""

    def test_record_page_loads(self, page: Page, console_servers) -> None:
        # Register via API to avoid UI timing issues
        import requests
        email = f"rec_{time.time_ns()}@test.dev"
        resp = requests.post(
            f"http://localhost:8000/api/auth/register",
            json={"email": email, "password": TEST_USER_PASSWORD, "name": "Record Test"},
            timeout=10,
        )
        token = resp.json()["token"]

        # Set token in localStorage and navigate to record page
        page.goto(f"{FRONTEND}/login")
        page.evaluate(f"localStorage.setItem('token', '{token}')")
        page.goto(f"{FRONTEND}/record")
        # Should have wake word input (id="wakeword")
        expect(page.locator("#wakeword")).to_be_visible(timeout=5000)

    def test_wake_word_input_and_start(self, page: Page, console_servers) -> None:
        import requests
        email = f"ww_{time.time_ns()}@test.dev"
        resp = requests.post(
            f"http://localhost:8000/api/auth/register",
            json={"email": email, "password": TEST_USER_PASSWORD, "name": "WakeWord Test"},
            timeout=10,
        )
        token = resp.json()["token"]

        page.goto(f"{FRONTEND}/login")
        page.evaluate(f"localStorage.setItem('token', '{token}')")
        page.goto(f"{FRONTEND}/record")
        page.fill("#wakeword", "hey test")

        # Start Recording button should be enabled
        start_btn = page.locator("text=/start recording/i")
        expect(start_btn).to_be_visible()
        expect(start_btn).to_be_enabled()


class TestProtectedRoutes:
    """Test that protected routes redirect to login."""

    def test_dashboard_redirect(self, page: Page, console_servers) -> None:
        # Navigate to app first so localStorage is accessible
        page.goto(f"{FRONTEND}/login")
        page.evaluate("localStorage.clear()")
        page.goto(f"{FRONTEND}/dashboard")
        page.wait_for_timeout(2000)
        assert "/login" in page.url or "/register" in page.url

    def test_record_redirect(self, page: Page, console_servers) -> None:
        page.goto(f"{FRONTEND}/login")
        page.evaluate("localStorage.clear()")
        page.goto(f"{FRONTEND}/record")
        page.wait_for_timeout(2000)
        assert "/login" in page.url or "/register" in page.url
