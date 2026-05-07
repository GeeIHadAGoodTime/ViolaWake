"""Fixtures for opt-in live deployment tests.

These tests hit https://violawake.com and https://api.violawake.com by default.
They are skipped unless VIOLAWAKE_LIVE=1 is present in the environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
import wave
from collections.abc import AsyncGenerator, Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest


RUN_LIVE = os.getenv("VIOLAWAKE_LIVE") == "1"
DEFAULT_API_BASE_URL = "https://api.violawake.com"
DEFAULT_SITE_URL = "https://violawake.com"
TEST_PASSWORD = "LiveTest123!"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_AUTHED_USER_CACHE: AuthUser | None = None


@dataclass(frozen=True)
class AuthUser:
    email: str
    token: str
    user: dict[str, Any]

    @property
    def is_verified(self) -> bool:
        return bool(self.user.get("email_verified"))


@pytest.fixture(scope="session", autouse=True)
def require_live_opt_in() -> None:
    """Never run live deployment tests by accident."""
    if not RUN_LIVE:
        pytest.skip("Set VIOLAWAKE_LIVE=1 to run live deployment tests.")


@pytest.fixture(scope="session")
def api_base_url() -> str:
    return os.getenv("VIOLAWAKE_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


@pytest.fixture(scope="session")
def site_url() -> str:
    return os.getenv("VIOLAWAKE_SITE_URL", DEFAULT_SITE_URL).rstrip("/")


@pytest.fixture
def unique_email() -> str:
    suffix = uuid.uuid4().hex[:12]
    domain = os.getenv("VIOLAWAKE_LIVE_EMAIL_DOMAIN", "example.com")
    return f"live-{int(time.time())}-{suffix}@{domain}"


@pytest.fixture
def email_factory() -> Callable[[str], str]:
    def _make(label: str = "live") -> str:
        suffix = uuid.uuid4().hex[:12]
        domain = os.getenv("VIOLAWAKE_LIVE_EMAIL_DOMAIN", "example.com")
        return f"{label}-{int(time.time())}-{suffix}@{domain}"

    return _make


@pytest.fixture
async def http_session(api_base_url: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(base_url=api_base_url, timeout=20.0, follow_redirects=False) as client:
        yield client


async def register_live_user(
    client: httpx.AsyncClient,
    email: str,
    *,
    name: str = "Live Test User",
    password: str = TEST_PASSWORD,
    require_token: bool = True,
) -> AuthUser:
    response = await client.post(
        "/api/auth/register",
        json={"email": email, "password": password, "name": name},
    )
    if response.status_code == 429 and not require_token:
        pytest.skip(f"Live registration is rate-limited: {json_or_text(response)}")
    assert response.status_code in (200, 201), _response_debug(response)
    body = response.json()
    assert body.get("user", {}).get("email") == email
    if require_token:
        assert body.get("token"), body
    return AuthUser(email=email, token=str(body.get("token") or ""), user=body["user"])


@pytest.fixture
async def authed_user(http_session: httpx.AsyncClient, unique_email: str) -> AuthUser:
    global _AUTHED_USER_CACHE
    if _AUTHED_USER_CACHE is not None:
        return _AUTHED_USER_CACHE
    user = await register_live_user(http_session, unique_email, require_token=False)
    if not user.token:
        pytest.skip(
            "Live registration returned no auth token; token-dependent probes are blocked."
        )
    _AUTHED_USER_CACHE = user
    return user


@pytest.fixture
async def authed_token(authed_user: AuthUser) -> str:
    return authed_user.token


@dataclass(frozen=True)
class CleanVenv:
    root: Path
    python: Path
    pip: Path

    def run(
        self,
        args: list[str],
        *,
        timeout: int = 300,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        merged_env.update(
            {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            }
        )
        if env:
            merged_env.update(env)
        return subprocess.run(
            [str(self.python), *args],
            cwd=str(PROJECT_ROOT),
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )

    def pip_install(self, *packages: str, timeout: int = 300) -> subprocess.CompletedProcess[str]:
        return self.run(["-m", "pip", "install", *packages], timeout=timeout)


@pytest.fixture(scope="session")
def clean_venv(tmp_path_factory: pytest.TempPathFactory) -> CleanVenv:
    root = tmp_path_factory.mktemp("violawake-live-venv") / "venv"
    subprocess.run(
        [sys.executable, "-m", "venv", str(root)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    if sys.platform == "win32":
        python = root / "Scripts" / "python.exe"
        pip = root / "Scripts" / "pip.exe"
    else:
        python = root / "bin" / "python"
        pip = root / "bin" / "pip"
    return CleanVenv(root=root, python=python, pip=pip)


@pytest.fixture(scope="session")
def playwright_browser() -> Generator[Any, None, None]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not installed")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def browser_context(playwright_browser: Any) -> Generator[Any, None, None]:
    context = playwright_browser.new_context(
        viewport={"width": 1366, "height": 900},
        ignore_https_errors=False,
    )
    yield context
    context.close()


@pytest.fixture
def live_page(browser_context: Any) -> Generator[Any, None, None]:
    page = browser_context.new_page()
    page.live_console_errors = []
    page.live_request_failures = []
    page.live_5xx_responses = []

    def on_console(msg: Any) -> None:
        if msg.type in {"error", "warning"}:
            text = msg.text
            known_noise = (
                "favicon" in text.lower()
                or "cookieconsent_status" in text
                or "Failed to load resource: the server responded with a status of 404" in text
            )
            if not known_noise:
                page.live_console_errors.append(f"{msg.type}: {text}")

    def on_page_error(exc: Exception) -> None:
        page.live_console_errors.append(f"pageerror: {exc}")

    def on_request_failed(request: Any) -> None:
        page.live_request_failures.append(f"{request.method} {request.url}: {request.failure}")

    def on_response(response: Any) -> None:
        if response.status >= 500:
            page.live_5xx_responses.append(f"{response.status} {response.url}")

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)
    yield page
    page.close()


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def _response_debug(response: httpx.Response) -> str:
    try:
        body: object = response.json()
    except Exception:
        body = response.text[:1000]
    return f"{response.request.method} {response.request.url} -> {response.status_code}: {body}"


def json_or_text(response: httpx.Response) -> object:
    try:
        return response.json()
    except Exception:
        return response.text


def make_wav_bytes(
    *,
    duration_s: float = 0.75,
    sample_rate: int = 16_000,
    frequency_hz: float | None = 440.0,
    amplitude: float = 0.4,
    channels: int = 1,
) -> bytes:
    sample_count = int(duration_s * sample_rate)
    if frequency_hz is None:
        mono = np.zeros(sample_count, dtype=np.float32)
    else:
        t = np.linspace(0, duration_s, sample_count, endpoint=False)
        mono = (np.sin(2 * np.pi * frequency_hz * t) * amplitude).astype(np.float32)

    if channels == 2:
        samples = np.column_stack([mono, mono])
    else:
        samples = mono
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

    import io

    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


def parse_summary_json(text: str) -> dict[str, Any]:
    """Extract a JSON object emitted by a subprocess helper."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise AssertionError(f"No JSON object found in subprocess output:\n{text[-2000:]}")
