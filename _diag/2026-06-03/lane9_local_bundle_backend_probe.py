"""Serve the built frontend bundle and prove auth traffic targets production API."""

from __future__ import annotations

import contextlib
import http.server
import json
import socket
import threading
import time
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
DIST = ROOT / "console" / "frontend" / "dist"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
API_LOGIN = "https://api.violawake.com/api/auth/login"
SPA_EXACT_ROUTES = {
    "/login",
    "/register",
    "/verify-email",
    "/forgot-password",
    "/reset-password",
    "/dashboard",
    "/record",
    "/billing",
    "/teams",
}
SPA_PREFIX_ROUTES = (
    "/record/",
    "/training/",
    "/model/",
    "/account/",
    "/teams/",
)


class SpaHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def log_message(self, _format: str, *args: object) -> None:
        return

    def send_head(self):  # noqa: ANN201 - inherited signature is untyped.
        path = self.translate_path(self.path)
        if not Path(path).exists():
            route = self.path.split("?", 1)[0]
            if route in SPA_EXACT_ROUTES or route.startswith(SPA_PREFIX_ROUTES):
                self.path = "/app/index.html"
            else:
                self.path = "/index.html"
        return super().send_head()


def free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main() -> int:
    port = free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), SpaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    result: dict[str, object] = {
        "servedDist": str(DIST),
        "localUrl": f"http://127.0.0.1:{port}/login",
        "expectedApiLogin": API_LOGIN,
    }
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(executable_path=CHROME_EXE, headless=True)
            page = browser.new_page()
            console_errors: list[str] = []
            api_requests: list[str] = []
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.on(
                "request",
                lambda request: api_requests.append(f"{request.method} {request.url}")
                if "api.violawake.com" in request.url
                else None,
            )
            page.goto(result["localUrl"], wait_until="networkidle")
            page.fill("#email", f"lane9-nonexistent-{time.time_ns()}@example.com")
            page.fill("#password", "WrongPass123!")
            page.locator('button[type="submit"]').wait_for(state="visible", timeout=5000)
            page.wait_for_function(
                "() => !document.querySelector('button[type=\"submit\"]')?.disabled",
                timeout=5000,
            )
            try:
                with page.expect_response(
                    lambda response: response.url == API_LOGIN and response.request.method == "POST",
                    timeout=15000,
                ) as response_info:
                    page.click('button[type="submit"]')
                response = response_info.value
                result["observedApiLogin"] = response.url
                result["status"] = response.status
            except PlaywrightTimeoutError as exc:
                result["error"] = str(exc)
                result["body"] = page.locator("body").inner_text(timeout=5000)[:1000]
                result["apiRequests"] = api_requests
            result["consoleErrors"] = console_errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    print(json.dumps(result, indent=2))
    observed_requests = result.get("apiRequests", [])
    if result.get("observedApiLogin") == API_LOGIN:
        return 0
    if isinstance(observed_requests, list) and f"POST {API_LOGIN}" in observed_requests:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
