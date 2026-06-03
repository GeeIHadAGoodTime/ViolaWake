"""Prove the deployed frontend sends auth traffic to the live backend."""

from __future__ import annotations

import json
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
LOGIN_URL = "https://violawake.com/login"
API_LOGIN = "https://api.violawake.com/api/auth/login"


def main() -> int:
    result: dict[str, object] = {
        "loginUrl": LOGIN_URL,
        "expectedApiLogin": API_LOGIN,
    }
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME_EXE, headless=True)
        page = browser.new_page()
        console_errors: list[str] = []
        page.on(
            "console",
            lambda message: console_errors.append(message.text)
            if message.type == "error"
            else None,
        )
        page.goto(LOGIN_URL, wait_until="networkidle")
        result["finalUrl"] = page.url
        try:
            page.fill("#email", f"lane9-nonexistent-{time.time_ns()}@example.com", timeout=5000)
            page.fill("#password", "WrongPass123!", timeout=5000)
            page.wait_for_function(
                "() => !document.querySelector('button[type=\"submit\"]')?.disabled",
                timeout=5000,
            )
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
        result["consoleErrors"] = console_errors
        result["bodyExcerpt"] = page.locator("body").inner_text(timeout=5000)[:500]
        browser.close()

    print(json.dumps(result, indent=2))
    return 0 if result.get("observedApiLogin") == API_LOGIN and not console_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
