"""Lane 9 live render probe for the deployed ViolaWake frontend."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

from playwright.sync_api import sync_playwright


ORIGIN = "https://violawake.com"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REQUIRED_PATHS = [
    "/",
    "/register",
    "/login",
    "/dashboard",
    "/pricing",
    "/faq",
    "/blog",
    "/about",
    "/contact",
    "/privacy",
    "/terms",
    "/docs",
    "/compare/picovoice",
    "/compare/openwakeword",
]
ROUTE_EXPECTATIONS = {
    "/register": {"text": ["Create account"], "selectors": ["#name", "#email", "#password"]},
    "/login": {"text": ["Welcome back"], "selectors": ["#email", "#password"]},
    "/dashboard": {"text": ["Welcome back"], "selectors": ["#email", "#password"]},
    "/pricing": {"text": ["Developer", "Business", "Enterprise"]},
    "/faq": {"text": ["ViolaWake FAQ"]},
    "/blog": {"text": ["Technical notes"]},
    "/about": {"text": ["About ViolaWake"]},
    "/contact": {"text": ["General questions"]},
    "/privacy": {"text": ["Privacy Policy"]},
    "/terms": {"text": ["Terms of Service"]},
    "/docs": {"text": ["Quickstart"]},
    "/compare/picovoice": {"text": ["Picovoice Porcupine"]},
    "/compare/openwakeword": {"text": ["OpenWakeWord"]},
}


def normalize(href: str) -> str | None:
    parsed = urlparse(urljoin(ORIGIN, href))
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if origin != ORIGIN:
        return None
    path = parsed.path
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def is_page_url(href: str) -> bool:
    path = urlparse(href).path
    return not re.search(
        r"\.(png|jpg|jpeg|gif|svg|webp|ico|css|js|mjs|map|json|txt|xml|pdf|zip|onnx|wasm)$",
        path,
        re.IGNORECASE,
    )


def summarize(values: list[str], limit: int = 4) -> list[str]:
    return [re.sub(r"\s+", " ", value)[:180] for value in values[:limit]]


def route_path(href: str) -> str:
    return urlparse(href).path.rstrip("/") or "/"


def main() -> int:
    max_pages = int(os.environ.get("LANE9_MAX_PAGES", "120"))
    required_only = os.environ.get("LANE9_REQUIRED_ONLY") == "1"
    seen: set[str] = set()
    queue: deque[str] = deque(normalize(path) or path for path in REQUIRED_PATHS)
    results: list[dict[str, object]] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path=CHROME_EXE, headless=True)
        while queue and len(seen) < max_pages:
            url = queue.popleft()
            if url in seen:
                continue
            seen.add(url)

            page = browser.new_page()
            console_errors: list[str] = []
            page_errors: list[str] = []
            request_failures: list[str] = []
            five_hundreds: list[str] = []

            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "requestfailed",
                lambda request: request_failures.append(
                    f"{request.method} {request.url} {request.failure or ''}"
                ),
            )
            page.on(
                "response",
                lambda response: five_hundreds.append(f"{response.status} {response.url}")
                if response.status >= 500
                else None,
            )

            status: int | None = None
            title = ""
            body_text = ""
            link_count = 0
            goto_error = ""
            final_url = url
            expected_text_missing: list[str] = []
            expected_selector_missing: list[str] = []

            try:
                response = page.goto(url, wait_until="networkidle", timeout=30000)
                status = response.status if response else None
                page.wait_for_timeout(400)
                final_url = page.url
                title = page.title()
                body_text = page.locator("body").inner_text(timeout=5000)
                expectations = ROUTE_EXPECTATIONS.get(route_path(url))
                if expectations:
                    expected_text_missing = [
                        text for text in expectations.get("text", []) if text not in body_text
                    ]
                    for selector in expectations.get("selectors", []):
                        if page.locator(selector).count() == 0:
                            expected_selector_missing.append(selector)
                links = page.locator("a[href]").evaluate_all("(anchors) => anchors.map((a) => a.href)")
                link_count = len(links)
                if not required_only:
                    for href in links:
                        normalized = normalize(href)
                        if normalized and is_page_url(normalized) and normalized not in seen:
                            queue.append(normalized)
            except Exception as exc:  # noqa: BLE001 - audit output wants the browser failure text.
                goto_error = str(exc)
            finally:
                page.close()

            render_error = bool(
                re.search(
                    r"something went wrong|application error|hydration failed|uncaught runtime error|failed to load",
                    body_text,
                    re.IGNORECASE,
                )
            )
            blank = len(body_text.strip()) == 0
            results.append(
                {
                    "url": url,
                    "finalUrl": final_url,
                    "status": status,
                    "title": title,
                    "bodyChars": len(body_text),
                    "linkCount": link_count,
                    "consoleErrors": summarize(console_errors),
                    "pageErrors": summarize(page_errors),
                    "requestFailures": summarize(request_failures),
                    "fiveHundreds": summarize(five_hundreds),
                    "expectedTextMissing": expected_text_missing,
                    "expectedSelectorMissing": expected_selector_missing,
                    "renderError": render_error,
                    "blank": blank,
                    "gotoError": goto_error,
                }
            )
        browser.close()

    failures = [
        result
        for result in results
        if result["gotoError"]
        or (isinstance(result["status"], int) and result["status"] >= 500)
        or result["blank"]
        or result["renderError"]
        or result["expectedTextMissing"]
        or result["expectedSelectorMissing"]
        or result["consoleErrors"]
        or result["pageErrors"]
        or result["fiveHundreds"]
    ]

    print(
        json.dumps(
            {
                "chromeExecutable": CHROME_EXE,
                "origin": ORIGIN,
                "pagesChecked": len(results),
                "failures": len(failures),
                "results": results,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
