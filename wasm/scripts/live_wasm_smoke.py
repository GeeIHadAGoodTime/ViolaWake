#!/usr/bin/env python
"""Short deployed WASM demo smoke for https://violawake.com/wasm/demo/."""

from __future__ import annotations

import json
import time
from argparse import ArgumentParser


def parse_args() -> object:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://violawake.com/wasm/demo/")
    parser.add_argument("--origin", default="https://violawake.com")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(f"playwright is not installed: {exc}") from exc

    url = args.url
    onnx_urls: list[str] = []
    console_errors: list[str] = []
    request_failures: list[str] = []
    status_text = ""
    score_text = ""
    log_text = ""
    wait_error = ""
    started = time.perf_counter()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
            ],
        )
        context = browser.new_context(viewport={"width": 1366, "height": 900})
        context.grant_permissions(["microphone"], origin=args.origin)
        page = context.new_page()
        page.set_default_timeout(20_000)
        page.on("request", lambda request: onnx_urls.append(request.url) if ".onnx" in request.url else None)
        page.on("requestfailed", lambda request: request_failures.append(f"{request.method} {request.url}: {request.failure}"))
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            page.get_by_role("button", name="Start").click(timeout=20_000)
            page.wait_for_function(
                "() => document.querySelector('#statusText')?.textContent?.includes('Listening')",
                timeout=20_000,
            )
        except Exception as exc:
            wait_error = str(exc)

        status_text = page.locator("#statusText").inner_text(timeout=5_000)
        score_text = page.locator("#scoreValue").inner_text(timeout=5_000)
        log_text = page.locator("#log").inner_text(timeout=5_000)
        elapsed_ms = (time.perf_counter() - started) * 1000
        context.close()
        browser.close()

    result = {
        "url": url,
        "status_text": status_text,
        "score_text": score_text,
        "log_text": log_text,
        "wait_error": wait_error,
        "onnx_request_count": len(onnx_urls),
        "onnx_urls": onnx_urls,
        "console_errors": console_errors,
        "request_failures": request_failures,
        "elapsed_ms": elapsed_ms,
        "pass": len(onnx_urls) >= 3 and not console_errors and not request_failures and "Listening" in status_text,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
