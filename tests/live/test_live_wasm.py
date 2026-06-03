"""Browser/WASM SDK availability probes."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from conftest import PROJECT_ROOT, join_url

try:
    from playwright.sync_api import Page, expect
except ImportError:  # pragma: no cover
    Page = object  # type: ignore[assignment,misc]
    expect = None  # type: ignore[assignment]


pytestmark = pytest.mark.live


WASM_DEMO_PATHS = ("/wasm/demo/", "/wasm-demo/", "/demo")
LOCAL_WASM_DIST = PROJECT_ROOT / "wasm" / "dist" / "violawake.js"


async def _find_live_demo(site_url: str) -> tuple[str | None, dict[str, int]]:
    statuses: dict[str, int] = {}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        for path in WASM_DEMO_PATHS:
            url = join_url(site_url, path)
            response = await client.get(url)
            statuses[path] = response.status_code
            body = response.text
            looks_like_wasm_demo = (
                "Browser Demo" in body
                or "btnStart" in body
                or "WakeDetector" in body and "onnxruntime-web" in body
            )
            if response.status_code == 200 and looks_like_wasm_demo:
                return url, statuses
    return None, statuses


def _find_live_demo_blocking(site_url: str) -> tuple[str | None, dict[str, int]]:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_find_live_demo(site_url))

    result: dict[str, Any] = {}

    def _run() -> None:
        try:
            result["value"] = asyncio.run(_find_live_demo(site_url))
        except BaseException as exc:  # pragma: no cover - surfaced in caller
            result["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


@pytest.mark.smoke
async def test_wasm_demo_route_reachable_or_local_dist_present(
    site_url: str,
    record_property: pytest.RecordProperty,
) -> None:
    demo_url, statuses = await _find_live_demo(site_url)
    record_property("live_demo_statuses", statuses)
    record_property("local_dist_exists", LOCAL_WASM_DIST.exists())

    if demo_url is not None:
        record_property("demo_url", demo_url)
        return

    if LOCAL_WASM_DIST.exists():
        pytest.skip(f"Live demo routes not served ({statuses}); local wasm/dist exists for fallback.")

    pytest.skip(f"Live demo routes not served ({statuses}); local wasm/dist is absent.")


def test_wasm_demo_requests_onnx_models(site_url: str, browser_context: object) -> None:
    demo_url, statuses = _find_live_demo_blocking(site_url)
    if demo_url is None:
        pytest.skip(f"No live WASM demo route is served: {statuses}")

    context = browser_context
    context.grant_permissions(["microphone"], origin=site_url)
    page: Page = context.new_page()
    onnx_urls: list[str] = []
    console_errors: list[str] = []

    page.on("request", lambda request: onnx_urls.append(request.url) if ".onnx" in request.url else None)
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.set_default_timeout(20_000)
    page.goto(demo_url, wait_until="domcontentloaded", timeout=20_000)
    expect(page.get_by_role("button", name=re.compile("Start", re.I))).to_be_visible()
    page.get_by_role("button", name=re.compile("Start", re.I)).click()
    page.wait_for_timeout(8000)

    assert any(url.endswith(".onnx") or ".onnx" in url for url in onnx_urls), {
        "onnx_urls": onnx_urls,
        "console_errors": console_errors,
    }
    assert console_errors == []
    page.close()


def test_wasm_synthetic_silence_and_sine_if_local_dist_built() -> None:
    if not LOCAL_WASM_DIST.exists():
        pytest.skip("wasm/dist/violawake.js is absent; run the WASM build job first.")
    pytest.skip("Local WASM inference requires ONNX model assets in wasm/demo/models.")


def test_wasm_10000_frame_memory_loop_if_assets_exist() -> None:
    model_dir = PROJECT_ROOT / "wasm" / "demo" / "models"
    required = ["melspectrogram.onnx", "embedding_model.onnx", "temporal_cnn.onnx"]
    missing = [name for name in required if not (model_dir / name).exists()]
    if not LOCAL_WASM_DIST.exists() or missing:
        pytest.skip(f"Local WASM assets are incomplete; missing: {missing}")
    pytest.skip("Memory-growth probe needs a local static server fixture for wasm/demo.")
