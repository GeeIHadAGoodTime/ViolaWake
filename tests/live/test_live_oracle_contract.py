"""Offline guards for live backend oracle request-shape drift."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session", autouse=True)
def require_live_opt_in() -> None:
    """This module is static and must run without live production access."""


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_live_billing_oracle_uses_documented_checkout_route() -> None:
    """The live smoke must probe the route served by source and OpenAPI."""
    live_api = _read("tests/live/test_live_api.py")
    live_readme = _read("tests/live/README.md")

    assert "/api/billing/checkout" in live_api
    assert "checkout-session" not in live_api
    assert "checkout-session" not in live_readme


def test_full_pipeline_download_token_uses_resource_id() -> None:
    """The full live flow must match DownloadTokenRequest's schema."""
    full_pipeline = _read("tests/live/full_pipeline_e2e.py")

    assert '"resource_id": model_id' in full_pipeline
    assert '"model_id": model_id' not in full_pipeline
