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


def test_live_oracle_probes_no_removed_billing_routes() -> None:
    """Billing/Stripe routes were removed (free service, 2026-07-31); the
    live smoke must not probe routes the backend no longer serves."""
    live_api = _read("tests/live/test_live_api.py")
    live_readme = _read("tests/live/README.md")

    assert "/api/billing/" not in live_api
    assert "/api/billing/" not in live_readme
    assert "checkout" not in live_api.lower()


def test_full_pipeline_download_token_uses_resource_id() -> None:
    """The full live flow must match DownloadTokenRequest's schema."""
    full_pipeline = _read("tests/live/full_pipeline_e2e.py")

    assert '"resource_id": model_id' in full_pipeline
    assert '"model_id": model_id' not in full_pipeline
