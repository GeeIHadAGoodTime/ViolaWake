"""Direct HTTP probes against the deployed ViolaWake API."""

from __future__ import annotations

import os

import httpx
import pytest

from conftest import (
    TEST_PASSWORD,
    AuthUser,
    auth_headers,
    json_or_text,
    make_wav_bytes,
    register_live_user,
)


pytestmark = pytest.mark.live


@pytest.mark.smoke
async def test_health_endpoint(http_session: httpx.AsyncClient) -> None:
    response = await http_session.get("/api/health")
    assert response.status_code == 200, json_or_text(response)


@pytest.mark.smoke
async def test_register_new_user_returns_token(
    http_session: httpx.AsyncClient,
    unique_email: str,
    record_property: pytest.RecordProperty,
) -> None:
    user = await register_live_user(http_session, unique_email)
    record_property("email_verified", user.is_verified)
    assert user.token.count(".") == 2


@pytest.mark.smoke
async def test_register_duplicate_email_rejected(
    http_session: httpx.AsyncClient,
    unique_email: str,
) -> None:
    await register_live_user(http_session, unique_email, require_token=False)
    duplicate = await http_session.post(
        "/api/auth/register",
        json={
            "email": unique_email,
            "password": TEST_PASSWORD,
            "name": "Duplicate Live Test",
        },
    )
    assert duplicate.status_code in (400, 409), json_or_text(duplicate)
    assert "registered" in str(json_or_text(duplicate)).lower()


@pytest.mark.smoke
async def test_login_bad_credentials_returns_401(http_session: httpx.AsyncClient) -> None:
    response = await http_session.post(
        "/api/auth/login",
        json={"email": "missing-live-user@example.com", "password": "WrongPass123!"},
    )
    assert response.status_code == 401, json_or_text(response)


@pytest.mark.smoke
async def test_me_with_token_returns_profile(
    http_session: httpx.AsyncClient,
    authed_user: AuthUser,
) -> None:
    response = await http_session.get("/api/auth/me", headers=auth_headers(authed_user.token))
    assert response.status_code == 200, json_or_text(response)
    body = response.json()
    assert body["email"] == authed_user.email
    assert "created_at" in body


@pytest.mark.smoke
async def test_billing_checkout_actual_route_configuration(
    http_session: httpx.AsyncClient,
    authed_user: AuthUser,
    record_property: pytest.RecordProperty,
) -> None:
    """Probe the route used by the deployed frontend/source code."""
    response = await http_session.post(
        "/api/billing/checkout",
        headers=auth_headers(authed_user.token),
        json={"tier": "developer"},
    )
    record_property("status_code", response.status_code)
    record_property("body", str(json_or_text(response))[:500])

    assert response.status_code in (200, 503), json_or_text(response)
    if response.status_code == 200:
        checkout_url = response.json().get("checkout_url")
        assert isinstance(checkout_url, str) and checkout_url.startswith("https://checkout.stripe.com")
    else:
        assert "configured" in str(json_or_text(response)).lower()


@pytest.mark.smoke
async def test_recordings_auth_required_and_fresh_user_list_empty(
    http_session: httpx.AsyncClient,
    authed_user: AuthUser,
) -> None:
    unauth = await http_session.get("/api/recordings")
    assert unauth.status_code in (401, 403), json_or_text(unauth)

    authed = await http_session.get("/api/recordings", headers=auth_headers(authed_user.token))
    assert authed.status_code == 200, json_or_text(authed)
    assert authed.json() == []


@pytest.mark.smoke
async def test_sql_injection_email_rejected_without_server_error(
    http_session: httpx.AsyncClient,
) -> None:
    response = await http_session.post(
        "/api/auth/register",
        json={
            "email": "x' OR 1=1 --@x.com",
            "password": TEST_PASSWORD,
            "name": "SQL Injection Probe",
        },
    )
    assert response.status_code in (400, 422), json_or_text(response)
    assert response.status_code < 500


@pytest.mark.smoke
async def test_path_traversal_file_route_not_served(
    http_session: httpx.AsyncClient,
    authed_user: AuthUser,
) -> None:
    response = await http_session.get(
        "/api/files/%2e%2e/%2e%2e/etc/passwd",
        headers=auth_headers(authed_user.token),
    )
    assert response.status_code in (400, 401, 403, 404), json_or_text(response)
    assert response.status_code != 200


@pytest.mark.smoke
async def test_jwt_tamper_rejected(
    http_session: httpx.AsyncClient,
    authed_user: AuthUser,
) -> None:
    token = authed_user.token
    replacement = "a" if token[-1] != "a" else "b"
    tampered = f"{token[:-1]}{replacement}"
    response = await http_session.get("/api/auth/me", headers=auth_headers(tampered))
    assert response.status_code == 401, json_or_text(response)


@pytest.mark.rate_limit
async def test_login_rate_limit_eventually_returns_429(
    http_session: httpx.AsyncClient,
    record_property: pytest.RecordProperty,
) -> None:
    if os.getenv("VIOLAWAKE_LIVE_RATE_LIMIT") != "1":
        pytest.skip("Set VIOLAWAKE_LIVE_RATE_LIMIT=1 to burn live login rate-limit budget.")

    statuses: list[int] = []
    hit_at: int | None = None
    for idx in range(12):
        response = await http_session.post(
            "/api/auth/login",
            json={
                "email": f"ratelimit-{idx}@example.com",
                "password": "WrongPass123!",
            },
        )
        statuses.append(response.status_code)
        if response.status_code == 429:
            hit_at = idx + 1
            break

    record_property("statuses", statuses)
    record_property("rate_limit_hit_at", hit_at)
    assert hit_at is not None, statuses


async def test_free_tier_upload_limit_probe(
    http_session: httpx.AsyncClient,
    authed_user: AuthUser,
) -> None:
    if os.getenv("VIOLAWAKE_LIVE_UPLOAD_QUOTA") != "1":
        pytest.skip("Set VIOLAWAKE_LIVE_UPLOAD_QUOTA=1 to upload many files to the live deployment.")
    if not authed_user.is_verified:
        pytest.skip("Upload quota probe requires a verified account.")

    uploaded_ids: list[int] = []
    rejection: httpx.Response | None = None
    wav_bytes = make_wav_bytes(duration_s=0.75, frequency_hz=440.0)
    try:
        for idx in range(51):
            response = await http_session.post(
                "/api/recordings/upload",
                headers=auth_headers(authed_user.token),
                files={"file": (f"quota_{idx:02d}.wav", wav_bytes, "audio/wav")},
                data={"wake_word": "live quota probe"},
            )
            if response.status_code == 200:
                uploaded_ids.append(int(response.json()["recording_id"]))
                continue
            rejection = response
            break

        assert rejection is not None, "No quota/rate rejection after 51 uploads"
        detail = str(json_or_text(rejection)).lower()
        assert rejection.status_code in (403, 429), json_or_text(rejection)
        assert any(word in detail for word in ("quota", "limit", "rate")), json_or_text(rejection)
    finally:
        for recording_id in uploaded_ids:
            await http_session.delete(
                f"/api/recordings/{recording_id}",
                headers=auth_headers(authed_user.token),
            )
