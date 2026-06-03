# Lane 8 Audit Report - SaaS Console Backend

Date: 2026-06-03
Branch: `audit-2026-06-03/l8-backend`
Commit landed: `2d2a509 fix(live): align backend oracle with API contract`

## Verdict

MUST-FIX / BLOCKED. This lane is not PASS.

Blocking reason: the public API became unreachable during the audit. `GET https://api.violawake.com/api/health` returned 200 earlier, then later returned Cloudflare 530 / error 1033 from both local curl and external fetch. Per the success criteria, health must be 200. Per SC correction A3, I did not restart, stop, deploy, or modify production containers or tunnel config.

## Fix Landed

Fixed a live-oracle contract drift:

- Removed the stale live smoke probe for `/api/billing/checkout-session`. Live OpenAPI and source expose `/api/billing/checkout`.
- Fixed `tests/live/full_pipeline_e2e.py` to request download tokens with `resource_id`, matching `DownloadTokenRequest`.
- Added offline ratchet `tests/live/test_live_oracle_contract.py`.

Ratchet:

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: live-backend-oracle-contract
contract: Live backend proof scripts must use backend/OpenAPI-backed routes and request field names.
detector: tests/live/test_live_oracle_contract.py
own_tests:
  - tests/live/test_live_oracle_contract.py::test_live_billing_oracle_uses_documented_checkout_route
  - tests/live/test_live_oracle_contract.py::test_full_pipeline_download_token_uses_resource_id
```

Pre-fix evidence:

```text
cmd /c git show HEAD^^:tests/live/test_live_api.py | findstr /n "checkout-session billing/checkout"
86:    The current source tree exposes /api/billing/checkout. If the deployed API
87:    does not expose /api/billing/checkout-session, this test fails and records
91:        "/api/billing/checkout-session",
111:        "/api/billing/checkout",

cmd /c git show HEAD^^:tests/live/full_pipeline_e2e.py | findstr /n "download-token model_id resource_id"
173:        r = client.post("/api/auth/download-token", json={"action": "model_download", "model_id": model_id})
```

Post-fix evidence:

```text
cmd /c findstr /n "download-token resource_id billing/checkout checkout-session" tests\\live\\full_pipeline_e2e.py tests\\live\\test_live_api.py tests\\live\\test_live_oracle_contract.py
tests\\live\\full_pipeline_e2e.py:173:        r = client.post("/api/auth/download-token", json={"action": "model_download", "resource_id": model_id})
tests\\live\\test_live_api.py:87:        "/api/billing/checkout",
tests\\live\\test_live_oracle_contract.py:27:    assert "/api/billing/checkout" in live_api
tests\\live\\test_live_oracle_contract.py:28:    assert "checkout-session" not in live_api
tests\\live\\test_live_oracle_contract.py:29:    assert "checkout-session" not in live_readme
tests\\live\\test_live_oracle_contract.py:36:    assert '"resource_id": model_id' in full_pipeline

cmd /c python -m pytest tests/live/test_live_oracle_contract.py -q -o addopts=
..                                                                       [100%]
2 passed in 7.38s
```

## Live Evidence

Initial health was good:

```text
curl.exe -sS -i https://api.violawake.com/api/health
HTTP/1.1 200 OK
...
{"status":"ok","uptime_s":162172.992,"ready":true,"version":"0.0.0"}
```

Later health failed:

```text
cmd /c curl.exe -sS -i -A "Mozilla/5.0 audit" https://api.violawake.com/api/health
HTTP/1.1 530 <none>
...
error code: 1033
```

External fetch also failed:

```text
open https://api.violawake.com/api/health
Failed to fetch https://api.violawake.com/api/health: (530) Unknown Status Code
```

Live OpenAPI before the 530 incident:

```text
curl.exe -sS -A "Mozilla/5.0 audit" https://api.violawake.com/openapi.json | python -c "... route count ..."
live routes 54
POST /api/billing/checkout
POST /api/billing/webhook
GET /api/billing/subscription
...
```

Prompt-specified `/api/openapi.json` is not the deployed schema path:

```text
curl.exe -sS -i https://api.violawake.com/api/openapi.json
HTTP/1.1 404 Not Found
...
{"detail":"Not Found"}
```

Read-only production stack evidence before the host degraded:

```text
docker ps --format "{{.Names}}\t{{.Status}}\t{{.Image}}" | Select-String -Pattern "wakeword|viola"
wakeword-backend-1    Up 45 hours (healthy)    wakeword-backend
wakeword-decoder-1    Up 45 hours (healthy)    wakeword-decoder
wakeword-postgres-1   Up 45 hours (healthy)    postgres:16-alpine
wakeword-tunnel-1     Up 45 hours              cloudflare/cloudflared:latest

docker inspect wakeword-backend-1 --format "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}"
running healthy
```

Read-only Docker inspection later hung after the 530 appeared, so I could not prove current container state or tunnel logs without production intervention.

## Auth / Replay Surface

No unauthenticated billing/user-data route accepted in the read-only probes I could run before the 530:

```text
GET /api/billing/subscription
HTTP/1.1 401 Unauthorized
{"detail":"Not authenticated"}

GET /api/models/1/download
HTTP/1.1 401 Unauthorized
{"detail":"Missing authentication token"}

GET /api/teams/1 with Authorization: Bearer wrong-team-invalid-token
HTTP/1.1 401 Unauthorized
{"detail":"Invalid token: Not enough segments"}

POST /api/billing/webhook without stripe-signature
HTTP/1.1 400 Bad Request
{"detail":"Missing stripe-signature header."}

POST /api/email/inbound without shared secret
HTTP/1.1 403 Forbidden
{"detail":"Forbidden"}
```

Relevant code anchors:

- Billing auth/rate routes: `console/backend/app/routes/billing.py:315`, `:391`, `:457`, `:498`, `:524`
- Billing webhook idempotency: `console/backend/app/routes/billing.py:177`, `:435`
- One-time download token decode: `console/backend/app/auth.py:292`, model use at `console/backend/app/routes/models.py:215`, `:230`, `:337`
- Team membership requires joined membership: `console/backend/app/auth.py:479`, `:499`; team routes use it at `console/backend/app/routes/teams.py:203`, `:218`, `:292`, `:324`, `:347`, `:382`, `:401`, `:434`

## Inbound Email Worker

Static review: Worker points at the backend inbound route and sends the shared secret header:

- Worker URL config: `workers/support-email/wrangler.toml:10`
- Worker POST: `workers/support-email/src/index.ts:48`
- Secret header: `workers/support-email/src/index.ts:52`
- Human forward: `workers/support-email/src/index.ts:72`

Backend route:

- Shared-secret guard: `console/backend/app/routes/inbound_email.py:76`, called at `:91`
- Sender extraction: `console/backend/app/routes/inbound_email.py:58`, `:100`
- 24h dedupe by sender: `console/backend/app/routes/inbound_email.py:106`
- Auto-reply send: `console/backend/app/routes/inbound_email.py:130`

Existing local tests cover successful dedupe and shared-secret rejection in `console/tests/test_inbound_email.py:42` and `:77`, but rerunning that file in this audit timed out under the host resource issue. I did not claim it passed in this run.

## Blocked Probes

The following were blocked by `SC_AUDIT_ROUND_1_CORRECTIONS.md`:

- Full live sign-up -> upload -> training -> model download: blocked. No pre-existing test account/staging target was documented; creating live users/uploads/jobs would write production DB state.
- Live billing checkout/card flow: blocked. No founder-provisioned test account/staging path; no real billing transactions allowed.
- Wrong-team live data leak probe: blocked. It requires pre-existing users/teams/models or production DB writes to create them.
- Real inbound Worker success with shared secret: blocked. The secret is not available for a safe live fixture, and posting with the real secret would create production inbound-email state.
- Tunnel-down negative probe: blocked. Stopping `wakeword-tunnel-1` is explicitly prohibited. The live system nevertheless entered the 530/1033 state during audit, which proves the failure mode is real and currently launch-blocking.

## Verification

Passed:

```text
cmd /c python -m py_compile tests/live/test_live_api.py tests/live/full_pipeline_e2e.py tests/live/test_live_oracle_contract.py

cmd /c python -m pytest tests/live/test_live_oracle_contract.py -q -o addopts=
..                                                                       [100%]
2 passed in 7.38s

cmd /c git diff --check
success
```

Not passed / not completed:

- `python -m pytest tests/live --collect-only -q` timed out.
- Combined `tests/live` plus `console/tests` collection hit an existing conftest import-name mismatch between `tests/conftest.py` and `console/tests/conftest.py`.
- `console/tests/test_inbound_email.py` timed out under the host resource issue.

## Mandatory Self-Audit Gate

Five surfaces I did not exhaustively probe:

1. Full live customer flow: blocked because it writes production account, recording, job, model, and possibly billing state without an approved test account.
2. Cross-team live leakage: blocked because creating two users, a team, and a model would write production state.
3. Real Stripe checkout/webhook completion: blocked because test-account/staging was not provisioned and real billing is prohibited.
4. Inbound Worker with the real shared secret: blocked because the secret is not available as a safe fixture and a real call would create production inbound-email/autoreply state.
5. Current tunnel container logs/alerting: blocked by host/tool resource failures after the 530 started; Docker read-only inspection hung, and SC A3 forbids recovery actions such as restart or redeploy.

