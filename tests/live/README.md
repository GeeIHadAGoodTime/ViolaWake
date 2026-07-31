# ViolaWake Live Deployment Tests

This suite verifies the deployed ViolaWake frontend, API, PyPI package, email, and WASM claims. Every test is marked `live` and is skipped unless `VIOLAWAKE_LIVE=1` is set.

## Targets

- Site: `VIOLAWAKE_SITE_URL`, default `https://violawake.com`
- API: `VIOLAWAKE_API_BASE_URL`, default `https://api.violawake.com`
- PyPI package: `violawake`, expected `>=0.2.2`

## Run

Smoke, intended runtime about 2-5 minutes:

```bash
VIOLAWAKE_LIVE=1 bash tests/live/run_smoke.sh
```

Collect all tests without running them:

```bash
pytest tests/live/ --collect-only
```

Full suite, intended runtime about 15 minutes plus any optional model downloads:

```bash
VIOLAWAKE_LIVE=1 pytest tests/live --no-cov -ra -vv
```

## Environment

- `VIOLAWAKE_LIVE=1`: required opt-in for all live tests.
- `VIOLAWAKE_SITE_URL`: override deployed frontend URL.
- `VIOLAWAKE_API_BASE_URL`: override deployed API URL.
- `VIOLAWAKE_LIVE_EMAIL_DOMAIN`: domain for throwaway email addresses, default `example.com`.
- `VIOLAWAKE_LIVE_EMAIL`: manual inbox address for email verification.
- `VIOLAWAKE_LIVE_MAILOSAUR_KEY`: Mailosaur API key for automated verification email checks.
- `VIOLAWAKE_LIVE_MAILOSAUR_SERVER_ID`: Mailosaur server id; required with the Mailosaur key.
- `VIOLAWAKE_LIVE_WEBHOOK_URL`: optional webhook polling URL if inbound email is externally routed there.
- `VIOLAWAKE_LIVE_RATE_LIMIT=1`: enables tests that intentionally consume auth rate-limit budget.
- `VIOLAWAKE_LIVE_UPLOAD_QUOTA=1`: enables the high-volume upload/quota probe against the live deployment.
- `VIOLAWAKE_LIVE_STT=1`: enables faster-whisper model download for the STT probe.

## Coverage

- `test_live_api.py`: health, registration, duplicate registration, login failure, `/me`, recordings auth, SQL injection validation, path traversal, JWT tamper, optional rate-limit and upload quota probes.
- `test_live_website.py`: landing, privacy/terms, registration, login, forgot password, cookies, bogus route behavior, mobile viewport, console/network errors.
- `test_live_sdk.py`: clean-venv `pip install violawake`, optional `violawake[all]`, detector/VAD/VoicePipeline/STT/TTS behavior, edge cases.
- `test_live_wasm.py`: live demo route reachability and optional ONNX/browser memory probes when assets exist.
- `test_live_email.py`: Resend verification through Mailosaur/webhook when configured, otherwise records whether live registration auto-verifies.

## Interpreting Common Failures

- Registration returns `email_verified=true`: Resend is disabled or email sending failed and the backend auto-verified the user.
- Email test skips with an unverified user: Resend may be configured, but no automated inbox was provided to verify delivery.
- `pip install violawake[all]` fails: the published extras are not clean-machine installable. The smoke path only checks the core install.
- WASM tests skip: the live site did not serve `/wasm-demo/` or `/demo`, and local `wasm/dist` assets were not present.
- Bogus route test fails: the SPA is redirecting unknown routes to `/` instead of rendering a 404 UI.
