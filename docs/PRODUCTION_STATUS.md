# Production Status

Living doc. Last verified end-to-end: **2026-05-07** (post-deploy includes: Stripe LIVE mode active; daily Postgres backups → R2; CSP/HSTS headers via Cloudflare Pages `_headers`; CSP via backend middleware; WASM browser SDK click-to-run at violawake.com/wasm/demo/; SDK auto-downloads OWW backbones on first init; rate limit raised to 100/hr keyed on CF-Connecting-IP; webhook idempotency Postgres-backed; security regression tests; legal pages updated for Stripe + Resend disclosure + retention windows; accessibility audit completed with trivial fixes applied). Update this file's date and the relevant rows whenever the live state changes.

This is the **canonical** post-launch status. Do not add running notes to `LAUNCH_READINESS.md`, `PROGRESS.md`, or `FUNCTIONAL_GAP_ANALYSIS.md` for post-launch state — those captured the pre-launch sprint. New facts go here.

---

## What's live

| Layer | Status | Where | Last deploy |
|---|---|---|---|
| Frontend | ✅ live | Cloudflare Pages, project `violawake`, `violawake.com` | 2026-07-03 (commit `c85977a`, deploy `5b2de9f4`: bug-report button on every page + canonical `/app/` SPA rewrite target fixing the /verify-email & /reset-password deep-link 308 strip) |
| Backend | ✅ live | `wakeword-backend-1` on the Hetzner box `167.233.233.33`, checkout `/opt/viola/Wakeword`, behind Cloudflare Tunnel `violawake-api-server` → `api.violawake.com` | **Automatic since 2026-07-31**: `violawake-deploy.timer` reconciles the container toward `origin/master` every 10 min (`scripts/deploy_backend.py`). The deployed commit is readable from the running image: `docker image inspect $(docker inspect wakeword-backend-1 --format '{{.Image}}') --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'` |
| Postgres | ✅ live | `wakeword-postgres-1` on the same box, internal to the `wakeword_default` network | volume `pgdata`, survives recreations |
| Cloudflare Tunnel | ✅ live | Container `cloudflared-wake-server`, tunnel `violawake-api-server` `a4961724-2b7b-49a1-8711-e088245be4c4` | The old `violawake-api` tunnel `7dbef1da-…` was deleted in the #326 estate cleanup (commit `06b35f8`) — it had no DNS route |
| SDK on PyPI | ✅ live | `violawake` v0.2.4 | 2026-05-07 (manually published via twine — `release.yml` had a chicken-and-egg bug where `pypi-publish.needs: [..., github-release]` and `github-release` 404'd on `fetch_release_models.py`, blocking PyPI even for valid wheels. Both fixed in same commit.) |
| Support inbox | ✅ live | Cloudflare Worker `violawake-agentic-inbox`, `support-inbox.violawake.com` (Access-protected); R2 `violawake-agentic-inbox`; `hello@violawake.com` Email Routing → worker | 2026-07-03 (version `189b07e1`) |

## Support inbox — agentic-inbox (2026-07-03)

Customer-support inbox for `hello@violawake.com`, mirroring useviola.com's setup but
**standalone** (own worker / R2 bucket / Access app / service token — shares nothing
with NOVVIOLA). Source vendored at `infra/agentic-inbox/`; founder-side ops skill at
`.claude/skills/support-inbox/`; deploy + rollback runbook + all resource ids at
`infra/agentic-inbox/DEPLOY.md`.

- **Worker** `violawake-agentic-inbox` deployed (version `189b07e1`) at custom domain
  `support-inbox.violawake.com`; bindings MAILBOX/EMAIL_AGENT/EMAIL_MCP (DOs), EMAIL
  (CF Email Service), BUCKET (R2 `violawake-agentic-inbox`), AI.
- **Access** self-hosted app (AUD `0ebf81ca…`) + non-identity service-token policy.
  Verified: no token → **403**, service token → **200**.
- **Inbound cutover**: `hello@violawake.com` Email Routing rule
  `5d6083a078794d4bb98d5e10a007b3cc` flipped from the old `violawake-support-email`
  worker to this one. The worker re-forwards every inbound to
  `violavoiceassistant@gmail.com` (`EMAIL_FORWARD_COPY_TO`), preserving the Gmail copy
  (CF allows one action per rule, so the worker does the forward). Catch-all left as
  `drop` (only `hello@` captured, as before).
- **Verified live**: real test emails to `hello@violawake.com` captured with full
  bodies + readable via the Access-authed REST API; audited outbound reply to a
  founder-controlled address returned `{"status":"sent"}` and filed in `sent`.
- **Rollback** (one API call): flip rule `5d6083a…` back to `violawake-support-email`
  (left deployed, not deleted). Old Console `/api/email/inbound` auto-ack pipeline
  intact.
- **Behavior change flagged**: the old worker auto-acknowledged every sender with a
  Resend "ticket VW-XXXX" reply; the agentic-inbox model is draft-and-approve (no
  auto-send), so that auto-ack no longer fires — aligns with the founder's
  approval-gated goal.
- **Known**: a concurrent Cloudflare Durable-Object storage incident on 2026-07-03
  intermittently 500'd both this inbox AND the useviola prod inbox (same platform
  error); the worker re-throws on capture failure so CF/sender MTAs retry (a
  during-incident message landed on retry after recovery — delayed, not dropped). The
  physical Gmail-copy arrival is founder-confirmable in `violavoiceassistant@gmail.com`.

## Verified end-to-end (2026-05-07)

- ✅ `pip install violawake[oww]` on a clean venv → import + load `temporal_cnn` model + score silence/sine/noise. All scored < 0.5 (no false positives).
- ✅ `https://violawake.com/` renders the polished landing page; pricing tiers Free/$29/$99/Custom show; cookie banner + privacy + terms render.
- ✅ Frontend bundle has `https://api.violawake.com/api` baked in (no same-origin `/api` fallback).
- ✅ `GET /api/health` → 200.
- ✅ `POST /api/auth/register` with new email → 201, returns user without token (verification flow placeholder).
- ✅ `POST /api/auth/login` with valid creds → 200 + JWT.
- ✅ `GET /api/auth/me` with token → 200.
- ✅ **Billing REMOVED 2026-07-31 (founder decision: ViolaWake is a free service).** All `/api/billing/*` routes, Stripe checkout/webhook/portal, and paid tiers are gone from the codebase. The Stripe webhook endpoint and the two ViolaWake price objects in the shared Stripe dashboard can be deleted/archived by the founder (no active subscribers existed at removal: all 72 subscription rows were tier=free with no stripe_subscription_id).
- ✅ `GET /api/billing/subscription` → returns user's free-tier state.
- ✅ `GET /api/recordings` (authed) → `[]` for new user.
- ✅ `POST /api/auth/login` with 4 wrong passwords → 401, then 5+ → 429 (slowapi rate limit).
- ✅ `DELETE /api/auth/account` without password body → 422 (security sprint enforces password confirm).
- ✅ `DELETE /api/auth/account` with wrong password → 401.
- ✅ `POST /api/recordings/upload` with 16 MB body → 413.
- ✅ Total live API routes: 44 (includes new `/api/teams/accept`, `/api/teams/{team_id}/leave`, `/api/auth/change-password`).

## NOT verified (and what would verify it)

- ✅ **API tunnel healthy** (was ❌ on 2026-06-03, when tunnel `violawake-api`
  `7dbef1da-…` reported `down` and the API returned HTTP 530). Re-measured
  2026-07-31: `https://api.violawake.com/api/health` returns HTTP 200 from
  outside Docker, served through the *replacement* tunnel
  `violawake-api-server` `a4961724-…` (container `cloudflared-wake-server`).
  The `7dbef1da` tunnel that was down no longer exists — it was deleted in the
  #326 Cloudflare estate cleanup. Re-verify with
  `curl -sS https://api.violawake.com/api/health`.
- ❌ **Nightly R2 backups are current.** Read-only R2 listing on 2026-06-03
  found latest Postgres/app-data backup objects dated 2026-05-10. Verify the
  Windows scheduled task, run `scripts/backup_to_r2_wrangler.sh`, then run
  `python scripts/backup_restore_drill.py --max-age-hours 36 --env-file
  .env.production --env-file /j/CLAUDE/PROJECTS/FewerJobs/.env` before moving
  this back to verified.
- ✅ **Email actually sends. Resend configured 2026-05-07 20:31 UTC.** Domain `violawake.com` verified on Resend (DNS records autoconfigured via Cloudflare integration). `VIOLAWAKE_RESEND_API_KEY` set in `.env.production`, backend restarted. Verified live by registering two test users post-restart and observing backend log `violawake.email: Sent email to ... for subject Verify your ViolaWake email` followed by `email_verified=False` on the resulting user (confirming the email-required path is now active rather than the auto-verify fallback). Real-inbox delivery to a live mailbox not yet confirmed — recommend one register-and-verify with a real mailbox before announcing.
- ❌ **Account lockout actually triggers.** New code adds `failed_login_count` + `locked_until` columns and sets them on bad attempts. We confirmed the rate-limit (slowapi) blocks after 4 attempts, but we have NOT confirmed the per-account lockout (which would persist across IP changes). To verify: 5 wrong logins on one account, then a 6th from a fresh IP should still 401 with "Account temporarily locked".
- ❌ **Full training pipeline against the live console.** Upload 10 silent/synthetic WAVs → start training → wait for SSE completion → download ONNX → load locally → push silence → expect score < 0.5. Blocked previously by the registration rate limit (10/hour) eating the test budget; do this when bumping the limit.
- ❌ **True-positive wake detection.** Verified: silence/sine/noise score < 0.5 (no false positives). NOT verified: an actual "Viola" utterance scores > 0.5 (true positive). Needs a recorded sample or live mic.
- ⚠️ **WASM browser SDK demo loads at `https://violawake.com/wasm/demo/`** (deployed 2026-05-07 — files copied into `console/frontend/public/wasm/`). However the demo requires the user to manually paste a "Model base URL" pointing at OWW backbone files; there is no default URL. To make it click-to-run, host `melspectrogram.onnx`, `embedding_model.onnx`, `temporal_cnn.onnx` somewhere CORS-friendly and default the URL field to that.

## Operational levers (only the operator can change)

| Lever | Where | Current value | Suggested for launch |
|---|---|---|---|
| Cloudflare API token | FewerJobs `.env` (`CLOUDFLARE_API_TOKEN`) | active, no expiry | Has DNS:Edit + zone:read scopes — can auto-add Resend DNS records when needed. Lacks Pages:Edit and email_routing — auto-deploy + email routing still need a wider-scoped token, OR keep manual. Stored as GH secret `CLOUDFLARE_API_TOKEN` for the deploy-pages.yml workflow (which currently won't fully succeed without Pages:Edit). |
| `VIOLAWAKE_RESEND_API_KEY` | `.env.production` | **set** as of 2026-05-07 20:31 UTC; domain `violawake.com` verified on Resend; `Sent email to ...` lines visible in backend logs | rotate if leaked; otherwise leave |
| `VIOLAWAKE_STRIPE_SECRET_KEY` | `.env.production` | UNUSED since 2026-07-31 (billing removed; config field deleted) | safe to remove from the env file |
| Registration rate limit | `console/backend/app/rate_limit.py` `REGISTER_LIMIT` | `100/hour` (raised from 10/hour 2026-05-07) — and now correctly per-end-user-IP via `CF-Connecting-IP` instead of per-CF-edge-IP | leave unless you start seeing register spam |
| Login rate limit | same file, `LOGIN_LIMIT` | works | leave |
| Robots / sitemap | `console/frontend/public/robots.txt` | sitemap line points to `console.violawake.com` (typo) | should be `violawake.com/sitemap.xml` |
| OG image | `console/frontend/public/og-image.png` | placeholder PNG | upgrade to a real branded 1200×630 |

## Known follow-ups (test debt, not blocking launch)

These are tracked as separate work; none of them affect the live user journey.

1. ~~~25 stale test mocks~~ **RESOLVED 2026-05-07.** All 28 failures + 7 errors fixed across `test_backend.py`, `test_billing.py`, `test_health_monitoring.py`, `test_job_queue.py`, `test_teams.py`. Final: `147 passed, 2 skipped, 0 failed, 0 errors` (was 136 before security regression tests landed). The 2 skips are intentional (rate-limit-header tests; conftest disables the limiter globally for the suite).
2. ~~Five ruff lint errors~~ **RESOLVED 2026-05-07.** `ruff check src/` and `ruff format --check src/` are now clean. CI's lint job for the SDK should pass.
3. **Pages docs workflow** fails because GitHub Pages isn't enabled on the repo. Either enable it or remove the `docs.yml` workflow.
4. **`tools/fetch_release_models.py`** uses `gh` CLI primary + GitHub API fallback. If GitHub Releases for the SDK package are populated correctly, releases work; verify on next tag-push.
5. **Hash-mismatch warning from `openwakeword`** at SDK runtime: `OWW backbone hash mismatch: expected 70d164290c1d095d, got e8444299a314fbb2`. Means the openwakeword package was updated upstream and the hash check warns but does not fail. Could degrade accuracy on real wake detection. Decide: pin a specific openwakeword version, or update the expected hash, or remove the check.

## How to keep this doc honest

When you deploy or change live state:

1. Update the **Last verified** date at the top.
2. Update the relevant row(s) under "What's live" with the new commit / version / date.
3. Move items between "Verified" and "NOT verified" as facts change.
4. If an "Operational lever" changes (e.g., you turn on Resend), update its row and add a new line under "Verified end-to-end" describing what now works.
5. If a "Known follow-up" gets resolved, delete it.

If you ever discover the live state has drifted from this doc, **trust the live probe (`curl`, OpenAPI, smoke suite) over this file** and update the doc to match what you observed.
