# ViolaWake Functional Gap Analysis

**Date:** 2026-03-28
**Auditor scope:** All code under `J:\CLAUDE\PROJECTS\Wakeword`
**Method:** Full source read of every system referenced in the audit checklist

---

## Executive Summary

ViolaWake is approximately **70% ready for production use as an open-source SDK** and approximately **45% ready as a hosted SaaS product**. The SDK core (wake detection, model download, training pipeline, PyPI packaging) is solid and shippable. The Console backend has real auth, real billing wiring, real training queues, and real storage abstraction. The gaps that remain are not scaffolding gaps -- they are *operational readiness* gaps: a broken Alembic migration, a WASM package that has never been built, a model release pipeline that is still a TODO stub, and a frontend that is missing the teams UI it advertises on the pricing page.

---

## Critical Gaps (P0) -- Users bounce immediately

### P0-1: Alembic migration is missing 3 tables the ORM declares

**What:** The initial Alembic migration (`console/backend/alembic/versions/20260326_0001_a1b2c3d4e5f6_initial_schema.py`) creates 6 tables: `users`, `recordings`, `trained_models`, `training_jobs`, `subscriptions`, `usage_records`. But the ORM in `console/backend/app/models.py` declares 7 tables, plus column additions:

- `teams` table -- entirely missing from migration
- `team_members` table -- entirely missing from migration
- `recordings.team_id` column -- missing from migration
- `trained_models.team_id` column -- missing from migration

**Why it matters:** Anyone deploying with `alembic upgrade head` (the standard path) will get a database that crashes the moment any teams endpoint is hit. The backend imports and wires `teams.router` unconditionally, so even listing teams returns a 500 because the table does not exist.

**Effort:** S (add a second migration with the missing CREATE TABLEs and ALTER TABLEs)

**Files:**
- `console/backend/alembic/versions/20260326_0001_a1b2c3d4e5f6_initial_schema.py`
- `console/backend/app/models.py` (reference -- correct)

---

### P0-2: WASM package has never been built

**What:** The `wasm/` directory has TypeScript source, a `package.json`, a `rollup.config.mjs`, and a demo `index.html`. But:

1. `wasm/dist/` does not exist -- the build has never been run
2. `wasm/node_modules/` does not exist -- `npm install` has never been run
3. The demo HTML imports from `../dist/violawake.js` which does not exist
4. The demo expects model files at `./models/` which do not exist
5. No CI job builds or tests the WASM package

**Why it matters:** Anyone cloning the repo and opening the demo will get a blank page with a console error. The WASM SDK is advertised in the README and represents the browser deployment story -- the most differentiating feature vs competitors. A non-functional demo is worse than no demo.

**Effort:** M (run `npm install && npm run build`, create a `wasm/demo/models/` directory with instructions or symlinks, add a CI job for the WASM build)

**Files:**
- `wasm/package.json`
- `wasm/demo/index.html`
- `.github/workflows/ci.yml` (needs a WASM build job)

---

### P0-3: Release model pipeline is a TODO stub

**What:** The `tools/fetch_release_models.py` script, called during the GitHub Release workflow, contains this:

```python
print(
    "TODO: implement artifact-store download support for S3/GCS with MODEL_STORE_TOKEN; "
    "using local fallback for now."
)
```

It attempts a local fallback from `models/` in the repo root. If those files are missing (and they are not checked into git per `.gitignore`/sdist excludes), the release workflow fails silently or with a non-obvious error. The `MODEL_STORE_TOKEN` secret is referenced but never consumed.

**Why it matters:** The entire `release.yml` workflow -- the mechanism for publishing new model versions to GitHub Releases -- depends on this script. No working release pipeline means no new model versions can be distributed to users via `violawake-download`.

**Effort:** M (implement S3/GCS download or simply copy models from a CI artifact cache; update `release.yml` to upload them)

**Files:**
- `tools/fetch_release_models.py`
- `.github/workflows/release.yml` (step: "Fetch model files from model artifact store")

---

### P0-4: Two model registry entries have placeholder SHA-256 hashes

**What:** `viola_mlp_oww` and `viola_cnn_v4` in `src/violawake_sdk/models.py` have:
```python
sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT"
```

The `download_model()` function correctly refuses to download these. But:
- `check_registry_integrity(strict=True)` will raise `RuntimeError` if called with these present
- The model-verify CI workflow runs `scripts/verify_models.py --ci` which may or may not handle this gracefully

These are marked DEPRECATED and labeled "never released", so the impact is contained. But they create noise in CI and confuse users who browse the registry.

**Why it matters:** Registry integrity checks in CI will flag these every run. New contributors will wonder why they exist.

**Effort:** S (remove both entries entirely since they are deprecated and were never uploaded)

**Files:**
- `src/violawake_sdk/models.py` (lines 59-86)

---

## Important Gaps (P1) -- Hurt credibility or prevent real use

### P1-1: Frontend has no Teams UI despite backend having full Teams API

**What:** The backend has a complete teams system: create team, invite member, join via token, remove member, change roles, share models, list team models (8 endpoints in `console/backend/app/routes/teams.py`). The frontend has zero team pages. The only mention of teams is on the Pricing page: `"Team management (coming soon)"`.

**Why it matters:** The backend investment is wasted if users cannot access it. The pricing page advertises it as a Business-tier feature. Users who upgrade expecting team management will be disappointed.

**Effort:** L (new pages: TeamList, TeamDetail, TeamInvite, TeamSettings; new API client functions; new routes in App.tsx)

**Files:**
- `console/frontend/src/App.tsx` (no team routes)
- `console/frontend/src/api.ts` (no team API functions)
- `console/frontend/src/pages/` (no team pages)
- `console/backend/app/routes/teams.py` (backend is complete)

---

### P1-2: Team invite does not send an email -- returns token in HTTP response

**What:** In `console/backend/app/routes/teams.py`, the `invite_member` endpoint:
```python
# In production, this token would be emailed. We return it in the response
# so callers can forward it (e.g., the email service layer).
return MessageResponse(message=f"Invite token issued: {invite_token}")
```

The `EmailService` has methods for verification, password reset, welcome, training complete, and quota warning -- but NOT for team invites. The invite token is returned raw in the JSON response body.

**Why it matters:** Team invites only work if someone manually copies the token from the API response and sends it to the invitee. This is not a viable user flow.

**Effort:** S (add `send_team_invite()` to `EmailService`, call it from the invite endpoint)

**Files:**
- `console/backend/app/routes/teams.py` (line 186)
- `console/backend/app/email_service.py` (missing `send_team_invite`)

---

### P1-3: Docker frontend serves Vite dev server in production

**What:** `console/Dockerfile.frontend`:
```dockerfile
CMD ["npx", "vite", "--host", "--port", "5173"]
```

This runs the Vite development server in the Docker container. For production, the frontend should be built with `npm run build` and served via a static file server (nginx, or the backend itself serving the built assets).

**Why it matters:** The Vite dev server is not production-ready: no gzip, no caching headers, HMR overhead, slower TTFB, potential security issues. Anyone deploying via docker-compose gets a dev-quality frontend.

**Effort:** S (multi-stage Dockerfile: build stage runs `npm run build`, serve stage uses nginx or `serve`)

**Files:**
- `console/Dockerfile.frontend`
- `console/docker-compose.yml`

---

### P1-4: Stripe price IDs are hardcoded placeholders

**What:** In `console/backend/app/routes/billing.py`:
```python
TIER_PRICE_MAP: dict[str, str] = {
    "developer": "stripe_price_developer",
    "business": "stripe_price_business",
}
```

These map to `settings.stripe_price_developer` and `settings.stripe_price_business` which default to empty strings. The `_price_id_for_tier` function raises HTTP 503 when these are empty. This means checkout is permanently broken until someone creates Stripe products and configures the env vars.

**Why it matters:** The billing UI (frontend Billing page, Pricing page with checkout buttons) is fully built but will always 503 without manual Stripe setup. There are no setup instructions in the README or docker-compose for billing configuration.

**Effort:** S (add billing setup docs; ensure `docker-compose.yml` includes placeholder comments for Stripe env vars)

**Files:**
- `console/backend/app/routes/billing.py`
- `console/backend/app/config.py` (Stripe config fields)
- `console/docker-compose.yml` (missing Stripe env vars)

---

### P1-5: No `verify_models.py` script exists

**What:** The `model-verify.yml` CI workflow runs:
```yaml
- name: Verify all models
  run: python scripts/verify_models.py --ci
```

But `scripts/verify_models.py` does not exist in the `scripts/` directory. The directory listing shows:
```
audit_deps.py, benchmark.py, benchmark_regression_check.py, build_clean_eval_set.py,
far_frr_analysis.py, fetch_release_models.py, live_head_to_head.py, merge_worktrees.py,
quality_gate.py, setup_github_repo.sh, update_model_registry.py
```

No `verify_models.py`.

**Why it matters:** The model-verify CI workflow will fail on every run with `FileNotFoundError`.

**Effort:** S (create the script; it should iterate MODEL_REGISTRY, download each, verify SHA-256, and report)

**Files:**
- `.github/workflows/model-verify.yml` (line 79)
- `scripts/` (missing `verify_models.py`)

---

### P1-6: No `generate_docs.py` script exists

**What:** The CI workflow `ci.yml` (console-backend job, line 135) and `docs.yml` (line 46) both run:
```yaml
- name: Verify API docs generation
  run: python scripts/generate_docs.py
```

This script does not exist in the `scripts/` directory.

**Why it matters:** The console-backend CI job and the docs deployment workflow both fail.

**Effort:** S (create the script; likely just calls `pdoc` on the SDK source)

**Files:**
- `.github/workflows/ci.yml` (line 135)
- `.github/workflows/docs.yml` (line 46)
- `scripts/` (missing `generate_docs.py`)

---

### P1-7: Email delivery requires external Resend API key with no fallback

**What:** The `EmailService` uses Resend (https://api.resend.com/emails) for all transactional email. When `VIOLAWAKE_RESEND_API_KEY` is not set (the default), all emails silently no-op:

```python
if not self.enabled:
    self._warn_disabled()
    logger.info("Skipping email send to %s ...")
    return False
```

This means: email verification links are never sent, password reset links are never sent, welcome emails are never sent, training completion notifications are never sent.

The registration flow still returns a JWT token, so users can log in. But they cannot verify their email, which means any endpoint using `get_verified_user` (recordings, training, billing, teams) returns HTTP 403.

**Why it matters:** Without Resend configured, new users register but cannot use any protected feature. There is no console-side email verification bypass for development.

**Effort:** S (add a development bypass: when `env=development` and email is disabled, auto-verify users on registration; document Resend setup for production)

**Files:**
- `console/backend/app/email_service.py`
- `console/backend/app/routes/auth.py` (register endpoint)
- `console/backend/app/auth.py` (`get_verified_user` dependency)

---

### P1-8: WASM demo double-processes audio frames

**What:** In `wasm/demo/index.html`, the audio processing loop calls both `getScore()` and `detect()` on the same frame:
```javascript
const score = await detector.getScore(frame);
const detected = await detector.detect(frame);
```

Both methods internally call `backbone.pushAudio()`, which advances the streaming state. This means each 20ms frame is fed through the pipeline twice, producing incorrect scores and potentially doubling the internal embedding buffer position.

**Why it matters:** The demo -- the primary showcase for browser detection -- will produce wrong scores and unreliable detection.

**Effort:** S (call `detect()` only, and read `detector.lastScore` for the display, or restructure to call `getScore()` and apply the decision gate manually)

**Files:**
- `wasm/demo/index.html` (lines 344-345)

---

## Nice-to-Have Gaps (P2) -- Polish items

### P2-1: No WASM CI job

**What:** The CI pipeline has jobs for lint, unit tests (3 platforms x 3 Python versions), console backend tests, console frontend build, integration tests, and benchmarks. But no job builds or typechecks the WASM package.

**Effort:** S (add a job that runs `cd wasm && npm ci && npm run typecheck && npm run build`)

**Files:**
- `.github/workflows/ci.yml`

---

### P2-2: mypy is non-blocking in CI

**What:** The lint job has `continue-on-error: true` for mypy:
```yaml
- name: Run mypy
  continue-on-error: true
  run: mypy src/violawake_sdk --exclude 'training|tools'
```

**Effort:** S (fix mypy errors and remove `continue-on-error`)

**Files:**
- `.github/workflows/ci.yml` (line 51-52)

---

### P2-3: Coverage floor is 50% -- low for a production SDK

**What:** CI enforces `--cov-fail-under=50` for unit tests and `--cov-fail-under=65` for release validation. For an SDK with security-sensitive model verification and cryptographic checks, 50% is low.

**Effort:** M (add mock-based tests for audio pipeline, model download, SHA verification)

**Files:**
- `.github/workflows/ci.yml` (line 97)
- `.github/workflows/release.yml` (line 57)

---

### P2-4: No OAuth / social login

**What:** Auth is email+password only. No Google, GitHub, or other OAuth providers.

**Effort:** L

**Files:**
- `console/backend/app/auth.py`
- `console/backend/app/routes/auth.py`

---

### P2-5: No model versioning or A/B comparison in Console

**What:** Users can train multiple models but cannot compare them side-by-side. The `ModelPerformance` page shows one model at a time. There is no version history or rollback.

**Effort:** M

**Files:**
- `console/frontend/src/pages/ModelPerformance.tsx`
- `console/frontend/src/pages/Dashboard.tsx`

---

### P2-6: No GPU training lane

**What:** The training service runs on CPU via the job queue. The competitive analysis acknowledges this: "No GPU-backed paid training lanes." For larger models or datasets, CPU training can take 30+ minutes (the configured timeout).

**Effort:** L (requires infrastructure: GPU worker pool, queue routing by tier)

**Files:**
- `console/backend/app/job_queue.py`
- `console/backend/app/services/training_service.py`

---

### P2-7: Console has no forgot-password link on the login page

**What:** The backend has a full forgot-password flow (`/api/auth/forgot-password`, `/api/auth/reset-password`). The frontend has a `ResetPassword.tsx` page. But there is no link from the Login page to trigger the forgot-password flow -- users must navigate to `/reset-password` manually. Also there is no `ForgotPassword.tsx` page that asks for the email address; the existing `ResetPassword.tsx` only handles the token-based reset step.

**Effort:** S (add a ForgotPassword page and link it from Login)

**Files:**
- `console/frontend/src/pages/Login.tsx` (missing link)
- `console/frontend/src/pages/` (missing ForgotPassword page)
- `console/frontend/src/App.tsx` (missing route)

---

### P2-8: 100+ test MP3 files in repo root

**What:** The repository root contains 130+ `test_*.mp3` files (various TTS voices in multiple languages). These appear to be test artifacts, not production assets. They inflate the repo size and are not excluded from git.

**Effort:** S (add to `.gitignore`, remove from tracking)

**Files:**
- Repository root (`test_*.mp3`)

---

### P2-9: `docker-compose.yml` uses deprecated `version: "3.9"` key

**What:** Docker Compose v2+ ignores the `version` key and prints a deprecation warning.

**Effort:** S (remove the `version` line)

**Files:**
- `console/docker-compose.yml` (line 1)

---

### P2-10: No rate limiting documentation for API consumers

**What:** The backend has rate limiting on registration (100/hr/IP), login (5/min/IP), email verification (20/5min/IP), password reset (5/5min/IP), and recording uploads (50/hr/user). None of these are documented in API docs or communicated to frontend developers.

**Effort:** S (add rate limit headers to responses, document in API reference)

**Files:**
- `console/backend/app/rate_limit.py`
- `console/backend/app/routes/auth.py`

---

## Summary Table

| ID | Gap | Severity | Effort | Blocks Users? |
|----|-----|----------|--------|---------------|
| P0-1 | Alembic migration missing teams tables | Critical | S | Yes -- 500 on any teams endpoint |
| P0-2 | WASM package never built | Critical | M | Yes -- demo is broken |
| P0-3 | Release model pipeline is TODO stub | Critical | M | Yes -- no new model releases |
| P0-4 | Placeholder SHA-256 hashes in registry | Critical | S | No -- deprecated models, but CI noise |
| P1-1 | No Teams frontend UI | Important | L | Partially -- backend works, no UI |
| P1-2 | Team invite returns token, no email | Important | S | Yes -- unusable invite flow |
| P1-3 | Docker serves Vite dev server | Important | S | No -- works but not production-grade |
| P1-4 | Stripe price IDs need manual setup | Important | S | Yes -- billing always 503 |
| P1-5 | Missing verify_models.py script | Important | S | No -- CI fails silently |
| P1-6 | Missing generate_docs.py script | Important | S | No -- CI fails, no docs deployed |
| P1-7 | Email requires Resend with no dev fallback | Important | S | Yes -- new users locked out |
| P1-8 | WASM demo double-processes frames | Important | S | No -- wrong scores in demo |
| P2-1 | No WASM CI job | Nice-to-have | S | No |
| P2-2 | mypy non-blocking | Nice-to-have | S | No |
| P2-3 | 50% coverage floor | Nice-to-have | M | No |
| P2-4 | No OAuth / social login | Nice-to-have | L | No |
| P2-5 | No model comparison UI | Nice-to-have | M | No |
| P2-6 | No GPU training lane | Nice-to-have | L | No |
| P2-7 | No forgot-password link on login | Nice-to-have | S | No |
| P2-8 | 130+ test MP3s in repo root | Nice-to-have | S | No |
| P2-9 | Deprecated docker-compose version key | Nice-to-have | S | No |
| P2-10 | No rate limit documentation | Nice-to-have | S | No |

---

## What IS solid

To be fair about what works well:

- **SDK core** (`WakeDetector`, `AsyncWakeDetector`, `DetectorConfig`, `VADEngine`, `NoiseProfiler`, `PowerManager`, `FusionStrategy`) -- all real implementations with real tests
- **Model download** -- SHA-256 verification, atomic writes, auto-download, size validation, HTTPS-only enforcement
- **Training pipeline** -- real PyTorch training via `_train_mlp_on_oww`, epoch callbacks, cancellation support, timeout protection
- **Console backend** -- real FastAPI app with JWT auth, rate limiting, bcrypt password hashing, Alembic migrations, async SQLite/PostgreSQL, Cloudflare R2 storage, Resend email, Stripe billing (when configured), SSE training progress streams, retention cleanup
- **Console frontend** -- real React app with auth context, protected routes, recording session, training status polling, billing management, model performance visualization
- **CI/CD** -- 6 workflow files covering lint, unit tests (9 matrix entries), integration tests, benchmarks, console tests, docs deployment, model verification, and release automation
- **PyPI packaging** -- correct `pyproject.toml` with 10 optional extras, proper hatch build config, entry points for CLI tools, sdist excludes

The codebase is substantially more complete than most open-source SDK projects at this stage. The gaps identified above are real but mostly small-to-medium fixes.
