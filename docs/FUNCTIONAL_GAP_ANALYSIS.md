# ViolaWake Functional Gap Analysis

**Date:** 2026-03-28
**Auditor scope:** All code under `J:\CLAUDE\PROJECTS\Wakeword`
**Method:** Full source read of every system referenced in the audit checklist

> **Updated 2026-04-05:** Many items resolved during security hardening sprint (20 fixes across 2 adversarial audit rounds). Resolved items marked with ~~strikethrough~~.
>
> **Updated 2026-05-07:** Live Playwright + source audit found 7 more items resolved in code that this doc had still listed OPEN. Resolved-since-2026-04-05 items: P0-2 (WASM built and demo serves models 200), P1-1 (Teams UI shipped), P1-2 (team-invite endpoint calls `send_team_invite`; raw-token leak gated behind a test-mode helper), P1-5 (`scripts/verify_models.py` exists), P1-6 (`scripts/generate_docs.py` exists), P1-8 (WASM demo no longer double-processes audio), P2-7 (forgot-password link + page exist).

---

## Executive Summary

ViolaWake is approximately **70% ready for production use as an open-source SDK** and approximately **45% ready as a hosted SaaS product**. The SDK core (wake detection, model download, training pipeline, PyPI packaging) is solid and shippable. The Console backend has real auth, real billing wiring, real training queues, and real storage abstraction. The gaps that remain are not scaffolding gaps -- they are *operational readiness* gaps: a broken Alembic migration, a WASM package that has never been built, a model release pipeline that is still a TODO stub, and a frontend that is missing the teams UI it advertises on the pricing page.

---

## Critical Gaps (P0) -- Users bounce immediately

### ~~P0-1: Alembic migration is missing 3 tables the ORM declares~~ RESOLVED (2026-04-05)

**Status:** Migration now exists for `teams`, `team_members`, `recordings.team_id`, and `trained_models.team_id`. Deploying with `alembic upgrade head` creates all required tables.

**Files:**
- `console/backend/alembic/versions/20260326_0001_a1b2c3d4e5f6_initial_schema.py`
- `console/backend/app/models.py` (reference -- correct)

---

### ~~P0-2: WASM package has never been built~~ RESOLVED (2026-05-07)

**Status:** WASM bundle and demo are deployed at `https://violawake.com/wasm/demo/` and serve all three ONNX models with HTTP 200. Verified live: page loads with title "ViolaWake — Browser Demo", default model URL pre-filled (`../models`), assets resolve.

**Files (now present in deployed `dist/`):**
- `console/frontend/public/wasm/dist/violawake.js`, `violawake.cjs`, `index.d.ts`, `detector.d.ts`, `features.d.ts`
- `console/frontend/public/wasm/models/temporal_cnn.onnx`, `melspectrogram.onnx`, `embedding_model.onnx`
- `console/frontend/public/wasm/demo/index.html`

---

### ~~P0-3: Release model pipeline is a TODO stub~~ RESOLVED (2026-05-06)

**Status:** `tools/fetch_release_models.py` now downloads release `.onnx` assets from GitHub Releases using `gh release download` when available, with a urllib GitHub API fallback. The script accepts `--tag`, preserves release workflow compatibility with `--version`, and writes to `--output`.

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

### ~~P0-4: Two model registry entries have placeholder SHA-256 hashes~~ RESOLVED (2026-05-06)

**Status:** `viola_mlp_oww` and `viola_cnn_v4` are not present in `MODEL_REGISTRY`; active tests and docs now reference `temporal_cnn` instead.

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

### ~~P1-1: Frontend has no Teams UI despite backend having full Teams API~~ RESOLVED (2026-05-07)

**Status:** Teams UI is shipped end-to-end. Routes registered in `App.tsx:101-122` (`/teams`, `/teams/accept`, `/teams/:teamId`). Pages: `Teams.tsx` (list + create modal + empty state), `TeamDetail.tsx` (detail + invite + member management), `TeamAccept.tsx` (token-based acceptance). Nav link in `Layout.tsx:28`. All Teams API client functions wired in `api.ts:361-451` (`createTeam`, `listTeams`, `getTeam`, `inviteMember`, `joinTeam`, `acceptTeamInvite`, `removeMember`, `updateMemberRole`, `leaveTeam`, `shareModel`, `listTeamModels`, `deleteTeam`).

---

### ~~P1-2: Team invite does not send an email -- returns token in HTTP response~~ RESOLVED (2026-05-07)

**Status:** `EmailService.send_team_invite()` is implemented (`email_service.py:100-117`) and called from the invite endpoint (`routes/teams.py:259`). In production the response message is just "Invitation sent" via `_test_invite_message()` which only includes the raw token under the test-mode helper path. Note: actual email delivery still requires `VIOLAWAKE_RESEND_API_KEY` to be set (separate operational gap, see P1-7).

---

### ~~P1-3: Docker frontend serves Vite dev server in production~~ RESOLVED (2026-04-05)

**Status:** Dockerfile.frontend now uses a multi-stage build (node build stage -> nginx:alpine serve stage). Production frontend is served via nginx with proper caching headers, gzip, and SPA fallback.

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

### ~~P1-5: No `verify_models.py` script exists~~ RESOLVED (2026-05-07)

**Status:** `scripts/verify_models.py` is present.

---

### ~~P1-6: No `generate_docs.py` script exists~~ RESOLVED (2026-05-07)

**Status:** `scripts/generate_docs.py` is present.

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

### ~~P1-8: WASM demo double-processes audio frames~~ RESOLVED (2026-05-07)

**Status:** `console/frontend/public/wasm/demo/index.html:349-352` now calls `detector.detect(frame)` once and reads `detector.lastScore` for display, with an explicit comment: `// detect() already advances the streaming detector state. Read lastScore instead of calling getScore(frame) for the same audio.`

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

### ~~P2-7: Console has no forgot-password link on the login page~~ RESOLVED (2026-05-07)

**Status:** `Login.tsx:128-131` links to `/forgot-password`. `ForgotPassword.tsx` is a real page that posts to `/api/auth/forgot-password` and shows a security-aware "If an account exists for that email, a reset link has been sent" response. Route registered in `App.tsx:49`.

---

### P2-8: 100+ test MP3 files in repo root

**What:** The repository root contains 130+ `test_*.mp3` files (various TTS voices in multiple languages). These appear to be test artifacts, not production assets. They inflate the repo size and are not excluded from git.

**Effort:** S (add to `.gitignore`, remove from tracking)

**Files:**
- Repository root (`test_*.mp3`)

---

### ~~P2-9: `docker-compose.yml` uses deprecated `version: "3.9"` key~~ RESOLVED (2026-04-05)

**Status:** No `version` key in current compose files. Deprecation warning eliminated.

**Files:**
- `console/docker-compose.yml`

---

### P2-10: No rate limiting documentation for API consumers

**What:** The backend has rate limiting on registration (100/hr/IP), login (5/min/IP), email verification (20/5min/IP), password reset (5/5min/IP), and recording uploads (50/hr/user). None of these are documented in API docs or communicated to frontend developers.

**Effort:** S (add rate limit headers to responses, document in API reference)

**Files:**
- `console/backend/app/rate_limit.py`
- `console/backend/app/routes/auth.py`

---

## Summary Table

| ID | Gap | Severity | Effort | Blocks Users? | Status |
|----|-----|----------|--------|---------------|--------|
| P0-1 | ~~Alembic migration missing teams tables~~ | Critical | S | ~~Yes~~ | **RESOLVED** |
| P0-2 | ~~WASM package never built~~ | Critical | M | ~~Yes~~ | **RESOLVED** (2026-05-07) |
| P0-3 | ~~Release model pipeline is TODO stub~~ | Critical | M | Yes -- no new model releases | RESOLVED |
| P0-4 | ~~Placeholder SHA-256 hashes in registry~~ | Critical | S | No -- deprecated models, but CI noise | RESOLVED |
| P1-1 | ~~No Teams frontend UI~~ | Important | L | ~~Partially~~ | **RESOLVED** (2026-05-07) |
| P1-2 | ~~Team invite returns token, no email~~ | Important | S | Email path wired; delivery requires Resend (P1-7) | **RESOLVED** (2026-05-07) |
| P1-3 | ~~Docker serves Vite dev server~~ | Important | S | ~~No~~ | **RESOLVED** |
| P1-4 | Stripe price IDs need manual setup | Important | S | No -- LIVE prices configured 2026-05-07 | RESOLVED |
| P1-5 | ~~Missing verify_models.py script~~ | Important | S | ~~No~~ | **RESOLVED** (2026-05-07) |
| P1-6 | ~~Missing generate_docs.py script~~ | Important | S | ~~No~~ | **RESOLVED** (2026-05-07) |
| P1-7 | Email requires Resend with no dev fallback | Important | S | Yes -- production deploy still has Resend unset | OPEN |
| P1-8 | ~~WASM demo double-processes frames~~ | Important | S | ~~No~~ | **RESOLVED** (2026-05-07) |
| P2-1 | No WASM CI job | Nice-to-have | S | No | OPEN |
| P2-2 | mypy non-blocking | Nice-to-have | S | No | OPEN |
| P2-3 | 50% coverage floor | Nice-to-have | M | No | OPEN |
| P2-4 | No OAuth / social login | Nice-to-have | L | No | OPEN |
| P2-5 | No model comparison UI | Nice-to-have | M | No | OPEN |
| P2-6 | No GPU training lane | Nice-to-have | L | No | OPEN |
| P2-7 | ~~No forgot-password link on login~~ | Nice-to-have | S | ~~No~~ | **RESOLVED** (2026-05-07) |
| P2-8 | 130+ test MP3s in repo root | Nice-to-have | S | No | OPEN |
| P2-9 | ~~Deprecated docker-compose version key~~ | Nice-to-have | S | ~~No~~ | **RESOLVED** |
| P2-10 | No rate limit documentation | Nice-to-have | S | No | OPEN |

---

## What IS solid

To be fair about what works well:

- **SDK core** (`WakeDetector`, `AsyncWakeDetector`, `DetectorConfig`, `VADEngine`, `NoiseProfiler`, `PowerManager`, `FusionStrategy`) -- all real implementations with real tests
- **Model download** -- SHA-256 verification, atomic writes, auto-download, size validation, HTTPS-only enforcement
- **Training pipeline** -- real PyTorch training via temporal CNN (Console training path), epoch callbacks, cancellation support, timeout protection
- **Console backend** -- real FastAPI app with JWT auth, rate limiting, bcrypt password hashing, Alembic migrations, async SQLite/PostgreSQL, Cloudflare R2 storage, Resend email, Stripe billing (when configured), SSE training progress streams, retention cleanup
- **Console frontend** -- real React app with auth context, protected routes, recording session, training status polling, billing management, model performance visualization
- **CI/CD** -- 6 workflow files covering lint, unit tests (9 matrix entries), integration tests, benchmarks, console tests, docs deployment, model verification, and release automation
- **PyPI packaging** -- correct `pyproject.toml` with 10 optional extras, proper hatch build config, entry points for CLI tools, sdist excludes

The codebase is substantially more complete than most open-source SDK projects at this stage. The gaps identified above are real but mostly small-to-medium fixes.
