# ViolaWake Console: Build vs Buy Audit

**Date:** 2026-03-28
**Auditor:** Senior Architect (automated code review of both codebases)
**Scope:** Every major subsystem in `console/backend/` and `console/frontend/`
**NOVVIOLA reference:** `J:\PROJECTS\NOVVIOLA_fixed3_patched\NOVVIOLA`

---

## Executive Summary

The ViolaWake Console codebase is remarkably well-built for a project this young. Most subsystems are **KEEP** — the hand-rolled implementations are lean, correct, and appropriately scoped. The two areas where OSS replacement would pay off are **rate limiting** (swap in slowapi when you need Redis) and **frontend form validation** (adopt react-hook-form to reduce boilerplate). Everything else is either already using the right library (bcrypt, python-jose, stripe SDK, boto3, SQLAlchemy async) or has custom code that is cleaner than the library alternative would be.

**Nothing should be borrowed from NOVVIOLA.** The NOVVIOLA auth system is 35+ files designed for a desktop-plus-cloud hybrid with magic links, MFA, field encryption, PKCE OAuth, brute-force guards, and privilege rotation. ViolaWake Console is a straightforward SaaS with email/password auth. Pulling NOVVIOLA code would import 10x the complexity for zero benefit.

---

## 1. Authentication & Session Management

### What ViolaWake Has (READ)

| File | Lines | What It Does |
|------|-------|-------------|
| `backend/app/auth.py` | 347 | bcrypt password hashing (with SHA-256 pre-hash for >72-byte passwords), JWT creation/decode via `python-jose`, purpose-scoped tokens (verify email, reset password, download, team invite), one-time download tokens with in-memory JTI tracking, `get_current_user` / `get_verified_user` FastAPI dependencies |
| `backend/app/routes/auth.py` | 311 | Register, login, me, verify-email, forgot-password, reset-password, delete-account (full GDPR cascade deletion of recordings, models, jobs, subscriptions) |
| `backend/app/middleware.py` | 273 | Request ID generation (UUID hex), structured JSON logging, security headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy), Sentry integration with PII scrubbing, error classification |
| `frontend/src/contexts/AuthContext.tsx` | 152 | React context for auth state, token validation on mount via /me call, post-auth redirect with open-redirect prevention |
| `frontend/src/api.ts` | 414 | JWT stored in localStorage, client-side expiry check with 30s buffer, automatic 401 redirect, manual JWT decode without library dependency |

### Current Implementation Quality: 4/5 stars

**Strengths:**
- `_prep_password()` correctly handles bcrypt's 72-byte limit by pre-hashing with SHA-256. This is a detail most hand-rolled auth systems miss.
- Purpose-scoped JWTs prevent token confusion attacks — a verify-email token cannot be used to reset a password.
- One-time download tokens use JTI tracking with pruning to prevent replay.
- `_client_ip()` implements an explicit trusted-proxy-count policy instead of blindly trusting X-Forwarded-For.
- Account deletion cascade properly cleans up all associated data across storage + DB.
- Frontend prevents open redirect on the `return` query parameter.
- Production enforces `VIOLAWAKE_SECRET_KEY` via Pydantic model validator.

**Security Gaps:**
1. **No refresh tokens.** JWTs have a 24-hour expiry and cannot be revoked server-side. If a token leaks, it is valid for 24 hours. Acceptable for launch; add a token blacklist or switch to opaque session tokens post-launch if needed.
2. **JWT in localStorage.** Vulnerable to XSS. HttpOnly cookies would be more secure, but localStorage is standard for SPAs with Bearer auth and the security headers middleware mitigates XSS surface.
3. **No CSRF protection.** Not needed with Bearer tokens (CSRF only applies to cookie-based auth), but if you ever add cookie auth, you will need it.
4. **Download token JTI store is in-memory.** Restarting the server invalidates all outstanding download tokens. This is fine at current scale but means tokens cannot be shared across multiple server processes.

### NOVVIOLA Comparison

NOVVIOLA's `auth/` directory has **35 files** including: magic link authentication, session management with sliding-window refresh (30-day default, 90-day max, idle timeout), brute-force guard with per-IP progressive backoff and per-account lockout, field-level encryption (AES-256-GCM for database fields), PKCE OAuth flow for Google/Apple, MFA support, GDPR data export, audit logging, privilege rotation, and per-user rate limiting. It uses opaque session tokens (hashed with SHA-256 before storage) instead of JWTs.

This is a mature, enterprise-grade auth system for a product that supports desktop, mobile, and cloud clients with social login. ViolaWake Console needs none of this complexity. Its email/password + JWT auth is exactly right for a developer-facing SaaS console.

### OSS Alternatives Evaluated

| Library | What It Does | Verdict |
|---------|-------------|---------|
| **FastAPI-Users** | Full auth lifecycle: register, login, verify, reset, OAuth, JWT + cookie strategies, SQLAlchemy/MongoDB adapters | **SKIP.** It would replace ~350 lines of auth.py and ~311 lines of routes/auth.py, but you would lose control over token scoping (download tokens, team invites), the SHA-256 pre-hash for long passwords, and the custom cascade deletion. FastAPI-Users is opinionated and adding custom token types requires subclassing internals. |
| **Authlib** | OAuth 2.0 / OIDC client and server | **SKIP for now.** Only needed when you add "Sign in with Google/GitHub." When that time comes, Authlib is the right choice — it handles PKCE, state parameters, and token exchange correctly. |
| **PyJWT vs python-jose** | JWT encoding/decoding | **KEEP python-jose.** It is already in requirements.txt and working. PyJWT would be equivalent. Neither has a security advantage over the other. python-jose's API is marginally more convenient for multi-algorithm support. |
| **passlib[bcrypt]** | Password hashing abstraction | **SKIP.** Direct bcrypt usage (as implemented) is simpler and has fewer dependencies. passlib adds value when you need to support multiple hash algorithms or migrate between them. ViolaWake only uses bcrypt. |

### Recommendation: **KEEP**

The auth system is correct, lean, and well-adapted to the product's needs. The code reads cleanly, handles edge cases (72-byte password limit, open redirect prevention, trusted proxy extraction), and the 658 lines across auth.py + routes/auth.py are easier to audit than a FastAPI-Users dependency tree. No changes needed for launch.

---

## 2. Payment / Billing

### What ViolaWake Has (READ)

| File | Lines | What It Does |
|------|-------|-------------|
| `backend/app/routes/billing.py` | 538 | Stripe Checkout Session creation, webhook handling (checkout.session.completed, subscription.updated, subscription.deleted, invoice.payment_failed), subscription CRUD, usage metering with per-period counters, Billing Portal session creation, tier-based quota enforcement |
| `backend/app/models.py` (Subscription, UsageRecord) | ~33 | Subscription model with stripe_customer_id, stripe_subscription_id, tier, status, current_period_end. UsageRecord with unique constraint on (user_id, action, period_start). |
| `frontend/src/pages/Billing.tsx` | 222 | Subscription display, usage bar, Stripe Billing Portal redirect |

### Current Implementation Quality: 5/5 stars

**Strengths:**
- Uses **Stripe Checkout** (not Elements or Payment Intents). This is the correct choice for a SaaS subscription product — Checkout is fully PCI-compliant, handles 3D Secure, and Stripe maintains the payment UI.
- **Webhook signature verification** is properly implemented using `stripe.Webhook.construct_event()`. The raw body is read before JSON parsing (required for signature verification).
- **Status mapping** from Stripe subscription statuses to internal statuses is comprehensive (covers active, past_due, canceled, unpaid, incomplete, incomplete_expired, trialing, paused).
- **Idempotent subscription creation** — `_get_or_create_subscription` ensures exactly one subscription row per user.
- **Tier downgrade on deletion** — when Stripe sends subscription.deleted, the user is correctly downgraded to free tier.
- **Billing Portal** — delegates subscription management to Stripe's hosted portal, which handles cancellation, upgrade, payment method update, and invoice history.
- **Quota enforcement** is a FastAPI dependency (`check_training_quota`) that can be injected into any route.
- **Graceful degradation** — billing features are gated behind `settings.billing_enabled`, and `_get_stripe()` uses deferred import so the app starts even without the stripe package.
- **Stripe customer creation** links the user ID in metadata for reconciliation.

**Security:**
- Webhook endpoint correctly reads raw body for signature verification.
- Stripe signature header is required (returns 400 if missing).
- Webhook secret is required (returns 503 if not configured).
- No sensitive billing data is stored locally (only Stripe IDs).

### NOVVIOLA Comparison

NOVVIOLA's `services/payments/` is a **completely different model**: an encrypted local vault (AES-256-GCM) that stores actual card numbers for use by a browser automation agent to fill payment forms on third-party sites (like Dominos). This is not SaaS billing — it is a desktop payment assistant. Zero overlap with ViolaWake Console's needs.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **stripe-python SDK** | **Already using it correctly.** The implementation follows Stripe's recommended patterns for Checkout + Webhooks + Billing Portal. |
| **dj-stripe** | Django-specific. Not applicable. |
| **Custom Stripe Elements** | **Not needed.** Checkout is the right pattern here. Elements would add frontend complexity (card input handling, PCI scope) for no benefit. |

### Recommendation: **KEEP**

This is textbook Stripe Checkout integration. It follows every best practice from the Stripe docs. The webhook handler is robust, signature verification is correct, and the Billing Portal delegation means you never have to build a payment method management UI. Zero changes needed.

---

## 3. Rate Limiting

### What ViolaWake Has (READ)

`backend/app/rate_limit.py` — 60 lines. In-memory sliding window rate limiter using a dict of timestamp lists. Called explicitly from route handlers with per-route limits (e.g., 5 login attempts per minute, 100 registrations per hour, 20 email verifications per 5 minutes).

### Current Implementation Quality: 3/5 stars

**Strengths:**
- Simple, correct sliding window algorithm.
- Returns standard rate-limit headers (X-RateLimit-Remaining, X-RateLimit-Reset).
- Per-route configuration allows fine-grained limits.
- Thread-safe enough for a single-process deployment (Python GIL).

**Weaknesses:**
- **In-memory only.** State is lost on restart and cannot be shared across multiple server processes. Acceptable for a single-process Railway deployment, but breaks with horizontal scaling.
- **No cleanup of old entries.** The `_rate_store` dict grows unboundedly. At ViolaWake's traffic levels this is not a problem (rate limit keys are IP-based so the dict maxes out at the number of unique IPs seen), but it is technically a memory leak.
- **Explicit call-site invocation** rather than middleware — developers must remember to add `check_rate_limit()` to each route. This is also a strength (fine-grained control) but a maintenance risk.

### OSS Alternatives Evaluated

| Library | What It Does | Verdict |
|---------|-------------|---------|
| **slowapi** | FastAPI rate limiting built on `limits`. Supports in-memory, Redis, Memcached backends. Decorator-based (`@limiter.limit("5/minute")`). | **FUTURE UPGRADE.** When you need horizontal scaling or Redis-backed rate limits, swap to slowapi. It is a drop-in addition (middleware + decorators) that takes ~30 minutes to integrate. |
| **fastapi-limiter** | Redis-based rate limiting for FastAPI. | **SKIP.** Requires Redis as a hard dependency. slowapi supports both memory and Redis backends. |

### Recommendation: **KEEP for launch, REPLACE when scaling**

The 60-line implementation is correct for a single-process deployment. When you add a second process or move to Redis, swap to **slowapi** (estimated effort: 2-3 hours, including tests). The call sites do not need to change much — slowapi's decorator pattern is simpler than the current explicit calls.

---

## 4. Email

### What ViolaWake Has (READ)

`backend/app/email_service.py` — 198 lines. Custom EmailService class that sends transactional emails via **Resend** (raw HTTP POST to `api.resend.com/emails`). Supports: verification, password reset, welcome, training complete, team invite, quota warning. Inline HTML templates with a reusable `_render_email()` method. Graceful no-op when API key is not configured.

### Current Implementation Quality: 4/5 stars

**Strengths:**
- Uses `httpx.AsyncClient` correctly with timeout.
- HTML is properly escaped using `html.escape()` (prevents XSS in email content).
- Button URLs are escaped with `escape(button_url, quote=True)` — prevents attribute injection.
- Clean singleton pattern with `get_email_service()`.
- Graceful fallback when email is not configured (auto-verifies email for development).

**Weaknesses:**
- **No retry on failure.** If Resend returns a transient error, the email is lost. For critical emails (verification, password reset), this could block the user. Low risk because Resend's uptime is excellent, but a single retry would be trivial to add.
- **Inline HTML templates.** Fine for 6 email types, but would become unwieldy at 15+. Jinja2 templates would scale better.
- **Not using the resend-python SDK.** The code makes raw HTTP calls to the Resend API. The SDK would give you type safety, automatic retries, and version tracking. Minor issue — the raw calls work correctly.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **resend-python** | **Optional upgrade.** Would replace the `_send_email()` method with `resend.Emails.send()`. Adds automatic retries and error types. ~15 minutes to swap. Not blocking. |
| **fastapi-mail** | **SKIP.** Designed for SMTP, not API-based email services like Resend. Wrong abstraction. |
| **Jinja2 templates** | **FUTURE.** When the email count exceeds ~10 types, extract templates to `.html` files. Not needed at 6 types. |

### Recommendation: **KEEP**

The email service is clean, correct, and well-scoped. The optional upgrade to `resend-python` SDK is a nice-to-have, not a blocker.

---

## 5. Database & ORM

### What ViolaWake Has (READ)

| File | Lines | What It Does |
|------|-------|-------------|
| `backend/app/database.py` | 73 | SQLAlchemy 2.0 async engine + session factory, auto-migration for simple schema changes (ALTER TABLE ADD COLUMN), dual backend support (SQLite for dev, PostgreSQL for production via `VIOLAWAKE_DB_URL`), `init_db()` with `create_all` |
| `backend/app/models.py` | 146 | 7 models using SQLAlchemy 2.0 `Mapped` annotations: User, Team, TeamMember, Recording, TrainingJob, TrainedModel, Subscription, UsageRecord |
| `backend/app/config.py` | 160 | Pydantic-settings with full env var configuration |

### Current Implementation Quality: 4/5 stars

**Strengths:**
- Uses SQLAlchemy 2.0 mapped columns with proper type annotations.
- Async session management with automatic commit/rollback in `get_db()`.
- Correct foreign key constraints with `ondelete="CASCADE"` where appropriate.
- Unique constraints on (team_id, user_id) and (user_id, action, period_start).
- `_ensure_schema_updates()` provides a lightweight migration path for simple column additions without requiring Alembic.
- Dual backend (SQLite + PostgreSQL) with a single configuration toggle.

**Weaknesses:**
- **No Alembic.** Schema migrations are done with raw `ALTER TABLE` in `_ensure_schema_updates()`. This works for adding nullable columns but will not handle column renames, type changes, or index additions. When the schema starts evolving regularly, Alembic becomes necessary.
- **No indexes on some foreign keys.** Most FKs have `index=True`, which is good. The `TrainingJob.model_id` FK lacks an index, but it is rarely queried by model_id so this is not a real issue.

### NOVVIOLA Comparison

NOVVIOLA uses SQLAlchemy for its auth database but with a very different pattern — it has a custom SQLite repository layer (`auth/database.py`) with explicit migration management. NOVVIOLA also has a separate PostgreSQL database for cloud deployment. Nothing is directly reusable.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **SQLAlchemy async** | **Already using it.** Correctly configured with `async_sessionmaker` and `create_async_engine`. |
| **Alembic** | **ADD WHEN NEEDED.** `alembic init` + `alembic revision --autogenerate` takes ~30 minutes to set up. Do it when the next schema change cannot be expressed as `ALTER TABLE ADD COLUMN`. |
| **SQLModel** | **SKIP.** SQLModel (Pydantic + SQLAlchemy) would merge models.py and schemas.py, but it has known issues with relationship handling in async contexts and is less mature than SQLAlchemy 2.0's native Mapped types. |

### Recommendation: **KEEP, add Alembic when schema complexity grows**

The database layer is solid. The main gap (no Alembic) is a known tradeoff for launch speed. Estimated effort to add Alembic: 1-2 hours.

---

## 6. Job Queue / Background Tasks

### What ViolaWake Has (READ)

`backend/app/job_queue.py` — ~500+ lines. Custom persistent job queue using **aiosqlite** with:
- Priority queuing (free < developer < business)
- Circuit breaker per user (pause after 3 consecutive failures, 5-minute backoff)
- Job lifecycle: pending -> running -> completed/failed/cancelled
- SSE-based progress streaming
- Cancellation support (for account deletion)
- Persistent state across restarts (SQLite DB)
- Concurrency control (configurable max_concurrent_jobs)

### Current Implementation Quality: 5/5 stars

**Strengths:**
- **Persistent queue.** Jobs survive server restarts. This is critical for training jobs that take minutes.
- **Circuit breaker.** Prevents a single user's broken recordings from consuming all worker capacity.
- **Priority.** Paid users get priority without starving free users.
- **Cancellation.** Clean cancellation with timeout for account deletion.
- **SSE streaming.** Real-time progress to the frontend without polling.
- **Concurrency control.** Prevents the server from being overwhelmed by concurrent training jobs.

**This is the strongest subsystem in the codebase.** It handles exactly the domain-specific needs (ML training jobs with progress, cancellation, and priority) that no generic job queue library would handle out of the box.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **Celery + Redis** | **OVERKILL.** Adds Redis as a dependency, requires a separate worker process, and you would still need to build the priority, circuit breaker, SSE streaming, and cancellation logic on top. |
| **arq** | **OVERKILL.** Requires Redis. Same problem as Celery. |
| **dramatiq** | **OVERKILL.** Requires RabbitMQ or Redis. |
| **Huey** | Supports SQLite backend, but lacks async support, priority queuing, and circuit breaker. You would need to rebuild half the current features. |

### Recommendation: **KEEP**

This is the right architecture. The custom queue is purpose-built for ML training job management and would be harder to replicate with any generic queue library. The aiosqlite backend is appropriate for the traffic level and deployment model (single server on Railway).

---

## 7. Storage

### What ViolaWake Has (READ)

`backend/app/storage.py` — 282 lines. Protocol-based storage abstraction with two backends:
- **LocalStorageBackend**: Filesystem storage for development.
- **R2StorageBackend**: Cloudflare R2 via boto3 S3-compatible API with presigned URLs.

Features: upload, download, delete, exists, presigned URLs. Storage key normalization with path traversal protection. Legacy path migration for backward compatibility.

### Current Implementation Quality: 5/5 stars

**Strengths:**
- **Protocol-based abstraction.** Adding a new backend (S3, GCS) requires implementing 5 methods.
- **Path traversal prevention.** `_validate_legacy_path()` ensures paths are within managed storage roots. `_normalize_storage_key()` rejects `..` and empty segments.
- **Presigned URLs.** R2 backend generates time-limited download URLs. Local backend returns internal file routes.
- **Automatic backend selection.** Picks R2 when configured, falls back to local.
- **Companion config handling.** `build_companion_config_identifier()` manages the `.config.json` sidecar for trained models.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **boto3** | **Already using it** for R2. Correct configuration with S3v4 signatures and auto region. |
| **fsspec / universal-pathlib** | **SKIP.** Would add an abstraction layer over an abstraction layer. The Protocol-based approach is cleaner for 2 backends. |

### Recommendation: **KEEP**

Clean, secure, well-abstracted. No changes needed.

---

## 8. Frontend Auth & State

### What ViolaWake Has (READ)

| File | Lines | What It Does |
|------|-------|-------------|
| `frontend/src/contexts/AuthContext.tsx` | 152 | React context with login/register/logout, auto-validate token on mount, error handling |
| `frontend/src/api.ts` | 414 | Typed API client using native `fetch`, JWT management in localStorage, client-side expiry check, 401 auto-redirect |
| `frontend/src/components/ProtectedRoute.tsx` | 27 | Auth guard with loading spinner |
| `frontend/src/hooks/useAuth.ts` | Re-exports from AuthContext |

### Current Implementation Quality: 4/5 stars

**Strengths:**
- **Zero external dependencies.** The frontend uses only `react`, `react-dom`, and `react-router-dom`. No axios, no react-query, no auth library. This is intentional minimalism.
- **Client-side JWT decode without a library.** The `decodeJwtPayload()` function is 10 lines and handles base64url decoding correctly.
- **30-second expiry buffer.** Prevents the race condition where a token expires between the client check and the server processing the request.
- **Open redirect prevention.** `getPostAuthRedirect()` rejects paths that don't start with `/` or start with `//`.
- **Type-safe API client.** Every endpoint function has typed request and response generics.

**Weaknesses:**
- **No automatic token refresh.** When the JWT expires, the user is redirected to login. For a developer console with 24-hour tokens this is acceptable, but not ideal for long sessions.
- **`alert()` for session expiry.** The `handleSessionExpiry()` function uses `window.alert()` which is a poor UX. Replace with the existing toast notification system.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **@auth0/auth0-react** | **SKIP.** Only useful if using Auth0 as identity provider. ViolaWake has its own auth backend. |
| **next-auth** | **SKIP.** Requires Next.js. ViolaWake uses Vite + React. |
| **TanStack Query (react-query)** | **FUTURE UPGRADE.** Would simplify data fetching, caching, and refetching for the dashboard and billing pages. Adds ~13KB gzipped. Worth adding when the number of data-fetching pages exceeds 5-6. Currently at exactly that threshold. |
| **axios** | **SKIP.** The native `fetch` wrapper in api.ts is cleaner than axios for this use case. No benefit from switching. |

### Recommendation: **KEEP**

The zero-dependency approach is correct for launch. The only actionable improvement is replacing `alert()` with the toast system for session expiry, which is a 2-line change. TanStack Query is a good post-launch addition when data fetching patterns become more complex.

---

## 9. Frontend UI Components

### What ViolaWake Has (READ)

| File | What It Is |
|------|-----------|
| `styles/global.css` | ~800 lines of custom CSS with CSS variables, dark theme, responsive layout, form styles, card styles, animations |
| `components/Layout.tsx` | Navbar with auth-aware navigation, email verification banner |
| `components/Toast.tsx` | Toast notification system |
| `components/ErrorBoundary.tsx` | React error boundary |
| `components/AudioRecorder.tsx` | Web Audio API recorder for wake word samples |
| `components/RecordingSession.tsx` | Multi-recording session management |
| `components/TrainingProgress.tsx` | SSE-connected training progress display |
| `components/ModelCard.tsx` | Model display card |

### Current Implementation Quality: 4/5 stars

**Strengths:**
- **CSS variables for theming.** All colors, spacing, and transitions are defined as variables. Switching to a light theme or adjusting the design system takes minutes.
- **No CSS framework dependency.** The 800-line global.css is comprehensive and the bundle is small.
- **Responsive design.** Media queries for mobile layouts.
- **Dark theme.** Appropriate for a developer tool.
- **Custom components match the product's needs.** AudioRecorder, RecordingSession, and TrainingProgress are domain-specific and cannot be replaced by any UI library.

**Weaknesses:**
- **Global CSS.** All styles are in one file. As the app grows, this will cause naming collisions. CSS Modules or a utility-first approach would help.
- **No form library.** Login, Register, ForgotPassword, and ResetPassword each have manual validation logic (validate function, touched state, error display). This is ~30 lines of boilerplate per form.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **Tailwind CSS** | **SKIP for now.** Would require reworking all 800 lines of CSS. The current CSS variables approach is functionally equivalent for a product of this size. Tailwind's value increases with team size (consistency) and component count (>50). |
| **shadcn/ui + Radix UI** | **SKIP.** Beautiful components, but would require adopting Tailwind as a dependency and reworking the design system. Not worth the churn for launch. |
| **React Hook Form** | **CONSIDER.** Would eliminate ~120 lines of repetitive validation boilerplate across 4 auth forms. Adds ~8KB gzipped. Reasonable post-launch improvement. |
| **Headless UI** | **SKIP.** Only needed for complex accessible components (combobox, listbox, dialog). The current components are simple enough. |

### Recommendation: **KEEP, consider React Hook Form post-launch**

The custom CSS and components are well-crafted and appropriate for the product's scope. The only worthwhile OSS addition is React Hook Form to reduce form validation boilerplate, but it is not blocking.

---

## 10. Cookie Consent / GDPR

### What ViolaWake Has (READ)

`frontend/src/components/CookieConsent.tsx` — 61 lines. Fixed-position banner at the bottom of the page. Stores consent in localStorage. Shows once per browser. Links to privacy policy. Accept button dismisses.

### Current Implementation Quality: 3/5 stars

**Strengths:**
- Simple, correct, non-blocking.
- Links to privacy policy.
- Persists consent in localStorage.

**Weaknesses:**
- **Accept-only.** GDPR requires the ability to reject non-essential cookies, or at minimum, separate consent for analytics vs. essential cookies. Since ViolaWake currently only uses essential cookies (auth, payment), this is technically compliant — but adding analytics later would require rework.
- **No cookie categorization.** No distinction between essential and non-essential.
- **No "manage preferences" option.** EU regulators increasingly expect granular consent.

### OSS Alternatives Evaluated

| Library | Verdict |
|---------|---------|
| **react-cookie-consent** | **FUTURE UPGRADE.** 280K weekly downloads, actively maintained. Provides accept/reject/manage, category-based consent, callback hooks. Would replace the 61-line component with a more compliant one. Adds ~5KB. |
| **vanilla-cookieconsent** | **SKIP.** Framework-agnostic but requires more integration work in React. |
| **Osano / OneTrust** | **SKIP.** Enterprise-grade hosted solutions. Overkill for a developer tool. |

### Recommendation: **KEEP for launch, upgrade to react-cookie-consent when adding analytics**

The current implementation is legally sufficient because ViolaWake only uses essential cookies (auth JWT). When you add analytics (Mixpanel, PostHog, GA), upgrade to react-cookie-consent for proper category-based consent. Estimated effort: 1-2 hours.

---

## 11. Middleware & Observability

### What ViolaWake Has (READ)

| File | Lines | What It Does |
|------|-------|-------------|
| `backend/app/middleware.py` | 273 | Request ID, structured JSON logging, security headers, Sentry integration with PII scrubbing, error classification, exception handlers |
| `backend/app/monitoring.py` | 207 | Error tracking (bounded deque), error classification (expected vs unexpected), version detection from pyproject.toml, health check utilities |

### Current Implementation Quality: 5/5 stars

This is production-grade observability. Specific details:

- **Sentry `before_send` hook** scrubs sensitive fields (api_key, authorization, password, secret, token) before sending to Sentry.
- **`send_default_pii=False`** prevents Sentry from collecting IP addresses and user data.
- **Error classification** distinguishes expected errors (rate limit, timeout, user input) from unexpected errors (bugs, config issues) with appropriate log levels.
- **Request logging** excludes health check endpoints to reduce noise.
- **Security headers** cover the OWASP basics.

### Recommendation: **KEEP**

Nothing to change. This is well-implemented.

---

## Summary Decision Matrix

| # | Subsystem | Quality | Recommendation | Action Needed | Effort |
|---|-----------|---------|---------------|---------------|--------|
| 1 | Auth & Sessions | 4/5 | **KEEP** | Replace `alert()` with toast for session expiry | 10 min |
| 2 | Payment / Billing | 5/5 | **KEEP** | None | - |
| 3 | Rate Limiting | 3/5 | **KEEP now, REPLACE later** | Swap to slowapi when adding Redis | 2-3 hrs |
| 4 | Email | 4/5 | **KEEP** | Optional: swap raw HTTP for resend-python SDK | 15 min |
| 5 | Database & ORM | 4/5 | **KEEP** | Add Alembic when schema changes become complex | 1-2 hrs |
| 6 | Job Queue | 5/5 | **KEEP** | None | - |
| 7 | Storage | 5/5 | **KEEP** | None | - |
| 8 | Frontend Auth | 4/5 | **KEEP** | Consider TanStack Query post-launch | 4-6 hrs |
| 9 | Frontend UI | 4/5 | **KEEP** | Consider React Hook Form for auth forms | 2-3 hrs |
| 10 | Cookie Consent | 3/5 | **KEEP now, REPLACE later** | Upgrade to react-cookie-consent when adding analytics | 1-2 hrs |
| 11 | Middleware | 5/5 | **KEEP** | None | - |

### Items borrowed from NOVVIOLA: **NONE**

NOVVIOLA is a desktop voice assistant with 35-file auth system, local payment vault, browser automation, multi-room audio, and MCP tool orchestration. ViolaWake Console is a straightforward SaaS web app. The architectural distance between them is too great for code sharing to be productive. NOVVIOLA's patterns are instructive as design references, but the actual code would import complexity without benefit.

### Bottom Line

Ship it. The codebase is launch-ready. The subsystems that scored 3/5 (rate limiting, cookie consent) are correct for current scale and have clear upgrade paths when scale demands it. The subsystems that scored 5/5 (billing, job queue, storage, middleware) are genuinely impressive and should not be touched.
