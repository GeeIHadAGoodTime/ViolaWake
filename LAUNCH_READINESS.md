# ViolaWake Console -- Launch Readiness Audit

**Auditor:** Claude Opus 4.6
**Date:** 2026-03-28
**Scope:** Full production chain audit for paid product launch

---

## Executive Summary

**7 out of 12 areas are launch-ready (YES or PARTIAL).** 5 areas have blockers.

The codebase is remarkably complete for a pre-launch product. Auth, billing, training pipeline, storage, and the frontend UX are all real implementations -- not stubs. The main blockers are operational (nginx missing API proxy, no OG image, no cookie consent, rate limiting is in-memory only) rather than architectural. Most blockers are fixable in a single focused day.

| # | Area | Verdict | Key Finding |
|---|------|---------|-------------|
| 1 | Infrastructure & Hosting | **PARTIAL** | Docker works, PostgreSQL supported, but nginx.conf missing API proxy |
| 2 | Domain & SSL | **PARTIAL** | CORS ready, but needs nginx API proxy and domain DNS |
| 3 | Auth & Security | **YES** | bcrypt, JWT, rate limiting, email verification, account deletion all implemented |
| 4 | Email | **YES** | Resend integration complete, all 6 email types implemented |
| 5 | Stripe Billing | **YES** | Full Stripe Checkout + webhooks + quota enforcement + billing portal |
| 6 | Training Pipeline | **YES** | Persistent job queue, progress streaming, timeout, cancellation, error reporting |
| 7 | Storage | **PARTIAL** | Local + R2 both implemented, but R2 is optional and untested path for launch |
| 8 | Frontend Polish | **YES** | Professional landing page, recording UX, real pricing, loading/error/empty states |
| 9 | Legal | **PARTIAL** | Real Privacy Policy + Terms, but no cookie consent banner |
| 10 | Monitoring & Error Handling | **PARTIAL** | Sentry ready, structured JSON logs, error tracker, but no alerting configured |
| 11 | SEO & Marketing | **NO** | Missing OG image, robots.txt, sitemap |
| 12 | Table Stakes | **PARTIAL** | Account deletion exists, password reset exists, but no password change (while logged in) |

---

## 1. Infrastructure & Hosting -- PARTIAL

**Evidence reviewed:** `docker-compose.yml`, `Dockerfile.backend`, `Dockerfile.frontend`, `backend/app/database.py`, `backend/run.py`, `frontend/nginx.conf`

### What works
- **Docker Compose** correctly orchestrates backend + frontend with health check dependency.
- **Dockerfile.backend** installs ViolaWake SDK with `[training]` extras, system deps (portaudio), and runs `run.py`.
- **Dockerfile.frontend** uses multi-stage build (node build -> nginx:alpine). Clean and correct.
- **PostgreSQL support** is real. `database.py` checks `VIOLAWAKE_DB_URL` and falls back to SQLite. Comment on line 14-16 explicitly says "Railway deployments should set VIOLAWAKE_DB_URL to postgresql+asyncpg://...".
- **Alembic migrations** exist with proper env.py. Two migration files present.
- **Health checks** are thorough: `/api/health`, `/api/health/live`, `/api/health/ready`, `/api/health/details`. Checks database, training queue, storage directories, and billing config. This is production-grade.
- **asyncpg** is in requirements.txt for PostgreSQL async driver.

### BLOCKER: nginx.conf missing API proxy
`frontend/nginx.conf` serves static files and does SPA fallback to `index.html`, but has **no `location /api` proxy block**. In production, the frontend will not be able to reach the backend API.

**Fix:** Add to `nginx.conf`:
```nginx
location /api/ {
    proxy_pass http://backend:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

### Railway deployment
Can deploy to Railway with:
- Backend: Dockerfile.backend, set `VIOLAWAKE_DB_URL` to Railway PostgreSQL addon
- Frontend: Dockerfile.frontend, needs the nginx fix above
- Alternatively, Railway can run both as separate services with networking

**Estimate:** 15 minutes to fix nginx.conf.

---

## 2. Domain & SSL -- PARTIAL

**Evidence reviewed:** `config.py` (cors_origins, effective_cors_origins, console_base_url)

### What works
- **CORS** is production-aware. `effective_cors_origins` automatically appends `https://console.violawake.com` and `https://violawake.com` when `VIOLAWAKE_ENV=production`.
- **CORS_ORIGINS** env var accepts comma-separated origins and handles parsing edge cases.
- **console_base_url** configurable for Stripe redirect URLs.
- **`trusted_proxy_count`** setting exists for correct IP extraction behind reverse proxies (Railway, Cloudflare).

### What needs setup
- Buy domain (violawake.com or similar)
- Point DNS to hosting provider
- SSL is provided free by Railway/Render/Fly.io, or by Cloudflare if using it as CDN
- Set `VIOLAWAKE_CORS_ORIGINS` and `VIOLAWAKE_CONSOLE_BASE_URL` to production domain

**Estimate:** 30 minutes DNS setup + propagation time.

---

## 3. Auth & Security -- YES

**Evidence reviewed:** `auth.py` (full), `routes/auth.py` (full), `rate_limit.py`, `middleware.py`, `schemas.py`, `config.py`

### Password hashing
- **bcrypt** with `gensalt()` (default 12 rounds). Correct.
- **Long password handling**: SHA-256 pre-hash for passwords > 72 bytes to avoid bcrypt truncation. This is the correct approach.
- Password validation: 8-128 character range enforced by Pydantic schema.

### JWT
- **Secret key**: Auto-generated via `secrets.token_urlsafe(32)` in dev. **Crashes on startup** if not set in production (`VIOLAWAKE_ENV=production` without `VIOLAWAKE_SECRET_KEY` raises `ValueError`). This is the correct behavior -- it forces you to set a real key.
- **Algorithm**: HS256 (standard).
- **Token lifetime**: 24 hours (configurable).
- **Purpose-scoped tokens**: Email verification (48h), password reset (2h), download (60s single-use) all use separate `purpose` claims. One-time download tokens track JTIs in memory with expiry pruning.

### Rate limiting
- **Login**: 5 attempts/minute per IP. Correct for brute force prevention.
- **Registration**: 100/hour per IP.
- **Email verification**: 20/5 minutes per IP.
- **Password reset**: 5/5 minutes per IP (forgot) and 10/5 minutes per IP (reset).
- **Recording upload**: 50/hour per user.
- Rate limit headers (`X-RateLimit-Remaining`, `X-RateLimit-Reset`) are set on responses.

### SHOULD-FIX: In-memory rate limiting
Rate limits are stored in a Python dict (`_rate_store`). This resets on restart and does not work across multiple workers. For a single-worker Railway deployment this is acceptable at launch. For multi-worker, would need Redis.

### Security headers
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- Request IDs on every response via `X-Request-ID`.

### Auth tokens
Bearer tokens in `Authorization` header (not cookies). Token stored in `localStorage` on the frontend. Client-side JWT expiry check with 30-second buffer prevents stale requests.

### CSRF
Not needed for this architecture. CSRF is a cookie-based session attack. This uses `Authorization: Bearer` headers, which are not auto-attached by the browser.

### Account lockout
No explicit account lockout after N failed attempts (only IP-based rate limiting). The 5/minute IP rate limit serves this purpose for launch.

### Email verification
Required for recording, training, and billing features via `get_verified_user` dependency. Unverified users can log in and see the dashboard but cannot do anything meaningful.

### Account deletion
Full implementation at `DELETE /api/auth/account`: deletes recordings from storage, models from storage, training jobs, usage records, subscriptions, and the user row. GDPR-adequate.

### Password reset
Full flow: forgot-password -> email with token -> reset-password endpoint. Proper rate limiting on both endpoints.

---

## 4. Email -- YES

**Evidence reviewed:** `email_service.py` (full), `routes/auth.py`

### Implementation
- **Provider**: Resend (via HTTP API). Clean async implementation with `httpx`.
- **Graceful degradation**: When `VIOLAWAKE_RESEND_API_KEY` is not set, emails are skipped and users are auto-verified. Logs a warning once. This is correct for development.
- **FROM address**: `ViolaWake <noreply@violawake.com>` -- requires domain verification in Resend.

### Emails implemented
1. **Verification email** -- on registration
2. **Welcome email** -- on email verification
3. **Password reset** -- on forgot-password request
4. **Training complete** -- with model download CTA
5. **Team invite** -- with join link
6. **Quota warning** -- near tier limit

### Email templates
Inline HTML with consistent branding (ViolaWake blue, clean layout, CTA button). Not fancy but professional enough for launch.

### Resend costs
- **Free tier**: 3,000 emails/month, 100/day limit.
- For launch this is more than enough. A $20/mo plan gives 50,000 emails/month if you scale.
- Domain verification: Need to add DNS records (DKIM, SPF) for `violawake.com` in Resend dashboard. Free.

**Estimate:** 15 minutes to set up Resend + verify domain.

---

## 5. Stripe Billing -- YES

**Evidence reviewed:** `routes/billing.py` (full), `config.py`, `models.py`, `frontend/src/pages/Billing.tsx`, `frontend/src/pages/Pricing.tsx`

### What needs to be created in Stripe Dashboard
1. **Two Products**: "ViolaWake Developer" ($29/mo) and "ViolaWake Business" ($99/mo)
2. **Two Prices**: Monthly recurring prices for each product
3. **Webhook endpoint**: Point to `https://yourdomain.com/api/billing/webhook`
4. **Webhook events to subscribe**: `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_failed`

### What works
- **Checkout flow**: Creates Stripe Checkout Session with proper metadata (user_id, tier). Redirects to `{console_base_url}/billing?session_id={CHECKOUT_SESSION_ID}` on success, `/pricing` on cancel.
- **Webhook handler**: Signature verification via `stripe.Webhook.construct_event`. Handles all 4 critical events.
- **Subscription management**: `checkout.session.completed` creates/updates subscription row. `subscription.updated` handles renewals and tier changes. `subscription.deleted` downgrades to free. `payment_failed` marks as `past_due`.
- **Billing portal**: Creates Stripe Billing Portal session for self-service subscription management (cancel, update payment method, view invoices).
- **Quota enforcement**: `check_training_quota` is a FastAPI dependency on the training route. Checks current usage against tier limit. Returns 403 with upgrade URL when limit is hit.
- **Usage tracking**: `record_usage` creates/increments `UsageRecord` per user per billing period (monthly).
- **Feature gating**: `billing_enabled` property checks for `stripe_secret_key`. When billing is disabled, 503 is returned from billing endpoints. Free tier still works.

### Tier limits
| Tier | Models/month | Price |
|------|-------------|-------|
| Free | 3 | $0 |
| Developer | 20 | $29/mo |
| Business | Unlimited | $99/mo |
| Enterprise | Unlimited (custom) | Custom |

### Frontend billing page
- Shows current tier, status, period end date
- Usage bar with models used / models limit
- "Manage Subscription" button opens Stripe Billing Portal
- "Subscription activated!" banner on return from Stripe Checkout
- Loading, error, and retry states all implemented

### Frontend pricing page
- 4 pricing cards (Free, Developer, Business, Enterprise)
- Checkout flow for paid tiers (redirects to Stripe)
- Enterprise card links to `mailto:enterprise@violawake.com`
- 7 FAQ items with accordion
- Bottom CTA for free tier

**Estimate:** 30 minutes to create Stripe products, prices, and webhook.

---

## 6. Training Pipeline -- YES

**Evidence reviewed:** `services/training_service.py` (full), `job_queue.py` (full), `routes/training.py`, `routes/jobs.py`

### Architecture
- **Persistent job queue** backed by aiosqlite (separate from main DB). Jobs survive restarts.
- **Worker pool**: Configurable `max_concurrent_jobs` (default: 2). Training runs in `asyncio.to_thread()` to avoid blocking the event loop.
- **Circuit breaker**: 3 consecutive failures pause the queue for 5 minutes. Prevents cascading failures.

### Training process
1. Recordings downloaded from storage to temp directory
2. Minimum 5 valid WAV files required
3. Calls `violawake_sdk.tools.train._train_mlp_on_oww` with progress callback
4. Progress reported per-epoch via callback -> stored in queue -> pushed via SSE to frontend
5. Training artifact (ONNX model + config JSON) stored in storage backend
6. d-prime metric extracted from config and stored on model row

### Error handling
- **Timeout**: 30-minute watchdog (configurable). Raises `RuntimeError` on timeout.
- **Cancellation**: `TrainingCancelledError` checked at multiple points during training. User can cancel via API.
- **Missing recordings**: Warns and skips missing files. Fails if < 5 valid files.
- **Training failure**: Error stored on job row, reported to user via SSE and status endpoint.
- **Cleanup**: Temp directory is always cleaned up in `finally` block.

### SSE streaming
- `EventSourceResponse` via `sse-starlette`
- Download token authentication for browser SSE (can't set headers on EventSource)
- 30-second heartbeat pings to keep connection alive
- Terminal events (completed/failed/cancelled) end the stream

### Resource requirements
Training uses CPU-only by default. A single training job on a 2-vCPU Railway instance should complete in 2-5 minutes. Memory usage depends on recording count but typically < 1GB.

---

## 7. Storage -- PARTIAL

**Evidence reviewed:** `storage.py` (full), `retention.py` (full), `config.py`

### Local storage
- Recordings stored at `data/recordings/{user_id}/{wake_word}/{filename}`
- Models stored at `data/models/{user_id}/{filename}`
- Path traversal prevention: validates that resolved paths are inside managed storage roots.
- Companion config files (`.config.json`) stored alongside models.

### R2 storage
- Full Cloudflare R2 implementation via boto3 S3 API.
- Presigned URLs for downloads.
- Configured via `VIOLAWAKE_R2_ENDPOINT`, `VIOLAWAKE_R2_ACCESS_KEY_ID`, `VIOLAWAKE_R2_SECRET_ACCESS_KEY`, `VIOLAWAKE_R2_BUCKET`.

### R2 costs
- **Free tier**: 10 GB storage, 10 million Class B (read) operations/month, 1 million Class A (write) operations/month.
- For launch, R2 free tier is more than enough. Models are ~102KB each. Even 1,000 models = ~100MB.

### Retention
- Recordings: auto-deleted after 90 days (configurable, 0 = disabled)
- Models: auto-deleted after 365 days (configurable, 0 = disabled)
- Retention loop runs every 24 hours in background task
- Active training job recordings are protected from deletion
- Admin endpoint `POST /api/admin/cleanup` for manual trigger

### SHOULD-FIX: Volume persistence
Docker Compose uses a named volume `backend-data`. On Railway, filesystem is ephemeral unless you attach a persistent volume or use PostgreSQL + R2. For launch, either:
- Use R2 for storage (recommended) -- 15 minutes to set up
- Or attach a Railway persistent volume to `/app/data`

---

## 8. Frontend Polish -- YES

**Evidence reviewed:** All 17 pages/components in `frontend/src/`, `global.css` (1500+ lines), `index.html`

### Landing page
- Hero section with code sample showing SDK usage
- Comparison table (ViolaWake vs Picovoice) -- effective competitive positioning
- "How it works" 3-step section (Record, Train, Deploy)
- Social proof stats (d-prime, latency, samples, model size)
- Pricing preview with 3 tiers
- Footer with Product/Company links, Privacy Policy, Terms of Service

### Recording UX
- Wake word input with validation (min 2 chars)
- 10-slot recording grid with progress bar
- Each recording shows duration, play button
- Review phase: can re-record any individual sample
- Upload progress with percentage
- Error recovery with "Try Again" and "Start Over" buttons

### Training status
- Real-time SSE progress with epoch counter and loss values
- Completion triggers download button + dashboard link
- Error display for failed jobs

### Dashboard
- Model grid with cards showing: wake word name, d-prime badge (color-coded: Excellent/Good/Needs work), date, file size
- Download, View Performance, View Details, Delete actions per model
- Delete confirmation dialog with focus trap and keyboard handling
- Empty state with CTA to record first wake word
- Loading spinner, error with retry

### Model performance page
- Separate page at `/model/:modelId/performance` for detailed metrics

### Auth pages
- Login, Register, Forgot Password, Reset Password, Email Verification
- All with proper form validation and error display

### Teams
- Teams list, team detail, invite members, share models
- Full CRUD with role-based access (owner, admin, member)

### Mobile responsive
CSS includes `@media` breakpoints. Navbar, pricing grid, model grid, and recording slots all have responsive layouts.

### Missing
- No dark/light mode toggle (dark only, which is fine for dev tools)
- No favicon.svg OG image (favicon exists as SVG but no social sharing image)

---

## 9. Legal -- PARTIAL

**Evidence reviewed:** `Privacy.tsx`, `Terms.tsx`

### Privacy Policy
- **Real and specific**, not boilerplate. Covers:
  - Account info collected (email, name, hashed password)
  - Voice recordings (stored during training, 90-day auto-delete)
  - Billing info (Stripe handles cards, only metadata stored)
  - SDK privacy (no telemetry, no audio sent to servers)
  - Data retention with specific timeframes
  - Third-party services (Stripe)
  - User rights (access, correct, delete)
  - Contact email: privacy@violawake.com
- **Last updated**: March 28, 2026 (current)

### Terms of Service
- **Real and specific**. Covers:
  - Service description (SDK vs Console distinction)
  - Account registration (16+ age requirement)
  - Acceptable use (no surveillance, no unauthorized recording)
  - IP ownership (users own their recordings and models)
  - Payments and billing (Stripe, refund policy)
  - Service availability (no SLA guarantee)
  - Liability limitation ($100 or 12-month fees cap)
  - Indemnification
  - Termination (by user or by ViolaWake)
  - Governing law (Delaware)
  - Contact email: legal@violawake.com
- **Last updated**: March 28, 2026 (current)

### BLOCKER: No cookie consent
EU users require cookie consent under GDPR/ePrivacy. The app uses `localStorage` for JWT tokens (not cookies), but Stripe and potentially Sentry set cookies. A simple consent banner is needed.

### SHOULD-FIX: GDPR data export
Privacy policy mentions right to access data, but there is no "export my data" endpoint. Account deletion exists (`DELETE /api/auth/account`). For launch, the deletion endpoint plus the email contact is likely sufficient. A full GDPR export can come later.

---

## 10. Monitoring & Error Handling -- PARTIAL

**Evidence reviewed:** `middleware.py`, `monitoring.py`, `config.py`

### What works
- **Structured JSON logging** via custom `JsonFormatter`. Every log entry has timestamp, level, logger name, message, request_id.
- **Sentry integration** ready. `init_sentry()` configures FastAPI integration with PII scrubbing (`send_default_pii=False`, `before_send` scrubs sensitive fields). Gated by `VIOLAWAKE_SENTRY_DSN`.
- **Error classification**: Exceptions classified as expected/unexpected with reason codes (user_input, rate_limit, timeout, config, data, bug). Each class gets appropriate log level.
- **Error tracker**: In-memory bounded deque (200 events) exposed via `/api/health/details` endpoint.
- **Request logging**: Every non-health request logged with method, path, route template, status code, duration_ms, request_id.
- **Unhandled exception middleware**: Returns clean 500 with request_id, never leaks stack traces to users.

### SHOULD-FIX: No alerting
No PagerDuty/Slack/email alerts for failed training jobs or health degradation. Sentry will catch unhandled exceptions, but training failures are handled exceptions. For launch, Sentry + periodic health check polling is sufficient.

---

## 11. SEO & Marketing -- NO

**Evidence reviewed:** `index.html`, `frontend/public/`

### What works
- `<title>ViolaWake Console</title>`
- `<meta name="description">` with relevant content
- Open Graph `og:title`, `og:description`, `og:type`
- Twitter Card meta tags
- `favicon.svg` exists

### BLOCKER: Missing OG image
No `og:image` or `twitter:image`. Social sharing will show a blank preview. For a paid product, this is table stakes.

**Fix:** Create a 1200x630 OG image showing the ViolaWake Console and add:
```html
<meta property="og:image" content="https://console.violawake.com/og-image.png" />
<meta name="twitter:image" content="https://console.violawake.com/og-image.png" />
```

### SHOULD-FIX: Missing robots.txt and sitemap
No `robots.txt` or `sitemap.xml` in `public/`. Without these, search engine indexing is uncontrolled.

**Fix:** Add `public/robots.txt`:
```
User-agent: *
Allow: /
Sitemap: https://console.violawake.com/sitemap.xml
```

Add `public/sitemap.xml` with landing, pricing, privacy, terms pages.

**Estimate:** 1 hour (20 min OG image, 10 min robots.txt, 10 min sitemap, 20 min testing).

---

## 12. Table Stakes for a Paid Product -- PARTIAL

### Account deletion -- YES
`DELETE /api/auth/account` deletes all user data: recordings, models, training jobs, usage records, subscriptions, and the user row. Storage files are cleaned up.

### Password change (while logged in) -- NO
There is a password **reset** flow (via email token), but no endpoint for changing password while logged in. Users would need to go through the forgot-password flow to change their password.

**Fix:** Add `PATCH /api/auth/password` endpoint that takes `current_password` and `new_password`. ~20 minutes.

### Usage dashboard -- YES
`/billing` page shows models used / models limit with progress bar, billing period dates, current tier and status.

### API key management -- NOT NEEDED (for now)
The product is a web console, not an API service. API keys are not needed for launch.

### Documentation/help -- PARTIAL
- SDK docs link to GitHub
- FAQ on pricing page (7 questions)
- No in-app help/docs section
- Contact emails in footer (hello@violawake.com)

For launch, the landing page + pricing FAQ + GitHub docs is sufficient.

---

## BLOCKERS -- Must Fix Before Accepting Money

| # | Blocker | Impact | Fix Time |
|---|---------|--------|----------|
| B1 | nginx.conf missing `/api` proxy | Frontend cannot reach backend in production | 15 min |
| B2 | No OG image for social sharing | Blank preview when shared on Twitter/LinkedIn/Slack | 30 min |
| B3 | No cookie consent banner | GDPR non-compliance for EU users (Stripe sets cookies) | 1 hour |

**Total blocker fix time: ~2 hours**

---

## SHOULD-FIX -- Hurts Credibility But Doesn't Block Launch

| # | Issue | Impact | Fix Time |
|---|-------|--------|----------|
| S1 | In-memory rate limiting | Resets on restart; no cross-worker support | 2 hours (Redis) |
| S2 | No password change (while logged in) | Users must use forgot-password flow | 30 min |
| S3 | No robots.txt / sitemap.xml | Search engines don't know what to index | 20 min |
| S4 | No GDPR data export endpoint | EU users can't download their data | 2 hours |
| S5 | Storage volume persistence on Railway | Filesystem data lost on redeploy without R2 or persistent volume | 15 min (R2 setup) |
| S6 | No Resend domain verification | Emails may go to spam without DKIM/SPF | 15 min |
| S7 | run.py uses reload=True outside production | Not harmful but wasteful in staging | 5 min |

**Total should-fix time: ~5 hours**

---

## NICE-TO-HAVE -- Polish Items

| # | Item | Notes |
|---|------|-------|
| N1 | Email templates with unsubscribe header | List-Unsubscribe for email deliverability |
| N2 | Webhook retry logging | Log and display failed webhook attempts |
| N3 | Training queue position indicator | Show "You are #3 in queue" |
| N4 | Model comparison view | Side-by-side d-prime comparison |
| N5 | In-app documentation/getting started guide | Help users integrate the SDK |
| N6 | Multi-worker rate limiting (Redis) | Only needed when scaling beyond 1 worker |
| N7 | Dark/light mode toggle | Currently dark-only |
| N8 | Mobile hamburger menu | Nav links wrap on small screens |

---

## Concrete $30 Launch Plan

### Services and Costs

| Service | Cost | Purpose |
|---------|------|---------|
| **Domain** (Namecheap/Porkbun) | $8-12/year | `violawake.com` or `violawake.dev` |
| **Railway.app Hobby** | $5/month | Backend + PostgreSQL addon (free 500MB) |
| **Cloudflare** (free plan) | $0 | DNS, CDN, SSL, DDoS protection |
| **Cloudflare R2** (free tier) | $0 | Object storage (10GB free) |
| **Resend** (free tier) | $0 | Transactional email (3,000/month) |
| **Stripe** | 2.9% + 30c per transaction | Payment processing |
| **Sentry** (free tier) | $0 | Error tracking (5K events/month) |
| **Total month 1** | ~$13-17 | Well under $30 |

### Setup Steps (in order)

**Step 1: Domain (10 min)**
1. Buy domain on Porkbun/Namecheap
2. Add to Cloudflare (free plan) -- change nameservers

**Step 2: Resend Email (15 min)**
1. Sign up at resend.com
2. Verify domain (add DNS records Cloudflare tells you)
3. Get API key

**Step 3: Stripe (30 min)**
1. Create Stripe account (already done if you have one)
2. Create Product "ViolaWake Developer" with $29/mo price
3. Create Product "ViolaWake Business" with $99/mo price
4. Note both Price IDs (`price_...`)
5. Create webhook endpoint pointing to `https://api.yourdomain.com/api/billing/webhook`
6. Select events: `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_failed`
7. Note webhook signing secret

**Step 4: Cloudflare R2 (15 min)**
1. Enable R2 in Cloudflare dashboard
2. Create bucket `violawake`
3. Create R2 API token (read/write for that bucket)
4. Note endpoint URL, access key, secret key

**Step 5: Railway (30 min)**
1. Create Railway project
2. Add PostgreSQL addon (free 500MB)
3. Deploy backend service from Dockerfile.backend
4. Set environment variables:
   ```
   VIOLAWAKE_ENV=production
   VIOLAWAKE_SECRET_KEY=<generated>
   VIOLAWAKE_DB_URL=<Railway PostgreSQL URL>
   VIOLAWAKE_STRIPE_SECRET_KEY=sk_live_...
   VIOLAWAKE_STRIPE_WEBHOOK_SECRET=whsec_...
   VIOLAWAKE_STRIPE_PRICE_DEVELOPER=price_...
   VIOLAWAKE_STRIPE_PRICE_BUSINESS=price_...
   VIOLAWAKE_RESEND_API_KEY=re_...
   VIOLAWAKE_CONSOLE_BASE_URL=https://console.yourdomain.com
   VIOLAWAKE_CORS_ORIGINS=https://console.yourdomain.com,https://yourdomain.com
   VIOLAWAKE_R2_ENDPOINT=https://<account-id>.r2.cloudflarestorage.com
   VIOLAWAKE_R2_ACCESS_KEY_ID=...
   VIOLAWAKE_R2_SECRET_ACCESS_KEY=...
   VIOLAWAKE_R2_BUCKET=violawake
   VIOLAWAKE_SENTRY_DSN=<optional>
   VIOLAWAKE_TRUSTED_PROXY_COUNT=1
   ```
5. Deploy frontend service from Dockerfile.frontend (after fixing nginx.conf)
6. Point Cloudflare DNS records to Railway

**Step 6: Fix Blockers (2 hours)**
1. Fix nginx.conf API proxy
2. Create OG image, add to public/ and index.html
3. Add cookie consent banner (simple React component)

**Step 7: Verify (30 min)**
1. Register account, verify email
2. Record 10 samples, start training
3. Download model
4. Test Stripe checkout (use test mode first, then switch to live)
5. Test billing portal
6. Check health endpoint
7. Share link on social media -- verify OG image appears

### Estimated Total Hours to Fix All Blockers

| Task | Hours |
|------|-------|
| Fix nginx.conf API proxy | 0.25 |
| Create OG image | 0.5 |
| Add cookie consent banner | 1.0 |
| Service setup (domain, Stripe, Resend, R2, Railway) | 2.0 |
| Deploy and verify | 1.0 |
| **Total** | **~5 hours** |

If also fixing SHOULD-FIX items, add 5 more hours for a total of ~10 hours of work.

---

## Architecture Strengths Worth Noting

Things that are notably well-done and would typically be missing in a pre-launch product:

1. **Production config validation**: Server crashes on startup if `VIOLAWAKE_SECRET_KEY` is not set in production mode. This prevents deploying with dev secrets.
2. **One-time download tokens**: Browser downloads (which cannot set Authorization headers) use short-lived, single-use JTI-tracked tokens. This is a non-trivial auth problem solved correctly.
3. **Circuit breaker on training queue**: Prevents cascade failure if training consistently fails.
4. **Retention cleanup**: Background task with active-job protection. Recordings tied to running jobs are never deleted.
5. **Comprehensive health checks**: Database, queue, storage, billing -- all checked with degraded/error status levels.
6. **Error classification**: Not just logging exceptions but classifying them into expected/unexpected with appropriate log levels. Health endpoint exposes error distribution.
7. **WAV validation and normalization**: Uploads are validated at the byte level (RIFF header, fmt chunk, PCM/float format), then converted to mono 16kHz if needed. No reliance on user having the right format.
8. **Account deletion**: Full cascade delete including storage files -- not just database rows.
