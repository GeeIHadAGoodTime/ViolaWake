# ViolaWake Pre-Launch Checklist

Last reviewed: 2026-03-26

This checklist is based on the current repository state:

- `pyproject.toml` is already set to version `0.1.0`.
- `src/violawake_sdk/models.py` still contains placeholder SHA-256 values.
- `.github/workflows/release.yml` expects `RELEASE_NOTES.md`, `tools/fetch_release_models.py`, and `tools/update_model_registry.py` — all now present.
- The backend health endpoint is `GET /api/health`.
- The backend initializes schema with `Base.metadata.create_all()` on startup; there is no Alembic migration workflow yet.
- The frontend reads `VITE_API_URL` at build time and already has routes for `/`, `/pricing`, `/privacy`, `/terms`, `/login`, `/register`, `/dashboard`, and `/billing`.
- `LICENSE` and `SECURITY.md` already exist at repo root.
- The backend currently writes recordings and trained models to local filesystem paths under `console/backend/data/`; Railway containers are ephemeral, so durable storage must be addressed before public launch.

## Critical Gaps To Resolve Before Launch

- [ ] Add a durable storage plan for user recordings and trained models
  How: the current backend uses `VIOLAWAKE_UPLOAD_DIR` and `VIOLAWAKE_MODELS_DIR` on local disk. On Railway, either mount a persistent volume or move artifacts to durable object storage before taking real users.

- [x] Make the tag-based release workflow executable end to end
  Done: `RELEASE_NOTES.md`, `tools/fetch_release_models.py`, and `tools/update_model_registry.py` created.

- [x] Verify Railway build context matches `console/Dockerfile.backend`
  Done: moved `railway.toml` to repo root with `dockerfilePath = "console/Dockerfile.backend"` so Docker build context is the repo root, matching the Dockerfile COPY paths.

## SDK Release

- [ ] Bump version in `pyproject.toml` to `0.1.0`
  How: `pyproject.toml` already says `0.1.0`; confirm any mirrored version strings and the git tag are aligned, then commit the release-ready version set.

- [ ] Replace SHA-256 placeholders in `models.py` with real hashes
  How: compute hashes for every shipped model asset with `Get-FileHash <file> -Algorithm SHA256` and replace the placeholder values in `src/violawake_sdk/models.py` before publishing.

- [ ] Upload `.onnx` model files to GitHub Release `v0.1.0`
  How: attach all release assets needed by the registry URLs, including `viola_mlp_oww.onnx`, `oww_backbone.onnx`, `viola_cnn_v4.onnx`, `kokoro-v1.0.onnx`, and `voices-v1.0.bin`; either automate this in the current release workflow or upload them manually to the GitHub Release.

- [ ] Run full test suite locally (`pytest tests/unit/ -v`)
  How: from repo root, install dev dependencies, run `pytest tests/unit/ -v`, and do not tag until failures and coverage regressions are resolved.

- [ ] Create `RELEASE_NOTES.md` for `v0.1.0`
  How: add a root-level `RELEASE_NOTES.md` because `.github/workflows/release.yml` uses it as `body_path`; summarize highlights, breaking changes, install steps, and known limitations.

- [ ] Push tag `v0.1.0` to trigger release workflow
  How: after merging the final release commit, create and push the annotated tag with `git tag -a v0.1.0 -m "ViolaWake v0.1.0"` and `git push origin v0.1.0`; confirm PyPI trusted publishing and any required GitHub secrets such as `MODEL_STORE_TOKEN` are configured first.

- [ ] Verify PyPI publish succeeded
  How: watch the `Release` workflow and confirm the `Publish to PyPI` job completes successfully for tag `v0.1.0`, then verify the package page exists on PyPI.

- [ ] Test `pip install violawake` from PyPI
  How: create a clean virtualenv, run `pip install violawake`, import the package, and smoke-test a documented entry point such as `violawake-download --model viola_mlp_oww`.

## Console Backend (Railway)

- [ ] Create Railway project
  How: create a new service for the backend, connect this GitHub repo. `railway.toml` is at repo root with `dockerfilePath = "console/Dockerfile.backend"`, so Railway's build context is the repo root.

- [ ] Provision PostgreSQL addon
  How: add Railway PostgreSQL and use its connection string as `VIOLAWAKE_DB_URL`; do not rely on the default SQLite path in production.

- [ ] Set all `VIOLAWAKE_*` env vars
  How: add every variable below in Railway. Use explicit production values instead of development defaults.

  | Variable | Production value / note |
  | --- | --- |
  | `VIOLAWAKE_ENV` | `production` |
  | `VIOLAWAKE_PORT` | `8000` |
  | `VIOLAWAKE_SECRET_KEY` | Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
  | `VIOLAWAKE_ALGORITHM` | Usually `HS256` |
  | `VIOLAWAKE_ACCESS_TOKEN_EXPIRE_HOURS` | Keep `24` unless product policy changes |
  | `VIOLAWAKE_CORS_ORIGINS` | Include the frontend origin, for example `https://violawake.com` |
  | `VIOLAWAKE_DB_URL` | Railway PostgreSQL async URL, for example `postgresql+asyncpg://...` |
  | `VIOLAWAKE_DATA_DIR` | Optional, only if you mount persistent storage |
  | `VIOLAWAKE_DB_PATH` | Optional if `VIOLAWAKE_DB_URL` is set; otherwise point at a persistent volume |
  | `VIOLAWAKE_UPLOAD_DIR` | Required if you keep filesystem-backed recordings; must be durable |
  | `VIOLAWAKE_MODELS_DIR` | Required if you keep filesystem-backed trained models; must be durable |
  | `VIOLAWAKE_TRAINING_TIMEOUT` | Default `1800` unless you want a stricter cap |
  | `VIOLAWAKE_MAX_CONCURRENT_JOBS` | Tune to Railway CPU/RAM budget; default is `2` |
  | `VIOLAWAKE_STRIPE_SECRET_KEY` | Stripe secret key for the live or test environment you are launching |
  | `VIOLAWAKE_STRIPE_WEBHOOK_SECRET` | Stripe webhook signing secret for `/api/billing/webhook` |
  | `VIOLAWAKE_STRIPE_PRICE_DEVELOPER` | Stripe Price ID for the `$29/mo` Developer plan |
  | `VIOLAWAKE_STRIPE_PRICE_BUSINESS` | Stripe Price ID for the `$99/mo` Business plan |
  | `VIOLAWAKE_CONSOLE_BASE_URL` | `https://violawake.com` if the frontend is on the apex domain |

- [ ] Create Stripe account + products + webhook
  How: create the Developer and Business recurring prices in Stripe, copy the live/test Price IDs into Railway env vars, and configure the webhook endpoint as `https://api.violawake.com/api/billing/webhook` for at least `checkout.session.completed`, `customer.subscription.updated`, `customer.subscription.deleted`, and `invoice.payment_failed`.

- [ ] Deploy and verify `/api/health` returns `200`
  How: deploy the Railway service, open `https://api.violawake.com/api/health`, and confirm it returns a healthy JSON payload after startup.

- [ ] Run database migration check
  How: because the app currently uses `Base.metadata.create_all()` instead of Alembic, boot the backend against a fresh PostgreSQL database, confirm all expected tables are created, and record the schema snapshot before launch.

## Console Frontend (Cloudflare Pages)

- [ ] Build frontend with `VITE_API_URL` set to production backend
  How: in Cloudflare Pages build settings, set `VITE_API_URL=https://api.violawake.com/api`, then build `console/frontend` with `npm run build` so the shipped bundle points at the production API namespace.

- [ ] Deploy to Cloudflare Pages
  How: configure Pages to build `console/frontend`, run `npm ci && npm run build`, and publish the `dist/` directory.

- [ ] Verify all routes work (`/`, `/pricing`, `/login`, `/register`, `/dashboard`, `/billing`)
  How: test both client-side navigation and direct hard-refresh loads for each route. SPA fallback routing is already configured in `console/frontend/public/_redirects` (`/* /index.html 200`).

- [ ] Test Stripe checkout flow end to end
  How: sign up a test account, upgrade from `/pricing`, complete Checkout, confirm the webhook updates subscription state, then verify `/billing` and the customer portal reflect the new tier.

## Domain & DNS

- [ ] Register `violawake.com` (or verify it's already registered)
  How: confirm registrar ownership, renewal settings, and access to DNS management before launch day.

- [ ] Point `violawake.com` -> Cloudflare Pages
  How: add the custom domain in Cloudflare Pages and create the required DNS records so the apex domain serves the frontend.

- [ ] Point `api.violawake.com` -> Railway
  How: add the custom domain in Railway, create the CNAME or target record Railway provides, and verify the backend is reachable on the final hostname.

- [ ] SSL certificates (Cloudflare + Railway auto-provision)
  How: wait for both platforms to issue certificates, then confirm HTTPS works cleanly on both `https://violawake.com` and `https://api.violawake.com`.

## Legal

- [ ] Privacy policy at `/privacy` (verify content)
  How: the route already exists; do a legal and factual review so the page matches the actual stack you ship, especially storage and subprocess claims such as Stripe, Cloudflare, and any mention of Cloudflare R2.

- [ ] Terms of service at `/terms` (verify content)
  How: the route already exists; review liability, billing, refund, acceptable-use, and termination language with final company details and support contacts.

- [ ] Apache 2.0 `LICENSE` file in repo root
  How: `LICENSE` already exists; verify it remains in the repo root, is included in sdist/wheel artifacts, and matches the README licensing statements.

## Security

- [ ] `SECURITY.md` in repo root (verify)
  How: `SECURITY.md` already exists; confirm the contact address, SLAs, and supported-version policy are correct for the public launch.

- [ ] GitHub security advisories enabled
  How: in GitHub repo settings, verify the Security tab is active and advisory reporting/private vulnerability reporting is enabled for `GeeIHadAGoodTime/ViolaWake`.

- [ ] Rate limiting on auth endpoints (verify implementation)
  How: verify the current in-memory limiter in `console/backend/app/routes/auth.py` still meets production needs behind Railway and Cloudflare; if horizontal scaling is planned, replace it with a shared store such as Redis before launch.

- [ ] No secrets committed to git (verify `.gitignore`)
  How: `.gitignore` already excludes `.env`, `.env.local`, `.env.*.local`, `*.key`, and `secrets.json`; still run a final secret scan over tracked files before launch.

## Marketing / Launch Day

- [ ] Hacker News Show HN post draft
  How: prepare a short post focused on the open-source angle, on-device privacy, ONNX inference, and the Console workflow, then have it ready to publish once the release and site are live.

- [ ] GitHub repo description and topics set
  How: update the repository metadata on GitHub with a concise description and relevant topics such as `wake-word`, `onnx`, `speech-recognition`, `voice-assistant`, and `python`.

- [ ] GitHub repo README badges working
  How: verify the existing badges in `README.md` resolve after launch, especially the PyPI badge once `violawake` is live on PyPI.

- [ ] Star the repo from founding accounts
  How: use team and personal accounts after launch so the repository has initial social proof without automating engagement.
