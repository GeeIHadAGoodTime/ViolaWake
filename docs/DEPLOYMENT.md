# Deployment

How `https://violawake.com` and `https://api.violawake.com` actually run in production. **Pushing to GitHub does NOT auto-deploy anything** — both pieces are manual.

Last verified end-to-end: **2026-05-07**. If you're reading this and the architecture has changed, update this file with the date.

---

## Architecture

```
                    GitHub: GeeIHadAGoodTime/ViolaWake
                                 │
                                 ▼  (git push origin master — manual)
                          (no auto-deploy)
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
  Frontend deploy (manual)                Backend deploy (manual)
  ────────────────────────                ─────────────────────
  cd console/frontend                     cd /                          
  VITE_API_URL=… npm run build            docker compose build backend
  wrangler pages deploy dist              docker compose up -d backend
            │                                         │
            ▼                                         ▼
  Cloudflare Pages                        Local Docker (your machine)
  project: violawake                      stack project: wakeword
  serves: violawake.com                     wakeword-backend-1   (uvicorn)
                                            wakeword-postgres-1  (data)
                                            wakeword-tunnel-1    (cloudflared)
                                                  │
                                                  ▼
                                        Cloudflare Tunnel
                                        violawake-api
                                        7dbef1da-74e3-4d7f-bba9-aad4a3e72150
                                                  │
                                                  ▼
                                        api.violawake.com (Cloudflare edge)
```

### Why this architecture

- **Cloudflare Pages for the frontend** — fast static hosting, free tier, instant rollback per deployment.
- **Local Docker + Cloudflare Tunnel for the backend** — keeps the host machine off the public internet, no fixed IP needed, no cloud-host bill while pre-revenue. When ready to scale, the same container image can be dropped onto Railway / Fly / a VPS without changing the application.
- **Separate Postgres in the same compose stack** — ViolaWake's data is isolated from NOVVIOLA's data and from any other project. Volume `pgdata` survives container recreations.
- **Why not auto-deploy?** GitHub Actions don't have access to the local Docker host (the backend lives there) and Cloudflare Pages was set up without git auto-build. Both are intentional during the operator-launch phase to keep the deploy gate human.

---

## Backend deploy (api.violawake.com)

### Prerequisites

- Docker Desktop running on the host machine (currently `75.86.16.150`)
- `.env.production` exists at the repo root with the required env vars (see `Required env vars` below). This file is git-ignored — never commit secrets.
- The cloudflared tunnel container has been bootstrapped once with `CLOUDFLARE_TUNNEL_TOKEN`. Subsequent deploys do not need to re-bootstrap it.

### Deploy

**Building on a Windows checkout: watch for CRLF.** Incident (2026-07-14): a
build from a Windows git worktree with `core.autocrlf=true` silently converted
`console/backend/entrypoint.sh` to CRLF line endings. The image built fine, but
the container crash-looped in production with
`exec /app/entrypoint.sh: no such file or directory` -- a corrupted shebang
(`#!/bin/sh\r\n`), not a missing file. `.gitattributes` now pins `*.sh`/
`entrypoint.sh` to `eol=lf` regardless of local git config, but if you ever see
that exact error after a Windows-built image, this is almost certainly it --
prefer building directly on the target Linux host (`/opt/viola/Wakeword` on the
box) over shipping a Windows-built image. Verify with
`docker run --rm --entrypoint sh <image> -c "head -c20 /app/entrypoint.sh | od -c"`
before deploying -- expect `\n` after `sh`, never `\r\n`.

```bash
cd /j/CLAUDE/PROJECTS/Wakeword

# 1. Verify you're on the commit you want to deploy
git log --oneline -3

# 2. Build the new image from the current working tree
docker compose -f docker-compose.production.yml build backend

# 3. Pre-recreate guard — refuse if a training job is in flight.
#    Recreating wakeword-backend-1 kills any RUNNING job; the resume path
#    re-queues PENDING jobs but a single slow progress event during the
#    new container's warmup can still flip them to status=failed
#    (Job 51 incident on 2026-05-07). Wait for the queue to drain, OR
#    pass --force / VIOLAWAKE_DEPLOY_FORCE=1 for an emergency hotfix
#    where queued customer work being killed is the lesser evil.
python scripts/check_in_flight_jobs.py || {
    echo "Deploy blocked. Re-run with --force when intentional.";
    exit 1;
}

# 4. Recreate the running container with the new image
docker compose -f docker-compose.production.yml up -d backend

# 5. Watch healthcheck (typically <30s)
docker inspect wakeword-backend-1 --format='{{.State.Health.Status}}'
# expect: healthy

# 6. Verify the live API now reflects the new code
curl -sS https://api.violawake.com/api/health
curl -sS https://api.violawake.com/openapi.json | python -c "import sys,json;d=json.load(sys.stdin);print('routes:',len(d['paths']))"
```

The Cloudflare Tunnel container reconnects to the new backend automatically — no tunnel config change is needed.

### Rollback

If the new container is unhealthy or breaks production, roll back to the previous image:

```bash
docker images wakeword-backend                # find the prior image SHA
docker compose -f docker-compose.production.yml stop backend
docker compose -f docker-compose.production.yml rm -f backend
docker tag wakeword-backend:<prior-sha> wakeword-backend:latest
docker compose -f docker-compose.production.yml up -d backend
```

### Database migrations

`docker compose up -d backend` runs `alembic upgrade head` via the entrypoint (see `console/backend/entrypoint.sh`). Migrations live in `console/backend/alembic/versions/`. Review the migration before deploying — additive column adds are safe, destructive ones (drop column, drop table, type change) require a maintenance window.

### Required env vars (`.env.production`)

```
# Database (Postgres in the same compose stack)
POSTGRES_PASSWORD=...

# Cloudflare Tunnel auth (one-time bootstrap)
CLOUDFLARE_TUNNEL_TOKEN=...

# App
VIOLAWAKE_ENV=production
VIOLAWAKE_SECRET_KEY=...                  # JWT signing — rotate to invalidate all sessions
VIOLAWAKE_ADMIN_TOKEN=...                 # Admin endpoints; treat like a root key
VIOLAWAKE_CONSOLE_BASE_URL=https://violawake.com
VIOLAWAKE_CORS_ORIGINS=https://violawake.com,https://www.violawake.com
VIOLAWAKE_PORT=8000

# Stripe (test or live mode — match across all four)
VIOLAWAKE_STRIPE_SECRET_KEY=...
VIOLAWAKE_STRIPE_WEBHOOK_SECRET=...
VIOLAWAKE_STRIPE_PRICE_DEVELOPER=price_...
VIOLAWAKE_STRIPE_PRICE_BUSINESS=price_...

# Email (Resend) — leave unset for auto-verify dev fallback
VIOLAWAKE_RESEND_API_KEY=...

# Limits / retention
VIOLAWAKE_MAX_CONCURRENT_JOBS=4
VIOLAWAKE_TRAINING_TIMEOUT=900
VIOLAWAKE_RECORDING_RETENTION_DAYS=30
VIOLAWAKE_MODEL_RETENTION_DAYS=90
VIOLAWAKE_POST_TRAINING_RETENTION_HOURS=72
VIOLAWAKE_NEGATIVES_CORPUS_DIR=/app/negatives
VIOLAWAKE_ALGORITHM=HS256
VIOLAWAKE_ACCESS_TOKEN_EXPIRE_HOURS=24
```

If a key is missing, the backend either no-ops the feature gracefully (Stripe → 503 with "Billing features require a configured Stripe secret key"; Resend → silently auto-verifies users) or fails to start (DB URL, secret key). Check container logs with `docker logs wakeword-backend-1` if the healthcheck fails.

### Corpus

Training requires a mounted universal speech-negative corpus. Generic speech negatives come from LibriSpeech/MUSAN file access, not Edge-TTS. Without this corpus, training fails fast with a clear error instead of silently falling back to flaky network TTS.

Operator default: mount the in-repo corpus directory read-only:

```yaml
./corpus:/app/corpus:ro
```

Alternative: download the smaller starter corpus, then mount that path:

```bash
violawake-download-corpus
```

```yaml
~/.violawake/corpus:/app/corpus:ro
```

`violawake-download-corpus` currently installs LibriSpeech `dev-clean` under `~/.violawake/corpus/librispeech/dev-clean`. MUSAN can be added under the same corpus root when the larger speech/music/noise set is available.

---

## Frontend deploy (violawake.com)

### Prerequisites

- `wrangler` installed and logged in: `wrangler whoami` should show your Cloudflare account.
- Node 20+ (for `npm run build`).

### Deploy

```bash
cd /j/CLAUDE/PROJECTS/Wakeword/console/frontend

# 1. Build with the production API URL baked into the bundle.
#    THIS IS NOT OPTIONAL — Vite bakes import.meta.env.VITE_API_URL at build time.
#    If you forget, the bundle calls /api on violawake.com → 405.
VITE_API_URL=https://api.violawake.com/api npm run build

# 2. Verify the right URL was baked in (sanity check)
grep -c "https://api.violawake.com/api" dist/assets/*.js   # expect: at least 1

# 3. Deploy
wrangler pages deploy dist --project-name violawake --branch master --commit-dirty=true

# 4. Verify the production URL serves the new bundle
curl -sS https://violawake.com/ | grep -oE '/assets/index-[A-Za-z0-9]+\.js'
```

Each `wrangler pages deploy` produces a unique preview URL (e.g. `https://05488c49.violawake.pages.dev`). The production alias `violawake.com` is updated automatically when deploying to the production branch (`--branch master`). Roll back via the Pages dashboard.

### Frontend pages list

Live as of 2026-05-07: `/` `/login` `/register` `/forgot-password` `/reset-password` `/verify-email` `/dashboard` `/record` `/training/:jobId` `/billing` `/pricing` `/teams` `/teams/:teamId` `/teams/accept` `/account/password` `/privacy` `/terms` `/landing`.

---

## Cloudflare Tunnel

### Tunnel inventory

| Name | UUID | Purpose | How it's run |
|---|---|---|---|
| `violawake-api` | `7dbef1da-74e3-4d7f-bba9-aad4a3e72150` | Routes `api.violawake.com` to local backend | `wakeword-tunnel-1` Docker container |
| `Viola_app` | `97f23a85-78fc-40e5-978c-426c752015d1` | NOVVIOLA — routes `api.useviola.com` etc | `cloudflared.exe` Windows Service |

The two tunnels run independently. **Restarting one does NOT affect the other.** This is the basis of the NOVVIOLA-vs-ViolaWake decoupling described in `CLAUDE.md`.

### Routes config

The current tunnel uses **remote-managed config** (Cloudflare dashboard → Zero Trust → Tunnels). The local `~/.cloudflared/config.yml.bak.dead-violawake-tunnel-2026-04-26` is from a DEAD legacy tunnel and should not be used as a reference for current behavior.

To inspect or change routes:
- View: dash.cloudflare.com → Zero Trust → Networks → Tunnels → `violawake-api` → Public Hostname
- Modify: same UI, or POST to the Cloudflare API `/accounts/{id}/cfd_tunnel/{tunnel_id}/configurations`

### Common tunnel issues

- **Backend container restart kills health for ~30s** — the tunnel keeps connections open and reconnects automatically; tunnel reconfiguration is not needed.
- **Tunnel container down (wakeword-tunnel-1 stopped)** — `api.violawake.com` returns Cloudflare 1033 / 530. Fix: `docker compose up -d tunnel`.
- **Tunnel token expired** — replace `CLOUDFLARE_TUNNEL_TOKEN` in `.env.production`, restart the tunnel container.

---

## SDK release (PyPI)

The SDK is **separate** from the SaaS deploy. Releasing the SDK does not affect the live console.

```bash
cd /j/CLAUDE/PROJECTS/Wakeword

# 1. Bump version in pyproject.toml
# 2. Update CHANGELOG.md and RELEASE_NOTES.md
# 3. Tag and push
git tag v0.2.3
git push origin v0.2.3
# .github/workflows/release.yml runs:
#   - validate tag matches pyproject.toml
#   - build wheel + sdist
#   - tools/fetch_release_models.py downloads models for the release artifact
#   - publish to PyPI via OIDC trusted publishing
#   - create GitHub Release with RELEASE_NOTES.md body
```

After release, verify on a clean machine:
```bash
python -m venv /tmp/test_venv && source /tmp/test_venv/bin/activate
pip install "violawake[oww]"
python -c "from openwakeword.utils import download_models; download_models()"
python -c "from violawake_sdk import WakeDetector; import numpy as np; d=WakeDetector(model='temporal_cnn'); print(d.process(np.zeros(1280, dtype=np.float32)))"
```

Note: `pip install violawake` (without `[oww]`) installs the base SDK but `WakeDetector` will raise `ModelNotFoundError` on first use. The SDK's wake-word feature requires the `openwakeword` extra **plus** a one-time `download_models()` call to fetch the OWW backbone ONNX files (the openwakeword PyPI wheel does not bundle them).

---

## Pre-flight checklist (before any deploy)

- [ ] `git status` clean (or you've explicitly noted what's uncommitted and why)
- [ ] `git log --oneline -3` shows the commits you intend to deploy
- [ ] For backend: `.env.production` exists at repo root with all required keys
- [ ] For frontend: `VITE_API_URL` is being passed to `npm run build`
- [ ] Migration review (if Alembic added a new file) — additive only, or maintenance window scheduled
- [ ] You've decided how to verify the deploy is live (which endpoint / which behavior change to probe)
- [ ] You know how to roll back

## Post-deploy verification (run after every deploy)

```bash
# Backend
curl -sS -o /dev/null -w "health: %{http_code}\n" https://api.violawake.com/api/health
curl -sS https://api.violawake.com/openapi.json | python -c "import sys,json;d=json.load(sys.stdin);print('routes:',len(d['paths']))"

# Frontend
curl -sS https://violawake.com/ | grep -oE '/assets/index-[A-Za-z0-9]+\.js'

# Live smoke (gated; never run in CI)
cd /j/CLAUDE/PROJECTS/Wakeword
VIOLAWAKE_LIVE=1 bash tests/live/run_smoke.sh
```

If smoke regresses, see `tests/live/RESULTS_<DATE>.md` for the prior baseline.
