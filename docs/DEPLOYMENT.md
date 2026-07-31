# Deployment

How `https://violawake.com` and `https://api.violawake.com` actually run in production.

**The backend deploys itself.** A commit merged to `master` reaches the running API within about ten minutes, unattended, via the reconciler in `scripts/deploy_backend.py`. The frontend is still a manual/dispatch deploy.

Last verified end-to-end: **2026-07-31** (backend host, tunnel and deploy path re-measured on the box; the pre-2026-07-31 version of this page named the wrong host). If you're reading this and the architecture has changed, update this file with the date.

---

## Architecture

```
                    GitHub: GeeIHadAGoodTime/ViolaWake
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
  Frontend deploy                          Backend deploy
  (manual / workflow_dispatch)             (AUTOMATIC, pull-based)
  ───────────────────────────              ────────────────────────
  cd console/frontend                      violawake-deploy.timer, every 10 min
  VITE_API_URL=… npm run build             on the host, runs
  wrangler pages deploy dist                 scripts/deploy_backend.py
            │                              which reconciles the running
            │                              container toward origin/master
            ▼                                         │
  Cloudflare Pages                                    ▼
  project: violawake                       Hetzner box 167.233.233.33
  serves: violawake.com                    checkout /opt/viola/Wakeword
                                           compose project `wakeword`
                                             wakeword-backend-1   (uvicorn)
                                             wakeword-postgres-1  (data)
                                             wakeword-decoder-1   (sidecar)
                                                  │
                                                  ▼
                                        Cloudflare Tunnel
                                        violawake-api-server
                                        a4961724-2b7b-49a1-8711-e088245be4c4
                                        (container cloudflared-wake-server)
                                                  │
                                                  ▼
                                        api.violawake.com (Cloudflare edge)
```

### Why this architecture

- **Cloudflare Pages for the frontend** — fast static hosting, free tier, instant rollback per deployment.
- **Docker + Cloudflare Tunnel for the backend** — keeps the host off the public internet, no fixed IP needed, no load balancer. The same image can be dropped onto any VPS without changing the application.
- **Separate Postgres in the same compose stack** — ViolaWake's data is isolated from every other project on the host. Volume `pgdata` survives container recreations.
- **Pull-based deploy, not a CI push.** The reconciler runs *on* the host and pulls; nothing inbound is opened and no CI credential lives on the box. This repo is **public**, so a self-hosted GitHub runner with a Docker socket would let a fork's pull request execute as root on a host that also serves other production stacks. Polling every ten minutes costs a couple of `git`/`docker inspect` calls and avoids that entire class.
- **The old note here said "no auto-deploy, intentional, keeps the gate human".** It stopped being true in practice: what it actually produced was four backend fixes merged on 2026-07-29/30 that nobody could confirm were live, and on 2026-07-31 a merged customer-facing billing fix still not serving two hours after it landed. The human gate is preserved where it belongs — destructive migrations still refuse to deploy unattended — not on every ordinary fix.

---

## Backend deploy (api.violawake.com)

### The automatic path (normal case: do nothing)

`violawake-deploy.timer` fires `scripts/deploy_backend.py` every 10 minutes on the host. With no drift it is a no-op. With drift it:

1. resolves `origin/master` and compares it against the **revision label of the image the running container is actually using** (`org.opencontainers.image.revision`) — the tag `:latest` cannot answer "what is serving?", the label can;
2. refuses if the deploy checkout is dirty, if free disk is under the floor, or if a new alembic revision is destructive (see below);
3. **defers** (does not deploy) while a training job is RUNNING or PENDING — recreating the container kills it (Job 51, 2026-05-07);
4. fast-forwards `/opt/viola/Wakeword`, builds the image labelled with the target commit, and **import-preflights it** (`python -c "import app.main"` in the real compose environment, `--no-deps`) before any traffic moves;
5. recreates the container, then requires container health **and** `https://api.violawake.com/api/health` = 200 **and** the running image's revision label = the target commit;
6. rolls back to the previously running image on any failure, and pages either way.

Watch it:

```bash
systemctl list-timers violawake-deploy.timer
journalctl -u violawake-deploy.service -n 100
python scripts/deploy_backend.py --dry-run          # what would it do right now?
cat /var/lib/violawake-deploy/journal.jsonl | tail -3
```

Install or repair it on a host (idempotent — re-run it to undo any hand-edit of the units):

```bash
sudo infra/deploy/install.sh
sudo infra/deploy/install.sh --uninstall
```

Host-specific settings live in `/etc/violawake-deploy.env` (off-VCS), notably `VIOLAWAKE_DEPLOY_ALERT_SINK` — the JSONL red-alert inbox that deploy failures and stale drift are appended to. Records carry `business: violawake`, so the operator's ops-ticket bridge turns them into issues on this repo.

### When it deliberately will not deploy

| Situation | What happens | What you do |
|---|---|---|
| Training job in flight | Deferred, retried next tick; pages if drift outlives `--stale-hours` (default 6) | Nothing, or `--force` for an emergency hotfix where killing queued customer work is the lesser evil |
| New alembic revision drops/renames/retypes a column or table, or runs raw SQL | Refused, paged | Review it, then deploy by hand with `--allow-destructive-migrations` inside a maintenance window |
| Free disk under 15 GiB | Refused, paged | Reclaim space (`docker system df`), then let the next tick run |
| Deploy checkout dirty or diverged | Refused, paged | Resolve it by hand on the box — the reconciler never force-moves a checkout it did not make |

### Manual deploy (hotfix, or when the timer is off)

Same script, run directly; there is no separate hand-written sequence to keep in sync any more:

```bash
cd /opt/viola/Wakeword
python scripts/deploy_backend.py --force                       # ignore in-flight jobs
python scripts/deploy_backend.py --allow-destructive-migrations
python scripts/deploy_backend.py --target-ref origin/some-branch
```

### Prerequisites

- Docker running on the host (currently the Hetzner box `167.233.233.33`, checkout `/opt/viola/Wakeword`)
- `.env.production` exists at the repo root with the required env vars (see `Required env vars` below). This file is git-ignored — never commit secrets.
- The cloudflared tunnel container has been bootstrapped once with `CLOUDFLARE_TUNNEL_TOKEN`. Subsequent deploys do not need to re-bootstrap it.

### The underlying commands (what the reconciler runs, for reference)

Kept because they are still the right thing to run when you are debugging the
deploy itself. **Do not run these as the routine deploy** — the reconciler adds
the guards, the revision label, the verification and the rollback, and a
hand-run `docker compose build` with `VIOLAWAKE_BUILD_SHA` unset labels the
image `unknown`, which the reconciler then treats as "not verifiable" rather
than as up to date.

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

**The live stack is composed of TWO files.** `wakeword-backend-1` on the box was created with
`-f docker-compose.production.yml -f docker-compose.viola-bridge.yml`, so any `compose` command
that omits the second file is operating on a different merged config than the running container.
The reconciler always passes both; if you are driving compose by hand, pass both too.

### Rollback

The reconciler rolls itself back on a failed verification. To do it by hand, the previously
running image is pinned before every build:

```bash
docker image inspect ghcr.io/geeihadagoodtime/wakeword-backend:rollback \
  --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'   # what you'd go back to
docker tag ghcr.io/geeihadagoodtime/wakeword-backend:rollback \
          ghcr.io/geeihadagoodtime/wakeword-backend:latest
docker compose -f docker-compose.production.yml -f docker-compose.viola-bridge.yml up -d backend
docker inspect wakeword-backend-1 --format='{{.State.Health.Status}}'
```

Then stop the timer (`systemctl stop violawake-deploy.timer`) or the next tick will roll forward
to `origin/master` again — a rollback is a statement about the code, so revert the commit on
master rather than leaving the host pinned behind it.

### Database migrations

`docker compose up -d backend` runs `alembic upgrade head` via the entrypoint (see `console/backend/entrypoint.sh`). Migrations live in `console/backend/alembic/versions/`. Additive revisions deploy automatically. Destructive ones (drop column, drop table, drop constraint, rename table, type change, raw `op.execute`) are refused by the reconciler and need a human in a maintenance window with `--allow-destructive-migrations` — see `DESTRUCTIVE_MIGRATION_PATTERNS` in `scripts/deploy_backend.py`. Note that an image rollback does **not** roll a migration back.

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
