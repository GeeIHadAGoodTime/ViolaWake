# Lane 10 audit report - Infrastructure & DevOps

Date: 2026-06-03
Worktree: `J:\CLAUDE\PROJECTS\Wakeword-l10-devops`
Branch: `audit-2026-06-03/l10-devops`
Verdict: MUST-FIX

## Binding corrections

`_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` was not present in this
worktree. It was found and read read-only from
`J:\CLAUDE\PROJECTS\Wakeword\_diag\2026-06-03\SC_AUDIT_ROUND_1_CORRECTIONS.md`.
I applied section A and section B4: no production `up -d`; deploy PASS is
build reproducibility plus read-only tunnel/DNS inspection.

## Finding 1 - live API is down behind the documented tunnel

MUST-FIX. `docs/DEPLOYMENT.md:215` documents tunnel `violawake-api`
(`7dbef1da-74e3-4d7f-bba9-aad4a3e72150`) for `api.violawake.com`;
`docs/DEPLOYMENT.md:231` documents tunnel-down as Cloudflare 1033/530.
Read-only Cloudflare inspection found DNS pointing to the documented tunnel,
but the tunnel itself reports `down`.

Command:

```text
Cloudflare API:
GET /zones?name=violawake.com
GET /zones/<zone_id>/dns_records?name=api.violawake.com&type=CNAME
GET /accounts/<account_id>/cfd_tunnel/7dbef1da-74e3-4d7f-bba9-aad4a3e72150
GET /accounts/<account_id>/cfd_tunnel/7dbef1da-74e3-4d7f-bba9-aad4a3e72150/configurations
```

Output excerpt:

```json
{
  "dns_records": [{
    "type": "CNAME",
    "name": "api.violawake.com",
    "content": "7dbef1da-74e3-4d7f-bba9-aad4a3e72150.cfargotunnel.com",
    "proxied": true
  }],
  "tunnel": {
    "id": "7dbef1da-74e3-4d7f-bba9-aad4a3e72150",
    "name": "violawake-api",
    "status": "down",
    "remote_config": true
  },
  "tunnel_config": {
    "ingress": [
      {"service": "http://backend:8000", "hostname": "api.violawake.com"},
      {"service": "http_status:404"}
    ]
  }
}
```

Command:

```text
fetch https://api.violawake.com/api/health
fetch https://api.violawake.com/openapi.json
fetch https://violawake.com/
```

Output excerpt:

```json
[
  {"url": "https://api.violawake.com/api/health", "status": 530},
  {"url": "https://api.violawake.com/openapi.json", "status": 530},
  {"url": "https://violawake.com/", "status": 200}
]
```

Fix implemented: updated `docs/PRODUCTION_STATUS.md` to record the 2026-06-03
outage and the proof needed before the backend is marked live again.

Blocked production action: reconnecting the tunnel likely requires
`docker compose -f docker-compose.production.yml up -d tunnel` or equivalent,
which section B4 forbids in this audit.

## Finding 2 - backups are stale and no scratch restore proof exists

MUST-FIX. `docs/RUNBOOK.md:73` documents nightly R2 backups. R2 contains only
three backup dates, and the newest Postgres/app-data pair is 2026-05-10.
On 2026-06-03 that is 596.4 hours old, so scheduled backups are silently
failing or were never installed.

Command:

```text
Cloudflare API:
GET /accounts/<account_id>/r2/buckets/violawake-backups
GET /accounts/<account_id>/r2/buckets/violawake-backups/objects?prefix=postgres/&limit=1000
GET /accounts/<account_id>/r2/buckets/violawake-backups/objects?prefix=app-data/&limit=1000
```

Output excerpt:

```json
{
  "bucket": {"status": 200, "success": true},
  "postgres": {
    "count": 3,
    "latest": [
      {"key": "postgres/2026-05-10.sql.gz", "size": 17382},
      {"key": "postgres/2026-05-09.sql.gz", "size": 17712},
      {"key": "postgres/2026-05-08.sql.gz", "size": 17617}
    ]
  },
  "app_data": {
    "count": 3,
    "latest": [
      {"key": "app-data/2026-05-10.tar.gz", "size": 53394855}
    ]
  }
}
```

The pre-existing restore docs were unsafe for drills: they piped the backup
into `wakeword-postgres-1` and wiped `/app/data` in `wakeword-backend-1`.
Those are production writes, not scratch restore drills.

File evidence before fix:

```text
docs/RUNBOOK.md:125 gunzip -c "restore/${DATE}.sql.gz" | docker exec -i wakeword-postgres-1 psql -U violawake -d violawake
docs/RUNBOOK.md:130 docker exec wakeword-backend-1 sh -lc 'mkdir -p /app/data && find /app/data -mindepth 1 -maxdepth 1 -exec rm -rf {} +'
```

Fix implemented:

- Added `scripts/backup_restore_drill.py`, which lists/downloads the latest R2
  Postgres backup via Cloudflare API, enforces a max backup age, restores into
  a generated scratch Postgres container, queries the restored DB, and removes
  the scratch container.
- Updated `docs/RUNBOOK.md:101` and `docs/RUNBOOK.md:115` so freshness checks
  and restore drills use the new script and explicitly do not write to
  `wakeword-postgres-1`.
- Updated `docs/PRODUCTION_STATUS.md` to mark current backups as not verified.

Verification:

```text
python scripts/backup_restore_drill.py --inspect-only --max-age-hours 36 --env-file J:/CLAUDE/PROJECTS/Wakeword/.env.production --env-file J:/CLAUDE/PROJECTS/FewerJobs/.env
```

Output:

```text
Latest backup: r2://violawake-backups/postgres/2026-05-10.sql.gz
Backup age: 596.4 hours
backup_restore_drill failed: Latest backup is older than 36 hours
```

Relaxed inspection to prove the latest object is at least downloadable and
decompressible:

```text
python scripts/backup_restore_drill.py --inspect-only --max-age-hours 1000 --env-file J:/CLAUDE/PROJECTS/Wakeword/.env.production --env-file J:/CLAUDE/PROJECTS/FewerJobs/.env
```

Output:

```text
Latest backup: r2://violawake-backups/postgres/2026-05-10.sql.gz
Backup age: 596.4 hours
Downloaded 17382 compressed bytes; decompressed to 103303 SQL bytes
SQL inspection OK: create_tables=11 copy_sections=11
```

Scratch restore was not completed because Docker server calls are currently
not responding:

```text
docker --version
Docker version 29.4.3, build 055a478

docker info --format Server={{.ServerVersion}}
Command failed: docker info --format Server={{.ServerVersion}}
```

## Finding 3 - documented deploy build is not reproducible today

MUST-FIX. Section B4 allows build only. The documented build command did not
finish and did not produce the two image IDs required for the corrected PASS.

Command:

```text
docker compose -f docker-compose.production.yml build backend
docker image inspect wakeword-backend --format "FIRST_IMAGE_ID={{.Id}}"
```

Output:

```text
command timed out after 904040 milliseconds
```

After aborting the stale build process, Docker server reads remained
unreliable:

```text
docker --version
Docker version 29.4.3, build 055a478

docker info --format Server={{.ServerVersion}}
Command failed: docker info --format Server={{.ServerVersion}}
```

Additional gap: `.dockerignore:29` explicitly unignores `.env.production`,
so production secrets are included in the Docker build context even if the
Dockerfile does not copy that file. `.dockerignore` is not listed in Lane 10
ownership, so I did not edit it in this branch.

## Finding 4 - CI has been red on master for the last 7 days

MUST-FIX. `docs/LANE_LEDGER.md:629` requires all CI workflows green on trunk.
The GitHub Actions history for branch `master` has daily CI failures from
2026-05-27 through 2026-06-03. The latest run on 2026-06-03 failed in lint and
console backend tests.

Command:

```text
GET https://api.github.com/repos/GeeIHadAGoodTime/ViolaWake/actions/runs?branch=master&per_page=100
GET https://api.github.com/repos/GeeIHadAGoodTime/ViolaWake/actions/runs/26869040477/jobs?per_page=100
```

Output excerpt:

```json
[
  {"name": "CI", "event": "schedule", "conclusion": "failure", "sha": "fa2bd3b", "created_at": "2026-06-03T07:00:53Z"},
  {"name": "CI", "event": "schedule", "conclusion": "failure", "sha": "fa2bd3b", "created_at": "2026-06-02T06:49:28Z"},
  {"name": "Model Verification", "event": "schedule", "conclusion": "success", "sha": "fa2bd3b", "created_at": "2026-06-01T11:21:50Z"},
  {"name": "CI", "event": "schedule", "conclusion": "failure", "sha": "fa2bd3b", "created_at": "2026-05-27T06:36:52Z"}
]
```

Latest failed jobs:

```json
[
  {"name": "Lint", "failed_steps": ["Run ruff (lint)"]},
  {"name": "Console backend tests", "failed_steps": ["Run console backend tests"]}
]
```

Not fixed in this branch: the failing source is outside the Lane 10 owned
workflow surface. Also, this local audit branch is ahead of `origin/master`;
the red GitHub history is for the latest pushed `master` commit `fa2bd3b`.

## Finding 5 - stale deploy helper carried a hard-coded secret and wrong hosting path

MUST-FIX. `scripts/deploy_launch.py` is Lane 10-owned and previously embedded
`PROD_SECRET_KEY = "..."` plus Railway-era deployment steps. That conflicts
with `docs/DEPLOYMENT.md`, which documents local Docker plus Cloudflare Tunnel.

Fix implemented: replaced `scripts/deploy_launch.py` with a fail-closed
deprecated helper that exits 1 and points to `docs/DEPLOYMENT.md`.

Verification:

```text
python scripts/deploy_launch.py
```

Output:

```text
scripts/deploy_launch.py is deprecated. Use docs/DEPLOYMENT.md for current manual deploy steps.
```

## Negative probes

- Backup silent-failure probe: current R2 state is a real broken shape. The new
  drill catches it by failing when the latest dated backup is older than 36
  hours.
- Restore safety probe: the old runbook restored directly into production
  containers; updated docs require scratch restore and warn against production
  writes during drills.
- Deploy image-tag probe: not run. Docker server calls were unresponsive after
  the documented build timed out; a nonexistent-image deploy probe would not be
  meaningful until Docker is healthy.

## Planned gate

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: backup-restore-drill-freshness
contract: Latest R2 Postgres backup must be fresh and restorable into scratch Postgres without writing production containers.
detector: scripts/backup_restore_drill.py
own_tests:
  - TBD - orchestrator should add a fixture-backed test proving stale latest backup exits nonzero.
  - TBD - orchestrator should add a fixture-backed test proving a fresh SQL dump reaches the scratch restore query path.
```

## Verification commands for edited files

```text
python -m py_compile scripts/backup_restore_drill.py scripts/deploy_launch.py
```

Output:

```text
<no stdout/stderr, exit 0>
```

```text
git diff --stat
```

Output excerpt:

```text
docs/PRODUCTION_STATUS.md | 12 ++
docs/RUNBOOK.md           | 61 +++++----
scripts/deploy_launch.py  | 334 +++-------------------------------------------
scripts/backup_restore_drill.py | added
```

## MANDATORY self-audit gate

- I did not reconnect or restart the tunnel. Section B4 forbids production
  `up -d` or container restart; the outage is therefore reported as blocked on
  founder authorization.
- I did not complete a scratch Postgres restore. Docker server calls stopped
  responding after the documented build timed out; I proved download and gzip
  SQL integrity, but not a restored query.
- I did not fix the red CI jobs. The latest pushed `master` failures are lint
  and console backend test failures outside the Lane 10 owned workflow files.
- I did not edit `.dockerignore` even though it includes `.env.production` in
  build context. The file is not listed in Lane 10 ownership, so this is
  surfaced as a gap for orchestrator routing.
- I did not verify UptimeRobot or any external alert delivery. The runbook
  documents how to create a monitor, but no API token or dashboard access for
  UptimeRobot was available in this audit.
