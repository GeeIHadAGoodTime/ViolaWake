# AUDIT — Lane 10: Infrastructure & DevOps

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l10-devops
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies, and § B4 is binding: NO `up -d` against production. PASS
> condition (a) becomes build-reproducibility + read-only tunnel/DNS
> inspection, not a live deploy.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 10):
*"Are deploys reproducible, backups taken on schedule, CI green on
every PR, and the production stack observable?"*

## Success criteria — binary verdict
PASS = (a) the documented deploy steps, run today from a clean shell,
land the expected image SHA on the live URL; (b) Postgres backups
exist and are verifiably restorable into a scratch container;
(c) every workflow under `.github/workflows/` is green on the current
trunk; (d) the production stack emits enough signal to detect a
documented outage.

MUST-FIX = (a) the documented deploy steps fail today; (b) backups are
silently failing or not restorable; (c) a CI workflow has been red on
trunk; (d) the production stack has a blind spot for a documented
failure mode (tunnel down, container crashed, disk full).

NOT MUST-FIX: optimization without measured regression, missing future
infra.

## Sources
- `docs/LANE_LEDGER.md` § 10
- `CLAUDE.md`
- Files owned (see ledger § 10 "Owns")
- `docs/DEPLOYMENT.md`, `docs/OPERATIONS_RUNBOOK.md`, `docs/RUNBOOK.md`,
  `docs/PRODUCTION_STATUS.md`

## Investigate
- Walk the documented deploy on a dry run. Does it still work today?
- Inspect the most recent successful backup (R2 / local). Can you
  restore it into a scratch container?
- Pull CI run history for the last 7 days; surface any red runs.
- Map documented failure modes to actual alerts/checks. Find blind
  spots.

Find every gap. Zero is the bar.

## Decide & implement
One topic branch. `Ratchet:` for class fixes (e.g. a periodic restore
drill check-in). `Ratchet-Exempt:` for single fixes. Don't push,
don't merge.

## Prove it
For deploys: actual command output. For backup restore: actual
restored DB queried successfully.

## Report
`_diag/2026-06-03/audit_lane_10_report.md`. Verdict + fixes +
MANDATORY self-audit gate.

## Scaffolding
- Docker compose: `docker-compose.production.yml`,
  `docker-compose.viola-bridge.yml`.
- Railway: `railway.json`, `railway.toml`.
- DO NOT touch NOVVIOLA containers when probing the bridge.
