# AUDIT — Lane 8: SaaS Console — Backend

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l8-backend
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies, and § B3 is binding: if no test account / staging exists,
> the live billing-flow probe is BLOCKED — surface it, don't burn real
> billing.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 8):
*"Does `api.violawake.com` correctly serve sign-up → sample upload →
training job → model download → billing → email flows under load?"*

This is the **paying-customer surface** — auth, billing, rate-limiting
errors here are real customer impact.

## Success criteria — binary verdict
PASS = (a) `GET https://api.violawake.com/api/health` returns 200;
(b) the full sign-up → training-job → model-download flow completes on
a live integration run (use a test account; don't burn real billing);
(c) auth boundaries (unauthenticated billing, cross-team access,
rate-limit bypass) all reject; (d) the inbound email worker
(`workers/support-email/`) accepts a documented inbound and processes
it.

MUST-FIX = (a) a route silently accepts an unauthenticated request
that touches user data or billing; (b) a cross-team data leak; (c) a
billing endpoint can be replayed; (d) a 5xx on a documented happy
path; (e) tunnel container down without alert.

NOT MUST-FIX: log-formatting nits, performance optimizations without
measured regression, missing future routes.

## Sources
- `docs/LANE_LEDGER.md` § 8
- `CLAUDE.md`
- Files owned (see ledger § 8 "Owns")
- `docs/DEPLOYMENT.md`, `docs/PRODUCTION_STATUS.md`,
  `docs/OPERATIONS_RUNBOOK.md`
- Live: `https://api.violawake.com/api/health`,
  `/api/openapi.json`

## Investigate
- Hit the live API (read-only first). Compare the live `openapi.json`
  to the routes in `console/backend/app/routes/`.
- For each route, identify its auth requirement and probe it with
  (a) no auth, (b) wrong-team auth, (c) replay. Use test accounts
  only — DO NOT touch real billing.
- Trace the sign-up → training → download flow end-to-end on a test
  account.
- Audit the inbound-email Worker against the route it pairs with.

Find every gap. Zero is the bar. Be adversarial here — paying users
are the ones who'll notice first.

## Decide & implement
One topic branch, one commit per fix. `Ratchet:` for class fixes
(e.g. a test asserting every billing/auth route requires the right
auth shape). `Ratchet-Exempt:` for single fixes.

Do NOT push, do NOT merge.

## Prove it
For each fix: the failing curl/request + the passing curl/request,
with response bodies.

## Report
`_diag/2026-06-03/audit_lane_08_report.md`. Verdict + fixes +
MANDATORY self-audit gate.

## Scaffolding
- Tunnel: `violawake-api`, UUID
  `7dbef1da-74e3-4d7f-bba9-aad4a3e72150`.
- Docker stack: `wakeword-backend-1`, `wakeword-postgres-1`,
  `wakeword-tunnel-1`. NEVER touch NOVVIOLA containers
  (`viola-api`, `viola-postgres-local`, etc).
- Use test accounts; do not burn real billing.
