# AUDIT — Lane 9: SaaS Console — Frontend

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l9-frontend
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. Lane 9 additional: **if you find an incorrect public
> claim (numeric or factual) on a rendered page, the FIX routes to
> Lane 11 (Marketing & Developer Docs) or Lane 5 (Evaluation) — record
> the suggested correction in your report, don't edit those lanes'
> files.** Lane 9 fixes are limited to the frontend code itself
> (rendering, build chain, bundle).

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 9):
*"Does `violawake.com` correctly render the sign-up, console,
comparison, pricing, and docs pages, and talk to the live backend?"*

## Success criteria — binary verdict
PASS = (a) `npm run build` with `VITE_API_URL=https://api.violawake.com/api`
produces a bundle that talks to the production backend (the 2026-05-07
regression must stay caught — construct + run the probe: omit
`VITE_API_URL`, build, grep the bundle for `/api`, assert the
absent-env case is detected);
(b) every page reachable from the deployed `https://violawake.com/` —
sign-up, console, `compare/*`, pricing, FAQ, blog, about, contact,
privacy, terms, docs — renders without console errors on the latest
stable Chrome (use headless if no browser available — `curl` + grep
for known stable strings as the minimum probe);
(c) the accessibility audit baseline (`tests/live/ACCESSIBILITY_AUDIT_2026-05-07.md`)
does not regress on the pages it covered.

MUST-FIX = any page 5xx; client-side render error; bundle ships
without `VITE_API_URL` baked; a11y regression on a covered page.

NOT MUST-FIX: copy phrasing, new-page suggestions, styling
preferences.

## Sources
- `docs/LANE_LEDGER.md` § 9
- `CLAUDE.md` (especially "How public copy is written")
- Files owned (ledger § 9 "Owns")
- Live: `https://violawake.com/` and all its linked pages

## Investigate
Read live pages first. Build locally next. Construct + run the
`VITE_API_URL` absence probe per § A1.

## Decide, prove, report
One topic branch. Report at
`_diag/2026-06-03/audit_lane_09_report.md`. Mandatory self-audit gate.
Gate spec per § A2.
