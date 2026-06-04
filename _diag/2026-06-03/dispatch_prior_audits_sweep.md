# SWEEP — Prior audit findings

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-prior-audits-sweep
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. **This sweep is RECOMMENDATIONS-ONLY** — every fix is
> routed to the lane that owns the affected surface.

> **Path uncertainty:** the cleanup audit (Lane 12 housekeeping,
> running in parallel) is moving these top-level audit docs INTO
> `docs/`. They may be at either location depending on cleanup's merge
> timing. Handle both: read from `<repo-root>/<NAME>.md` AND from
> `docs/<NAME>.md`, take whichever exists; if both exist, prefer the
> one in `docs/` and note the duplicate.

## Mission
*"Take every finding from prior audit / readiness / gap-analysis docs
and triage it: RESOLVED, OPEN, WONTFIX, or SUPERSEDED."*

The project has accumulated audit artifacts that nobody has gone back
to close out: `ADVERSARY_AUDIT.md`, `AUDIT_2026_03_28.md`,
`FUNCTIONAL_GAP_ANALYSIS.md`, `ACCURACY_MISSION.md`,
`BUILD_VS_BUY_AUDIT.md`, `E2E_READINESS.md`, `LAUNCH_READINESS.md`,
`PRE_LAUNCH_CHECKLIST.md`, `PROGRESS.md`. Each contains findings
that may or may not still be open.

## Success criteria — binary verdict
PASS = every numbered/bulleted finding in each prior audit doc is
classified with evidence:
- **RESOLVED** = code/state shows the fix is in (cite file:line).
- **OPEN** = code/state shows the issue persists (cite file:line).
- **WONTFIX** = there's a documented decision (in CHANGELOG, an ADR,
  or a prior commit) that the project elected not to fix.
- **SUPERSEDED** = a later change made the finding moot (cite the
  superseding change).

MUST-FIX = a finding classified as OPEN with severity P0/P1 (or
equivalent) that has not been actioned into any lane's current scope.
For each such finding, name the owning lane (per the ledger) and
recommend it for action.

NOT MUST-FIX: low-severity OPEN findings without evidence of customer
impact; copy nits inside old audits.

## Sources
- `docs/LANE_LEDGER.md` (lane ownership for routing)
- `CLAUDE.md`
- The prior audit docs above (handle path uncertainty per the
  binding note)

## Investigate
For each finding, read it → grep/inspect current code → classify with
evidence.

## Output
Write a recommendations-only report at
`_diag/2026-06-03/audit_prior_findings_report.md`:
- Per source doc: a table of findings × classifications × evidence.
- Aggregated MUST-FIX list (P0/P1 OPEN findings) with owning lane.
- Mandatory five-bullet self-audit gate.

Commit on a topic branch. No code edits to the affected lanes — write
recommendations only.
