# AUDIT — ADRs (Architecture Decision Records)

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-adr-audit
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. **This audit is RECOMMENDATIONS-ONLY** — if an ADR
> conflicts with current code, the FIX (whether to the ADR or to the
> code) belongs to the owning lane; you record the conflict and the
> recommended resolution, you do NOT edit the offending code.

## Mission
*"Does each ADR (`docs/adr/ADR-001` through `ADR-005`) accurately
describe the CURRENT state of the system, or has reality drifted from
the locked decision?"*

ADRs are load-bearing — they're the locked decisions the codebase is
built on, and other lanes' "Sources" sections cite them. A wrong ADR
silently propagates wrong assumptions across every lane.

## Success criteria — binary verdict
PASS = for each of ADR-001 (ONNX runtime), ADR-002 (OWW feature
extractor), ADR-003 (Python-first), ADR-004 (open-core licensing),
ADR-005 (packaging):
(a) the ADR's "decision" still matches current code/architecture, with
cited file:line evidence;
(b) the ADR's "consequences" are observable in the codebase
(positively or as documented technical debt);
(c) the ADR isn't silently superseded by a later commit/PR.

MUST-FIX = an ADR's locked decision is materially violated by current
code without an explicit "superseded by ADR-NNN" pointer.

NOT MUST-FIX: ADR copy nits, requests for new ADRs, stylistic
preferences.

## Sources
- `docs/LANE_LEDGER.md` (which lanes own which surfaces)
- `CLAUDE.md`
- `docs/adr/ADR-001-onnx-runtime.md` through `ADR-005-packaging.md`
- Current code (read what each ADR makes a claim about)

## Investigate
For each ADR, identify the claim → grep/read the code that should
embody it → record agreement or drift with file:line evidence.

## Output
Write a recommendations-only report at
`_diag/2026-06-03/audit_adrs_report.md`:
- Per ADR: status (CURRENT / DRIFT / SUPERSEDED-BY-X / TECHNICAL-DEBT)
  with cited evidence.
- For each DRIFT: which lane owns the resolution, what the recommended
  fix shape looks like.
- Mandatory five-bullet self-audit gate.

Commit the report on a topic branch (no code edits expected). If you
DO end up fixing something, scope it tightly and document why it
couldn't be deferred to the owning lane.
