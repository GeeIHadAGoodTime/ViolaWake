# HETEROGENEOUS SC RE-AUDIT

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-het-sc-audit
Worktree off `master`. Don't touch master, don't push, don't merge.

> **This is a RECOMMENDATIONS-ONLY audit.** No code edits, no prompt
> edits. You produce a report only.

## Mission
*The orchestrator (Claude) did the SC self-audit on its own dispatch
prompts and on the corrections file. CLAUDE.md says the SC audit
should be done by a HETEROGENEOUS agent. You are that heterogeneous
agent (codex GPT-5.5 vs the Claude orchestrator).*

*Apply the same SC audit binary verdict bar to the orchestrator's
work as the orchestrator applied to itself, and surface anything the
orchestrator missed.*

## The bar (binding — same as the orchestrator's)
A dispatch prompt's SC is PASS iff all six hold:

1. **Catches plausibly broken implementations.** A codex doing
   shallow work and claiming PASS gets caught.
2. **Probes are realistic broken shapes.**
3. **Baseline is runnable with documented resources.** PASS can be
   established without missing creds/files/infra.
4. **Lane file ownership doesn't overlap another lane.**
5. **SC doesn't force work that belongs to another lane.**
6. **Reviewer's binary question is not trivially gameable.**

MUST-FIX if any of 1–6 fails. NOT MUST-FIX: stylistic phrasing,
additional scope ideas, completeness nits.

Cap: this is round 2 of the SC audit (round 1 was the orchestrator's
self-audit). Per CLAUDE.md, no further rounds — surface what you find
and STOP; don't loop.

## Artifacts to audit
- `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` — the
  orchestrator's round-1 corrections (the orchestrator's claimed
  fixes to its own SCs)
- `_diag/2026-06-03/dispatch_*.md` — the dispatch prompts for each
  lane audit (after the corrections were appended)
- `docs/LANE_LEDGER.md` — the lane definitions the dispatch prompts
  reference
- `CLAUDE.md` — the project rules

## What to look for (open lens — these are EXAMPLES, not a checklist)
- An SC that looks strong but a codex could narrow-interpret to PASS
- A probe shape the orchestrator named but didn't tell codex to
  CONSTRUCT (corrections file § A1 was meant to fix this — check it
  did, comprehensively)
- A baseline that requires resources I (codex) cannot access
  (creds, deleted files, missing infra)
- A file the orchestrator put in two lanes' "Owns" blocks
- An SC that forces work that belongs to a different lane's bounded
  context
- Subtle ways "MUST-FIX = plausibly broken passes" could be gamed
- Anything the orchestrator's self-audit missed because of self-
  validation bias

Don't enumerate "look for X, Y, Z" — find every gap, at any layer,
at any severity, zero is the bar.

## Output
Write `_diag/2026-06-03/audit_het_sc_report.md` with:
- Per lane (l1-wake, l2-companions, l3-wasm, l4-training, l5-eval,
  l6-cli, l7-distro, l8-backend, l9-frontend, l10-devops,
  l11-marketing) + the corrections file itself: PASS or MUST-FIX with
  cited evidence.
- Aggregate verdict: did the orchestrator's round-1 fixes hold the
  bar?
- Mandatory five-bullet self-audit gate (what you didn't probe and
  why).

Commit on this worktree's branch. NO edits to any other artifact.
