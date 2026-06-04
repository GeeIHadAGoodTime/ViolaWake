# AUDIT — Lane 1: Wake Detection

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l1-wake
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> It carries BINDING corrections that override anything in this prompt
> they contradict — especially § A (common to all lanes) and § B1
> (Lane 1 specific: a stronger PASS clause).

## Mission
The lane's capability question (from `docs/LANE_LEDGER.md` § 1):
*"Given live audio, does the SDK detect the configured wake word — and
reject everything else — at the documented threshold, on the documented
audio contract?"*

## Success criteria — binary verdict
PASS = the lane's success criteria and oracle SC (from the ledger) hold
on the current trunk.

MUST-FIX = an item where a plausibly real user or customer claim already
breaks, today, on the deployed/published artifacts. Examples of
MUST-FIX: a public number that doesn't reproduce; the 4-gate decision
policy is bypassable; the audio contract drifts silently between
training and inference; a confusable in the documented set actually
fires.

NOT MUST-FIX: stylistic preferences, "could be cleaner," hypothetical
edge cases without evidence, missing future features, documentation
nits.

## Sources — read the source, not a summary
- `docs/LANE_LEDGER.md` § 1 (the lane spec — your scope)
- `CLAUDE.md` (project rules, especially Investigation Discipline,
  Ratchet Rule, AGENT DISPATCH PRINCIPLES)
- Files this lane owns (see ledger § 1 "Owns")
- `docs/adr/ADR-001-onnx-runtime.md`, `ADR-002-oww-feature-extractor.md`
- `docs/PROVEN_TRAINING_RECIPE.md` (the inference-contract half)
- `benchmark_v2/BENCHMARK_REPORT_v2.md` for the public numbers this
  lane stands behind

## Investigate
Exercise the actual capability — don't grade the code from a desk.
Find every gap, at any layer, at any severity. Don't stop until
exhausted. Zero is the bar. Treat as if seeing for the first time; if
something looks well-tested, dig hardest there. Default fallbacks that
hide bugs are exactly the bugs.

## Decide & implement
For each MUST-FIX you find: fix it on a topic branch in this worktree,
one commit per fix. Class-level fixes (the failure was POSSIBLE because
of a structural enabler) ship a `Ratchet:` gate-id in the same commit;
update `quality/gates.yaml` (create the file if it doesn't exist).
Single-instance fixes use `Ratchet-Exempt: <enum>` per the CLAUDE.md
enum.

Do NOT push. Do NOT merge to master. Do NOT modify `CLAUDE.md` or
`docs/LANE_LEDGER.md`. Do NOT touch files outside this lane's "Owns"
glob in the ledger.

## Prove it
For each fix: actual command output and file:line evidence that the
verifying run passes. "Tests passed" is not evidence — show the output.

## Report
Write `_diag/2026-06-03/audit_lane_01_report.md`:
- Binary verdict (PASS, or `MUST-FIX:` followed by the list)
- Per fix: the gap, the file:line, evidence, commit SHA
- MANDATORY self-audit gate: "Before declaring complete, list five
  surfaces / failure modes / questions / tradeoffs you did NOT
  exhaustively probe, and explain why." Answer in five bullets.

## Operational scaffolding (not lens-boxing — these unblock work)
- Audio contract: 16 kHz mono, 20 ms frames (320 samples), OWW 96-dim
  embeddings, default threshold 0.80 (was 0.50 — see "Don't manufacture
  accuracy claims" in CLAUDE.md).
- Published numbers under this lane's claim: d'=8.577 / EER=0.8% from
  production eval set; EER=5.49% from `benchmark_v2/`.
- Reference model: the SHA pinned in `src/violawake_sdk/models.py`.
