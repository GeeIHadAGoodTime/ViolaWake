# AUDIT — Lane 11: Marketing & Developer Docs

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l11-marketing
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. If `scripts/generate_docs.py` produces a giant diff vs
> committed HTML, document the gap — don't commit the regen unless the
> source truly drifted.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 11):
*"Does every outward-facing artifact — README, PyPI description,
generated API docs, Show-HN draft, SEO content — match the current
state of the product and reproduce its claims?"*

## Success criteria — binary verdict
PASS = (a) every numeric claim in `README.md`, the comparison pages,
and the PyPI description traces to a named script + corpus in Lane 5;
(b) API docs (`docs/api/` HTML) regenerate from current source without
diff; (c) `docs/REGISTRY.md` lists every authoritative doc and nothing
authoritative lives outside it; (d) the public copy follows the
"How public copy is written" rules in CLAUDE.md.

MUST-FIX = (a) a numeric claim on a public page has no Lane 5
reproducer; (b) a public symbol exists in the SDK but is missing from
API docs; (c) a public page contains the forbidden patterns from
CLAUDE.md "How public copy is written" (meta-process narration, dated
Corrections, Self-Certification footers, "Not Offered" paragraphs,
links to internal review docs).

NOT MUST-FIX: copy phrasing preferences, hypothetical future content.

## Sources
- `docs/LANE_LEDGER.md` § 11
- `CLAUDE.md` ("How public copy is written", "Don't manufacture
  accuracy claims")
- Files owned (ledger § 11 "Owns")
- The live pages on `https://violawake.com/` (use curl/fetch to read
  what users actually see — not the local source)
- `docs/COMPETITIVE_ANALYSIS.md`, `docs/PRD.md`,
  `docs/BUSINESS_PLAN.md`, `docs/SHOW_HN_DRAFT.md`

## Investigate
- For each numeric claim on a public page, find its origin script in
  Lane 5. If not found, that's a MUST-FIX.
- For each `docs/REGISTRY.md` entry, confirm the file exists. For
  each authoritative doc, confirm it's in the registry.
- Grep public copy for the CLAUDE.md forbidden patterns.
- Diff API doc HTML against current source.

Find every gap. Zero is the bar.

## Decide & implement
One topic branch. Each fix is one commit. `Ratchet:` for class fixes
(e.g. CI gate that greps public copy for un-cited numbers, or for the
forbidden meta-process phrases). `Ratchet-Exempt:` for single edits.

Do NOT push, do NOT merge.

## Prove it
For each fix: the before/after copy + the cited reproducer (where
relevant).

## Report
`_diag/2026-06-03/audit_lane_11_report.md`. Verdict + fixes +
MANDATORY self-audit gate.

## Scaffolding
- The live site is `https://violawake.com/`. Read the live HTML,
  not the local `console/frontend/src/`.
- The comparison page: `https://violawake.com/compare/picovoice`.
- Lane 11 owns the DOCS; Lane 5 owns the BENCHMARK that produces the
  numbers. If a number is wrong, the fix is usually in Lane 5; in
  Lane 11 the fix is removing/correcting the claim.
