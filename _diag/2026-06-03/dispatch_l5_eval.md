# AUDIT — Lane 5: Evaluation & Benchmarking

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l5-eval
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. If a fix routes to Lane 11 (public copy edit), record
> the suggested diff in your report — don't edit Lane 11 files.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 5):
*"Are the public accuracy claims reproducible from this repo on the
corpora checked into this repo?"*

This is the **public claim instrument** — every number on the public
comparison pages, the README headline, and `COMPETITIVE_ANALYSIS.md`
must trace back to a script and a corpus in this lane. If a claim
can't be reproduced from this repo today, it's a MUST-FIX (either
reproduce it or remove the claim).

## Success criteria — binary verdict
PASS = (a) every headline number on a public page is reproducible by
running a named script in this lane against a corpus in this lane at a
pinned model SHA; (b) `benchmark_regression_check.py` runs cleanly on
the current trunk; (c) per-category FAR/FRR is published.

MUST-FIX = (a) a public number whose origin script can't be found; (b)
a script that no longer runs / no longer produces the number it
references; (c) a corpus referenced by a script that's missing or
silently changed; (d) a contamination between train and eval corpora.

NOT MUST-FIX: nit on copy phrasing, suggested-but-unmeasured
improvements, hypothetical edge cases.

## Sources
- `docs/LANE_LEDGER.md` § 5
- `CLAUDE.md` (especially "Don't manufacture accuracy claims")
- `README.md` (headline claims to verify)
- `docs/COMPETITIVE_ANALYSIS.md`
- `benchmark_v2/BENCHMARK_REPORT_v2.md` and the scripts there
- Files owned by this lane (ledger § 5 "Owns")

## Investigate
For each public number, find its origin script. Run that script
yourself, get the actual current output, compare to the claim. Find
every gap. Zero is the bar.

The four shapes that destroy public trust here: a number with no
script; a number whose script no longer produces it; a script whose
corpus has silently drifted; a benchmark that secretly shares code
with the SDK so a buggy SDK + a buggy benchmark agree.

## Decide & implement
One topic branch. For each reproducible-claim failure: either fix the
reproducer or open a fix PR with the claim removed/corrected. Each fix
is one commit. `Ratchet:` for the class fix (e.g. a CI gate that
greps public docs for un-cited numbers); `Ratchet-Exempt:` for a
single-claim correction.

Do NOT push, do NOT merge, do NOT modify `CLAUDE.md` or
`docs/LANE_LEDGER.md`.

## Prove it
For each verified claim: the command run + the actual output cited
inline.

## Report
`_diag/2026-06-03/audit_lane_05_report.md`. Verdict + fixes +
MANDATORY self-audit gate (five bullets, what you did NOT probe and
why).

## Scaffolding
- Public claims to verify (non-exhaustive — find others by reading
  README / public pages): d'=8.577 / EER=0.8%; EER=5.49% vs OWW alexa
  EER=8.24%; "98.7% accuracy" anywhere it appears.
- Audio contract: 16 kHz mono, 20 ms frames.
- Reference model: SHA-pinned in `src/violawake_sdk/models.py`.
