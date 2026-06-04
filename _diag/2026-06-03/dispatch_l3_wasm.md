# AUDIT — Lane 3: Browser Wake Detection (WASM)

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l3-wasm
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. If no parity tolerance is documented in repo, default
> to score-space L_inf ≤ 1e-3 and call it out in the report.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 3):
*"Does the in-browser TypeScript detector produce the same scores as
the Python SDK on the same audio?"*

## Success criteria — binary verdict
PASS = WASM and Python agree to within the documented tolerance on a
shared audio subset; bundle size and first-detection latency stay under
their documented bars; live smoke test green.

MUST-FIX = parity disagreement beyond tolerance on real audio; bundle
load fails on the deployed site; the documented audio contract isn't
honored on the JS side (frame stride, sample rate, normalization).

## Sources
- `docs/LANE_LEDGER.md` § 3
- `CLAUDE.md`
- Files owned by this lane (`wasm/`, `console/frontend/dist/wasm/`,
  `tests/live/test_live_wasm.py`)
- Live site: `https://violawake.com` (the deployed bundle reaches the
  user via this surface)

## Investigate
Run the detector — both in Node (against the build) and against the
deployed page. Compare scores against the Python SDK on a shared
corpus subset (10 samples is enough to surface contract drift). Find
every gap, at any layer. Zero is the bar.

## Decide & implement
One topic branch. `Ratchet:` for class-level fixes (gate that catches
the disagreement shape), `Ratchet-Exempt:` for single-instance. Don't
push, don't merge.

## Prove it
For parity: a script + its output showing scores within tolerance.
Bundle metrics: actual byte size + actual measured latency.

## Report
`_diag/2026-06-03/audit_lane_03_report.md`. Verdict + fixes +
MANDATORY self-audit gate (five bullets).

## Scaffolding
- Python SDK is the reference; WASM must match it, not vice versa.
- Audio contract is the same: 16 kHz mono, 20 ms frames, OWW 96-dim.
- A 10-sample shared corpus is sufficient to detect contract drift.
