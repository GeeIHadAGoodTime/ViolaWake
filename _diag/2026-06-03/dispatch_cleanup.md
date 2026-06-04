# CLEANUP — Lane 12 housekeeping

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-cleanup
This is a worktree off `master`. Don't touch master, don't push.

## Mission
Resolve the cruft block in `docs/LANE_LEDGER.md` so every lane's file
ownership stays disjoint. This is mechanical scope work — NOT an audit.

## What to do
Read `docs/LANE_LEDGER.md` → "Cruft" section. It lists every stray file
and its disposition. Execute the dispositions:

- The ~170 stray `test_*.mp3` files at repo root, plus
  `test_long.mp3`, `test_short.mp3`, `test_model.config.json`,
  `test_model.onnx` → MOVE (don't delete — preserve evidence) into
  `_diag/2026-06-03/cleanup/old_tts_sweep/`. Use `git mv` to preserve
  history.
- `_write_wake_detector.py`, `diagnostic_embedding_analysis.py` (top
  level) → MOVE to `_diag/2026-06-03/cleanup/scratch/` via `git mv`.
- Empty file `python` at top level → DELETE via `git rm`.
- Top-level docs that belong in `docs/` per the ledger:
  `ACCURACY_MISSION.md`, `ADVERSARY_AUDIT.md`, `BUILD_VS_BUY_AUDIT.md`,
  `E2E_READINESS.md`, `FUNCTIONAL_GAP_ANALYSIS.md`, `LAUNCH_READINESS.md`,
  `PROGRESS.md` → `git mv` into `docs/`. If any internal link in the
  repo points at the old top-level path, update it in the same commit.
- Top-level `SECURITY.md` is a duplicate of `docs/SECURITY.md`. If
  contents are identical, `git rm` the top-level and add a one-line
  pointer file at top level pointing to `docs/SECURITY.md`. If contents
  differ, surface the diff in the report and do NOT delete.

## What NOT to touch
- `src/violawake/` compat shim (marked CONFIRM in the ledger — needs
  founder sign-off, leave alone).
- Anything under another lane's owned globs (use the ledger's "Owns"
  blocks as the boundary).
- `CLAUDE.md`, `docs/LANE_LEDGER.md`.

## How to commit
One commit per logical group (mp3 sweep, scratch scripts, empty python,
doc relocations, SECURITY resolution). Each commit message names what
moved/deleted and why; use `Ratchet-Exempt: single-instance-data` for
each (per CLAUDE.md Ratchet enum).

Do NOT push. Do NOT merge to master.

## Report
Write `_diag/2026-06-03/cleanup_report.md` with:
- For each disposition: what was moved/deleted, how many files, the
  commit SHA, any surprises.
- Confirmation that `git ls-files` count went down by the expected
  amount.
- Anything you noticed in passing that doesn't fit any lane (new cruft
  for the next pass).
- Self-audit (MANDATORY): list five housekeeping concerns you did NOT
  exhaustively probe (e.g. `.gitignore` quality, gitignored cruft in
  `dist/`, etc.) and explain why each was out of scope here.
