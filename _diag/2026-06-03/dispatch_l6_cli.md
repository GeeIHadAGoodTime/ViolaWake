# AUDIT — Lane 6: SDK CLI & Sample Tools

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l6-cli
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies — especially § A1: construct and RUN the probe (e.g.
> remove a CLI entry from `[project.scripts]`, prove the verification
> path catches it).

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 6):
*"Can a user run `violawake-train`, `violawake-eval`,
`violawake-collect`, `violawake-download` and have each command do
what its `--help` says, on a clean install?"*

## Success criteria — binary verdict
PASS = on a fresh venv with `pip install -e .` (in this worktree),
each documented CLI command runs to its documented outcome; `--help`
matches the docs.

MUST-FIX = a CLI that crashes on documented invocation; a CLI whose
`--help` lies about its behavior; a CLI that silently produces no
output on success; the `examples/` scripts don't run unmodified after
`pip install "violawake[oww]"` + `download_models()`.

NOT MUST-FIX: cosmetic help-text wording, missing future flags.

## Sources
- `docs/LANE_LEDGER.md` § 6
- `CLAUDE.md`
- Files owned by this lane
- `pyproject.toml` `[project.scripts]` (the published entry points)
- `examples/basic_detection.py`, `async_detection.py`

## Investigate
Make a clean venv. Install. Run every documented CLI. Run the
examples. Find every gap. Zero is the bar.

## Decide & implement
One topic branch, one commit per fix. `Ratchet:` for class-level fixes
(e.g. a CI smoke that runs every entry point from `[project.scripts]`
in a fresh venv). `Ratchet-Exempt:` for single-CLI fixes. Don't push,
don't merge.

## Prove it
Actual venv creation log + actual command output for each verified
CLI.

## Report
`_diag/2026-06-03/audit_lane_06_report.md`. Verdict + fixes +
MANDATORY self-audit gate.

## Scaffolding
- Entry points are declared in `pyproject.toml` `[project.scripts]`.
- A clean venv: `python -m venv .venv-audit && .venv-audit\Scripts\activate
  && pip install -e .` from the worktree.
- The `[oww]` extra requires `download_models()` first-run.
