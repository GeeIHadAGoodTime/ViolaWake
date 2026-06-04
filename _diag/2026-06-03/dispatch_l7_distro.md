# AUDIT — Lane 7: Public API & Distribution

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-l7-distro
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies, and § B2 carries the pre-publish handling: if violawake
> isn't on PyPI yet, audit the locally-built wheel from `python -m
> build`, not a "pip install" that would fail by construction.

## Mission
Lane capability question (`docs/LANE_LEDGER.md` § 7):
*"Does `pip install violawake==<version>` give a user a working SDK
with the documented public API surface, the right models available via
`ModelCache`, and a CHANGELOG entry?"*

## Success criteria — binary verdict
PASS = (a) the latest published PyPI version installs clean on
Python 3.10/3.11/3.12 on at least one OS; (b) every `ModelSpec` in
`src/violawake_sdk/models.py` resolves (URL reachable, SHA matches);
(c) `from violawake_sdk import *` does not raise; (d) CHANGELOG is
current.

MUST-FIX = (a) `pip install violawake` fails on a clean venv; (b) a
`ModelSpec` URL is 404; (c) a `ModelSpec` SHA-256 mismatches the
actual artifact; (d) a public symbol was removed without a CHANGELOG
entry; (e) the wheel ships a private path as if public.

NOT MUST-FIX: missing future model entries, cosmetic README in PyPI
description.

## Sources
- `docs/LANE_LEDGER.md` § 7
- `CLAUDE.md` (especially "Don't manufacture accuracy claims" and
  "Three of them" launch-surface block)
- Files owned by this lane
- `pyproject.toml`
- `CHANGELOG.md`, `RELEASE_NOTES.md`
- `docs/adr/ADR-005-packaging.md`, `ADR-003-python-first.md`,
  `ADR-004-open-core.md`

## Investigate
- Pull the latest PyPI version (use `pip download violawake==<v>` to
  fetch the actual published wheel, not the source).
- Install it into a fresh venv. Import. Run the documented quick-start
  from the README.
- For every `ModelSpec` in `src/violawake_sdk/models.py`: HEAD the URL,
  download a byte range, verify the SHA-256 against the spec.
- Diff the actual public surface (`dir(violawake_sdk)`) against the
  documented one (API docs + README + ADRs).
- Diff CHANGELOG entries against the commit log since the last release.

Find every gap. Zero is the bar.

## Decide & implement
One topic branch, one commit per fix. `Ratchet:` for class fixes
(e.g. a CI step that validates every `ModelSpec` SHA + URL on every
PR). `Ratchet-Exempt:` for single fixes.

## Prove it
Command output for each verified step.

## Report
`_diag/2026-06-03/audit_lane_07_report.md`. Verdict + fixes +
MANDATORY self-audit gate.

## Scaffolding
- PyPI URL: `https://pypi.org/project/violawake/`
- `pip download violawake==<v> --no-deps -d /tmp/wheel` to fetch
  without installing.
- Audio contract canon lives in `src/violawake_sdk/_constants.py`.
