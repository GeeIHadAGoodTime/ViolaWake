# BUILD — Quality-gate framework bootstrap

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-gates-bootstrap
Worktree off `master`. Don't touch master, don't push, don't merge.

> **READ FIRST:** `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`.
> § A applies. This is the work § A2 referenced: bootstrap the
> `quality/gates.yaml` framework and the mechanical-enforcement
> scripts CLAUDE.md describes, so the running audit lanes can append
> gate entries via the orchestrator's integration commit.

## Mission
Bootstrap the gate framework CLAUDE.md mandates:

- `quality/gates.yaml` — the gate registry. Schema per CLAUDE.md
  Ratchet Rule: `gate_id`, contract, detector script reference,
  `own_tests` list. The file starts with a header and an empty
  registry; lanes add entries via orchestrator integration commits.
- `scripts/check_no_direct_main_commits.py` — the pre-commit hook per
  CLAUDE.md "Worktree Isolation - Mechanical enforcement": refuses
  any non-merge commit from the main checkout regardless of branch.
  Merges with `MERGE_HEAD` present are allowed.
- `scripts/check_ratchet_rule.py` — the pre-commit hook per
  CLAUDE.md "Ratchet Rule - Mechanical enforcement": on fix-like
  commits (`fix|hotfix|security|incident|hardening|regression`)
  touching production code, require either `Ratchet: <gate-id>`
  (gate-id must exist in `quality/gates.yaml` AND commit must touch
  the gate's surface) or `Ratchet-Exempt: <closed-enum-reason>`
  (enum values: `docs-only`, `external-dep-bump`,
  `single-instance-data`, `revert-related`).
- `.githooks/pre-commit` — wires both checkers. Document how to
  install (`git config core.hooksPath .githooks` in README's dev
  setup).
- A CI workflow (`.github/workflows/gates.yml`) that runs both
  checkers on every PR.

## Success criteria — binary verdict
PASS = all five artifacts above exist and:
(a) `check_no_direct_main_commits.py` REJECTS a non-merge commit from
    the main checkout (construct + run this probe);
(b) `check_no_direct_main_commits.py` ALLOWS a non-merge commit from
    a worktree (construct + run);
(c) `check_no_direct_main_commits.py` ALLOWS a merge commit from the
    main checkout (construct + run);
(d) `check_ratchet_rule.py` REJECTS a fix-like commit touching
    production code with no Ratchet trailer (construct + run);
(e) `check_ratchet_rule.py` ACCEPTS the same commit with
    `Ratchet-Exempt: docs-only` (construct + run);
(f) `check_ratchet_rule.py` REJECTS a `Ratchet-Exempt:` with a
    free-text reason (closed enum enforcement);
(g) `quality/gates.yaml` validates against its own documented schema.

MUST-FIX = any probe (a)–(g) does the wrong thing.

NOT MUST-FIX: future gate ideas, additional enum values, ergonomic
improvements.

## Sources
- `CLAUDE.md` ("Ratchet Rule" and "Worktree Isolation - Mechanical
  enforcement" sections)
- `docs/LANE_LEDGER.md`

## Investigate, decide, prove
Build the artifacts. For each PASS condition, construct the probe and
show its exact behavior. Cite commit SHAs.

## Output
Write `_diag/2026-06-03/audit_gates_bootstrap_report.md` with the
probe outputs + verdict + mandatory self-audit gate.

## Important
This worktree IS allowed to create `quality/gates.yaml` — the § A2
"do not create" rule is for the AUDIT lanes (so concurrent worktrees
don't conflict). This bootstrap lane is the canonical creator. The
schema and seed file you ship will be the one the orchestrator
appends to.
