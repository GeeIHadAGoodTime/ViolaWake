# BUILD — Post-audit integration plan

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-integration-plan
Worktree off `master`. Don't touch master, don't push, don't merge.

## Mission
Read every landed audit report under `_diag/2026-06-03/` (they live in
sibling worktrees, NOT this one — look in
`J:/CLAUDE/PROJECTS/Wakeword-*/[_diag/2026-06-03/audit_*_report.md` and
`cleanup_report.md`). Produce ONE integration plan: `_diag/2026-06-03/
INTEGRATION_PLAN.md`.

The plan turns 15 disjoint audit reports into one sequenced merge
playbook the orchestrator can execute.

## What the plan must contain (open-ended; expand as the reports warrant)

- **Branch inventory:** every audit branch + its head SHA + each
  commit's subject. Use `git -C` to inspect each sibling worktree.
- **Planned-gate YAML rollup:** every lane that wrote a planned gate
  spec (per `SC_AUDIT_ROUND_1_CORRECTIONS.md` § A2) — collect them
  into ONE block ready to land in `quality/gates.yaml`. Resolve any
  id collisions.
- **Cross-lane fix routing:** lane X's report says "fix is in lane Y";
  collate these into a per-lane to-do list keyed by owner lane.
- **BLOCKED items:** which need founder authorization (test accounts,
  Stripe test mode, production credentials, etc.).
- **Live-prod breakage list:** which surfaces are broken right now
  (Lane 8 api 530, Lane 3 wasm CSP, Lane 9 SPA routes) + which lane
  branch has the candidate fix + what remains before deploy.
- **Suggested merge order:** dependency-aware sequence — gates first,
  then cleanup, then lane branches, then integration commit with the
  planned-gates YAML. Justify each ordering edge.
- **Conflicts to expect:** any two branches that touched the same
  file. List the file + the conflicting branches + a one-line
  resolution recommendation.

## Constraints
- READ-only across sibling worktrees; write only in this worktree.
- Don't edit any other worktree's files.
- Don't merge anything.
- Don't push.
- The orchestrator will use the plan to drive the actual integration.

## Output
Commit `_diag/2026-06-03/INTEGRATION_PLAN.md` on
`post-audit-2026-06-03/integration-plan`. Mandatory five-bullet
self-audit gate at the end (what you didn't probe, why).
