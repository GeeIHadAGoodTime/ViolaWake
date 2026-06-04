# Dispatch ledger — 2026-06-03 audit fleet

Authoritative record of codex sessions dispatched for the Wakeword
post-ship audit. Update on each state change.

## Methodology gates honored

- **CLAUDE.md "no audit without success criteria"**: every dispatch
  has an SC binary verdict bar written upfront.
- **CLAUDE.md "SC audit before building"**: orchestrator's
  self-audit landed in `SC_AUDIT_ROUND_1_CORRECTIONS.md`;
  heterogeneous codex SC re-audit dispatched as `het-sc-audit`.
- **CLAUDE.md "cap at 2 rounds"**: SC audit capped at round 2 PASS.
- **CLAUDE.md "Worktree Isolation"**: each session in its own
  worktree off `master`. Don't push, don't merge to master.

## Cohort

### Wave 1 (initial 10, dispatched ~14:13 local)

| # | Lane | Worktree | bg-id | Status |
|---|------|----------|-------|--------|
| 0 | Cleanup (cruft → _diag/) | `Wakeword-cleanup` | bg8b2mzpn | DONE rc=0 |
| 1 | Lane 1 Wake Detection | `Wakeword-l1-wake` | b8tfjk6nq | running |
| 2 | Lane 2 Companions | `Wakeword-l2-companions` | btxb3zg7h | running |
| 3 | Lane 3 WASM | `Wakeword-l3-wasm` | bs6zehnq8 | running |
| 4 | Lane 5 Evaluation | `Wakeword-l5-eval` | bfsi2uw80 | running |
| 5 | Lane 6 CLI | `Wakeword-l6-cli` | b3wrnivfq | running |
| 6 | Lane 7 Distribution | `Wakeword-l7-distro` | bhf3xxnp6 | running |
| 7 | Lane 8 Backend | `Wakeword-l8-backend` | byqh6540f | running |
| 8 | Lane 10 DevOps | `Wakeword-l10-devops` | b7z4x4lib | running |
| 9 | Lane 11 Marketing | `Wakeword-l11-marketing` | bt9oqjd4d | running |

### Wave 2 (extension for max throughput)

| # | Lane | Worktree | bg-id | Status |
|---|------|----------|-------|--------|
| 10 | Quality-gates bootstrap | `Wakeword-gates-bootstrap` | b5i4t0o3j | running |
| 11 | Heterogeneous SC re-audit | `Wakeword-het-sc-audit` | bvnavzyij | running |
| 12 | Lane 4 Training | `Wakeword-l4-training` | bkhdjegez | FAILED rc=127 (OOM); deferred for retry |
| 13 | Prior-audits sweep | `Wakeword-prior-audits-sweep` | ba3f1obpq → retry | FAILED rc=1 (fork); retried |
| 14 | ADR audit | `Wakeword-adr-audit` | (this turn) | dispatched |
| 15 | Lane 9 Frontend | `Wakeword-l9-frontend` | — | DEFERRED (npm install RAM-heavy; await drain) |

### Resource ceiling note

Local machine sustains ~11 parallel codex sessions before fork/OOM.
Strategy: defer RAM-heavy dispatches (L4 corpus loading, L9 npm
install) until 2–3 wave-1 sessions complete and free resources.
Lighter dispatches (doc-only audits) can be retried sooner.

## Reports expected

Each session writes to `_diag/2026-06-03/`:

- `cleanup_report.md`
- `audit_lane_NN_report.md` (NN = 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11)
- `audit_adrs_report.md`
- `audit_prior_findings_report.md`
- `audit_gates_bootstrap_report.md`
- `audit_het_sc_report.md`

## Merge sequence (after reports land)

Per CLAUDE.md "Convergence" — merge fix waves sequentially in
dependency order. Planned order:

1. `cleanup` (root-cruft relocation) — DONE; merge first to free
   subsequent worktree merges from path conflicts.
2. `gates-bootstrap` — lands the `quality/gates.yaml` framework so
   subsequent merges can append gate entries.
3. `het-sc-audit` — recommendations-only; informs whether to dispatch
   any SC round-3 (cap normally says no).
4. `adr-audit` + `prior-audits-sweep` — recommendations-only; route
   findings to owning lanes.
5. Lane audits in any order they finish (each disjoint), with the
   orchestrator running the per-merge spot-check from
   `feedback_per_wave_verification_protocol`.
6. Single integration commit landing `quality/gates.yaml` populated
   with each lane's planned-gate YAML blocks from its report.

## Done-bar (CLAUDE.md "Convergence")

5 simultaneously-true conditions:
1. All 11 reports landed, each with explicit PASS or MUST-FIX list.
2. Every MUST-FIX is either resolved on its lane's branch or
   surfaced as `BLOCKED — requires founder` with the founder loop
   closed.
3. The heterogeneous SC re-audit returns no new structural MUST-FIX
   on the corrections file.
4. All wave-1 + wave-2 branches merge cleanly to master.
5. `git ls-files | wc -l` matches lane ownership sum (disjointness
   structurally verified post-merge).
