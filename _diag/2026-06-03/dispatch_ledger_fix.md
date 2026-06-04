# GOVERNANCE — Resolve ledger disjointness violations

Environment: Windows 11, bash shell. Project root: J:\CLAUDE\PROJECTS\Wakeword-ledger-fix
Worktree off `master`. Don't touch master, don't push, don't merge.

## Mission
The heterogeneous SC re-audit
(`J:/CLAUDE/PROJECTS/Wakeword-het-sc-audit/_diag/2026-06-03/audit_het_sc_report.md`)
found 5 specific file-ownership overlaps in `docs/LANE_LEDGER.md`.
This is **governance work** (orchestrator's purview) — not a lane
audit and NOT SC iteration. Cap-at-2-rounds applies to SCs, not to
factual disjointness errors in the ledger.

Apply surgical edits to `docs/LANE_LEDGER.md` that resolve EACH
overlap. Each resolution should:

- Pick ONE owner lane (whichever is the more natural bounded context)
- Remove the file from the OTHER lane's "Owns" block
- If the rejected owner had load-bearing rationale, document
  the cross-lane dependency in the chosen owner's section

## The 5 overlaps to fix

Reference: `audit_het_sc_report.md` lines 51-153 in the het-sc worktree.

1. **`docs/archive/`** — currently in both Lane 11 (Marketing) and
   Lane 12 (Governance).
2. **`docs/PROVEN_TRAINING_RECIPE.md`** — currently split "by half"
   between Lane 1 (inference-contract half) and Lane 4 (training
   half). File-by-half ownership is invalid.
3. **`console/frontend/dist/wasm/`** — Lane 3 (WASM) owns this; Lane 9
   (Frontend) owns `console/frontend/dist/` which contains it.
4. **`docs/api/`** — Lane 11 (Marketing) wants regen-check on it;
   Lane 8 (Backend) owns it as generated FastAPI OpenAPI docs.
5. **`docs/ROADMAP_10_OF_10.md`** — currently in both Lane 4
   (Training) and Lane 12 (Governance).

## Constraints
- ONE commit on `post-audit-2026-06-03/ledger-fix`.
- Touch ONLY `docs/LANE_LEDGER.md`. No other files.
- Don't change SC text, capability questions, or success criteria —
  this is disjointness/ownership only.
- Don't push, don't merge to master.
- `Ratchet-Exempt: single-instance-data` in commit message.

## Output
Commit the fix. Write a short report at
`_diag/2026-06-03/ledger_fix_report.md` (also committed) summarising:
- Per-overlap: which owner you picked and why
- Any cross-lane dependency notes added
- Verification that no other overlaps exist (run a sanity check by
  expanding "Owns" globs against `git ls-files`)
- Mandatory five-bullet self-audit gate
