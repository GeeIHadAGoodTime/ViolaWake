# Ledger Disjointness Fix Report

Source audit: `J:/CLAUDE/PROJECTS/Wakeword-het-sc-audit/_diag/2026-06-03/audit_het_sc_report.md`, lines 51-153.

## Per-overlap resolutions

1. `docs/archive/`
   - Owner picked: Lane 11, Marketing & Developer Docs.
   - Why: the tracked archive contains superseded benchmark/eval/public documentation registered from the docs surface, not governance process artifacts.
   - Dependency note added: Lane 12 may cite `docs/archive/`, but process and audit archives live in `_diag/`.

2. `docs/PROVEN_TRAINING_RECIPE.md`
   - Owner picked: Lane 4, Training & Augmentation.
   - Why: the file is the canonical training recipe and cannot be split by "inference-contract half" versus "training half."
   - Dependency note added: Lane 1 consumes the inference-contract portions and needs review on contract-affecting recipe changes.

3. `console/frontend/dist/wasm/`
   - Owner picked: Lane 3, Browser Wake Detection (WASM).
   - Why: the path is the WASM build output for the browser detector, while Lane 9 owns the React/Vite frontend around it.
   - Dependency note added: Lane 9 consumes the WASM assets through the frontend bundle, but `console/frontend/dist/wasm/` remains Lane 3-owned.

4. `docs/api/`
   - Owner picked: Lane 8, SaaS Console Backend.
   - Why: the path is generated FastAPI/OpenAPI backend documentation, not marketing copy.
   - Dependency note added: Lane 11 may verify regeneration as part of doc-sync/public-claim checks, but generated files and the OpenAPI source contract stay Lane 8-owned.

5. `docs/ROADMAP_10_OF_10.md`
   - Owner picked: Lane 12, Project Governance & Process.
   - Why: the file is a multi-subsystem product roadmap covering SDK, console, pricing, testing, documentation, packaging, and launch status, not only training/eval.
   - Dependency note added: Lane 4 depends on the training/eval milestones, but the roadmap remains Governance-owned for coordinated cross-lane updates.

## Verification

Ran a sanity check that parsed every `Owns` block in `docs/LANE_LEDGER.md`, expanded file and glob ownership against `git ls-files`, and reported duplicate tracked-file ownership across lanes.

Result:

```text
OK: 197 Owns entries expanded across 556 tracked files; no cross-lane duplicate tracked-file owners found.
```

## Self-audit gate

- [x] All five audit-reported overlaps were resolved in `docs/LANE_LEDGER.md`.
- [x] Each overlap now has exactly one owner lane.
- [x] Rejected ownership entries were removed from `Owns` blocks without editing SC text, capability questions, or success criteria.
- [x] Load-bearing rejected-owner rationale was preserved as cross-lane dependency notes in the retained owner's section.
- [x] No push, no merge to master; this report and the ledger edit are ready for one ledger-fix commit.
