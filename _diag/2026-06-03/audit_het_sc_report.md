# Heterogeneous SC Re-audit Report

Date: 2026-06-03
Agent: Codex heterogeneous reviewer
Scope: recommendations-only SC re-audit of round-1 corrections plus lane dispatch prompts.

## Input provenance

The requested `_diag/2026-06-03/*` artifacts were not present in this audit worktree. I read the dispatch prompts and correction file from the primary checkout at `J:\CLAUDE\PROJECTS\Wakeword\_diag\2026-06-03` because that is where the files exist. This report is the only artifact written in `J:\CLAUDE\PROJECTS\Wakeword-het-sc-audit`.

Command evidence:

```text
git ls-files _diag docs/LANE_LEDGER.md CLAUDE.md
CLAUDE.md
docs/LANE_LEDGER.md

corrections presence by lane worktree
Wakeword-l1-wake: False
Wakeword-l2-companions: False
Wakeword-l3-wasm: False
Wakeword-l4-training: False
Wakeword-l5-eval: False
Wakeword-l6-cli: False
Wakeword-l7-distro: False
Wakeword-l8-backend: False
Wakeword-l9-frontend: False
Wakeword-l10-devops: False
Wakeword-l11-marketing: False
Wakeword-het-sc-audit: False
```

The SC bar in `CLAUDE.md` requires binary PASS/MUST-FIX, and says MUST-FIX includes cases where a plausibly broken implementation could pass, probes are unrealistic, baseline resources are impossible, file ownership overlaps, lane scope breaks another lane, or the reviewer question is gameable (`CLAUDE.md:817-824`). It also caps this at two rounds (`CLAUDE.md:829`).

## Aggregate verdict

MUST-FIX. The orchestrator's round-1 fixes did not hold the bar.

The biggest failure is delivery: every lane prompt says to read `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` first (for example `dispatch_l1_wake.md:6`, `dispatch_l8_backend.md:6`, `dispatch_l10_devops.md:6`), but that correction file is absent from every lane worktree. The corrections are not appended to the lane prompts; they are an external override file. A worker running from the lane worktree can therefore execute the pre-correction prompt and still plausibly claim PASS.

## Corrections File

Verdict: MUST-FIX.

MUST-FIX: The correction file is not delivered into the lane worktrees it is supposed to control. The file says it overrides dispatch prompts (`SC_AUDIT_ROUND_1_CORRECTIONS.md:4`), adds the core "construct and run probes" rule (`SC_AUDIT_ROUND_1_CORRECTIONS.md:19-36`), and forbids `quality/gates.yaml` edits (`SC_AUDIT_ROUND_1_CORRECTIONS.md:42-63`). The command evidence above shows the file is absent from every lane worktree, so those corrections are not operationally binding.

MUST-FIX: The corrections leave a production-write conflict unresolved. A3 forbids writing to a production database (`SC_AUDIT_ROUND_1_CORRECTIONS.md:65-82`), while B3 still allows a full sign-up -> training -> download probe if a test account exists (`SC_AUDIT_ROUND_1_CORRECTIONS.md:153-154`). That full flow writes production state unless staging is explicitly required. A Lane 8 worker could either block too much or write prod state while believing B3 authorizes it.

MUST-FIX: B4 requires read-only inspection of live Cloudflare tunnel config and DNS routes (`SC_AUDIT_ROUND_1_CORRECTIONS.md:173-181`) but the dispatch does not document a credential-free command or authorized read-only credential path. PASS for Lane 10 can depend on missing external access.

## l1-wake

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 1's prompt depends on the correction file (`dispatch_l1_wake.md:6`), but that file is absent from the lane worktree.

MUST-FIX: Without the missing B1 override, the original prompt still treats "a public number that doesn't reproduce" as a Lane 1 MUST-FIX (`dispatch_l1_wake.md:21-24`) and sources benchmark public numbers (`dispatch_l1_wake.md:39-40`). The correction explicitly says to remove that example because it overlaps Lane 5 (`SC_AUDIT_ROUND_1_CORRECTIONS.md:124-126`). Because the override is not available in the lane worktree, the prompt can force Lane 1 into Lane 5's bounded context.

MUST-FIX: Lane ownership is not file-disjoint for the key training recipe doc. Lane 1 owns `docs/PROVEN_TRAINING_RECIPE.md` "the inference-contract half" (`docs/LANE_LEDGER.md:134-166`), while Lane 4 owns the same file's "training half" (`docs/LANE_LEDGER.md:283-310`). File ownership by "half" fails the file-disjoint lane bar.

## l2-companions

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 2 depends on the correction file for A1 probe construction (`dispatch_l2_companions.md:6-8`), but that file is absent from the lane worktree.

MUST-FIX: The baseline is not runnable from documented resources. The ledger requires Kokoro first-audio latency on "the reference hardware" and STT on "a fixed WAV" (`docs/LANE_LEDGER.md:228-242`), while the dispatch says to run real audio and measure latency (`dispatch_l2_companions.md:32-35`) without specifying the reference hardware or fixed WAV path. A worker can choose an ad hoc machine/audio sample and still claim PASS.

## l3-wasm

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 3 depends on the correction file (`dispatch_l3_wasm.md:6`), but that file is absent from the lane worktree.

MUST-FIX: Ownership overlaps Lane 9. Lane 3 owns `console/frontend/dist/wasm/` (`docs/LANE_LEDGER.md:258`), while Lane 9 owns `console/frontend/dist/` (`docs/LANE_LEDGER.md:549`). A generated WASM asset under `console/frontend/dist/wasm/` is in both lanes.

MUST-FIX: The baseline names resources/bars but not their concrete source. The ledger requires documented bundle-size and first-detection latency bars plus a 10-sample corpus (`docs/LANE_LEDGER.md:262-272`), and the dispatch makes those part of PASS (`dispatch_l3_wasm.md:15-18`), but the prompt does not name the bar values or the corpus path. A worker can invent the subset or skip the undocumented bars.

## l4-training

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 4 depends on the correction file (`dispatch_l4_training.md:6`), but that file is absent from the lane worktree.

MUST-FIX: The dispatch narrows away the lane's real SC. It says a full from-scratch retrain is not required and absence is not a MUST-FIX (`dispatch_l4_training.md:7-14`). The ledger's success criteria require a from-scratch retrain reaching documented d'/EER and same-seed reproducibility (`docs/LANE_LEDGER.md:314-316`). A pipeline that cannot train a model reproducibly could pass this prompt.

MUST-FIX: Ownership overlaps Governance and Lane 1. Lane 4 owns `docs/PROVEN_TRAINING_RECIPE.md` and `docs/ROADMAP_10_OF_10.md` (`docs/LANE_LEDGER.md:308-310`); Lane 1 also owns part of the training recipe (`docs/LANE_LEDGER.md:166`), and Governance also owns `docs/ROADMAP_10_OF_10.md` (`docs/LANE_LEDGER.md:697-711`).

## l5-eval

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 5 depends on the correction file (`dispatch_l5_eval.md:6`), but that file is absent from the lane worktree.

MUST-FIX: The dispatch can still force public-copy edits outside the evaluation lane. It says that for each reproducible-claim failure the worker should fix the reproducer or open a fix PR with the claim removed/corrected (`dispatch_l5_eval.md:53-58`), while Lane 11 owns outward-facing docs like README and public developer docs (`docs/LANE_LEDGER.md:653-668`). The prompt header tries to route Lane 11 fixes to the report (`dispatch_l5_eval.md:6-8`), but the later "Decide & implement" section contradicts that boundary.

## l6-cli

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 6 depends on the correction file for the CLI-entry negative probe (`dispatch_l6_cli.md:6-8`), but that file is absent from the lane worktree.

MUST-FIX: The SC forces a Lane 7-owned file. Lane 6's oracle SC says a removed CLI entry in `pyproject.toml` must fail (`docs/LANE_LEDGER.md:424-426`), and the dispatch sources `pyproject.toml` as the published entry point table (`dispatch_l6_cli.md:29-34`). But `pyproject.toml` is owned by Lane 7 (`docs/LANE_LEDGER.md:442-450`), while Lane 6 owns only CLI/tool modules, tests, and examples (`docs/LANE_LEDGER.md:401-415`). A real entry-point fix belongs to another lane.

## l7-distro

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 7 depends on the correction file for pre-publish handling (`dispatch_l7_distro.md:6-9`), but that file is absent from the lane worktree.

MUST-FIX: The platform baseline is weakened. The ledger baseline requires the latest published version to install on Python 3.10, 3.11, and 3.12 across Linux, Windows, and macOS (`docs/LANE_LEDGER.md:480-481`). The dispatch PASS only requires Python 3.10/3.11/3.12 "on at least one OS" (`dispatch_l7_distro.md:18-19`). A wheel broken on macOS or Linux could pass.

MUST-FIX: The ModelSpec SHA probe is gameable as written. The dispatch says to HEAD each URL, download a byte range, and verify SHA-256 (`dispatch_l7_distro.md:48`). A byte range cannot verify the SHA-256 of the full artifact. A truncated or corrupt object could pass a superficial range check.

## l8-backend

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 8 depends on B3 for test-account fallback (`dispatch_l8_backend.md:6-8`), but that file is absent from the lane worktree.

MUST-FIX: The live-flow baseline conflicts with the no-production-write correction. The dispatch requires full sign-up -> training-job -> model-download on a live integration run (`dispatch_l8_backend.md:20-22`) and instructs the worker to trace sign-up -> training -> download on a test account (`dispatch_l8_backend.md:45-52`). A3 forbids writing to a production database (`SC_AUDIT_ROUND_1_CORRECTIONS.md:65-82`). The SC does not require staging, so PASS cannot be established safely.

MUST-FIX: Billing test-account resources are not an established baseline. The live billing docs say Stripe is in LIVE MODE and real-card checkout is unverified (`docs/PRODUCTION_STATUS.md:28-40`), while the live test docs only enable checkout-card tests under `VIOLAWAKE_STRIPE_TEST_MODE=1` with test-mode keys (`tests/live/README.md:41-53`). The dispatch says to use test accounts but does not document that such an account or test-mode backend exists (`dispatch_l8_backend.md:49-52`, `dispatch_l8_backend.md:79`).

## l9-frontend

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 9 depends on the correction file (`dispatch_l9_frontend.md:6-8`), but that file is absent from the lane worktree.

MUST-FIX: Ownership overlaps Lane 3. Lane 9 owns `console/frontend/dist/` (`docs/LANE_LEDGER.md:549`), which contains Lane 3's `console/frontend/dist/wasm/` ownership (`docs/LANE_LEDGER.md:258`).

MUST-FIX: The browser baseline is weakened. The ledger requires latest stable Chrome, Firefox, and Safari (`docs/LANE_LEDGER.md:565-566`), while the dispatch only requires latest stable Chrome and allows a curl/grep fallback (`dispatch_l9_frontend.md:25-29`). A Firefox- or Safari-only render failure could pass.

## l10-devops

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 10 depends on B4 for the no-live-deploy rewrite (`dispatch_l10_devops.md:6-8`), but that file is absent from the lane worktree.

MUST-FIX: Without the unavailable correction, the original PASS still requires a live deploy landing the expected image SHA on the live URL (`dispatch_l10_devops.md:16-18`). That directly conflicts with B4's no-`up -d` replacement (`SC_AUDIT_ROUND_1_CORRECTIONS.md:167-181`) and lets a worker either touch production or fail an otherwise correct non-prod audit.

MUST-FIX: Even with B4, PASS can require missing external access. B4 requires read-only inspection of Cloudflare tunnel config and DNS routes (`SC_AUDIT_ROUND_1_CORRECTIONS.md:178-180`), while the ledger success criteria include deploy, backup restore, and CI green state (`docs/LANE_LEDGER.md:623-637`). The dispatch does not document the authorized read-only Cloudflare/GitHub/R2 credential path, so baseline PASS can depend on unavailable infra.

## l11-marketing

Verdict: MUST-FIX.

MUST-FIX: Universal correction-delivery failure applies. Lane 11 depends on the correction file (`dispatch_l11_marketing.md:6`), but that file is absent from the lane worktree.

MUST-FIX: The SC forces an API-doc ownership conflict. Lane 11 PASS requires API docs under `docs/api/` to regenerate without diff (`dispatch_l11_marketing.md:18-21`; ledger repeats this at `docs/LANE_LEDGER.md:670-674`), but Lane 8 owns `docs/api/` as generated FastAPI/OpenAPI docs (`docs/LANE_LEDGER.md:495-513`). A Lane 11 worker can be forced into a Lane 8 file.

MUST-FIX: Ownership overlaps Governance. Lane 11 owns `docs/archive/` (`docs/LANE_LEDGER.md:653-668`), and Governance also owns `docs/archive/` with a note to resolve it (`docs/LANE_LEDGER.md:697-705`). The ledger itself acknowledges the overlap, so the disjoint ownership bar fails.

## Mandatory Self-audit Gate

- I did not execute the lane agents' implementation probes. This was a recommendations-only SC audit, and the prompt explicitly prohibited code/prompt edits beyond the report.
- I did not read the multi-megabyte `dispatch_*.out` logs exhaustively. The requested audit artifacts were the correction file, dispatch prompts, lane ledger, and CLAUDE.md; worker output logs are downstream evidence, not the SC text under review.
- I did not log into Cloudflare, Stripe, GitHub, PyPI, or R2 dashboards. The SC issue is that the prompts require or imply those resources without a documented read-only path; using private dashboards would mask the baseline-resource problem.
- I did not mechanically expand every lane glob over `git ls-files`. The visible ledger overlaps cited above are enough to fail the binary disjointness bar; a full glob expander would be useful for cleanup but is not needed to establish MUST-FIX.
- I did not produce verdicts for non-lane dispatch prompts such as cleanup, ADR audit, gates bootstrap, prior-audits sweep, or this heterogeneous audit prompt. The required output scope was l1 through l11 plus the corrections file.
