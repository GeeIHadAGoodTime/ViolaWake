# Prior Audit Findings Sweep - Recommendations Only

Date: 2026-06-03
Branch: `audit-2026-06-03/prior-audits-sweep`
Mode: recommendations only. No lane-owned source files were edited.

## Scope And Path Handling

- Required correction note read: `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md`. Section A was treated as binding: no gate edits, no production writes, no deploy/restart, no push/merge/tag, and evidence stays inline.
- Source doc resolution followed the path-uncertainty rule. `docs/AUDIT_2026_03_28.md` and `docs/PRE_LAUNCH_CHECKLIST.md` were used from `docs/`; the other requested source docs were present at repo root. No requested source doc existed in both locations in this worktree.
- Lane routing uses `docs/LANE_LEDGER.md`. Relevant owners: Lane 1 wake detector, Lane 4 training pipeline, Lane 5 evaluation/benchmarking, Lane 7 packaging/release, Lane 8 backend/API, Lane 9 frontend/console, Lane 10 infrastructure/DevOps, Lane 11 docs/legal/copy, Lane 12 housekeeping.
- Read-only live checks were used where launch/readiness findings depended on current public state.

## Evidence Commands

- `git status --short --branch` returned `## audit-2026-06-03/prior-audits-sweep`; master was not checked out.
- `curl -L -s -o NUL -w "%{http_code}" https://api.violawake.com/api/health` returned `530` on 2026-06-03. `curl -L -s -o NUL -w "%{http_code}" https://api.violawake.com/openapi.json` also returned `530`.
- `curl -L -s -o NUL -w "%{http_code}" https://violawake.com/` returned `200`; `robots.txt`, `sitemap.xml`, and `og-image.png` also returned `200`.
- `python -m pip index versions violawake` reported latest published package `violawake (0.2.6)`.
- `git tag --list v0.2.2 v0.2.6` showed both tags. The GitHub release API for `v0.2.6` returned `200` and showed wheel/sdist assets.
- `git ls-files "test_*.mp3" "*.mp3"` returned no tracked root MP3 test artifacts.

## Aggregated MUST-FIX Recommendations

These are the remaining P0/P1 OPEN findings from the prior audit set. Some are already conceptually owned by the ledger, but they still need lane execution and proof before they can be closed.

| ID | Finding | Source | Current Classification | Owner Lane | Recommendation |
|---|---|---|---|---|---|
| MF-1 | Production API health/readiness is currently unavailable. This blocks health checks, checkout, login, training, and API-doc verification. | `LAUNCH_READINESS.md:66`, `docs/PRE_LAUNCH_CHECKLIST.md:70`, `docs/PRE_LAUNCH_CHECKLIST.md:88` | OPEN P0 | Lane 10 primary, Lane 8 affected | Restore the API/tunnel/backend path, then rerun read-only `/api/health` and `/openapi.json` probes. Current command evidence: both returned HTTP `530`; `docs/DEPLOYMENT.md:231` documents `530`/tunnel-down as an operations failure mode. |
| MF-2 | Real-speech and real speech-negative accuracy proof is not closed. The old contaminated eval was superseded, but the current public benchmark still caveats synthetic/TTS positives and the ledger still calls for production oracle probes. | `ACCURACY_MISSION.md:58`, `ACCURACY_MISSION.md:521`, `docs/LANE_LEDGER.md:127` | OPEN P1 | Lane 5 primary, Lane 1 affected | Run and publish a clean real-speaker plus real speech-negative oracle artifact with thresholds, per-category FAR, and production SDK path parity. |
| MF-3 | Batch scoring and production SDK scoring parity remains unproved in the audit trail. | `ACCURACY_MISSION.md:528`, `docs/LANE_LEDGER.md:183` | OPEN P1 | Lane 1 primary, Lane 5 affected | Add a lane-owned parity proof that the SDK, batch benchmark, and live detector consume the same model/backbone/policy and produce matching scores on fixtures. |
| MF-4 | The prior progress claim that all Playwright tests are green in CI is not supported by the current workflow. Backend CI ignores `console/tests/e2e`, and the local E2E runner installs Chromium only. | `PROGRESS.md:176`, `.github/workflows/ci.yml:133`, `console/run_e2e.py:35`, `docs/LANE_LEDGER.md:566` | OPEN P1 | Lane 9 primary, Lane 10 affected | Wire browser E2E into CI for the ledger-required browser set, or downgrade the readiness claim until lane proof exists. |

## ADVERSARY_AUDIT.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| F1/P6: `InferenceBackend.is_available()` missing from base interface. | RESOLVED | Base interface and implementations exist at `src/violawake_sdk/backends/base.py:124`, `src/violawake_sdk/backends/onnx_backend.py:95`, and `src/violawake_sdk/backends/tflite_backend.py:326`. README now documents the method at `README.md:686`. | Lane 1. No action. |
| F2: `backend.load(num_threads=...)` docs wrong or unsupported. | RESOLVED | TFLite backend accepts `**kwargs` at `src/violawake_sdk/backends/tflite_backend.py:308`; README scopes `num_threads` to TFLite at `README.md:688`. | Lane 1/Lane 11. No action. |
| F3/P5: augmentation count mismatch. | OPEN P2 | README has the corrected seven-type table at `README.md:752`, but a later training bullet still says `8 augmentation types` at `README.md:1111`. | Lane 11 with Lane 4. Correct stale docs only. |
| P1: account deletion absent. | RESOLVED | Account export/delete UI/API exists at `console/frontend/src/api.ts:258`, `console/backend/app/routes/account.py:212`, and deletion logic at `console/backend/app/routes/auth.py:365`. | Lane 8/Lane 9. No action. |
| P2: training cancellation/resume absent. | RESOLVED | Job SSE, cancel, and resume surfaces exist at `console/backend/app/routes/training.py:113`, `console/backend/app/routes/jobs.py:177`, `console/backend/app/routes/jobs.py:211`, and queue resume/cancel internals at `console/backend/app/job_queue.py:170`, `console/backend/app/job_queue.py:311`, `console/backend/app/job_queue.py:592`. | Lane 8. No action. |
| P3: documentation site not deployed. | RESOLVED | Docs generation workflow exists at `.github/workflows/docs.yml:46`; live `/docs/` returned `200`; docs shell exists at `docs/index.html:245`. | Lane 10/Lane 11. No action. |
| P4: model file sizes/hash metadata missing or placeholder. | RESOLVED | Registry has real SHA/size fields at `src/violawake_sdk/models.py:48` and `src/violawake_sdk/models.py:65`; tests require non-empty SHA and size at `tests/integration/test_sdk_surface.py:87`. | Lane 7. No action. |
| Priority gap list: core SDK/API/docs lies are blockers. | SUPERSEDED | The concrete blocker rows above are resolved except the low-severity augmentation copy nit. Public SDK now defaults to TemporalCNN at `README.md:735` and model verification is in `.github/workflows/model-verify.yml:79`. | Lane 11 owns remaining copy cleanup. |

## docs/AUDIT_2026_03_28.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| 1. Business tier claimed GPU acceleration. | RESOLVED | Current business positioning avoids a shipped-GPU claim; `docs/BUSINESS_PLAN.md:17` explicitly says Modal GPU training is not current reality. | Lane 11. No action. |
| 2. "Start Free Trial" charged immediately. | RESOLVED | Billing creates trial subscriptions with `trial_period_days` at `console/backend/app/routes/billing.py:356`; current pricing copy uses non-immediate language at `console/frontend/src/pages/Pricing.tsx:103`. | Lane 8/Lane 9. No action. |
| 3. Recordings deleted after training was promised but not implemented. | RESOLVED | Post-training soft delete and retention cleanup exist at `console/backend/app/job_queue.py:1237` and `console/backend/app/retention.py:100`. | Lane 8. No action. |
| 4. Priority training queue was FIFO. | RESOLVED | Tier priorities are active constants at `console/backend/app/job_queue.py:69`. | Lane 8. No action. |
| 5. Enterprise phantom features. | RESOLVED | Business plan distinguishes current reality from target integrations at `docs/BUSINESS_PLAN.md:17` and `docs/BUSINESS_PLAN.md:57`. | Lane 11. No action. |
| 6. TFLite claimed but no `.tflite` models shipped. | SUPERSEDED | Active docs/registry removed the placeholder TFLite model; model registry comments note TFLite removal at `src/violawake_sdk/models.py:88`. | Lane 7/Lane 11. No action. |
| 7. d-prime claim lacked synthetic qualifier. | RESOLVED | README now qualifies benchmark data as synthetic/TTS at `README.md:36` and `README.md:1067`. | Lane 11/Lane 5. No action. |
| 8. Email verification silently failed without Resend. | RESOLVED | Auth auto-verifies/fails open for dev when email is disabled/fails at `console/backend/app/routes/auth.py:140`; Resend production setup is documented at `docs/PRODUCTION_STATUS.md:3`. | Lane 8. No action on original finding. |
| 9. No client-side recording quality gates. | RESOLVED | Frontend recorder has quality messaging at `console/frontend/src/components/AudioRecorder.tsx:91`; backend recording validation has volume checks around `console/backend/app/routes/recordings.py:967`. | Lane 8/Lane 9. No action. |
| 10. Teams labeled coming soon while built. | RESOLVED | Teams routes/UI are live in `console/backend/app/routes/teams.py:34`, `console/frontend/src/pages/Teams.tsx:65`, and `console/frontend/src/App.tsx:113`. | Lane 9/Lane 8. No action. |
| 11. Console used legacy MLP instead of TemporalCNN. | RESOLVED | Production training path uses `_train_temporal_cnn`; docs call it required at `docs/PROVEN_TRAINING_RECIPE.md:98`; integration tests cover TemporalCNN at `tests/integration/test_feature_completeness.py:1286`. | Lane 4. No action. |
| 12. Default epochs mismatch. | RESOLVED | README training examples and CLI table consistently show `--epochs 50` at `README.md:879` and `README.md:910`. | Lane 4/Lane 11. No action. |
| 13. README used wrong `--word` flag. | RESOLVED | README and CLI table use `--word` at `README.md:879`; examples use `--word` at `README.md:909`. | Lane 11. No action. |
| 14. README train/eval flag mismatch. | RESOLVED | Current README documents `--positives` and `--eval-dir` at `README.md:879`; contamination check uses `--train` and `--eval` at `README.md:889`. | Lane 11. No action. |
| 15. Empty `SECRET_KEY` accepted. | RESOLVED | Production config validates secrets; production status includes security regression proof at `docs/PRODUCTION_STATUS.md:3`. | Lane 8/Lane 10. No action. |
| 16. WASM roadmap checked but dist not built. | RESOLVED | CI has `wasm-build` at `.github/workflows/ci.yml:162`; Pages deploy checks WASM assets at `.github/workflows/deploy-pages.yml:71`; public WASM files exist under `console/frontend/public/wasm/`. | Lane 7/Lane 10. No action. |
| 17. `reset()`/`close()` undocumented. | RESOLVED | README documents detector cleanup/reset API at `README.md:1144` and nearby API-reference rows. | Lane 11/Lane 1. No action. |
| 18. Competitive analysis stale. | RESOLVED | Current business/competitive docs qualify non-current integrations at `docs/BUSINESS_PLAN.md:17`; README benchmark comparison is current at `README.md:36`. | Lane 11. No action. |
| 19. Priority queue values stale. | RESOLVED WITH COPY NOTE | Code constants are current at `console/backend/app/job_queue.py:69`; one queue docstring still says older values at `console/backend/app/job_queue.py:214`. | Lane 11 with Lane 8. Low-severity cleanup, not must-fix. |
| AudioContext suspension/RMS/zero-energy/quality-gate bundle. | RESOLVED | Recorder error/help exists at `console/frontend/src/components/AudioRecorder.tsx:91`; backend validation is present at `console/backend/app/routes/recordings.py:967`; tests cover detector zero/edge cases at `tests/unit/test_wake_detector_edge_cases.py:76`. | Lane 1/Lane 8/Lane 9. No action. |

## FUNCTIONAL_GAP_ANALYSIS.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| P0-1: Alembic/migrations missing teams tables. | RESOLVED | Current ORM has teams/team members/models relationships at `console/backend/app/models.py:50` and `console/backend/app/models.py:62`; startup compatibility migration adds team columns at `console/backend/app/database.py:73`. | Lane 8. No action. |
| P0-2: WASM never built. | RESOLVED | CI `wasm-build` exists at `.github/workflows/ci.yml:162`; deploy probe checks `/wasm/dist/violawake.js` at `.github/workflows/deploy-pages.yml:71`. | Lane 7/Lane 10. No action. |
| P0-3: release model pipeline TODO. | RESOLVED | Release workflow exists at `.github/workflows/release.yml:57`; PyPI latest is 0.2.6; model verification workflow exists at `.github/workflows/model-verify.yml:79`. | Lane 7. No action. |
| P0-4: placeholder SHA hashes. | RESOLVED | Registry has real SHA/size fields at `src/violawake_sdk/models.py:48` and verification rejects placeholders at `scripts/verify_models.py:129`. | Lane 7. No action. |
| P1-1: Teams frontend UI absent. | RESOLVED | Teams routes and pages exist at `console/frontend/src/App.tsx:113`, `console/frontend/src/pages/Teams.tsx:65`, and `console/frontend/src/pages/TeamDetail.tsx:224`. | Lane 9. No action. |
| P1-2: team invite email absent. | RESOLVED | Invite email service exists at `console/backend/app/email_service.py:98`; team invite route calls it at `console/backend/app/routes/teams.py:257`. | Lane 8. No action. |
| P1-3: Docker frontend used Vite dev server. | RESOLVED | Frontend Dockerfiles run production build at `console/frontend/Dockerfile:14` and `console/Dockerfile.frontend:10`. | Lane 10/Lane 9. No action. |
| P1-4: Stripe price IDs placeholders. | RESOLVED | Price IDs are environment-mapped at `console/backend/app/routes/billing.py:40` and validated during checkout at `console/backend/app/routes/billing.py:99`; production status says live Stripe mode active at `docs/PRODUCTION_STATUS.md:3`. | Lane 8. Current live API outage is tracked separately as MF-1. |
| P1-5: `verify_models.py` missing. | RESOLVED | Script exists and checks SHA/size at `scripts/verify_models.py:67`; CI runs it at `.github/workflows/model-verify.yml:79`. | Lane 7. No action. |
| P1-6: `generate_docs.py` missing. | RESOLVED | Script is documented at `README.md:1243`; docs workflow runs `python scripts/generate_docs.py` at `.github/workflows/docs.yml:46`. | Lane 11/Lane 10. No action. |
| P1-7: email lacked dev fallback. | RESOLVED | Auth route auto-verifies accounts when email is disabled/fails in dev at `console/backend/app/routes/auth.py:140`. | Lane 8. No action on original finding. |
| P1-8: WASM demo double-processed frames. | RESOLVED | WASM demo and build are current; public demo file exists under `console/frontend/public/wasm/demo/index.html`; deploy checks WASM paths at `.github/workflows/deploy-pages.yml:71`. | Lane 7/Lane 9. No action. |
| P2-1: no WASM CI job. | RESOLVED | `.github/workflows/ci.yml:162` defines the WASM build job and uploads `wasm/dist` at `.github/workflows/ci.yml:188`. | Lane 10/Lane 7. No action. |
| P2-2: mypy non-blocking. | OPEN P2 | CI still has `continue-on-error: true` at `.github/workflows/ci.yml:51`. | Lane 10 with owning typed surfaces. Keep as quality backlog, not must-fix. |
| P2-3: coverage floor too low. | OPEN P2 | Unit coverage floor is still 50 in CI at `.github/workflows/ci.yml:97`; release workflow uses 65 at `.github/workflows/release.yml:57`. | Lane 10. Raise when lanes can absorb test failures. |
| P2-4: no OAuth. | OPEN P2 | No OAuth provider routes were found; auth routes are password/email based at `console/backend/app/routes/auth.py:299`. | Lane 8/Lane 9. Product backlog, not must-fix. |
| P2-5: no model versioning/A-B comparison UI. | OPEN P2 | Models can be listed/deleted/downloaded, but no comparison route/UI was found; model card delete/download surface at `console/frontend/src/components/ModelCard.tsx:216`. | Lane 9/Lane 8. Backlog. |
| P2-6: no GPU training lane. | WONTFIX FOR CURRENT LAUNCH | Current business docs explicitly say Modal GPU training is not current reality at `docs/BUSINESS_PLAN.md:17`. | Lane 4/Lane 10 if revived later. |
| P2-7: forgot-password link missing. | RESOLVED | UI routes and API calls exist at `console/frontend/src/App.tsx:52`, `console/frontend/src/api.ts:238`, and `console/backend/app/routes/auth.py:299`. | Lane 9/Lane 8. No action. |
| P2-8: root test MP3 clutter. | RESOLVED | `git ls-files "test_*.mp3" "*.mp3"` returned no tracked MP3 artifacts. | Lane 12. No action. |
| P2-9: docker-compose obsolete version key. | RESOLVED | Current compose files do not expose a stale top-level `version` finding in tracked production compose; volumes are explicit at `docker-compose.production.yml:122`. | Lane 10. No action. |
| P2-10: no rate-limit docs. | OPEN P2 | Slowapi limits exist at `console/backend/app/rate_limit.py:34`, but consumer-facing docs for limits were not found. | Lane 11 with Lane 8. Document limits/headers. |

## ACCURACY_MISSION.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| Primary metric: held-out real speech detection still open. | OPEN P1 | `ACCURACY_MISSION.md:58` marks real speech detection open; current README still qualifies public benchmark positives as synthetic/TTS at `README.md:36`; ledger still requires production oracle probes at `docs/LANE_LEDGER.md:127`. | Lane 5 primary, Lane 1 affected. MUST-FIX MF-2. |
| Primary metric: FAPH crisis from MLP era. | SUPERSEDED | Registry marks MLP-era FAR/FRR and clean eval reports as superseded by TemporalCNN at `docs/REGISTRY.md:91`; current README ships TemporalCNN as default at `README.md:735`. | Lane 5. No action on old MLP finding. |
| Primary metric: held-out true-positive rate. | SUPERSEDED WITH OPEN FOLLOW-UP | Current reports show TemporalCNN improvement at `docs/PROVEN_TRAINING_RECIPE.md:16`, but real-speaker parity remains open under MF-2. | Lane 5. Close old MLP item, keep real-speaker proof open. |
| Confusable rejection and TTS-only confusable bars. | OPEN P1 | Current docs claim confusable augmentation at `docs/PROVEN_TRAINING_RECIPE.md:100`, but ledger still requires negative probes and per-category bars at `docs/LANE_LEDGER.md:170`. | Lane 5/Lane 1. Include in MF-2 oracle work. |
| Critical contaminated eval: train/eval byte overlap. | RESOLVED FOR DETECTION, SUPERSEDED FOR CLAIMS | Contamination tooling is documented at `README.md:889`; duplicate audit artifacts exist at `experiments/real_speech_eval.json:31`; old real-speech claims are not acceptable launch proof. | Lane 5. No action on detection; rerun clean eval for MF-2. |
| Threshold implication: threshold 0.90 underperformed, <=0.80 needed. | SUPERSEDED | Production default threshold is now 0.80 at `src/violawake_sdk/_constants.py:121`; README examples use 0.80 at `README.md:63`. | Lane 1. No action. |
| R2 ACAV mining did not reduce FAPH. | SUPERSEDED | R2/MLP-era reports are archived and superseded by TemporalCNN at `docs/REGISTRY.md:88`. | Lane 5. No action. |
| Data leakage/memorization concern. | RESOLVED AS TOOLING, OPEN AS LAUNCH PROOF | `violawake-contamination-check` exists at `README.md:889`; ledger still requires contamination sentinels at `docs/LANE_LEDGER.md:324`. | Lane 5. Fold into MF-2. |
| Challenger finding 1: SDK inference path differed from measurement path. | OPEN P1 | SDK path now uses `WakeDetector` plus TemporalCNN, but `ACCURACY_MISSION.md:528` still asks if batch scoring matches production and ledger calls for current baseline parity at `docs/LANE_LEDGER.md:183`. | Lane 1/Lane 5. MUST-FIX MF-3. |
| Challenger finding 2: no TP eval on hardened model. | SUPERSEDED WITH OPEN FOLLOW-UP | TemporalCNN became the production baseline at `README.md:735`; clean real-speaker proof remains open under MF-2. | Lane 5. No separate action. |
| Challenger finding 3: mean-pooling ceiling hypothesis. | SUPERSEDED | TemporalCNN uses temporal windows over OWW embeddings and is documented as production default at `docs/PROVEN_TRAINING_RECIPE.md:52` and `README.md:735`. | Lane 4/Lane 5. No action. |
| Challenger finding 4: statistical invalidity from too little negative audio. | OPEN P1 | Current public README benchmark uses 700 negatives/180 TTS positives at `README.md:36`; ledger still requires richer oracle probes at `docs/LANE_LEDGER.md:170`. | Lane 5. MUST-FIX MF-2. |
| Challenger finding 5: 122 duplicate groups unaudited. | RESOLVED | Duplicate groups are recorded at `experiments/real_speech_eval.json:31`; contamination tooling exists at `README.md:889`. | Lane 5. No action beyond clean rerun. |
| Revised next step: duplicate audit. | RESOLVED | See `experiments/real_speech_eval.json:31` and `README.md:889`. | Lane 5. No action. |
| Revised next step: round 2 training. | SUPERSEDED | The current production recipe is TemporalCNN, not the old round-2 MLP line, at `docs/PROVEN_TRAINING_RECIPE.md:98`. | Lane 4/Lane 5. No action. |
| Revised next step: statistical corpus / 50-100h confidence. | OPEN P1 | No current 50-100h artifact was found; ledger still requires production eval and negative probes at `docs/LANE_LEDGER.md:170`. | Lane 5. MUST-FIX MF-2. |
| Revised next step: SDK inference path resolution. | OPEN P1 | Same as MF-3. Current implementation exists, but parity proof is not present in the audit trail. | Lane 1/Lane 5. MUST-FIX MF-3. |
| Revised next step: diverse speaker eval, domain diversity, Sierra outliers. | OPEN P1/P2 | Old real-speech report lists Sierra outliers and duplicate contamination at `experiments/REAL_SPEECH_EVAL.md:122` and `experiments/REAL_SPEECH_EVAL.md:150`. Current README still caveats real speakers at `README.md:36`. | Lane 5. Include in MF-2. |
| Q20 Common Voice / speech negatives. | OPEN P1 | Roadmap identifies Common Voice as speech negatives at `docs/ROADMAP_10_OF_10.md:282`, but no current closure artifact was found. | Lane 5. Include in MF-2. |
| Q25 ensemble diversity. | SUPERSEDED | Product default moved to single TemporalCNN; README lists ensemble APIs but not as launch blocker at `README.md:619`. | Lane 5. No must-fix. |
| Q26 OWW backbone compatibility. | OPEN P1 | `experiments/verify_sdk_path.py:5` exists as an experiment, but no current lane acceptance artifact proves production parity. | Lane 1/Lane 5. Include in MF-3. |
| Q27 group-aware split leak. | RESOLVED AS TRAINING PRACTICE, OPEN AS AUDIT PROOF | Proven recipe includes group-aware split at `docs/PROVEN_TRAINING_RECIPE.md:92`; MF-2 still requires clean published eval proof. | Lane 4/Lane 5. |

## BUILD_VS_BUY_AUDIT.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| Auth: no refresh tokens. | WONTFIX FOR CURRENT LAUNCH | Audit bottom line accepted current build-for-now posture; current auth remains token/password based with reset/change endpoints at `console/backend/app/routes/auth.py:299` and `console/backend/app/routes/auth.py:345`. | Lane 8. Revisit only if session requirements change. |
| Auth: JWT in localStorage. | WONTFIX FOR CURRENT LAUNCH | No documented later decision reverses this; frontend auth context remains SPA-token based at `console/frontend/src/contexts/AuthContext.tsx:40`. | Lane 8/Lane 9. Security backlog, not launch must-fix. |
| Auth: CSRF not needed with bearer tokens. | WONTFIX | The finding itself says CSRF is not needed for bearer-token flow; no cookie-session switch found. | Lane 8. No action. |
| Auth: download token JTI in-memory. | OPEN P2 | Download/token storage remains app-local; no Redis/shared replay store was found. | Lane 8/Lane 10. Backlog for multi-worker scale. |
| Rate limiting: in-memory only. | OPEN P2 | Slowapi limiter exists at `console/backend/app/rate_limit.py:34`, but no Redis/shared storage config was found. | Lane 8/Lane 10. Backlog unless multi-worker launch. |
| Rate limiting: no cleanup old entries. | SUPERSEDED | Custom in-memory dict finding was superseded by Slowapi shared limiter at `console/backend/app/rate_limit.py:34`. | Lane 8. No action. |
| Rate limiting: explicit call-site invocation. | SUPERSEDED | Rate limit middleware/handler is centralized in `console/backend/app/main.py:13` and route helpers use shared limiter at `console/backend/app/rate_limit.py:82`. | Lane 8. No action. |
| Email: no retry. | OPEN P2 | Email service sends directly; no retry/backoff layer was found around `console/backend/app/email_service.py:19`. | Lane 8. Backlog. |
| Email: inline HTML templates. | OPEN P2 | Email service constructs HTML inline around `console/backend/app/email_service.py:150`. | Lane 8/Lane 11. Backlog. |
| Email: raw HTTP rather than Resend SDK. | OPEN P2 | Email service remains local implementation at `console/backend/app/email_service.py:19`; no SDK migration found. | Lane 8. Backlog. |
| DB: no Alembic / schema evolution. | SUPERSEDED | Compatibility migration/startup schema handling exists at `console/backend/app/database.py:64`; release/ops docs now own deploy flow. | Lane 8/Lane 10. No action on original. |
| DB: missing `TrainingJob.model_id` index. | OPEN P2 | Current model indexes user/team fields at `console/backend/app/models.py:128`; no specific `model_id` index evidence was found. | Lane 8. Low-severity DB cleanup. |
| Frontend auth: no automatic token refresh. | WONTFIX FOR CURRENT LAUNCH | Same as no-refresh-token decision; password reset/change flows exist at `console/frontend/src/api.ts:238` and `console/frontend/src/api.ts:248`. | Lane 8/Lane 9. Backlog. |
| Frontend auth: `alert()` for session expiry. | RESOLVED | Auth context routes session-return handling without relying on alert at `console/frontend/src/contexts/AuthContext.tsx:40`. | Lane 9. No action. |
| UI: global CSS. | OPEN P2 | Global stylesheet remains large at `console/frontend/src/styles/global.css:2155`. | Lane 9. Backlog. |
| UI: no form library. | OPEN P2 | Forms are still local React state/components, for example `console/frontend/src/pages/ChangePassword.tsx:16`. | Lane 9. Backlog. |
| Cookie consent: accept-only, no preferences/categories. | OPEN P2 | Public cookie consent exists, but no preference-management UI was found. | Lane 9/Lane 11. Backlog if analytics cookies expand. |

## E2E_READINESS.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| PyTorch/training extras are large. | WONTFIX | Training extras are inherent to the chosen local training stack; docs still separate optional training deps at `console/README.md:81` and deployment prerequisites at `docs/DEPLOYMENT.md:179`. | Lane 4/Lane 7. Document only. |
| openWakeWord first load downloads backbone. | WONTFIX / DOCUMENTED | Training audit documents lazy OWW download at `docs/TRAINING_PIPELINE_AUDIT_2026-05-07.md:170`; deployment entrypoint preloads where needed. | Lane 7/Lane 10. Keep documented. |
| Free tier limit may block demos. | WONTFIX PRODUCT DECISION | Billing tiers enforce quotas; pricing copy explains tier limits at `console/frontend/src/pages/Pricing.tsx:29`. | Lane 8/Lane 9. Product decision. |
| ScriptProcessorNode deprecated. | OPEN P2 | Console recorder still calls `createScriptProcessor` at `console/frontend/src/components/AudioRecorder.tsx:226`; WASM demo also does at `console/frontend/public/wasm/demo/index.html:330`. | Lane 9. Replace with AudioWorklet when scheduled. |
| Windows path separator normalization. | RESOLVED | Storage key normalization handles path/key safety at `console/backend/app/storage.py:240` and storage backends isolate object identifiers. | Lane 8. No action. |

## LAUNCH_READINESS.md

| Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| API/backend health readiness. | OPEN P0 | Live `/api/health` and `/openapi.json` returned HTTP `530`; deployment docs identify `530`/tunnel down at `docs/DEPLOYMENT.md:231`. | Lane 10 primary, Lane 8 affected. MUST-FIX MF-1. |
| Domain/SSL/frontend availability. | PARTIAL: FRONTEND RESOLVED, API OPEN | `https://violawake.com/` returned `200`; API subdomain returned `530`. | Lane 10. Close frontend, fix API. |
| `nginx.conf`/deploy blocker. | SUPERSEDED | Current deployment docs describe Cloudflare Pages plus backend Docker/tunnel flow at `docs/DEPLOYMENT.md:22`; no nginx blocker remains in current architecture. | Lane 10. No action. |
| In-memory rate limiting should-fix. | OPEN P2 | Slowapi limiter exists at `console/backend/app/rate_limit.py:34`, but shared Redis storage was not found. | Lane 8/Lane 10. Backlog. |
| Resend domain/setup. | RESOLVED WITH LIVE-FLOW CAVEAT | Production status says Resend/domain verified at `docs/PRODUCTION_STATUS.md:3`; live verification is blocked by API `530`. | Lane 8/Lane 10. Re-verify after MF-1. |
| Stripe products/prices/webhook setup. | RESOLVED WITH LIVE-FLOW CAVEAT | Stripe live mode documented at `docs/PRODUCTION_STATUS.md:3`; code maps price IDs at `console/backend/app/routes/billing.py:40`; live checkout cannot be probed while API is `530`. | Lane 8/Lane 10. Re-verify after MF-1. |
| Durable storage / volume persistence. | RESOLVED FOR SINGLE NODE | Production compose has explicit backend data volume at `docker-compose.production.yml:43` and volume definition at `docker-compose.production.yml:122`; R2 backend exists at `console/backend/app/storage.py:146`. | Lane 10/Lane 8. No launch blocker unless scaling horizontally. |
| Dark/light theme missing. | WONTFIX / OPEN LOW | No documented decision to implement light theme was found; launch docs treated this as nice-to-have. | Lane 9. Not must-fix. |
| Favicon/OG image missing. | RESOLVED | Frontend meta references OG image at `console/frontend/index.html:17`; live `/og-image.png` returned `200`; generation script exists at `console/frontend/scripts/generate-og-image.py:15`. | Lane 9/Lane 11. No action. |
| Cookie consent. | RESOLVED FOR BASIC CONSENT | Cookie/legal pages exist; remaining preference-center gap is tracked under Build-vs-Buy as low severity. | Lane 9/Lane 11. No launch blocker. |
| GDPR export/delete. | RESOLVED | Export/delete API/UI exists at `console/backend/app/routes/account.py:25`, `console/backend/app/routes/account.py:212`, and `console/frontend/src/pages/Privacy.tsx:176`. | Lane 8/Lane 9. No action. |
| Alerting/monitoring. | OPEN P2 | Sentry init exists at `console/backend/app/middleware.py:106`, but no PagerDuty/Slack/alert routing evidence was found. | Lane 10. Backlog. |
| Robots/sitemap. | RESOLVED | Generated robots/sitemap code at `console/frontend/scripts/generate-marketing.mjs:519`; live `/robots.txt` and `/sitemap.xml` returned `200`. | Lane 11/Lane 10. No action. |
| Documentation/help partial. | OPEN P2 | Docs site exists, but in-app guide/help is still listed as a nice-to-have in launch readiness. | Lane 11/Lane 9. Backlog. |
| Email unsubscribe headers. | OPEN P2 | No unsubscribe-header implementation was found in `console/backend/app/email_service.py`. | Lane 8/Lane 11. Backlog. |
| Webhook retry logging. | OPEN P2 | Stripe webhook handling exists, but no retry queue evidence was found near `console/backend/app/routes/billing.py:447`. | Lane 8. Backlog. |
| Queue position display. | OPEN P2 | Queue/job state exists, but no explicit UI queue-position proof was found. | Lane 9/Lane 8. Backlog. |
| Model comparison. | OPEN P2 | No model comparison UI/API was found; see Functional P2-5. | Lane 9/Lane 8. Backlog. |
| Multi-worker rate limits. | OPEN P2 | Same as in-memory Slowapi item. | Lane 8/Lane 10. Backlog. |
| Mobile hamburger. | OPEN P2 | No current lane proof found; frontend polish backlog only. | Lane 9. Backlog. |
| `run.py` reload in production. | RESOLVED | Backend runner gates reload on dev mode at `console/backend/run.py:15`. | Lane 10. No action. |

## docs/PRE_LAUNCH_CHECKLIST.md

| Checklist Finding | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| Critical: durable storage before real users. | RESOLVED FOR SINGLE NODE | Compose volume exists at `docker-compose.production.yml:43`; R2 backend exists at `console/backend/app/storage.py:146`; upload volume config exists at `console/backend/app/config.py:36`. | Lane 10/Lane 8. Reassess for multi-replica only. |
| Critical: release workflow executable. | RESOLVED | Release workflow runs tests/build/publish at `.github/workflows/release.yml:57` and `.github/workflows/release.yml:162`; package 0.2.6 is published. | Lane 7. No action. |
| Critical: Railway build context. | SUPERSEDED | Current deployment docs describe Cloudflare Pages plus Docker backend flow at `docs/DEPLOYMENT.md:22`; Railway-specific checklist item is no longer canonical. | Lane 10. No action unless Railway returns. |
| SDK: bump version. | RESOLVED | `pyproject.toml:7` is 0.2.6 and PyPI latest is 0.2.6. | Lane 7. No action. |
| SDK: verify model SHA hashes. | RESOLVED | Registry SHA/size exists at `src/violawake_sdk/models.py:48`; model verify workflow at `.github/workflows/model-verify.yml:79`. | Lane 7. No action. |
| SDK: upload/attach model assets. | SUPERSEDED / RESOLVED | Current releases point to existing model assets; release API for `v0.2.6` returned wheel/sdist and model asset redirect for `v0.1.0` is live. | Lane 7. No action unless changing registry URLs. |
| SDK: write release notes. | RESOLVED | Release workflow consumes `RELEASE_NOTES.md` at `.github/workflows/release.yml:130`; current release exists. | Lane 7/Lane 11. No action. |
| SDK: tag release. | RESOLVED | `git tag --list v0.2.2 v0.2.6` showed both tags. | Lane 7. No action. |
| SDK: GitHub release. | RESOLVED | GitHub release API for `v0.2.6` returned `200` with assets. | Lane 7. No action. |
| SDK: PyPI release. | RESOLVED | `python -m pip index versions violawake` reported latest `0.2.6`. | Lane 7. No action. |
| SDK: clean install smoke test. | RESOLVED FROM PRIOR STATUS, NOT RERUN | `docs/PRODUCTION_STATUS.md:21` records clean venv import/model smoke success; this sweep did not rerun a new venv install. | Lane 7. Rerun only for release certification. |
| Backend: create deployment project. | SUPERSEDED | Current deploy path is documented at `docs/DEPLOYMENT.md:22`; Railway checklist item is not canonical. | Lane 10. No action. |
| Backend: provision PostgreSQL. | RESOLVED / DOCUMENTED | Deployment docs include DB URL/env setup; production status references Postgres backups at `docs/PRODUCTION_STATUS.md:3`. | Lane 10/Lane 8. No action from repo. |
| Backend: set env vars. | RESOLVED FROM DOCS, LIVE BLOCKED | Deployment docs list env setup; live API `530` prevents proof of the current runtime env. | Lane 10. Covered by MF-1. |
| Backend: deploy and health-check backend. | OPEN P0 | Live health endpoint returned `530`. | Lane 10. MUST-FIX MF-1. |
| Backend: run migrations. | OPEN UNTIL LIVE API RESTORED | Startup schema handling exists at `console/backend/app/database.py:64`, but live health failure prevents runtime confirmation. | Lane 10/Lane 8. Verify after MF-1. |
| Backend: verify API docs/openapi. | OPEN P0 | Live `/openapi.json` returned `530`. | Lane 10/Lane 8. MUST-FIX MF-1. |
| Frontend: set `VITE_API_URL`. | RESOLVED FOR BUILDS, LIVE API OPEN | Pages workflow/build docs set API URL at `docs/PRE_LAUNCH_CHECKLIST.md:97`; live frontend is `200`, API is `530`. | Lane 10/Lane 9. Rebuild only if API URL changed. |
| Frontend: Cloudflare Pages deploy. | RESOLVED | Live root returned `200`; Pages deploy workflow exists at `.github/workflows/deploy-pages.yml:50`. | Lane 10/Lane 9. No action. |
| Frontend: hard-refresh routes. | RESOLVED | Static redirects are generated at `console/frontend/scripts/generate-marketing.mjs:648`; public `_redirects` exists at `console/frontend/public/_redirects`. | Lane 9/Lane 10. No action. |
| Frontend: Stripe checkout flow. | OPEN DUE API 530 | Checkout code exists and trial setup exists at `console/backend/app/routes/billing.py:356`, but live API outage blocks a no-charge flow check. | Lane 8/Lane 10. Verify after MF-1. |
| Domain: buy/configure domain. | RESOLVED | Live `https://violawake.com/` returned `200`. | Lane 10. No action. |
| Domain: DNS for frontend/backend. | PARTIAL | Frontend resolves and serves `200`; API returns `530`. | Lane 10. Covered by MF-1. |
| Domain: SSL. | PARTIAL | Frontend SSL works; API SSL/tunnel path returns Cloudflare `530`. | Lane 10. Covered by MF-1. |
| Domain: update CORS. | OPEN UNTIL API RESTORED | Current API health is unavailable; CORS cannot be verified live. | Lane 10/Lane 8. Verify after MF-1. |
| Legal: privacy policy review. | RESOLVED WITH COPY NOTE | Privacy export/delete copy exists at `console/frontend/src/pages/Privacy.tsx:176`; Terms deletion copy conflicts with current 30-day backend behavior at `console/frontend/src/pages/Terms.tsx:238` versus `console/backend/app/routes/auth.py:386`. | Lane 11/Lane 8. Low-severity legal copy correction. |
| Legal: terms review. | OPEN P2 COPY ACCURACY | Same Terms/backend deletion-window mismatch above. | Lane 11. Backlog but should be cleaned before broad launch. |
| Legal: license. | RESOLVED | License/package metadata is present in repo; no active audit blocker found. | Lane 11/Lane 7. No action. |
| Security: `SECURITY.md`. | RESOLVED | Security doc exists and covers volume/security posture at `docs/SECURITY.md:15`. | Lane 11/Lane 10. No action. |
| Security: GitHub advisories. | OPEN EXTERNAL | GitHub repository settings are not inspectable from repo state in this sweep. | Lane 10/Lane 11. Human repo-owner check. |
| Security: rate limiting. | RESOLVED BASIC, OPEN SCALE | Slowapi rate limiting exists at `console/backend/app/rate_limit.py:34`; shared storage remains open P2. | Lane 8/Lane 10. Backlog. |
| Security: secrets scan. | OPEN PROCESS | No current secrets-scan command was run in this recommendations-only sweep. | Lane 10/Lane 12. Run before release certification. |
| Marketing: Show HN draft. | RESOLVED | Draft exists at `docs/SHOW_HN_DRAFT.md:11`. | Lane 11. No action. |
| Marketing: GitHub description/topics. | OPEN EXTERNAL | Repo metadata is external to the checkout. | Lane 11. Human repo-owner check. |
| Marketing: PyPI badges. | RESOLVED | Published package exists at PyPI version 0.2.6; README has install/readme material. | Lane 7/Lane 11. No action. |
| Marketing: star own repo. | WONTFIX / EXTERNAL | Not a code/readiness blocker and cannot be validated from repo state. | Lane 11. No action. |

## PROGRESS.md

| Finding / Claim | Classification | Evidence | Owner / Recommendation |
|---|---|---|---|
| Phase 1-4 foundation/golden path claims. | RESOLVED HISTORICAL | Current code includes backend/frontend/training/model verification surfaces cited throughout this report; no contradiction found for the historical completion claims. | Owning lanes. No action. |
| Gate 1 quality proof. | PARTIAL / OPEN P1 FOR CURRENT ORACLE | Unit/integration surfaces exist, but ledger still calls for production negative probes and oracle work at `docs/LANE_LEDGER.md:170`. | Lane 1/Lane 5. Covered by MF-2/MF-3. |
| Gate 2 packaging/release proof. | RESOLVED | PyPI latest 0.2.6 and release workflow present at `.github/workflows/release.yml:162`. | Lane 7. No action. |
| Gate 3 console/backend proof. | PARTIAL | Current backend/API code exists, but live API is `530`. | Lane 8/Lane 10. Covered by MF-1. |
| Gate 4 docs/deploy proof. | PARTIAL | Public frontend/docs/SEO assets are live; API deploy proof is open due `530`. | Lane 10/Lane 11. Covered by MF-1 for API. |
| Gate 5 security hardening. | PARTIAL | Sentry/CSP/security docs exist at `console/backend/app/middleware.py:106` and `docs/SECURITY.md:15`; advisory/settings/secrets checks remain external/process. | Lane 10/Lane 11. Process follow-up. |
| Gate 5 Firefox cross-browser pass. | OPEN P1 | Ledger requires Chrome/Firefox/Safari at `docs/LANE_LEDGER.md:566`; E2E runner installs Chromium only at `console/run_e2e.py:35`. | Lane 9/Lane 10. MUST-FIX MF-4. |
| Gate 5 README console quickstart. | RESOLVED | README console/training commands exist around `README.md:879` and deploy docs exist at `docs/DEPLOYMENT.md:179`. | Lane 11. No action. |
| Gate 5 CSS/UX polish. | OPEN P2 | Frontend exists and accessibility status is documented in `docs/PRODUCTION_STATUS.md:3`, but no current broad visual regression proof was rerun in this sweep. | Lane 9. Backlog/certification item. |
| Gate 5 all Playwright tests green in CI. | OPEN P1 | CI ignores `console/tests/e2e` at `.github/workflows/ci.yml:133`; E2E tests exist at `console/tests/e2e/test_browser_flow.py:2`. | Lane 9/Lane 10. MUST-FIX MF-4. |
| Blockers/security fixes listed as completed. | MOSTLY RESOLVED | Account deletion, retention, rate limiting, release, model registry, and legal surfaces are present; remaining exceptions are live API, oracle proof, external security settings, and low-severity copy/style backlog. | See MF list and per-doc OPEN rows. |

## Cross-Doc Open Low-Severity Backlog

- Lane 11: fix stale augmentation count (`README.md:1111`), stale priority docstring/copy (`console/backend/app/job_queue.py:214`), rate-limit docs, and Terms/backend deletion-window mismatch.
- Lane 8/Lane 10: decide whether to move Slowapi to shared storage before multi-worker scale; add webhook/email retry if operational volume warrants it.
- Lane 9: replace `createScriptProcessor` with AudioWorklet, add cookie preferences if non-essential cookies expand, and wire model comparison/queue position only if product wants them.
- Lane 10/Lane 12: run secrets scan and human repo settings checks during release certification, not as part of this recommendations-only sweep.

## Self-Audit Gate

- I did not edit lane-owned source, quality gates, production configs, or runtime code; this artifact is recommendations-only.
- I did not deploy, restart, write to production, push, merge, tag, or run destructive commands; live checks were read-only HTTP/package metadata probes.
- I did not treat missing proof as success. Live API health, openapi, checkout, migrations, CORS, and Stripe/email flows remain open where API `530` blocks verification.
- I did not run full test suites, full Playwright, Firefox/Safari, a clean PyPI venv install, or large model downloads; when proof was not current, the row says so.
- I did not verify external dashboard-only state such as GitHub advisories, repo metadata, Cloudflare dashboard config, Stripe dashboard config, or real inbox delivery; those are routed to the owning lanes/human operator.
