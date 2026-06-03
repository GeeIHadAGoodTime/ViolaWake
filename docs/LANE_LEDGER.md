# LANE LEDGER — ViolaWake SDK

The authoritative set of lanes for this project. Every agent — orchestrator
or worker — anchors to this file before dispatching work. A summary or chat
recollection is not a substitute.

This ledger applies the discipline from `CLAUDE.md` → "Project decomposition"
and "Lanes & the Lane Ledger." If you haven't read those two sections, stop
and read them now.

---

## Methodology — why this split, not another

This decomposition is grounded in three convergent disciplines, all of
which say the same thing in different vocabularies. The ViolaWake split
honors all three simultaneously:

- **PMBOK Work Breakdown Structure (WBS).** Two cardinal rules:
  the **100% rule** — every element of the project's scope appears in
  exactly one place in the WBS, and the union covers 100% of the
  scope — and **mutual exclusivity** — no element overlaps another; you
  shouldn't need input from another element to finish this one. ([PMBOK
  WBS overview][1]; [Wikipedia WBS — 100% rule + mutual
  exclusivity][2].)
- **DDD Bounded Contexts.** Each context has its own ubiquitous
  language, model, and rules; integration with other contexts goes
  through an explicit translation layer, not a shared internal model.
  ([Martin Fowler — Bounded Context][3]; [Context Mapper — Bounded
  Context][4].)
- **Business Capability Maps.** Capabilities are **nouns** — what the
  business does, not how it does it. The map is functional, stable, and
  deliberately conservative; capabilities change for strategic reasons,
  not per feature. ([Business Architecture Associates — Business
  Capability Map][5]; [Ardoq — capabilities vs processes vs value
  streams][6].)

The convergent rule across all three: **a project is a finite set of
disjoint capability areas, each owned end-to-end by one agent, each with
one oracle.** Lanes here are **noun-named**, **file-disjoint**,
**capability-mapped**, and **deliberately stable**. If a new feature
fits in an existing lane, it goes there; new lanes are minted only when
the product gains a strategically new capability.

[1]: https://en.wikipedia.org/wiki/Work_breakdown_structure
[2]: https://en.wikipedia.org/wiki/Work_breakdown_structure
[3]: https://www.martinfowler.com/bliki/BoundedContext.html
[4]: https://contextmapper.org/docs/bounded-context/
[5]: https://businessarchitectureassociates.com/wp-content/uploads/2020/10/5deecd9698d1d5cef8c9c313ea7b9316.pdf
[6]: https://www.ardoq.com/knowledge-hub/value-streams

---

## Status markers (read before scanning the table)

Every lane carries two orthogonal markers: a **scope status** (is the lane
in scope and decided?) and an **oracle status** (does the lane have an
instrument that catches a broken implementation?).

**Scope status — binding on what gets implemented.** Per CLAUDE.md, agents
must honor these. Shipping a CONFIRM/OPEN item as if decided is drift even
when the code is clean.

- `LOCKED` — decided; implement.
- `CONFIRM` — pending founder sign-off; surface before shipping.
- `OPEN` — undecided; do not implement.

**Oracle status — binding on what counts as "done."** No lane is done
without its oracle.

- `NEEDS-ORACLE` — no instrumented success-test exists yet. Building the
  oracle's SC + oracle is the lane's first work item.
- `ORACLE-DRAFT` — partial instrument exists (e.g. a benchmark, a smoke
  test) but doesn't meet the three-anchor bar (negative probes, baseline,
  heterogeneous-review).
- `ORACLE-LIVE` — three-anchor bar met; oracle runs in CI on every commit
  for the lane's surface.
- `EXHAUSTED` — oracle is live AND two consecutive heterogeneous adversarial
  rounds returned zero P0/P1. Re-open only on a regression.

---

## Disjointness map (the 100% check)

Every file in the repo belongs to exactly one lane. This is the structural
test: `git ls-files | wc -l` MUST equal the sum of `git ls-files` filtered
to each lane's path globs. A file that fits two lanes is a sign the lanes
are mis-cut — re-cut, don't dual-own.

The file-glob assignments are listed in each lane's **Owns** section
below. The `_cruft/` block at the end captures known-stray files that
belong nowhere yet and are scheduled for relocation or deletion — they
are NOT a lane.

---

## The lanes

There are **twelve** lanes. Nine are **product capabilities** (what the SDK
+ console + browser detector do). Two are **operational** (infrastructure
and outward-facing copy). One is **process** (the orchestration artifacts
themselves).

| #  | Lane                                  | Scope    | Oracle         |
|----|---------------------------------------|----------|----------------|
| 1  | Wake Detection                        | `LOCKED` | `ORACLE-DRAFT` |
| 2  | Companion Engines & VoicePipeline     | `LOCKED` | `NEEDS-ORACLE` |
| 3  | Browser Wake Detection (WASM)         | `LOCKED` | `NEEDS-ORACLE` |
| 4  | Training & Augmentation               | `LOCKED` | `ORACLE-DRAFT` |
| 5  | Evaluation & Benchmarking             | `LOCKED` | `ORACLE-DRAFT` |
| 6  | SDK CLI & Sample Tools                | `LOCKED` | `NEEDS-ORACLE` |
| 7  | Public API & Distribution             | `LOCKED` | `NEEDS-ORACLE` |
| 8  | SaaS Console — Backend                | `LOCKED` | `ORACLE-DRAFT` |
| 9  | SaaS Console — Frontend               | `LOCKED` | `ORACLE-DRAFT` |
| 10 | Infrastructure & DevOps               | `LOCKED` | `NEEDS-ORACLE` |
| 11 | Marketing & Developer Docs            | `LOCKED` | `NEEDS-ORACLE` |
| 12 | Project Governance & Process          | `LOCKED` | N/A (process)  |

---

### 1. Wake Detection

**Capability question:** *"Given live audio, does the SDK detect the
configured wake word — and reject everything else — at the documented
threshold, on the documented audio contract?"*

**Scope:** `LOCKED`. **Oracle:** `ORACLE-DRAFT` — the production eval set
(d'=8.577, EER=0.8%) and `benchmark_v2/` (EER=5.49%) instrument the model,
but the lane lacks (a) negative probes for the four classes of regression
(threshold drift, sample-rate drift, backbone swap, 4-gate policy bypass),
(b) a heterogeneous-review pass, and (c) CI wiring of the production eval
on every commit.

**Owns (disjoint set):**
```
src/violawake_sdk/wake_detector.py
src/violawake_sdk/async_detector.py
src/violawake_sdk/confidence.py
src/violawake_sdk/ensemble.py
src/violawake_sdk/oww_backbone.py
src/violawake_sdk/speaker.py
src/violawake_sdk/noise_profiler.py
src/violawake_sdk/power_manager.py
src/violawake_sdk/audio.py
src/violawake_sdk/audio_source.py
src/violawake_sdk/backends/                       # ONNX + TFLite backends
src/violawake_sdk/security/cert_pinning.py        # model-download integrity
src/violawake/                                     # legacy compat shim
tests/unit/test_wake_detector*.py
tests/unit/test_wakeword_detector.py
tests/unit/test_async_detector.py
tests/unit/test_wake_decision_policy.py
tests/unit/test_confidence.py
tests/unit/test_ensemble.py
tests/unit/test_oww_backbone.py
tests/unit/test_noise_profiler.py
tests/unit/test_power_manager.py
tests/unit/test_audio*.py
tests/unit/test_speaker.py
tests/unit/test_tflite_backend.py
tests/unit/test_cert_pinning.py
tests/unit/test_detector_config.py
tests/integration/test_wake_detector_e2e.py
docs/adr/ADR-001-onnx-runtime.md
docs/adr/ADR-002-oww-feature-extractor.md
```

**Success criteria (lane SC):**
- Live wake-word recall ≥ documented bar on the production eval set at
  the production threshold (`0.80`).
- Per-category FAR on the documented confusables set (`alexa`, `hey
  siri`, music speech, ...) stays under its documented bar.
- The 4-gate decision policy is exercised end-to-end in a test; bypassing
  any single gate is caught by a negative probe.
- Audio-contract assertions (16 kHz mono, 20 ms frames, 96-dim OWW
  embeddings) trip CI if any consumer feeds non-conforming audio.

**Oracle SC (what the oracle must catch — written before building):**
- Negative probes: (a) threshold lowered to `0.50`, (b) audio fed at
  8/22/48 kHz, (c) OWW backbone swapped for a wrong version, (d) any one
  of the 4 decision-policy gates removed.
- Known-good baseline: current SHA-pinned `temporal_cnn` at threshold
  `0.80` passes per-category FAR + recall bars on both the production
  eval corpus and `benchmark_v2/`.
- Heterogeneous reviewer's binary question: "if a refactor silently drops
  one decision-policy gate while keeping the public API intact, would
  this oracle catch it?"

**Open investigations:** (none currently tracked here — when one is
dispatched, link its `audit/active/inv_*` dir)

---

### 2. Companion Engines & VoicePipeline

**Capability question:** *"Does the SDK ship working STT / TTS / VAD
engines, and a `VoicePipeline` composition that wires Wake → STT → TTS
correctly?"*

**Scope:** `LOCKED`. **Oracle:** `NEEDS-ORACLE` — unit tests exist for
each engine but there is no full live `VoicePipeline` oracle that proves
end-to-end behavior (audio in → spoken response out) with measured first-
audio latency.

**Owns:**
```
src/violawake_sdk/tts.py
src/violawake_sdk/tts_engine.py
src/violawake_sdk/stt.py
src/violawake_sdk/stt_engine.py
src/violawake_sdk/vad.py
src/violawake_sdk/vad_engine.py
src/violawake_sdk/pipeline.py
tests/unit/test_tts_engine.py
tests/unit/test_stt_engine.py
tests/unit/test_stt_engine_wav.py
tests/unit/test_stt_tts_engines.py
tests/unit/test_vad.py
tests/unit/test_vad_engine.py
tests/unit/test_pipeline.py
tests/unit/test_voice_pipeline.py
tests/integration/test_full_pipeline.py
tests/integration/test_pipeline.py
tests/integration/test_streaming_stt.py
```

**Success criteria:**
- VoicePipeline reaches "spoken response" in the documented latency
  budget (Kokoro: 0.3–0.8 s first-audio) on the reference hardware.
- STT segments are returned with timestamps; tested on a fixed WAV.
- VAD adapters (WebRTC / Silero / RMS) interchange without behavior
  change at the pipeline level.

**Oracle SC:**
- Negative probes: (a) STT engine swapped for a no-op returning empty
  text — pipeline must surface the error, not silently emit; (b) TTS
  engine misconfigured (wrong voice id) — must raise, not synthesize
  silence; (c) VAD always-on (returns "speech" forever) — pipeline must
  not deadlock.
- Baseline: the reference `examples/basic_detection.py` and an
  end-to-end pipeline example run clean on a clean venv.

---

### 3. Browser Wake Detection (WASM)

**Capability question:** *"Does the in-browser TypeScript detector
produce the same scores as the Python SDK on the same audio?"*

**Scope:** `LOCKED`. **Oracle:** `NEEDS-ORACLE` — a live smoke test
exists (`tests/live/test_live_wasm.py`) but parity with the Python SDK on
a shared corpus is not yet measured.

**Owns:**
```
wasm/
console/frontend/dist/wasm/                       # built assets only
tests/live/test_live_wasm.py
```

**Cross-lane dependencies:** Lane 9 consumes the committed WASM assets through
the frontend bundle, but `console/frontend/dist/wasm/` remains Lane 3-owned.
Frontend build changes that move or rename the WASM bundle must coordinate
with Lane 3.

**Success criteria:**
- WASM detector + Python SDK agree to within documented tolerance on a
  shared corpus subset (Python is the reference).
- Bundle size + first-detection latency stay under their documented bars.

**Oracle SC:**
- Negative probes: (a) features extracted at wrong frame stride —
  scores must diverge from Python (test catches divergence); (b) wrong
  model loaded — load fails fast.
- Baseline: 10-sample audio corpus, Python and WASM scores within
  tolerance.

---

### 4. Training & Augmentation

**Capability question:** *"Given labeled audio, does the training
pipeline produce a model that passes Evaluation & Benchmarking's bars —
reproducibly?"*

**Scope:** `LOCKED`. **Oracle:** `ORACLE-DRAFT` — the v2 recipe
(`docs/PROVEN_TRAINING_RECIPE.md`) and the integration test
(`tests/integration/test_training_e2e.py`) describe how to train, and the
`TRAINING_PIPELINE_AUDIT_2026-05-07.md` documents a recent audit. Missing
pieces: a reproducibility check that two runs on the same seed produce
equivalent models, and negative probes for the augmentation pipeline.

**Owns:**
```
src/violawake_sdk/training/                       # augment, evaluate,
                                                  # losses, temporal_model,
                                                  # weight_averaging
tests/unit/test_augment.py
tests/unit/test_losses.py
tests/unit/test_temporal_model.py
tests/unit/test_training_pipeline.py
tests/unit/test_weight_averaging.py
tests/unit/test_train.py
tests/unit/test_rir_augment.py
tests/unit/test_spec_augment.py
tests/integration/test_training_e2e.py
_training_corpus/
data/                                              # hf_cache, operator/, ...
corpus/                                            # librispeech, musan,
                                                  # OWW features
experiments/                                       # score-CSV exploration
docs/PROVEN_TRAINING_RECIPE.md                     # canonical training recipe
docs/TRAINING_PIPELINE_AUDIT_2026-05-07.md
```

**Cross-lane dependencies:** Lane 1 consumes the inference-contract portions
of `docs/PROVEN_TRAINING_RECIPE.md`; contract-affecting recipe changes need
Wake Detection review.

**Success criteria:**
- A from-scratch retrain on the documented corpus reaches the documented
  d'/EER on both the production eval set and `benchmark_v2/`.
- Reproducibility: two runs at the same seed produce models that score
  within tolerance on the held-out set.
- Augmentation pipeline assertions hold (RIR, SpecAugment, noise mix).

**Oracle SC:**
- Negative probes: (a) augmentation disabled — model must underperform a
  documented bar (catches "training pipeline silently bypassing
  augmentation"); (b) corpus contaminated with eval-set samples — the
  contamination check must flag.
- Baseline: the v2 recipe end-to-end on the canary corpus.

---

### 5. Evaluation & Benchmarking

**Capability question:** *"Are the public accuracy claims reproducible
from this repo on the corpora checked into this repo?"*

**Scope:** `LOCKED`. This is the **public claim instrument** — every
number on `violawake.com/compare/picovoice`, in the README, and in
`COMPETITIVE_ANALYSIS.md` must trace back to a script and corpus in this
lane. **Oracle:** `ORACLE-DRAFT` — `benchmark_v2/` runs reproducibly,
but `benchmark_regression_check.py` is not yet wired as a release gate.

**Owns:**
```
src/violawake_sdk/tools/evaluate.py
src/violawake_sdk/tools/streaming_eval.py
src/violawake_sdk/tools/confusables.py
src/violawake_sdk/tools/contamination_check.py
src/violawake_sdk/tools/test_confusables.py
src/violawake_sdk/cli/evaluate.py
benchmark_v2/
benchmark_oww/
eval_clean/
tools/benchmark.py
tools/benchmark_regression_check.py
tools/build_clean_eval_set.py
tools/far_frr_analysis.py
tools/live_head_to_head.py
examples/streaming_eval.py
tests/unit/test_evaluate_oww.py
tests/unit/test_benchmark.py
tests/unit/test_performance.py
tests/unit/test_confusables.py
tests/benchmarks/                                  # bench_latency, ...
tests/golden_path_test.py
docs/COMPETITIVE_ANALYSIS.md
docs/AUDIT_2026_03_28.md
docs/ACCURACY_MISSION.md (top-level)              # → schedule move into docs/
docs/ADVERSARY_AUDIT.md (top-level)               # → schedule move into docs/
docs/E2E_READINESS.md (top-level)                 # → schedule move into docs/
docs/FUNCTIONAL_GAP_ANALYSIS.md (top-level)       # → schedule move into docs/
docs/LAUNCH_READINESS.md (top-level)              # → schedule move into docs/
docs/S1.3_REQUIREMENTS_SYNTHESIS.md
docs/PROGRESS.md (top-level)                      # → schedule move into docs/
```

**Success criteria:**
- Every headline number on a public page is reproducible by running a
  named script in this lane against a checked-in corpus at a pinned
  model SHA.
- `benchmark_regression_check.py` runs in CI on every release tag and
  fails on regression beyond a documented delta.
- Per-category FAR/FRR is published and updated on retrain.

**Oracle SC:**
- Negative probes: (a) a `ModelSpec` with a wrong SHA — the benchmark
  must refuse to run; (b) a corpus row labeled "viola" but actually
  containing music — contamination check flags.
- Baseline: the v2 benchmark run reproduces `BENCHMARK_REPORT_v2.md`
  numbers within tolerance.

---

### 6. SDK CLI & Sample Tools

**Capability question:** *"Can a user run `violawake-train`,
`violawake-eval`, `violawake-collect`, and `violawake-download` and have
each command do what its `--help` says, on a clean install?"*

**Scope:** `LOCKED`. **Oracle:** `NEEDS-ORACLE` — `tests/unit/test_cli.py`
exists but there is no end-to-end "clean venv → each CLI runs to its
documented outcome" check.

**Owns:**
```
src/violawake_sdk/cli/download.py
src/violawake_sdk/tools/collect_samples.py
src/violawake_sdk/tools/download_corpus.py
src/violawake_sdk/tools/download_model.py
src/violawake_sdk/tools/expand_corpus.py
src/violawake_sdk/tools/generate_samples.py
src/violawake_sdk/tools/train.py
tests/unit/test_cli.py
tests/unit/test_download_corpus.py
tests/unit/test_model_download.py
examples/basic_detection.py
examples/async_detection.py
```

**Success criteria:**
- Each CLI command runs to completion on a clean venv with documented
  args.
- `--help` output matches what's published in the README and API docs.
- The `examples/` scripts run unmodified after `pip install
  "violawake[oww]"` + `download_models()`.

**Oracle SC:**
- Negative probes: a CLI is removed from `pyproject.toml`'s
  `[project.scripts]` — the live oracle must fail.
- Baseline: a fresh venv + `pip install -e .` + each CLI's documented
  invocation exits 0.

---

### 7. Public API & Distribution

**Capability question:** *"Does `pip install violawake==<version>` give a
user a working SDK with the documented public API surface, the right
models available via `ModelCache`, and a CHANGELOG entry?"*

**Scope:** `LOCKED`. **Oracle:** `NEEDS-ORACLE` — there is a release
workflow but no live "install + import + run" check on the published
wheel.

**Owns:**
```
src/violawake_sdk/__init__.py                     # public API contract
src/violawake_sdk/_constants.py                   # audio contract canon
src/violawake_sdk/_exceptions.py                  # error hierarchy
src/violawake_sdk/models.py                       # ModelSpec registry
src/violawake_sdk/py.typed
pyproject.toml
dist/                                              # built wheels (gitignored)
models/                                            # top-level shipped onnx
tools/fetch_release_models.py
tools/update_model_registry.py
tests/unit/test__constants.py
tests/unit/test_models.py
tests/integration/test_sdk_surface.py
tests/integration/test_feature_completeness.py
tests/live/test_live_sdk.py
docs/adr/ADR-005-packaging.md
docs/adr/ADR-003-python-first.md
docs/adr/ADR-004-open-core.md
CHANGELOG.md
RELEASE_NOTES.md
.github/workflows/release.yml
.github/workflows/model-verify.yml
```

**Success criteria:**
- A published version's public API matches the version's documented
  surface (no accidental removals; no undocumented additions).
- `ModelCache` resolves every registered `ModelSpec`'s URL + SHA on the
  live CDN.
- CHANGELOG is updated in the same commit as any user-visible behavior
  change.

**Oracle SC:**
- Negative probes: (a) a `ModelSpec` URL goes 404 — the live oracle
  fails; (b) a public symbol is removed — the surface test fails; (c) a
  user-visible behavior changes without a CHANGELOG entry — gate fails.
- Baseline: latest published version installs clean in a Python 3.10,
  3.11, 3.12 venv on Linux + Windows + macOS.

---

### 8. SaaS Console — Backend

**Capability question:** *"Does `api.violawake.com` correctly serve
sign-up → sample upload → training job → model download → billing → email
flows under load?"*

**Scope:** `LOCKED`. **Oracle:** `ORACLE-DRAFT` — `tests/live/test_live_*`
exercise the live API but coverage is partial (no load test, no auth
fuzzing).

**Owns:**
```
console/backend/                                   # FastAPI app + alembic
                                                  # + services + tests
console/decoder/                                   # decoder Docker service
workers/support-email/                             # inbound email Worker
tests/live/test_live_api.py
tests/live/test_live_billing.py
tests/live/test_live_email.py
tests/live/full_pipeline_e2e.py
tests/live/run_smoke.sh
tests/live/conftest.py
tests/live/README.md
tests/live/RESULTS_2026-05-06.md
tests/live/RESULTS_2026-05-07.md
.github/workflows/console-ci.yml
docs/api/                                          # generated FastAPI docs
                                                  # (NOT marketing pages —
                                                  # this is backend OpenAPI)
```

**Cross-lane dependencies:** Lane 11 may verify `docs/api/` regeneration as
part of doc-sync and public-claim checks, but the generated files and OpenAPI
source contract stay Lane 8-owned.

**Success criteria:**
- `GET /api/health` returns 200 from the live tunnel under nominal load.
- The full sign-up → training-job → model-download flow completes on a
  live integration run.
- Authn/authz tests (cert pinning, rate limit, role boundaries) pass
  before every deploy.

**Oracle SC:**
- Negative probes: (a) a route is removed from `console/backend/app/
  routes/` — the live smoke fails; (b) the `wakeword-tunnel-1` container
  is stopped — `api.violawake.com` returns 5xx within the documented
  detection window; (c) a billing route silently accepts an unauthenticated
  request — auth test fails.
- Baseline: the live smoke (`tests/live/run_smoke.sh`) is green against
  the deployed image SHA.

---

### 9. SaaS Console — Frontend

**Capability question:** *"Does `violawake.com` correctly render the
sign-up, console, comparison, pricing, and docs pages, and talk to the
live backend?"*

**Scope:** `LOCKED`. **Oracle:** `ORACLE-DRAFT` — `tests/live/
test_live_website.py` exists and the 2026-05-07 accessibility audit
landed, but coverage gaps remain (full responsive check, cross-browser).

**Owns:**
```
console/frontend/                                  # React + Vite +
                                                  # everything BUT
                                                  # dist/wasm/
tests/live/test_live_website.py
tests/live/ACCESSIBILITY_AUDIT_2026-05-07.md
.github/workflows/deploy-pages.yml
docs/SEO_AUDIT.md
docs/SEO_OUTREACH.md
docs/SEO_RUNBOOK.md
```

**Success criteria:**
- Build with `VITE_API_URL=https://api.violawake.com/api` produces a
  bundle whose live request lands on the backend (the 2026-05-07
  regression must stay caught).
- All comparison / pricing / docs pages render without console errors on
  the latest stable Chrome + Firefox + Safari.
- Accessibility audit baseline (2026-05-07) does not regress.

**Oracle SC:**
- Negative probes: (a) `VITE_API_URL` unset at build time — the live
  oracle must catch a 405 from same-origin `/api`; (b) any page returns
  a 5xx or a client-side render error — fails.
- Baseline: live `violawake.com` exercise covers every linked page on
  the deployed build.

---

### 10. Infrastructure & DevOps

**Capability question:** *"Are deploys reproducible, backups taken on
schedule, CI green on every PR, and the production stack observable?"*

**Scope:** `LOCKED`. **Oracle:** `NEEDS-ORACLE` — there are runbooks but
no instrument that proves "the documented deploy steps, run today,
produce the documented end state." Backup verification (`backup_to_r2*`)
is partially scripted; restore drills aren't.

**Owns:**
```
docker-compose.production.yml
docker-compose.viola-bridge.yml
railway.json
railway.toml
scripts/backup_postgres.sh
scripts/backup_postgres.cmd
scripts/backup_to_r2.py
scripts/backup_to_r2_wrangler.sh
scripts/backup-task.xml
scripts/check_in_flight_jobs.py
scripts/deploy_launch.py
scripts/verify_models.py
scripts/live_compare.py                            # ops monitoring,
                                                  # not benchmark
scripts/ai_debug_session.log                      # ← scheduled for cleanup
tools/audit_deps.py
tools/setup_github_repo.sh
tools/merge_worktrees.py
tools/quality_gate.py
logs/
.github/workflows/ci.yml
.github/workflows/docs.yml
docs/DEPLOYMENT.md
docs/OPERATIONS_RUNBOOK.md
docs/RUNBOOK.md
docs/PRODUCTION_STATUS.md
docs/PRE_LAUNCH_CHECKLIST.md
docs/SECURITY.md
SECURITY.md                                        # top-level — duplicate
                                                  # of docs/SECURITY.md;
                                                  # resolve to one source
```

**Success criteria:**
- A documented deploy of backend + frontend, executed from a clean
  shell, lands the expected image SHA on the live URL with `< 5 min`
  total downtime.
- Postgres backup runs on schedule, verified by a periodic restore
  drill into a scratch container.
- All CI workflows green on the trunk; failing workflow blocks the
  merge.

**Oracle SC:**
- Negative probes: (a) `docker-compose.production.yml` references a
  nonexistent image tag — deploy fails fast, not silently; (b) the
  backup script silently errors — alert fires within the documented
  window.
- Baseline: a documented deploy at a known SHA reproduces the live
  state.

---

### 11. Marketing & Developer Docs

**Capability question:** *"Does every outward-facing artifact —
README, PyPI description, generated API docs, Show-HN draft, SEO
content — match the current state of the product and reproduce its
claims?"*

**Scope:** `LOCKED`. **Oracle:** `NEEDS-ORACLE` — there is no gate that
catches "README claims a feature that doesn't exist" or "the comparison
table cites a benchmark that was renamed."

**Owns:**
```
README.md
LICENSE
CONTRIBUTING.md
docs/REGISTRY.md
docs/PRD.md
docs/ARCHITECTURE.md
docs/BUSINESS_PLAN.md
docs/SHOW_HN_DRAFT.md
docs/TEST_STRATEGY.md
docs/index.html
scripts/generate_docs.py
docs/archive/                                      # superseded docs
                                                  # (kept for trace)
```

**Cross-lane dependencies:** Lane 12 may cite `docs/archive/` during
governance reviews, but process and audit archives live in `_diag/`.
`docs/archive/` is only for superseded public/developer docs registered in
`docs/REGISTRY.md`.

**Success criteria:**
- Every numeric claim in `README.md`, the comparison pages, and the
  PyPI description traces to a named script + corpus in Lane 5.
- API docs (`docs/api/` HTML) regenerate from current source without
  diffs in CI.
- `docs/REGISTRY.md` lists every authoritative doc; nothing
  authoritative lives outside the registry.

**Oracle SC:**
- Negative probes: (a) README cites a number not produced by any Lane 5
  script — gate fails; (b) a public symbol exists in the SDK but is
  missing from API docs — gate fails.
- Baseline: a doc-sync pass on the current commit produces zero diff.

---

### 12. Project Governance & Process

**Capability question:** *"Is the orchestration discipline itself
healthy — is CLAUDE.md current, is this Lane Ledger accurate, are
audit findings tracked to closure?"*

**Scope:** `LOCKED`. **Oracle:** `N/A (process)` — Governance is not a
product capability and does not get an instrumented oracle in the same
sense. Its health is asserted by the orchestrator on every startup
check (see CLAUDE.md → "Orchestrator startup checklist").

**Owns:**
```
CLAUDE.md
docs/LANE_LEDGER.md                                # this file
audit/active/                                      # to be created;
                                                  # investigations live here
_diag/                                             # dated investigation
                                                  # artifacts
docs/ROADMAP_10_OF_10.md                          # product roadmap
```

**Cross-lane dependencies:** Lane 4 depends on the training/eval milestones in
`docs/ROADMAP_10_OF_10.md`, but the multi-subsystem roadmap remains
Governance-owned so roadmap updates stay coordinated across lanes. Lane 11
owns archived public/developer docs under `docs/archive/`; Governance archives
process and audit evidence under `_diag/`.

**Why this lane exists:** the LANE_LEDGER itself, this CLAUDE.md, audit
artifacts, and investigation dirs all need a home. Without one, they
silently grow into product-lane file lists and break the disjointness
rule.

**Process bar (in lieu of oracle SC):**
- Every lane in this ledger reflects current scope. New work that
  doesn't fit becomes an investigation INSIDE a lane, not a new lane.
- Every fix-like commit either ships a `Ratchet:` gate or carries an
  enumerated `Ratchet-Exempt:` reason.
- The orchestrator runs the startup checklist (CLAUDE.md → Orchestrator
  startup checklist) at the start of every session.

---

## Cruft — resolved 2026-06-03

The 2026-06-03 cleanup audit (committed under `audit-2026-06-03/cleanup`)
resolved the historical cruft block. Surviving open item:

```
# duplicate SDK namespace — CONFIRM with founder
src/violawake/                 # compat shim re-exporting violawake_sdk;
                               # currently in Lane 1. Keep for back-compat
                               # or deprecate in next major.
```

If cruft accumulates again, this block is the index — list what doesn't
belong to any lane and schedule its disposition.

---

## How to use this ledger

- **Before dispatching ANY work**, identify the lane it belongs to. If
  it doesn't fit one, the work is either (a) cleanup that goes in
  `_cruft/`-handling under Lane 12, or (b) the wrong scope.
- **Before declaring a lane "done"**, confirm both `Scope = LOCKED` and
  `Oracle = ORACLE-LIVE`. A green test is not done.
- **Before opening a new lane**, prove the new capability is genuinely
  outside every existing lane's bounded context. Default answer: NO. New
  features almost always fit an existing lane.
- **Disjointness check (run periodically):**
  ```bash
  # the sum of all lanes' tracked files must equal `git ls-files | wc -l`,
  # minus anything in the cruft block.
  git ls-files | wc -l
  ```
  Drift here is an early signal a lane has silently grown into another's
  territory.

---

**Last orchestrator review:** 2026-06-03 (this is the initial ledger).
Next review: when any lane changes `Scope` or `Oracle` status, OR on
every major release tag — whichever comes first.
