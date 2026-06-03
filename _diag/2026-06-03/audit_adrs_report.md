# ADR Audit Report - 2026-06-03

Scope: recommendations-only audit of ADR-001 through ADR-005 against current system reality.

Requested audit root: `J:\CLAUDE\PROJECTS\Wakeword-adr-audit`.

Audited source root: `J:\CLAUDE\PROJECTS\Wakeword`. The requested audit root contains the diagnostic inputs; the code, ADRs, lane ledger, and tests live in the sibling Wakeword source checkout. The source checkout was read-only for this audit.

Overall verdict: MUST-FIX.

Reason: ADR-001 and ADR-002 have material decision drift in current code without an explicit "Superseded by ADR-NNN" pointer. ADR-003 and ADR-004 are current. ADR-005's locked decision is current, but the consequences/implementation details contain stale packaging details.

## Correction Memo Compliance

`_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` Section A applies. This report is recommendations-only. I did not edit source code, ADRs, `quality/gates.yaml`, release workflows, model registry data, or production systems.

Evidence/probes used:

- `python -m pytest tests/unit/test_wake_decision_policy.py tests/unit/test_wake_detector_core.py tests/unit/test_oww_backbone.py tests/unit/test_models.py tests/unit/test_training_pipeline.py tests/integration/test_sdk_surface.py -q` -> `139 passed, 2 warnings in 19.31s`.
- `python -m pytest tests/unit/test_cli.py::TestTrainCLI -q` -> `12 passed in 5.33s`.
- `python -m pytest tests/integration/test_feature_completeness.py::TestBackends tests/integration/test_feature_completeness.py::TestModelRegistry tests/integration/test_feature_completeness.py::TestPackageMetadata -q` -> `21 passed in 1.60s`.
- Broken `ModelSpec` URL probe -> `HTTPError: 404 Client Error: Not Found for url: https://raw.githubusercontent.com/GeeIHadAGoodTime/ViolaWake/definitely-missing/broken.onnx`.
- Public SDK symbol removal probe -> `ImportError: cannot import name 'WakeDetector' from 'violawake_sdk'`.
- Broken threshold probe -> `broken-threshold-low-score-detects=True`, confirming why Lane 1's threshold-drift negative probe matters.
- Local wheel build -> `Successfully built violawake-0.2.6-py3-none-any.whl`; wheel inspection found `onnx_entries []` and `tflite_entries []`.

## Supersession Check

`docs/REGISTRY.md:46` says accepted ADRs are immutable and must be changed by creating a new superseding ADR. `docs/REGISTRY.md:50-54` lists ADR-001 through ADR-005 as `Accepted`. `docs/REGISTRY.md:110` says superseded ADRs must be marked `Superseded by ADR-XXX`.

No ADR-001 through ADR-005 file carries a superseded status, and the current registry does not point any of them at a newer ADR.

Relevant later-history signals:

- `72ed857 docs: archive MLP-era records, update docs for temporal_cnn era`
- `527be8b fix(training): enforce temporal-only wake model pipeline`
- `1d1952b fix(tests): update test refs from viola_mlp_oww to temporal_cnn, fix registry test`
- `4c46498 fix: silence quality gate, training pipeline consistency, remove TFLite placeholder`
- `520e121 fix(stt): preserve faster-whisper import failures`

Those commits explain the drift, but they do not add an explicit superseding ADR pointer.

## ADR-001 - ONNX Runtime for All Model Inference

Status: DRIFT.

Claim:

- `docs/adr/ADR-001-onnx-runtime.md:37` locks "Use ONNX Runtime ... for all model inference in the ViolaWake SDK."
- `docs/adr/ADR-001-onnx-runtime.md:39` says all models are `.onnx` and loaded exclusively via `onnxruntime.InferenceSession`; no TensorFlow, PyTorch, or JAX at inference time.
- Consequences start at `docs/adr/ADR-001-onnx-runtime.md:126`, including ONNX-centric distribution and optional training/GPU dependencies.

Current evidence:

- ONNX is still implemented: `src/violawake_sdk/backends/onnx_backend.py:3`, `src/violawake_sdk/backends/onnx_backend.py:28`, and `src/violawake_sdk/backends/onnx_backend.py:88` wrap and instantiate `onnxruntime.InferenceSession`.
- Current wake inference is no longer ONNX-only: `src/violawake_sdk/backends/__init__.py:30-41` exposes `onnx`, `tflite`, and `auto`; `src/violawake_sdk/backends/__init__.py:54-58` accepts the TFLite backend; `src/violawake_sdk/backends/__init__.py:73-91` auto-selects ONNX first and falls back to TFLite.
- TFLite is a real inference backend, not just documentation: `src/violawake_sdk/backends/tflite_backend.py:4-18` documents TFLite model files; `src/violawake_sdk/backends/tflite_backend.py:101-127` imports `tflite_runtime` or TensorFlow Lite; `src/violawake_sdk/backends/tflite_backend.py:297-324` implements `TFLiteBackend`.
- The public wake detector advertises pluggable runtimes: `src/violawake_sdk/wake_detector.py:278-305` describes ONNX/TFLite/auto; `src/violawake_sdk/wake_detector.py:319` allows `("onnx", "tflite", "auto")`; `src/violawake_sdk/wake_detector.py:327` defaults to `auto`; `src/violawake_sdk/wake_detector.py:577-597` resolves `.onnx` and `.tflite` files.
- Companion runtime surfaces are also not direct `onnxruntime.InferenceSession`: `src/violawake_sdk/tts.py:138-165` uses `kokoro_onnx.Kokoro`; `src/violawake_sdk/stt.py:43` and `src/violawake_sdk/stt.py:141-161` use `faster_whisper.WhisperModel`.
- User docs reflect current drift: `README.md:125`, `README.md:640`, and `README.md:679-700` document a TFLite alternative.

Consequence check:

- Positive consequence remains partially observable: ONNX Runtime remains the primary core dependency in `pyproject.toml:41`.
- Material contradiction is observable: `pyproject.toml:76-79` adds a `tflite` extra as an alternative backend, which violates the ADR's "all model inference" and "loaded exclusively" language.

Owning lane:

- Lane 1 owns wake detection and ADR-001: `docs/LANE_LEDGER.md:121-165`.

Recommended resolution:

- Create a superseding ADR, or update ADR-001's status to `Superseded by ADR-NNN`, describing the current architecture: ONNX Runtime is the default/core wake runtime, TFLite is an optional wake backend, OWW backbone assets may be package-managed, Kokoro and faster-whisper use their domain runtimes through wrapper libraries.
- If the intended locked decision is still ONNX-only, the owning lane must remove or quarantine TFLite/STT/TTS inference paths from the SDK surface and README. That would be a larger product/code change and should not be done by this audit lane.

## ADR-002 - OWW Feature Extractor

Status: DRIFT.

Claim:

- `docs/adr/ADR-002-oww-feature-extractor.md:29` says the shipped default later moved to `temporal_cnn`, but the core choice remains frozen OWW plus ViolaWake-owned wake head.
- `docs/adr/ADR-002-oww-feature-extractor.md:37` locks "Train only the MLP classification head, not the feature extractor."
- `docs/adr/ADR-002-oww-feature-extractor.md:62` says `viola_v4.onnx` remains in the model registry as a lightweight secondary option.
- `docs/adr/ADR-002-oww-feature-extractor.md:117-118` lists OWW backbone plus current default `temporal_cnn`.
- `docs/adr/ADR-002-oww-feature-extractor.md:142-148` lists MLP CPU training and a maintained fallback CNN path as consequences.

Current evidence:

- The frozen OWW backbone part is current: `src/violawake_sdk/oww_backbone.py:24-25` defines 96-dim OWW embeddings and 1,280-sample chunks; `src/violawake_sdk/oww_backbone.py:104-147` resolves pinned OWW backbone assets; `src/violawake_sdk/oww_backbone.py:171-179` creates an `OpenWakeWordBackbone`; `src/violawake_sdk/oww_backbone.py:227-297` produces 96-dim embeddings.
- Wake detection still uses OWW embeddings before the wake head: `src/violawake_sdk/wake_detector.py:540-542`, `src/violawake_sdk/wake_detector.py:648-650`, and `src/violawake_sdk/wake_detector.py:709-739`.
- The current production head is temporal CNN, not MLP: `src/violawake_sdk/tools/train.py:6` says "TemporalCNN classifier head on top of frozen OpenWakeWord"; `src/violawake_sdk/tools/train.py:1231-1258` marks TemporalCNN as the production architecture; `src/violawake_sdk/tools/train.py:1499-1505` builds `TemporalCNN`; `src/violawake_sdk/tools/train.py:1716-1717` records `architecture: temporal_cnn`.
- Legacy MLP training is intentionally removed: `src/violawake_sdk/tools/train.py:2055-2057`; `tests/unit/test_training_pipeline.py:12-17`.
- The deprecated MLP/CNN model-registry entries are intentionally absent: `tests/unit/test_models.py:54-57` asserts `viola_mlp_oww` and `viola_cnn_v4` are not in `MODEL_REGISTRY`.
- Current registry default is temporal CNN: `src/violawake_sdk/models.py:48-56` registers `temporal_cnn` and `oww_backbone`; `src/violawake_sdk/models.py:88-89` says TFLite conversion is not yet validated; `src/violawake_sdk/models.py:107-111` aliases `viola` to `temporal_cnn` and marks `oww_backbone` package-managed.

Consequence check:

- Positive consequence remains observable: the OWW backbone is frozen and reused.
- Material contradiction is observable: MLP-only training and maintained CNN fallback are no longer true in code or tests.

Owning lane:

- Lane 1 owns wake detection and ADR-002: `docs/LANE_LEDGER.md:121-165`.

Recommended resolution:

- Supersede ADR-002 with a decision that locks the actual current state: frozen OWW backbone plus a ViolaWake-owned temporal wake head, with legacy MLP/CNN paths treated as historical unless explicitly restored.
- If product ownership wants ADR-002 as written, the owning lane must restore MLP training and the lightweight CNN registry/runtime path, then add regression coverage. Current tests intentionally assert the opposite, so the lower-risk fix shape is an ADR supersession/update rather than code rollback.

## ADR-003 - Python First

Status: CURRENT.

Claim:

- `docs/adr/ADR-003-python-first.md:34` locks a Python-native SDK first and no C library for Phase 1.
- `docs/adr/ADR-003-python-first.md:36` says Python 3.10+ throughout, with ONNX Runtime as the inference layer and no project C extensions beyond PyPI wheels.
- `docs/adr/ADR-003-python-first.md:121-136` lists consequences: no MCU/C/C++ SDK, no easy language bindings, bindings deferred.

Current evidence:

- Packaging is Python-native: `pyproject.toml:8` describes a Python-native SDK; `pyproject.toml:35` requires Python `>=3.10`; `pyproject.toml:193-194` packages `src/violawake_sdk`.
- Public SDK imports are Python-first and tested: `tests/integration/test_sdk_surface.py:38-50` imports `WakeDetector`, `VoicePipeline`, `VADEngine`, model registry helpers, and CLI entrypoints.
- Local code search found no C/C++/Rust/Go/Java/Swift/Kotlin core or binding files: `rg --files | rg "\.(c|cc|cpp|h|hpp|rs|go|swift|kt|java)$|(^|/)include/|ffi|bindings"` returned no matches.
- The integration surface test passed in the 139-test audit run.

Consequence check:

- No C/C++ SDK or FFI binding layer is present.
- Python 3.10+ packaging and Python entrypoints are current.

Owning lane:

- Lane 7 owns public API/distribution and ADR-003: `docs/LANE_LEDGER.md:432-462`.

Recommendation:

- No ADR-003 fix is required for the Python-first decision.
- When ADR-001 is superseded, update or supersede the ONNX-runtime wording shared by ADR-003 so the runtime layer is not silently inconsistent with the current TFLite/STT/TTS surfaces.

## ADR-004 - Open-Core Licensing

Status: CURRENT.

Claim:

- `docs/adr/ADR-004-open-core.md:31` locks Apache 2.0 for SDK code, models, and training pipeline, with commercial differentiation from hosted managed Console service.
- `docs/adr/ADR-004-open-core.md:34-35` draws the split between developer-machine open source and hosted Console commercial service.
- `docs/adr/ADR-004-open-core.md:90-101` lists open SDK/training/model/docs/eval/tooling and paid managed training Console.
- `docs/adr/ADR-004-open-core.md:157-173` lists consequences around trust, forkability, and Console revenue.

Current evidence:

- `LICENSE:1` is Apache License 2.0. `LICENSE:150-159` includes third-party notices for OpenWakeWord, Kokoro, and ONNX Runtime.
- `pyproject.toml:10` points packaging at `LICENSE`; `pyproject.toml:26` uses the Apache Software License classifier.
- `README.md:7`, `README.md:22-24`, and `README.md:1252-1254` present the SDK/training/model posture as Apache 2.0 and disclose OWW.
- Console commercialization is implemented/documented: `console/frontend/src/pages/Pricing.tsx:14-19` says the SDK/CLI are Apache/free and the Console is the paid product; `console/frontend/src/pages/Terms.tsx:15-18` and `console/frontend/src/pages/Terms.tsx:88-97` distinguish Apache SDK use from Console terms; `console/backend/app/routes/billing.py` and `console/backend/app/config.py:83-87` implement billing surfaces.

Consequence check:

- Enterprise-friendly Apache licensing is observable in package metadata and repository license.
- Commercial differentiation is observable in Console pricing, terms, and billing code.

Owning lane:

- Lane 7 owns ADR-004 and public distribution/commercial boundary docs: `docs/LANE_LEDGER.md:432-462`.

Recommendation:

- No ADR-004 fix required.
- Optional cleanup only: if the project wants a separate `NOTICE` file instead of embedding third-party notices in `LICENSE`, Lane 7 can handle that as non-blocking packaging/license hygiene. This is not a MUST-FIX for this ADR audit.

## ADR-005 - Packaging

Status: TECHNICAL-DEBT.

Claim:

- `docs/adr/ADR-005-packaging.md:33` locks Python code on PyPI via `hatchling`, with model files distributed via GitHub Releases, SHA-256 verification, and download-on-demand model cache.
- `docs/adr/ADR-005-packaging.md:104-123` describes the GitHub Releases + model cache pattern.
- `docs/adr/ADR-005-packaging.md:176-182` describes the release workflow building wheel/sdist, running tests, uploading model assets, and publishing via PyPI.
- `docs/adr/ADR-005-packaging.md:192` says core includes `onnxruntime, numpy, pyaudio, scipy`.
- `docs/adr/ADR-005-packaging.md:202-213` lists consequences: fast pip install, no bundled models, SHA verification, first-run internet requirement, offline pre-seeding.

Current evidence:

- Hatchling packaging is current: `pyproject.toml:1-3`.
- Python code packaging excludes large/runtime artifacts: `pyproject.toml:161-191` excludes model/audio artifacts from sdist, and `pyproject.toml:193-194` packages only `src/violawake_sdk`.
- Current core dependencies do not include `pyaudio`: `pyproject.toml:37-45` lists core dependencies; `pyproject.toml:48-51` moves `pyaudio` to the `audio` extra.
- Model cache/registry is current: `src/violawake_sdk/models.py:3-5`, `src/violawake_sdk/models.py:26-61`, `src/violawake_sdk/models.py:153-157`, `src/violawake_sdk/models.py:272`, `src/violawake_sdk/models.py:393-437`, and `src/violawake_sdk/models.py:552-585`.
- OWW has a package-managed exception: `src/violawake_sdk/models.py:56-61`, `src/violawake_sdk/models.py:107-111`, and `src/violawake_sdk/models.py:355`.
- Release workflow builds via hatch and creates GitHub releases: `.github/workflows/release.yml:75-84`, `.github/workflows/release.yml:109-138`, and `.github/workflows/release.yml:142-164`.
- The workflow intentionally tolerates missing model assets: `.github/workflows/release.yml:109-116` says PyPI users do not depend on release assets being attached, and `fail_on_unmatched_files: false` appears at `.github/workflows/release.yml:125-138`.
- Local wheel verification found no bundled `.onnx` or `.tflite` entries.

Consequence check:

- Positive consequences are observable: fast package build, no model files bundled in the wheel, SHA verification and download-on-demand cache are implemented.
- Technical debt is observable: ADR-005's core dependency table is stale for `pyaudio`, and the "models are published as GitHub Releases assets" language is no longer fully true for the package-managed OWW backbone and tolerant release-asset workflow.

Owning lane:

- Lane 7 owns public API/distribution and ADR-005: `docs/LANE_LEDGER.md:432-462`.

Recommendation:

- Update/supersede ADR-005 wording narrowly: core dependencies exclude `pyaudio`; microphone capture is an `audio` extra; `temporal_cnn` is release/model-cache managed; `oww_backbone` is package-managed through OpenWakeWord with integrity checks; the release workflow may publish PyPI even when optional model-release attachment is absent.
- No code change is recommended from this audit; the implementation already matches the more mature packaging behavior.

## Lane-Owned Drift Resolutions

- ADR-001 DRIFT -> Lane 1. Recommended fix shape: superseding ADR for multi-runtime reality, or owner-lane removal of TFLite/domain-runtime surfaces if ONNX-only remains intended.
- ADR-002 DRIFT -> Lane 1. Recommended fix shape: superseding ADR for frozen OWW + temporal head reality, or owner-lane restoration of MLP/CNN fallback if the old decision remains intended.
- ADR-003 CURRENT -> Lane 7. No fix required, but harmonize runtime wording when ADR-001 is superseded.
- ADR-004 CURRENT -> Lane 7. No fix required.
- ADR-005 TECHNICAL-DEBT -> Lane 7. Recommended fix shape: stale ADR wording/table update or supersession; no code change.

## Mandatory Five-Bullet Self-Audit Gate

- I did not modify source code, ADRs, lane ledger, release workflows, production config, or `quality/gates.yaml`; this remained recommendations-only.
- I did not run production destructive operations, live billing operations, live release publishing, or production model-CDN mutation; evidence came from static reads, local tests, local probes, and a local wheel build.
- I did not exhaustively implement every Lane 1 oracle negative mutation because this audit lane does not own source/gate changes; I ran the available decision-policy tests and a local threshold-drift probe, and recorded the missing oracle work as lane-owned.
- I did not verify latest published PyPI install across the full Lane 7 baseline matrix; local package/build/surface probes were sufficient for ADR-currentness, while full distribution oracle certification remains Lane 7 work.
- I did not audit every downstream document that cites these ADRs; the mission was ADR-vs-current-code accuracy, and broader source-section/doc-sync remediation belongs to the owning lanes after ADR drift is resolved.
