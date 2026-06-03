# Lane 11 Audit Report - Marketing and Developer Docs

Date: 2026-06-03
Branch: `audit-2026-06-03/l11-marketing`
Verdict: MUST-FIX

## Verdict

MUST-FIX remains. Lane-owned public copy and the registry were corrected in this branch, but the release cannot pass Lane 11 because:

1. Live public pages still show unsupported production-eval numbers and comparison-page numbers.
2. `docs/api/` is stale against the SDK public surface: `AsyncVoicePipeline` is exported but missing from generated API HTML.
3. API doc regeneration could not be proven locally because the configured pdoc environment reports `pdoc installed: no`.

## Sources Read

- `docs/LANE_LEDGER.md` section 11.
- `CLAUDE.md` public-copy rules and ratchet rules.
- `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` section A.
- Live pages fetched with `curl.exe`: `/`, `/pricing`, `/compare/picovoice`, `/compare/openwakeword`, `/compare/snowboy`, `/docs`, `/faq`.
- Lane 5 benchmark sources: `benchmark_v2/run_benchmark.py`, `benchmark_v2/build_corpus.py`, `benchmark_v2/BENCHMARK_REPORT_v2.md`, `benchmark_v2/OPERATOR_BENCHMARK.md`.

## Implemented Fixes

### 1. Public README and PyPI Description Claims

PyPI uses the README as the long description:

```text
pyproject.toml:8:description = "Open-source wake word detection SDK with training pipeline - privacy-first, on-device, Python-native"
pyproject.toml:9:readme = "README.md"
```

Before:

```text
README.md:3: A production-tested wake word engine...
README.md:1055: Measured on i7-12700H, Windows 11, RTX 3060 (CPU inference)
README.md:669: 102 KB wake head + 1.33 MB shared OWW backbone = 1.43 MB total runtime footprint | ~8ms/frame
README.md:735: temporal_cnn ... ~25K
README.md:814: Record 10 voice samples in the browser
```

After:

```text
README.md:3: A wake word SDK with accessible training, ONNX inference, and a Python-first API.
README.md:36: cites benchmark_v2/BENCHMARK_REPORT_v2.md for 700-file negatives, 180 TTS positives, 5.49% EER vs 8.24% EER.
README.md:356: VAD table no longer publishes fixed latency estimates.
README.md:669: Architecture table no longer publishes fixed size/latency claims.
README.md:735: Architecture table no longer publishes parameter-count claims.
README.md:814: Web Console workflow now says "Record voice samples".
README.md:928: model sizes now point to violawake-list-models/model registry instead of hard-coded public copy.
README.md:1057: latency section now tells users to run pytest tests/benchmarks/bench_latency.py.
```

Reproducer for retained benchmark numbers:

```text
benchmark_v2/run_benchmark.py:40: CORPUS_DIR = Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_v2/corpus")
benchmark_v2/run_benchmark.py:232: collect_negatives()
benchmark_v2/run_benchmark.py:248: collect_positives(wake_word)
benchmark_v2/run_benchmark.py:417: writes EER table
benchmark_v2/run_benchmark.py:418: writes ROC AUC table
benchmark_v2/BENCHMARK_REPORT_v2.md:4: Shared negative corpus: 700 files
benchmark_v2/BENCHMARK_REPORT_v2.md:10: Matched positives: 180 viola, 180 alexa
benchmark_v2/BENCHMARK_REPORT_v2.md:19: EER 5.49% vs 8.24%
benchmark_v2/BENCHMARK_REPORT_v2.md:20: ROC AUC 0.9877 vs 0.9555
```

### 2. Show HN Draft Claims

Before:

```text
docs/SHOW_HN_DRAFT.md: d-prime=8.577, EER 0.8%, AUC 0.9993, 25K params, <5ms inference
docs/SHOW_HN_DRAFT.md: "big chungus" ... Grade A, zero false positives
```

After:

```text
docs/SHOW_HN_DRAFT.md:22: benchmark_v2/run_benchmark.py over benchmark_v2/corpus reports temporal_cnn at 5.49% EER versus openWakeWord Alexa at 8.24% EER on a shared 700-file negative corpus and 180 synthetic positives per system.
docs/SHOW_HN_DRAFT.md:24: quality gate reports EER, FAR, FRR, and ROC AUC before a model ships.
```

Reproducer: same `benchmark_v2/run_benchmark.py` and `benchmark_v2/BENCHMARK_REPORT_v2.md` lines listed above.

### 3. Forbidden Public-Copy Patterns

Before:

```text
docs/index.html: ViolaWake is a production-tested wake word SDK...
docs/COMPETITIVE_ANALYSIS.md: This is the biggest narrative correction.
```

After:

```text
docs/index.html:165: ViolaWake is a wake word SDK - Apache 2.0, ONNX-first, Python-native.
docs/COMPETITIVE_ANALYSIS.md:138: ViolaWake is no longer limited to "great training code, but you still need the CLI."
```

Local public-copy scan:

```text
python scripts/generate_docs.py --check-public-copy
Public copy check passed: 4 file(s)
```

Live forbidden-pattern scan:

```text
Live forbidden-pattern scan: no hits
```

### 4. Docs Registry Completeness

Before: authoritative docs were missing from `docs/REGISTRY.md`, including `CLAUDE.md`, `docs/LANE_LEDGER.md`, `docs/index.html`, `docs/api/index.html`, `docs/SECURITY.md`, and `BUILD_VS_BUY_AUDIT.md`.

After:

```text
docs/REGISTRY.md:21: Project Contract - CLAUDE.md
docs/REGISTRY.md:22: Lane Ledger - docs/LANE_LEDGER.md
docs/REGISTRY.md:30: Developer Docs Landing - docs/index.html
docs/REGISTRY.md:31: Generated API Docs - docs/api/index.html
docs/REGISTRY.md:43: Console Security Notes - docs/SECURITY.md
docs/REGISTRY.md:109: Build vs Buy Audit - BUILD_VS_BUY_AUDIT.md
```

Registry checks:

```text
All registry paths exist
All doc-meta LIVING/ADR files are listed in docs/REGISTRY.md
Contract-authoritative docs listed: CLAUDE.md, docs/LANE_LEDGER.md
```

### 5. Ratchet Checks Added

Implemented check-only modes in `scripts/generate_docs.py`:

```text
scripts/generate_docs.py:23: FORBIDDEN_PUBLIC_COPY_PATTERNS
scripts/generate_docs.py:34: UNSUPPORTED_LATENCY_SNAPSHOT
scripts/generate_docs.py:53: --check-public-copy
scripts/generate_docs.py:58: --check-api-public-surface
scripts/generate_docs.py:94: check_public_copy()
scripts/generate_docs.py:121: check_api_public_surface()
```

Negative public-copy probe:

```text
Public copy check failed:
- C:\Users\jihad\AppData\Local\Temp\lane11_bad_public_copy.md: forbidden public-copy pattern: Self-Certification(?: Note)?
- C:\Users\jihad\AppData\Local\Temp\lane11_bad_public_copy.md: forbidden public-copy pattern: Professional legal review is recommended
- C:\Users\jihad\AppData\Local\Temp\lane11_bad_public_copy.md: fixed latency table lacks a checked-in benchmark result
```

Negative API-doc probe:

```text
API public surface check failed:
- missing from docs/api: DetectorConfig
- missing from docs/api: AsyncWakeDetector
- missing from docs/api: WakeDecisionPolicy
- missing from docs/api: validate_audio_chunk
- missing from docs/api: ConfidenceResult
- missing from docs/api: ConfidenceLevel
- missing from docs/api: FusionStrategy
- missing from docs/api: NoiseProfiler
- missing from docs/api: PowerManager
- missing from docs/api: VADEngine
- missing from docs/api: TTSEngine
- missing from docs/api: STTEngine
- missing from docs/api: StreamingSTTEngine
- missing from docs/api: VoicePipeline
- missing from docs/api: AsyncVoicePipeline
- missing from docs/api: ViolaWakeError
- missing from docs/api: ModelNotFoundError
- missing from docs/api: AudioCaptureError
- missing from docs/api: ModelLoadError
- missing from docs/api: PipelineError
- missing from docs/api: VADBackendError
- missing from docs/api: list_models
- missing from docs/api: list_voices
```

Positive API-doc probe:

```text
API public surface check passed: 24 symbol(s)
```

Planned quality gates, not added to `quality/gates.yaml` per correction memo:

```yaml
quality:
  gate_id: lane11-public-copy-claims
  owner_lane: 11
  detector: python scripts/generate_docs.py --check-public-copy
  own_tests:
    - TBD - orchestrator should add a bad public-copy fixture containing Self-Certification and a fixed latency table
    - TBD - orchestrator should add a clean public-copy fixture
  cross_lane_deps:
    - lane5-benchmark-reproducer

quality:
  gate_id: lane11-api-public-surface-docs
  owner_lane: 11
  detector: python scripts/generate_docs.py --check-api-public-surface
  own_tests:
    - TBD - orchestrator should add a generated-docs fixture missing an exported symbol
    - TBD - orchestrator should add a generated-docs fixture containing all exported symbols
  cross_lane_deps:
    - lane8-sdk-public-surface

quality:
  gate_id: lane11-docs-registry-coverage
  owner_lane: 11
  detector: TBD - orchestrator should formalize the registry coverage PowerShell/Python scan used in this audit
  own_tests:
    - TBD - orchestrator should add missing-registry-entry and all-present fixtures
  cross_lane_deps: []
```

## Remaining Gaps

### A. Live Site Still Contains Unsupported Numbers

Command summary:

```text
https://violawake.com/ => 0.8%, 102 KB, 25,409, 8.58, d-prime
https://violawake.com/compare/picovoice => 0.8%, 10 hours, 102 KB, 8.58, 97%, d-prime
https://violawake.com/compare/openwakeword => 0.8%, 102 KB, 25,409, 8.58, d-prime
https://violawake.com/compare/snowboy => 102 KB, d-prime
```

These are what users see today. I did not edit live-site/frontend source in this lane because the audit scope says Lane 11 owns docs, while the live site source is outside Lane 11 ownership. The unsupported live production-eval claims remain MUST-FIX for release.

### B. API Docs Are Stale

SDK export exists:

```text
src/violawake_sdk/__init__.py:62: from violawake_sdk.pipeline import AsyncVoicePipeline, VoicePipeline
src/violawake_sdk/__init__.py:174: "AsyncVoicePipeline",
src/violawake_sdk/pipeline.py:508: class AsyncVoicePipeline:
```

Generated API HTML does not expose it as public API:

```text
docs/api/violawake_sdk.html:674: __all__ = [
```

Check output:

```text
python scripts/generate_docs.py --check-api-public-surface
API public surface check failed:
- missing from docs/api: AsyncVoicePipeline
```

Dry-run generation evidence:

```text
python scripts/generate_docs.py --dry-run
Command: C:\Users\jihad\viola-whisper\Scripts\python.exe -m pdoc --output-directory J:\CLAUDE\PROJECTS\Wakeword-l11-marketing\docs\api --docformat google violawake violawake_sdk
Output:  J:\CLAUDE\PROJECTS\Wakeword-l11-marketing\docs\api
pdoc installed: no
```

No giant generated HTML diff was committed. The gap is documented as required.

## Commands Run

```text
git worktree list
git -c core.fsmonitor=false -c status.showUntrackedFiles=no status --short --branch
git merge-base --is-ancestor master HEAD
curl.exe -L -sS https://violawake.com/
curl.exe -L -sS https://violawake.com/compare/picovoice
curl.exe -L -sS https://violawake.com/compare/openwakeword
curl.exe -L -sS https://violawake.com/compare/snowboy
python -m py_compile scripts/generate_docs.py
python scripts/generate_docs.py --check-public-copy
python scripts/generate_docs.py --check-api-public-surface
python scripts/generate_docs.py --dry-run
```

## Mandatory Self-Audit Gate

- [x] I cited exact before/after copy and command output for each fix.
- [x] I ran negative probes for the new public-copy and API-doc checks.
- [x] I did not edit `quality/gates.yaml`; planned gates are documented above.
- [x] I did not push, merge, or touch the master worktree.
- [x] I staged no live-site/frontend or Lane 8 generated API HTML fix to hide remaining missing evidence; remaining gaps are fail-closed in this report.

