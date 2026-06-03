# Lane 7 Audit Report - Public API & Distribution

Date: 2026-06-03  
Worktree: `J:\CLAUDE\PROJECTS\Wakeword-l7-distro`  
Branch: `audit-2026-06-03/l7-distro`  
Artifact source: PUBLISHED PyPI wheel, because `violawake` exists on PyPI.

## Verdict

MUST-FIX.

The latest published PyPI version, `violawake 0.2.6`, installs on Python
3.10/3.11/3.12, but the published wheel does not satisfy the lane contract:

1. The published wheel's `violawake-download` entry point fails because
   `violawake_sdk.tools` is missing from the wheel.
2. The README-documented `import violawake` compatibility package is missing
   from the published wheel.
3. ADR-005 documents a `ModelCache` class, but the published SDK does not
   expose one.
4. `WakewordDetector` is documented as a top-level compatibility alias, but
   it was missing from `violawake_sdk.__all__`.
5. `oww_backbone` is a `ModelSpec` whose URL returns 404, so the registry does
   not meet the "every ModelSpec URL + SHA resolves" criterion.
6. `RELEASE_NOTES.md` was stale for the current published release.

The branch fixes items 1-4 and 6 for the next wheel. Item 5 remains a real
MUST-FIX because the correct repair crosses the wake-backbone/package-managed
model boundary.

## Binding Sources

- Lane 7 question and ownership: `docs/LANE_LEDGER.md:432-466`.
- Lane 7 success criteria: `docs/LANE_LEDGER.md:468-474`.
- Lane 7 negative probes: `docs/LANE_LEDGER.md:476-480`.
- PyPI evidence rule: `CLAUDE.md:205-209`, `CLAUDE.md:923-924`.
- Pre-publish correction: `J:\CLAUDE\PROJECTS\Wakeword\_diag\2026-06-03\SC_AUDIT_ROUND_1_CORRECTIONS.md:128-142`.
- Negative-probe correction: `J:\CLAUDE\PROJECTS\Wakeword\_diag\2026-06-03\SC_AUDIT_ROUND_1_CORRECTIONS.md:19-35`.

## Published Wheel Evidence

Command:

```powershell
pip index versions violawake
```

Output excerpt:

```text
violawake (0.2.6)
Available versions: 0.2.6, 0.2.5, 0.2.4, 0.2.2, 0.2.1, 0.2.0, 0.1.0
LATEST:    0.2.6
```

Command:

```powershell
pip download violawake==0.2.6 --no-deps -d _diag/2026-06-03/wheel
```

Output excerpt:

```text
Saved .\_diag\2026-06-03\wheel\violawake-0.2.6-py3-none-any.whl
Successfully downloaded violawake
```

Wheel contents probe:

```text
top-levels ['violawake-0.2.6.dist-info', 'violawake_sdk']
violawake entries []
```

Clean install/import matrix from the downloaded published wheel:

```text
Python 3.10.20: py 3.10.20 violawake_sdk 0.2.6 exports 25
Python 3.11.9:  py 3.11.9 violawake_sdk 0.2.6 exports 25
Python 3.12.13: py 3.12.13 violawake_sdk 0.2.6 exports 25
```

Published quick-start command failure:

```powershell
_diag/2026-06-03/venv311/Scripts/violawake-download --model temporal_cnn
```

Output:

```text
ModuleNotFoundError: No module named 'violawake_sdk.tools'
```

Published compatibility import failure:

```powershell
_diag/2026-06-03/venv311/Scripts/python -c "import violawake; print(violawake.__version__)"
```

Output:

```text
ModuleNotFoundError: No module named 'violawake'
```

Published `ModelCache` failure:

```powershell
_diag/2026-06-03/venv311/Scripts/python -c "from violawake_sdk.models import ModelCache; print(ModelCache)"
```

Output:

```text
ImportError: cannot import name 'ModelCache' from 'violawake_sdk.models'
```

## ModelSpec URL/SHA Evidence

Registry source: `src/violawake_sdk/models.py:47-112`.

Command:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
$env:VIOLAWAKE_MODEL_DIR=(Resolve-Path _diag/2026-06-03/model-downloads).Path
python scripts/verify_models.py --ci --no-skip-package-managed --no-skip-deprecated --report _diag/2026-06-03/model-verify-all-report.json
```

Output:

```text
VERIFY temporal_cnn
  PASS
VERIFY oww_backbone
  FAIL
  ERROR: Download failed: HTTP Error 404: Not Found
VERIFY kokoro_v1_0
  PASS
VERIFY kokoro_voices_v1_0
  PASS
VERIFY temporal_convgru
  PASS
VERIFY r3_10x_s42
  PASS
SKIP viola: alias for temporal_cnn
Results: 5 passed, 1 failed, 1 skipped.
```

Full HEAD/range/SHA probe excerpt:

```text
MODEL temporal_cnn ... HEAD status=200 ... SHA match=True
MODEL oww_backbone ... HEAD status=404 ... GET/SHA ERROR HTTPError: 404 Client Error
MODEL kokoro_v1_0 ... HEAD status=200 ... SHA match=True
MODEL kokoro_voices_v1_0 ... HEAD status=200 ... SHA match=True
MODEL temporal_convgru ... HEAD status=200 ... SHA match=True
MODEL r3_10x_s42 ... HEAD status=200 ... SHA match=True
ALIAS viola -> temporal_cnn: skipped duplicate spec
```

Root cause line:

- `src/violawake_sdk/models.py:64` points `oww_backbone` at
  `https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources`,
  which now returns 404.

Observed upstream downloadable OWW artifacts:

```text
melspectrogram.onnx HEAD 200 size 1087958 SHA ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f
embedding_model.onnx HEAD 200 size 1326578 SHA 70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f
COMBINED ('melspectrogram.onnx','embedding_model.onnx') e8444299a314fbb2971d33b39ff6fce4838be0f4a8d98aa4cf87537ee1350454
```

This proves the pinned combined SHA is meaningful, but the current `ModelSpec`
cannot represent the two-file OWW backbone artifact. Keeping it as a single
downloadable `ModelSpec` is false.

## Implemented Fixes

### Public API and ModelCache

Files:

- `src/violawake_sdk/models.py:121-203` adds the documented `ModelCache`.
- `src/violawake_sdk/__init__.py:158` adds `WakewordDetector` to `__all__`.
- `tests/unit/test_models.py:97-117` covers `ModelCache`.
- `tests/integration/test_sdk_surface.py:40-56` covers `ModelCache`,
  `WakewordDetector`, and `import violawake`.

Verification:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python -m pytest tests/unit/test_models.py tests/integration/test_sdk_surface.py -q
```

Output:

```text
collected 66 items
tests\unit\test_models.py ................................................
tests\integration\test_sdk_surface.py ..................
66 passed, 1 warning
```

### Wheel Packaging and Release Ratchet

Files:

- `pyproject.toml:195` includes both `src/violawake_sdk` and `src/violawake`
  in wheels.
- `pyproject.toml:163` excludes `_diag/` from source distributions.
- `.github/workflows/release.yml:86-142` installs the built wheel and verifies
  imports, console scripts, tools modules, `ModelCache`, and
  `violawake-download --help`.

Verification:

```powershell
python -m build --wheel --no-isolation
```

Output:

```text
Successfully built violawake-0.2.6-py3-none-any.whl
wheel dist\violawake-0.2.6-py3-none-any.whl size 197387 entries 56
has violawake True
has tools True
```

Fixed-wheel matrix:

```text
Python 3.10.20: py 3.10.20 version 0.2.6 shim 0.2.6 True
Python 3.11.9:  wheel smoke ok 0.2.6 10 <class 'violawake_sdk.models.ModelCache'>
Python 3.12.13: py 3.12.13 version 0.2.6 shim 0.2.6 True
```

Fixed quick-start command:

```powershell
_diag/2026-06-03/venv-wheel-fixed2/Scripts/violawake-download --model temporal_cnn
```

Output:

```text
temporal_cnn: ...\model-cache-fixed-quickstart\temporal_cnn.onnx (0.1 MB)
Done. Models cached to ~/.violawake/models/
Downloading temporal_cnn: 100%|##########| 102k/102k
```

Fixed non-microphone quick-start construction:

```powershell
_diag/2026-06-03/venv-wheel-fixed2/Scripts/python -c "from violawake_sdk import WakeDetector; d=WakeDetector(model='temporal_cnn', threshold=0.80, confirm_count=3); print(type(d).__name__, d.threshold); d.close()"
```

Output:

```text
WakeDetector 0.8
```

### Workflow Branch Fixes

Files:

- `.github/workflows/model-verify.yml:9-14` now targets `master`.
- `.github/workflows/release.yml:230-248` gives `update-docs` access to the
  `validate` output and pushes the registry update to `master`.

YAML parse check:

```powershell
@'... yaml.safe_load(...) ...'@ | python -
```

Output:

```text
.github/workflows/release.yml parsed True
.github/workflows/model-verify.yml parsed True
```

### Release Docs

Files:

- `CHANGELOG.md:9-24` now records current unreleased SDK, packaging, and
  post-v0.2.6 changes.
- `RELEASE_NOTES.md:5-16` now includes v0.2.6 notes.

Changelog probe:

```text
commits_since_v0.2.6= 21
actual_unreleased_nonempty= True
negative_probe_empty_changelog_caught=missing changelog entries
```

## Negative Probes Required by SC

1. ModelSpec URL goes 404.

Command:

```powershell
python scripts/verify_models.py --ci --no-skip-package-managed --no-skip-deprecated --report _diag/2026-06-03/model-verify-all-report.json
```

Caught failure:

```text
VERIFY oww_backbone
  FAIL
  ERROR: Download failed: HTTP Error 404: Not Found
```

2. Public symbol removed.

Command:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
@'... remove WakeDetector from actual surface set ...'@ | python -
```

Output:

```text
actual_public_surface_missing= []
negative_probe_removed_WakeDetector_missing= ['WakeDetector']
public symbol removal probe caught missing WakeDetector
```

3. User-visible change without changelog entry.

Command:

```powershell
@'... compare git log v0.2.6..HEAD to Unreleased section and synthetic empty changelog ...'@ | python -
```

Output:

```text
commits_since_v0.2.6= 21
actual_unreleased_nonempty= True
negative_probe_empty_changelog_caught=missing changelog entries
```

## Remaining MUST-FIX

### `oww_backbone` cannot stay as a single downloadable `ModelSpec`

Qualification: MUST-FIX (ModelSpec URL is 404; every `ModelSpec` must resolve
URL + SHA).

Evidence:

- Code: `src/violawake_sdk/models.py:57-65`.
- Failure: `scripts/verify_models.py --ci --no-skip-package-managed
  --no-skip-deprecated` returns `VERIFY oww_backbone FAIL`.

Required repair:

- Split package-managed OpenWakeWord backbone integrity out of
  `MODEL_REGISTRY`, or extend the registry with a first-class multi-artifact
  spec that can represent `melspectrogram.onnx` plus `embedding_model.onnx`.
- Update the wake-backbone integrity path to consume that representation.

Why not completed in this lane commit:

- The current integrity consumer lives in `src/violawake_sdk/oww_backbone.py`,
  which is Lane 1 ownership per `docs/LANE_LEDGER.md:140`. Changing it from
  Lane 7 would violate the disjoint worktree audit rule. The report keeps the
  failure explicit rather than weakening the check or hiding the package-managed
  entry.

## Planned Gates

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: release-wheel-public-surface-smoke
contract: Built wheels must expose the documented top-level imports, compatibility shim, ModelCache, tools modules, and console scripts before PyPI publish.
detector: .github/workflows/release.yml inline step "Smoke test built wheel"
own_tests:
  - tests/integration/test_sdk_surface.py
  - tests/unit/test_models.py
```

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: all-modelspec-live-url-sha
contract: Every ModelSpec in MODEL_REGISTRY must have a live HTTPS artifact URL whose downloaded bytes match the pinned SHA-256.
detector: scripts/verify_models.py --ci --no-skip-package-managed --no-skip-deprecated
own_tests:
  - _diag/2026-06-03/model-verify-all-report.json
  - TBD - orchestrator should add a fixture-level 404/SHA mismatch unit test if this detector is promoted into quality/gates.yaml
```

## Self-Audit Gate

1. I did not run the microphone `stream_mic()` loop from the README because it
   is an infinite hardware-dependent loop. I verified the install, download
   command, OWW auto-download, and `WakeDetector(...)` construction instead.
2. I did not publish a new PyPI release. The published `0.2.6` wheel remains
   broken until a new release is cut from this branch and uploaded.
3. I did not change `oww_backbone.py` to repair the package-managed
   multi-artifact model representation because that file is Lane 1 owned.
4. I did not run GitHub Actions remotely. I parsed the edited workflow YAML and
   ran the wheel-smoke commands locally against the rebuilt wheel.
5. I did not exhaustively instantiate TTS/STT/VAD extras; the lane question is
   the public API/distribution surface, and optional engine behavior belongs to
   Lane 2. Import placeholders and documented top-level exports were verified.
