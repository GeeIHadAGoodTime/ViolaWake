# Audit Lane 01 Report - Wake Detection

Branch audited: `audit-2026-06-03/l1-wake`

Binary verdict: PASS after four fixes committed on this topic branch.

The stronger Lane 1 PASS clause from `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md:107`-`122` is satisfied:

- SDK-boundary audio-contract assertions are tested at `tests/unit/test_wake_detector_oracle.py:69`.
- The four required negative probes were constructed and caught detectable failures.
- Per-category benchmark v2 FAR bars are now documented at `docs/PROVEN_TRAINING_RECIPE.md:24` and enforced at `tests/unit/test_wake_detector_oracle.py:101`.
- The 4-gate decision policy is exercised end-to-end in one test at `tests/unit/test_wake_detector_oracle.py:86`.

`quality/gates.yaml` was not edited, per `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md:42`-`63`.

## Fix 1 - Audio Source Contract Drift

Gap: `FileSource` and `MicrophoneSource` could accept non-contract audio shapes, letting 8/22/48 kHz, stereo, or non-16-bit inputs reach inference under the published 16 kHz mono, 20 ms frame contract.

Fix commit: `c113c0dc283fc575db5e48e23c217150f5b3b642` (`Harden audio sources against contract drift`)

File:line evidence:

- Contract validator: `src/violawake_sdk/audio_source.py:44`
- Microphone boundary checks: `src/violawake_sdk/audio_source.py:109`
- WAV boundary checks: `src/violawake_sdk/audio_source.py:207`
- soundfile boundary checks: `src/violawake_sdk/audio_source.py:230`
- Rejection tests: `tests/unit/test_audio_source.py:240`, `:248`, `:256`, `:305`, `:349`

Verifier:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; $env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l1-wake\src'; python -m pytest -q -o addopts='' tests/unit/test_audio_source.py::TestFileSource::test_rejects_wrong_wav_sample_rate tests/unit/test_audio_source.py::TestFileSource::test_rejects_stereo_wav tests/unit/test_audio_source.py::TestFileSource::test_rejects_non_16bit_wav tests/unit/test_audio_source.py::TestFileSource::test_rejects_non_wav_wrong_sample_rate_via_soundfile tests/unit/test_audio_source.py::TestMicrophoneSourceContract
```

Output:

```text
..........                                                               [100%]
============================== warnings summary ===============================
C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474
  C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474: PytestConfigWarning: Unknown config option: asyncio_mode

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
10 passed, 1 warning in 0.15s
```

Planned gate:

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: audio-source-contract-strict
contract: SDK audio sources fail closed on non-16kHz mono 16-bit 320-sample-frame input.
detector: tests/unit/test_audio_source.py
own_tests:
  - tests/unit/test_audio_source.py::TestFileSource::test_rejects_wrong_wav_sample_rate
  - tests/unit/test_audio_source.py::TestMicrophoneSourceContract
```

## Fix 2 - OWW Backbone Hash Drift

Gap: OWW backbone integrity drift could be logged instead of blocking runtime load, which would silently break the training/inference feature-extractor contract.

Fix commit: `61f96169793c2f93baab9ece957c0e9f43418b68` (`Fail closed on OWW backbone hash drift`)

File:line evidence:

- Integrity verifier called before session use: `src/violawake_sdk/oww_backbone.py:176`
- Fail-closed hash comparison: `src/violawake_sdk/oww_backbone.py:184`, `:202`
- Match/mismatch tests: `tests/unit/test_oww_backbone.py:325`, `:338`

Verifier:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; $env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l1-wake\src'; python -m pytest -q -o addopts='' tests/unit/test_oww_backbone.py::TestBackboneIntegrity
```

Output:

```text
..                                                                       [100%]
============================== warnings summary ===============================
C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474
  C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474: PytestConfigWarning: Unknown config option: asyncio_mode

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
2 passed, 1 warning in 0.11s
```

Planned gate:

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: oww-backbone-integrity-fail-closed
contract: The installed OWW backbone hash must match the pinned registry hash before inference sessions load.
detector: tests/unit/test_oww_backbone.py
own_tests:
  - tests/unit/test_oww_backbone.py::TestBackboneIntegrity::test_integrity_mismatch_raises
  - tests/unit/test_oww_backbone.py::TestBackboneIntegrity::test_integrity_match_passes
```

## Fix 3 - Missing Lane Oracle Probes

Gap: Lane 1 did not have one SDK-boundary oracle that locked the 16 kHz / 20 ms / 320-sample / 96-dim contract, default threshold `0.80`, and all four policy gates.

Fix commit: `7f024b07745a00b5bff521888444f7f8473473f5` (`Add Lane 1 wake detector oracle probes`)

File:line evidence:

- Contract constants test: `tests/unit/test_wake_detector_oracle.py:69`
- Default threshold regression test: `tests/unit/test_wake_detector_oracle.py:77`
- Single end-to-end 4-gate test: `tests/unit/test_wake_detector_oracle.py:86`

Verifier:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; $env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l1-wake\src'; python -m pytest -q -o addopts='' tests/unit/test_wake_detector_oracle.py
```

Output:

```text
....                                                                     [100%]
============================== warnings summary ===============================
C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474
  C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474: PytestConfigWarning: Unknown config option: asyncio_mode

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
4 passed, 1 warning in 0.23s
```

Required negative probes:

Command:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l1-wake\src;J:\CLAUDE\PROJECTS\Wakeword-l1-wake'; @'
from __future__ import annotations

import importlib.util
import tempfile
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

import violawake_sdk._constants as constants
import violawake_sdk.wake_detector as wd
from violawake_sdk._exceptions import ModelLoadError
from violawake_sdk.audio_source import FileSource
from violawake_sdk.models import ModelSpec
from violawake_sdk.oww_backbone import OpenWakeWordBackbone, OpenWakeWordBackbonePaths

ROOT = Path.cwd()
ORACLE_PATH = ROOT / 'tests' / 'unit' / 'test_wake_detector_oracle.py'

def load_oracle(name: str):
    spec = importlib.util.spec_from_file_location(name, ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

original_constant = constants.DEFAULT_THRESHOLD
original_wd = wd.DEFAULT_THRESHOLD
try:
    constants.DEFAULT_THRESHOLD = 0.50
    wd.DEFAULT_THRESHOLD = 0.50
    oracle = load_oracle('oracle_threshold_broken')
    try:
        loud = (np.ones(320, dtype=np.int16) * 1200).tobytes()
        oracle.test_default_threshold_is_080_and_rejects_050_score(loud)
        raise RuntimeError('threshold_050 probe unexpectedly passed')
    except AssertionError as exc:
        print(f'PROBE threshold_050 caught: {exc.__class__.__name__}')
finally:
    constants.DEFAULT_THRESHOLD = original_constant
    wd.DEFAULT_THRESHOLD = original_wd

with tempfile.TemporaryDirectory() as td:
    for rate in (8000, 22050, 48000):
        path = Path(td) / f'wrong-{rate}.wav'
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(np.zeros(320, dtype=np.int16).tobytes())
        try:
            FileSource(path).start()
            raise RuntimeError(f'wrong_sample_rate_{rate} probe unexpectedly passed')
        except ValueError as exc:
            print(f'PROBE wrong_sample_rate_{rate} caught: {exc}')

with tempfile.TemporaryDirectory() as td:
    td_path = Path(td)
    melspec = td_path / 'melspectrogram.onnx'
    embed = td_path / 'embedding_model.onnx'
    melspec.write_bytes(b'fake-melspec')
    embed.write_bytes(b'fake-embedding')
    paths = OpenWakeWordBackbonePaths(melspectrogram=melspec, embedding_model=embed)
    bad_spec = ModelSpec(
        name='oww_backbone',
        url='https://example.invalid/openwakeword',
        sha256='0' * 64,
        size_bytes=melspec.stat().st_size + embed.stat().st_size,
        description='bad test backbone',
        version='test',
    )
    try:
        with patch('violawake_sdk.models.MODEL_REGISTRY', {'oww_backbone': bad_spec}):
            OpenWakeWordBackbone._verify_backbone_integrity(paths)
        raise RuntimeError('swapped_backbone probe unexpectedly passed')
    except ModelLoadError as exc:
        print(f'PROBE swapped_backbone caught: {exc}')

oracle = load_oracle('oracle_policy_probe')

def broken_evaluate(self, score: float, rms: float = 100.0, is_playing: bool = False) -> bool:
    return rms >= self.rms_floor and score >= self.threshold

with patch.object(wd.WakeDecisionPolicy, 'evaluate', broken_evaluate):
    try:
        loud = (np.ones(320, dtype=np.int16) * 1200).tobytes()
        silent = np.zeros(320, dtype=np.int16).tobytes()
        oracle.test_detect_exercises_all_four_policy_gates_end_to_end(loud, silent)
        raise RuntimeError('removed_policy_gate probe unexpectedly passed')
    except AssertionError as exc:
        print(f'PROBE removed_policy_gate caught: {exc.__class__.__name__}')
'@ | python -
```

Output:

```text
PROBE threshold_050 caught: AssertionError
PROBE wrong_sample_rate_8000 caught: FileSource(wrong-8000.wav) violates ViolaWake audio contract: expected 16000Hz/1ch/16bit 20ms PCM frames, got 8000Hz/1ch/16bit
PROBE wrong_sample_rate_22050 caught: FileSource(wrong-22050.wav) violates ViolaWake audio contract: expected 16000Hz/1ch/16bit 20ms PCM frames, got 22050Hz/1ch/16bit
PROBE wrong_sample_rate_48000 caught: FileSource(wrong-48000.wav) violates ViolaWake audio contract: expected 16000Hz/1ch/16bit 20ms PCM frames, got 48000Hz/1ch/16bit
PROBE swapped_backbone caught: OWW backbone hash mismatch: expected 0000000000000000..., got 14b1b699d2d1eef0... The installed openwakeword backbone differs from the pinned training/inference contract.
PROBE removed_policy_gate caught: AssertionError
```

Planned gate:

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: wake-detector-oracle-probes
contract: Lane 1 oracle locks threshold 0.80, SDK audio constants, 96-dim OWW embeddings, and all four policy gates.
detector: tests/unit/test_wake_detector_oracle.py
own_tests:
  - tests/unit/test_wake_detector_oracle.py::test_default_threshold_is_080_and_rejects_050_score
  - tests/unit/test_wake_detector_oracle.py::test_detect_exercises_all_four_policy_gates_end_to_end
```

## Fix 4 - Documented Per-Category FAR Bar Missing

Gap: The corrected PASS clause requires per-category FAR on the documented confusables set to stay under documented bars, and says missing bars are their own MUST-FIX. The repo had global false-accept targets and benchmark category summaries, but no explicit benchmark v2 per-category bars.

Fix commit: `5ea419de059bc771c1a59136229ee249a56dc6cd` (`Document Lane 1 per-category FAR bars`)

File:line evidence:

- New documented bar section: `docs/PROVEN_TRAINING_RECIPE.md:24`
- Enforced bars: `tests/unit/test_wake_detector_oracle.py:28`
- Category-bar oracle: `tests/unit/test_wake_detector_oracle.py:101`

Verifier:

```powershell
@'
import csv
import collections
from pathlib import Path
p = Path('benchmark_v2/violawake_scores_v2.csv')
rows = list(csv.DictReader(p.open(newline='')))
cats = collections.defaultdict(list)
for r in rows:
    if r['label'] == 'negative':
        cats[r['category']].append(float(r['score']))
print('threshold=0.80')
for k in sorted(cats):
    v = cats[k]
    fa = sum(s >= 0.80 for s in v)
    print(f'{k}: false_accepts={fa} total={len(v)} far={fa/len(v):.6f} max_score={max(v):.6f}')
'@ | python -
```

Output:

```text
threshold=0.80
adversarial_alexa: false_accepts=0 total=105 far=0.000000 max_score=0.362914
adversarial_viola: false_accepts=5 total=105 far=0.047619 max_score=0.898278
noise: false_accepts=0 total=20 far=0.000000 max_score=0.573148
speech: false_accepts=22 total=200 far=0.110000 max_score=0.951197
speech_existing: false_accepts=3 total=270 far=0.011111 max_score=0.874750
```

Planned gate:

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: wake-detector-per-category-far-bars
contract: Benchmark v2 negative categories must stay at or below documented default-threshold false-accept bars.
detector: tests/unit/test_wake_detector_oracle.py
own_tests:
  - tests/unit/test_wake_detector_oracle.py::test_benchmark_v2_negative_categories_stay_under_documented_bars
  - tests/unit/test_wake_detector_oracle.py::test_default_threshold_is_080_and_rejects_050_score
```

## Final Verification

Focused Lane 1 suite at HEAD:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'; $env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l1-wake\src'; python -m pytest -q -o addopts='' tests/unit/test_wake_detector_oracle.py tests/unit/test_audio_source.py tests/unit/test_oww_backbone.py tests/unit/test_wake_detector_core.py tests/unit/test_wake_detector_edge_cases.py tests/unit/test_wake_decision_policy.py
```

Output:

```text
........................................................................ [ 57%]
......................................................                   [100%]
============================== warnings summary ===============================
C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474
  C:\Users\jihad\viola-whisper\Lib\site-packages\_pytest\config\__init__.py:1474: PytestConfigWarning: Unknown config option: asyncio_mode

tests/unit/test_oww_backbone.py::TestResolveBackbonePaths::test_raises_when_models_missing
  C:\Users\jihad\viola-whisper\Lib\site-packages\requests\__init__.py:113: RequestsDependencyWarning: urllib3 (2.20.907) or chardet (7.4.3)/charset_normalizer (3.4.3) doesn't match a supported version!
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
126 passed, 2 warnings in 4.49s
```

Python lint:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l1-wake\src'; python -m ruff check src/violawake_sdk/audio_source.py src/violawake_sdk/oww_backbone.py tests/unit/test_audio_source.py tests/unit/test_oww_backbone.py tests/unit/test_wake_detector_oracle.py
```

Output:

```text
All checks passed!
```

## Mandatory Self-Audit Gate

- I did not run a physical microphone live-audio capture. The audit covered the SDK entry boundary and file-source contract because no specific capture device or hardware authorization was provided.
- I did not rerun the full production eval corpus for the public d'=8.577 / EER=0.8% claim. The binding correction moves public-number reproducibility out of this lane; this report uses the checked-in benchmark v2 score artifact for per-category bars.
- I did not exhaustively test every optional wrapper path (`AsyncWakeDetector`, `WakeDetector.from_source`, and higher-level voice pipeline callers). The fixes target the shared core contract and existing Lane 1 unit surface.
- I did not mutate real installed OpenWakeWord package files to prove hash failure. The swapped-backbone probe patched the pinned registry hash against temporary model files to avoid modifying package-managed assets.
- I did not audit Browser Wake Detection / WASM behavior. That is Lane 3 ownership, and this work stayed inside Lane 1 ownership plus the required `_diag` report.
