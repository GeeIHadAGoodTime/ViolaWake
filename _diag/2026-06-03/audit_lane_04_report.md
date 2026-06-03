# Lane 4 Audit Report - Training & Augmentation

Date: 2026-06-03  
Branch: `audit-2026-06-03/l4-training`  
Verdict: **PASS after fixes**

## Summary

Lane 4 passes the fast SC oracle after two local fixes:

- `4e57a13 Enforce training audio contract`
- `bcd739b Fix training recipe artifact references`

No full from-scratch retrain was run. Per the binding correction, this audit is
pipeline integrity, not retrain reproducibility against the milestone d'/EER
oracle.

## Fixes

1. **Training loader now fails fast on audio-contract drift.**
   - Code: `src/violawake_sdk/tools/train.py:215`, `src/violawake_sdk/tools/train.py:812`, `src/violawake_sdk/tools/train.py:840`
   - The TemporalCNN path now uses `_load_training_audio()` before OWW embedding extraction at `src/violawake_sdk/tools/train.py:1137` and `src/violawake_sdk/tools/train.py:1420`.
   - Tests: `tests/unit/test_train.py:158`, `tests/unit/test_train.py:172`, `tests/unit/test_train.py:370`

2. **Training recipe no longer references absent checked-in artifacts.**
   - Doc: `docs/PROVEN_TRAINING_RECIPE.md:7`, `docs/PROVEN_TRAINING_RECIPE.md:15`, `docs/PROVEN_TRAINING_RECIPE.md:121`
   - The proven ONNX model and embedding cache are documented as generated / out-of-band artifacts instead of local paths missing from this worktree.

## Evidence

### Required integration test

Command:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l4-training\src'; python -m pytest tests/integration/test_training_e2e.py -q
```

Output excerpt:

```text
collected 2 items
tests\integration\test_training_e2e.py .. [100%]
2 passed in 2.67s
```

### Augmentation known-input probe

Relevant code:

- Additive noise: `src/violawake_sdk/training/augment.py:194`
- SpecAugment: `src/violawake_sdk/training/augment.py:260`
- Synthetic RIR: `src/violawake_sdk/training/augment.py:334`
- RIR apply path: `src/violawake_sdk/training/augment.py:404`, `src/violawake_sdk/training/augment.py:435`
- Pipeline entry points: `src/violawake_sdk/training/augment.py:528`, `src/violawake_sdk/training/augment.py:553`

Command:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l4-training\src'; @'
import json
import numpy as np
from violawake_sdk.training.augment import AugmentConfig, AugmentationPipeline
sr = 16000
t = np.arange(sr, dtype=np.float32) / sr
audio = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
cfg = AugmentConfig(
    p_gain=0.0,
    p_time_stretch=0.0,
    p_pitch_shift=0.0,
    p_noise=1.0,
    p_time_shift=0.0,
    p_rir=1.0,
    p_spec_augment=1.0,
    noise_snr_range_db=(10.0, 10.0),
    spec_freq_mask_param=8,
    spec_time_mask_param=12,
    spec_n_freq_masks=2,
    spec_n_time_masks=2,
)
pipe = AugmentationPipeline(config=cfg, seed=123)
aug = pipe.augment_clip(audio, factor=1)[0]
spec = np.ones((40, 94), dtype=np.float32)
masked = pipe.augment_spectrogram(spec)
print(json.dumps({
    'waveform_input_len': int(len(audio)),
    'waveform_output_len': int(len(aug)),
    'waveform_changed': bool(not np.allclose(audio, aug, atol=1e-6)),
    'waveform_diff_rms': float(np.sqrt(np.mean((aug - audio) ** 2))),
    'waveform_peak': float(np.max(np.abs(aug))),
    'spec_shape': list(masked.shape),
    'spec_zero_count': int(np.sum(masked == 0.0)),
    'spec_changed': bool(not np.array_equal(spec, masked)),
}, indent=2))
'@ | python -
```

Output excerpt:

```json
{
  "waveform_input_len": 16000,
  "waveform_output_len": 16000,
  "waveform_changed": true,
  "waveform_diff_rms": 0.20638567209243774,
  "waveform_peak": 0.42796358466148376,
  "spec_shape": [40, 94],
  "spec_zero_count": 837,
  "spec_changed": true
}
```

Unit coverage command:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l4-training\src'; python -m pytest tests/unit/test_augment.py tests/unit/test_rir_augment.py tests/unit/test_spec_augment.py tests/unit/test_losses.py -q
```

Output excerpt:

```text
collected 85 items
tests\unit\test_augment.py ..................................
tests\unit\test_rir_augment.py ......................
tests\unit\test_spec_augment.py ...............
tests\unit\test_losses.py ..............
85 passed in 10.93s
```

### Contamination negative probe

Relevant code:

- Hash overlap detection: `src/violawake_sdk/tools/contamination_check.py:58`
- Public check entry point: `src/violawake_sdk/tools/contamination_check.py:200`
- CLI exits nonzero on overlap: `src/violawake_sdk/tools/contamination_check.py:275`, `src/violawake_sdk/tools/contamination_check.py:282`

Command:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l4-training\src'; @'
import subprocess, sys, tempfile, wave
from pathlib import Path
import numpy as np
root = Path(tempfile.mkdtemp(prefix='vw_contam_cli_probe_'))
train_dir = root / 'train'
eval_dir = root / 'eval'
train_dir.mkdir(); eval_dir.mkdir()
sr = 16000
t = np.arange(sr, dtype=np.float32) / sr
audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
pcm = (audio * 32767).astype(np.int16).tobytes()
for path in (train_dir / 'train_clip.wav', eval_dir / 'eval_clip.wav'):
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm)
cmd = [sys.executable, '-m', 'violawake_sdk.tools.contamination_check', '--train', str(train_dir), '--eval', str(eval_dir), '--method', 'hash']
result = subprocess.run(cmd, text=True, capture_output=True)
print('command:', ' '.join(cmd))
print('returncode:', result.returncode)
print('stdout:', result.stdout.strip())
print('stderr:', result.stderr.strip())
raise SystemExit(0 if result.returncode == 1 and 'overlap_count' in result.stdout else 1)
'@ | python -
```

Output excerpt:

```text
returncode: 1
"overlap_count": 1
"contamination_rate": 1.0
stderr: WARNING: 1 overlapping items found (100.0% contamination rate)
```

### Audio-contract negative probe

Relevant code:

- Contract constants: `src/violawake_sdk/tools/train.py:215`, `src/violawake_sdk/tools/train.py:217`
- Strict loader: `src/violawake_sdk/tools/train.py:840`
- Temporal embedding path uses strict loader: `src/violawake_sdk/tools/train.py:1137`

Command:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l4-training\src'; @'
import sys, tempfile, wave
from pathlib import Path
from types import ModuleType
import numpy as np
from violawake_sdk.tools import train
root = Path(tempfile.mkdtemp(prefix='vw_bad_rate_probe_'))
wav_path = root / 'bad_22khz.wav'
sr = 22050
t = np.arange(sr, dtype=np.float32) / sr
audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
pcm = (audio * 32767).astype(np.int16)
with wave.open(str(wav_path), 'wb') as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm.tobytes())
class _FakePreprocessor:
    def embed_clips(self, audio_batch, ncpu=1):
        raise AssertionError('embedding should not run for bad sample rate')
class _FakeModel:
    def __init__(self, inference_framework):
        self.preprocessor = _FakePreprocessor()
fake_openwakeword = ModuleType('openwakeword')
fake_model = ModuleType('openwakeword.model')
fake_model.Model = _FakeModel
sys.modules['openwakeword'] = fake_openwakeword
sys.modules['openwakeword.model'] = fake_model
try:
    train._extract_temporal_embeddings([wav_path], 'neg', verbose=False, seq_len=9)
except train.TrainingError as exc:
    print('CAUGHT:', exc)
    raise SystemExit(0)
print('NOT CAUGHT')
raise SystemExit(1)
'@ | python -
```

Output excerpt:

```text
CAUGHT: Training audio must be 16000 Hz mono; ...\bad_22khz.wav is 22050 Hz.
```

Regression tests:

```powershell
$env:PYTHONPATH='J:\CLAUDE\PROJECTS\Wakeword-l4-training\src'; python -m pytest tests/unit/test_train.py -q
```

Output excerpt:

```text
collected 22 items
tests\unit\test_train.py ...................... [100%]
22 passed in 7.83s
```

### Loss function behavior

Relevant code:

- FocalLoss definition: `src/violawake_sdk/training/losses.py:24`
- Label smoothing: `src/violawake_sdk/training/losses.py:71`
- Focal weighting and alpha weighting: `src/violawake_sdk/training/losses.py:80`, `src/violawake_sdk/training/losses.py:84`
- BCE computation: `src/violawake_sdk/training/losses.py:87`

Covered by `tests/unit/test_losses.py` in the 85-test command above, including
known-value BCE reduction, gamma effect, alpha effect, label smoothing, finite
edge cases, and gradient flow.

### Recipe reference check

Command:

```powershell
$missing = @(); $paths = @('experiments/train_temporal_j5.py','experiments/j5_temporal_results.json','experiments/head_to_head_eval.py','src/violawake_sdk/models.py','src/violawake_sdk/tools/train.py','src/violawake_sdk/training/temporal_model.py','src/violawake_sdk/training/losses.py','console/backend/app/services/training_service.py','console/backend/scripts/train_full_pipeline.py'); foreach ($p in $paths) { if (!(Test-Path $p)) { $missing += $p } }; Write-Output ('missing_count=' + $missing.Count); if ($missing.Count -gt 0) { $missing }; rg -n 'experiments/models/j5_temporal|embedding_cache_temporal' docs/PROVEN_TRAINING_RECIPE.md; if ($LASTEXITCODE -eq 1) { Write-Output 'stale_artifact_refs=0'; exit 0 }
```

Output:

```text
missing_count=0
stale_artifact_refs=0
```

## Planned Gate

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: training-audio-contract
contract: Detects training pipeline acceptance of non-16 kHz or non-mono audio before embedding extraction.
detector: TBD - orchestrator will write
own_tests:
  - tests/unit/test_train.py::TestTrainHelpers::test_load_training_audio_rejects_wrong_sample_rate
  - tests/unit/test_train.py::TestTrainHelpers::test_extract_temporal_embeddings_rejects_wrong_sample_rate_before_embedding
  - tests/unit/test_train.py::TestTrainHelpers::test_load_training_audio_accepts_16khz_mono
```

## Notes

- The active editable Python environment initially imported `violawake_sdk` from
  `J:\CLAUDE\PROJECTS\Wakeword`, not this worktree. All final verification above
  pins `PYTHONPATH=J:\CLAUDE\PROJECTS\Wakeword-l4-training\src`.
- `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` was read after it appeared
  in this worktree. It remains unmodified by this audit.

## Self-Audit Gate

- Did not run a full from-scratch retrain because the binding correction says it is not required for this lane SC; this audit covers the fast pipeline-integrity oracle.
- Did not run embedding-based contamination detection because the required deliberate contamination probe is caught by the hash method with a nonzero CLI exit; embedding duplicate search is slower and depends on OWW model availability.
- Did not audit Console backend training parity beyond confirming referenced files exist because Console backend behavior is Lane 8-owned; this lane only corrected the recipe reference and SDK training loader.
- Did not scan every checked-in or out-of-band corpus audio file for sample-rate drift because the corpus can be large/out-of-band; instead the loader now rejects drift at runtime and the 22 kHz negative probe proves the guard fires.
- Did not rerun public benchmark or production d'/EER bars because those are the milestone oracle / Lane 5 evaluation surface, not this fast Lane 4 SC.
