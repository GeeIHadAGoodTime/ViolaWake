# Audit Lane 02 Report - Companion Engines & VoicePipeline

## Binary verdict

**MUST-FIX.**

The negative probes exposed real silent-fallback gaps and two fixes landed on
`audit-2026-06-03/l2-companions`. The lane still cannot be called PASS because
the live Kokoro first-audio probe missed the documented 0.3-0.8 s budget under
this run, and the full live STT/VoicePipeline probe could not complete while the
host was resource-exhausted.

## Environment ownership

Command:
```powershell
python -c "import violawake_sdk.pipeline as p; print(p.__file__)"
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -c "import violawake_sdk.pipeline as p; print(p.__file__)"
```

Output:
```text
J:\CLAUDE\PROJECTS\Wakeword\src\violawake_sdk\pipeline.py
J:\CLAUDE\PROJECTS\Wakeword-l2-companions\src\violawake_sdk\pipeline.py
```

All verification commands below used:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path
```

## Negative probes run

Probe file: `tests/unit/test_voice_pipeline_oracle_probes.py:36`,
`tests/unit/test_voice_pipeline_oracle_probes.py:48`,
`tests/unit/test_voice_pipeline_oracle_probes.py:62`,
`tests/unit/test_voice_pipeline_oracle_probes.py:75`.

Pre-fix command:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m pytest tests/unit/test_voice_pipeline_oracle_probes.py::test_noop_stt_empty_text_is_a_pipeline_error tests/unit/test_voice_pipeline_oracle_probes.py::test_tts_wrong_voice_is_a_pipeline_error_from_command_path tests/unit/test_voice_pipeline_oracle_probes.py::test_vad_always_on_stops_recording_at_max_duration --no-cov -q
```

Pre-fix output excerpt:
```text
tests\unit\test_voice_pipeline_oracle_probes.py FF.                      [100%]
FAILED ... test_noop_stt_empty_text_is_a_pipeline_error
E           Failed: DID NOT RAISE <class 'violawake_sdk._exceptions.PipelineError'>
FAILED ... test_tts_wrong_voice_is_a_pipeline_error_from_command_path
E           Failed: DID NOT RAISE <class 'violawake_sdk._exceptions.PipelineError'>
Captured log: ERROR violawake_sdk.pipeline:pipeline.py:380 Command handler 'handler' failed
1 passed, 2 failed
```

Post-fix command:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m pytest tests/unit/test_voice_pipeline_oracle_probes.py tests/unit/test_stt_tts_engines.py::TestSTTEngineErrorHandling::test_import_error_preserves_transitive_dependency_failure --no-cov -q
```

Post-fix output:
```text
collected 5 items
tests\unit\test_voice_pipeline_oracle_probes.py ....                     [ 80%]
tests\unit\test_stt_tts_engines.py .                                     [100%]
5 passed in 10.63s
```

## Fix 1 - VoicePipeline silent fallback

Gap: `VoicePipeline` silently accepted a no-op STT engine returning empty text
and swallowed TTS misconfiguration under "Command handler failed." That violates
the ledger oracle: no-op STT must surface an error, wrong TTS voice must raise,
and always-on VAD must not deadlock.

Fix: `VoicePipeline` now emits an `error` event and raises `PipelineError` for
missing STT, STT import/prewarm failures, empty STT output, empty audio reaching
STT, missing/failing TTS, and empty TTS audio. TTS failures are no longer caught
as handler failures.

Files:
- `src/violawake_sdk/pipeline.py:30` adds the `error` event.
- `src/violawake_sdk/pipeline.py:228` makes `speak()` fail closed.
- `src/violawake_sdk/pipeline.py:297` makes `_transcribe_and_respond()` fail closed.
- `src/violawake_sdk/pipeline.py:349` rejects empty transcription.
- `src/violawake_sdk/pipeline.py:360` preserves STT unavailable failures.
- `src/violawake_sdk/pipeline.py:375` separates command handler failures from TTS failures.
- `src/violawake_sdk/pipeline.py:475` centralizes error event emission and `PipelineError`.
- `src/violawake_sdk/pipeline.py:522` stops swallowing STT import/prewarm failures.
- `tests/unit/test_voice_pipeline.py:331`, `tests/unit/test_voice_pipeline.py:359`,
  `tests/unit/test_voice_pipeline.py:562`, `tests/unit/test_voice_pipeline.py:588`,
  `tests/unit/test_voice_pipeline.py:601`, `tests/unit/test_voice_pipeline.py:655`,
  and `tests/unit/test_voice_pipeline.py:753` update existing expectations.
- `tests/unit/test_voice_pipeline_oracle_probes.py:36`, `:48`, `:62`, `:75`
  are the negative probes.

Evidence:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m pytest tests/unit/test_tts_engine.py tests/unit/test_stt_engine.py tests/unit/test_stt_engine_wav.py tests/unit/test_stt_tts_engines.py tests/unit/test_vad.py tests/unit/test_vad_engine.py tests/unit/test_pipeline.py tests/unit/test_voice_pipeline.py tests/unit/test_voice_pipeline_oracle_probes.py tests/integration/test_pipeline.py tests/integration/test_streaming_stt.py --no-cov -q
```

Output:
```text
collected 260 items
...
tests\integration\test_streaming_stt.py ......................           [100%]
260 passed in 45.59s
```

Commit: `f0d4b21ca23765b641db7d0dc8c00a01e7f58eb1`.

Planned gate:
```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: voice-pipeline-error-surfacing
contract: VoicePipeline must surface no-op STT, TTS misconfiguration, and always-on VAD probe failures instead of silently returning to idle or deadlocking.
detector: tests/unit/test_voice_pipeline_oracle_probes.py
own_tests:
  - tests/unit/test_voice_pipeline_oracle_probes.py::test_noop_stt_empty_text_is_a_pipeline_error
  - tests/unit/test_voice_pipeline_oracle_probes.py::test_tts_wrong_voice_is_a_pipeline_error_from_command_path
  - tests/unit/test_voice_pipeline_oracle_probes.py::test_vad_always_on_stops_recording_at_max_duration
```

## Fix 2 - STT import failure cause

Gap: direct `STTEngine.prewarm()` reported every `ImportError` from
`faster_whisper` as "faster-whisper is not installed." In the live probe, the
actual failure was a transitive `av` DLL/resource failure, so the old message
hid the real cause.

Fix: missing `faster_whisper` still gets the install hint, but installed package
or dependency import failures preserve the dependency/cause.

Files:
- `src/violawake_sdk/stt.py:153` catches `ModuleNotFoundError` separately.
- `src/violawake_sdk/stt.py:161` preserves failed dependency imports.
- `src/violawake_sdk/stt.py:166` preserves installed-but-broken import errors.
- `tests/unit/test_stt_tts_engines.py:182` proves the real import cause remains visible.

Evidence:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m pytest tests/unit/test_stt_tts_engines.py::TestSTTEngineErrorHandling::test_import_error_when_faster_whisper_missing tests/unit/test_stt_tts_engines.py::TestSTTEngineErrorHandling::test_import_error_preserves_transitive_dependency_failure --no-cov -q
```

Output:
```text
collected 2 items
tests\unit\test_stt_tts_engines.py ..                                    [100%]
2 passed in 6.23s
```

Commit: `520e121c0e03ceb34c632cde277944072942b339`.

Planned gate:
```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: stt-import-cause
contract: STTEngine must distinguish a missing faster-whisper package from installed-but-broken transitive imports.
detector: tests/unit/test_stt_tts_engines.py
own_tests:
  - tests/unit/test_stt_tts_engines.py::TestSTTEngineErrorHandling::test_import_error_when_faster_whisper_missing
  - tests/unit/test_stt_tts_engines.py::TestSTTEngineErrorHandling::test_import_error_preserves_transitive_dependency_failure
```

## Live capability probes

Probe script: `_diag/2026-06-03/lane_02_live_probe.py:57`,
`_diag/2026-06-03/lane_02_live_probe.py:97`,
`_diag/2026-06-03/lane_02_live_probe.py:150`,
`_diag/2026-06-03/lane_02_live_probe.py:225`.

Full live probe command:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python _diag\2026-06-03\lane_02_live_probe.py
```

Output:
```text
command timed out after 904079 milliseconds
```

TTS slice command:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; @'
import time
from violawake_sdk.tts import TTSEngine
print('before engine', flush=True)
tts=TTSEngine(voice='af_heart')
print('before load', flush=True)
t=time.perf_counter(); tts._get_kokoro(); print('loaded_ms', round((time.perf_counter()-t)*1000,1), flush=True)
text='Turn on the kitchen lights. Confirm when ready.'
for i in range(2):
    t=time.perf_counter(); gen=tts.synthesize_chunked(text); first=next(gen); print('first', i+1, round((time.perf_counter()-t)*1000,1), first.shape, flush=True); rest=list(gen); print('rest_chunks', len(rest), flush=True)
'@ | python -u -
```

Output:
```text
before engine
before load
loaded_ms 49567.6
first 1 27786.5 (22187,)
rest_chunks 1
first 2 1770.9 (22187,)
rest_chunks 1
```

This run does **not** satisfy the documented Kokoro first-audio budget
(`0.3-0.8 s`). The warm second chunk was `1.77 s`.

STT slice command:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; @'
from violawake_sdk.stt import STTEngine
stt=STTEngine(model='base', language='en')
stt.prewarm()
'@ | python -u -
```

Output excerpt:
```text
Traceback (most recent call last):
  File "...src\violawake_sdk\stt.py", line 153, in _get_model
    from faster_whisper import WhisperModel
  File "...\site-packages\av\__init__.py", line 16, in <module>
    from av._core import time_base, library_versions, ffmpeg_version_info
ImportError: DLL load failed while importing _core: The paging file is too small for this operation to complete.
...
ImportError: faster-whisper is not installed. Install with: pip install 'violawake[stt]'
```

After Fix 2, the direct error path is covered by
`tests/unit/test_stt_tts_engines.py:182`, but a clean live STT transcription
run remains unproven because the host hit system resource exhaustion.

Resource exhaustion evidence:
```text
Could not load file or assembly 'Microsoft.PowerShell.Security...'
Insufficient system resources exist to complete the requested service. (0x800705AA)

Starting the CLR failed with HRESULT 80004005.
```

## Additional verification

Source/probe ruff command:
```powershell
$env:PYTHONPATH=(Resolve-Path .\src).Path; python -m ruff check src/violawake_sdk/pipeline.py src/violawake_sdk/stt.py _diag/2026-06-03/lane_02_live_probe.py
```

Output:
```text
All checks passed!
```

Whole edited test-file ruff was not used as a gate because it reported
pre-existing style debt across the files (`SIM117`, `F401`, import ordering)
outside the narrow audit fix.

## Mandatory self-audit gate

- I did not prove live faster-whisper transcription on a fixed WAV because
  `av` failed to import under paging-file/resource exhaustion; I added a probe
  script and preserved the real error cause, but the live STT run needs a clean
  host.
- I did not prove a clean end-to-end `VoicePipeline.run()` with real STT and
  real TTS output because the full live probe timed out before completion; the
  script is present for rerun once host pressure drops.
- I did not install WebRTC or Silero VAD during this run because the machine was
  already resource-exhausted; RMS and the always-on VAD deadlock probe were
  exercised, but explicit WebRTC/Silero live adapters remain unproven.
- I did not modify `quality/gates.yaml` because
  `_diag/2026-06-03/SC_AUDIT_ROUND_1_CORRECTIONS.md` section A2 forbids lane
  agents from touching it; planned gate specs are included above.
- I did not modify `CLAUDE.md` or `docs/LANE_LEDGER.md` because the prompt
  forbids it; stale oracle status must be updated by the orchestrator after
  integration, not by this lane branch.
