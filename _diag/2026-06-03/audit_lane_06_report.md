# Audit Lane 06 Report — SDK CLI & Sample Tools

Date: 2026-06-03
Worktree: `J:\CLAUDE\PROJECTS\Wakeword-l6-cli`
Branch: `audit-2026-06-03/l6-cli`

## Verdict

PASS for the Lane 6 capability question after the fixes in this worktree:

> Can a user run `violawake-train`, `violawake-eval`, `violawake-collect`, `violawake-download` and have each command do what its `--help` says, on a clean install?

Strict notes:

- `violawake-download --model temporal_cnn` now exits 0 on a clean core install without `requests`/`tqdm`.
- `violawake-train`, `violawake-eval`, and `violawake-collect` now make their optional runtime requirements explicit in `--help` and fail closed with actionable dependency messages on a core install.
- `examples/basic_detection.py` and `examples/async_detection.py` run unmodified after `pip install -e ".[oww]"` plus `download_models()`.
- SC-CONFLICT / out-of-lane: `violawake-streaming-eval` still throws a traceback on a missing-audio invocation, but `docs/LANE_LEDGER.md` assigns `src/violawake_sdk/tools/streaming_eval.py` to Lane 5, and the binding correction says not to touch outside the lane owns list.
- Out-of-lane doc drift: README still advertises `violawake-train --architecture`, while the temporal-only CLI rejects it and Lane 6 tests assert rejection. I did not edit README because it is not in Lane 6 ownership in the ledger.

## Fixes

1. Clean-install downloader fix
   - Code: `src/violawake_sdk/tools/download_model.py:21`
   - Change: `violawake-download` falls back to the built-in verified downloader when optional `[download]` progress dependencies are absent.
   - Output fix: success now prints the actual `VIOLAWAKE_MODEL_DIR`, not a hard-coded `~/.violawake/models`.
   - Regression: `tests/unit/test_cli.py:416`

2. Sample collection fail-closed / Windows output fix
   - Code: `src/violawake_sdk/tools/collect_samples.py:6`, `:153`, `:156`, `:170`
   - Change: help names `[audio]`; status output is ASCII; zero-recorded requested sessions exit 1 instead of looking successful.
   - Regression: `tests/unit/test_cli.py:518`

3. Optional dependency help honesty
   - Code: `src/violawake_sdk/tools/train.py:6`, `src/violawake_sdk/tools/evaluate.py:4`, `src/violawake_sdk/tools/generate_samples.py:6`
   - Change: help epilogs now name `[training]` or `[generate]` where a core install cannot complete the heavy operation.

4. Example clean `[oww]` run
   - Code: `examples/basic_detection.py:8`, `:32`, `:36`
   - Change: the basic example still uses the microphone when available, but on a clean `[oww]` install without `pyaudio`, it runs a synthetic silence detector smoke and exits 0.

5. Installed entry-point ratchet
   - Code: `tests/unit/test_cli.py:26`, `:82`
   - Change: tests now verify installed distribution metadata and actual console-script `--help`, not only `python -m` modules.

## Negative Probe

Constructed broken variant:

```text
Removed this line from pyproject.toml:
violawake-download = "violawake_sdk.tools.download_model:main"
```

Verification path:

```text
.\.venv-audit\Scripts\python.exe -m pip install -e . --no-deps
.\.venv-audit\Scripts\python.exe -m pytest -o addopts= tests\unit\test_cli.py::TestInstalledProjectScripts::test_distribution_metadata_exposes_all_published_cli_scripts -q
```

Caught failure excerpt from `_diag/2026-06-03/audit_lane_06_negative_probe.log`:

```text
E       AssertionError: assert ['violawake-download'] == []
E         Left contains one more item: 'violawake-download'
FAILED tests/unit/test_cli.py::TestInstalledProjectScripts::test_distribution_metadata_exposes_all_published_cli_scripts
```

Restored `pyproject.toml`, reinstalled metadata, and the same test passed:

```text
1 passed, 1 warning in 0.51s
```

## Clean Venv Evidence

Final venv creation and install log: `_diag/2026-06-03/audit_lane_06_final_venv_install.log`

```text
$ python -m venv .venv-audit
exit=0
Python 3.11.9

$ .\.venv-audit\Scripts\python.exe -m pip install -e .
exit=0
Successfully installed flatbuffers-25.12.19 numpy-2.4.6 onnxruntime-1.26.0 packaging-26.2 protobuf-7.35.0 pysbd-0.3.4 scipy-1.17.1 violawake-0.2.6

$ .\.venv-audit\Scripts\python.exe -m pip list
violawake   0.2.6    J:\CLAUDE\PROJECTS\Wakeword-l6-cli
```

## CLI Evidence

Full log: `_diag/2026-06-03/audit_lane_06_final_cli_verification.log`

```text
violawake-train --help                 exit=0
violawake-eval --help                  exit=0
violawake-collect --help               exit=0
violawake-download --help              exit=0
violawake-download-corpus --help       exit=0
violawake-expand-corpus --help         exit=0
violawake-streaming-eval --help        exit=0
violawake-test-confusables --help      exit=0
violawake-contamination-check --help   exit=0
violawake-generate --help              exit=0
```

Clean-install `violawake-download --model temporal_cnn` with an empty `VIOLAWAKE_MODEL_DIR`:

```text
exit=0
Optional download progress dependencies are not installed; using built-in downloader.
temporal_cnn: J:\CLAUDE\PROJECTS\Wakeword-l6-cli\_diag\2026-06-03\cli_final_workspace\model_cache\temporal_cnn.onnx (0.1 MB)
Done. Models cached to J:\CLAUDE\PROJECTS\Wakeword-l6-cli\_diag\2026-06-03\cli_final_workspace\model_cache
stderr: Downloading model 'temporal_cnn' (102.4 KB)... done.
```

Other representative clean-core outcomes:

```text
violawake-download --list              exit=0  prints available models
violawake-download --list-cached       exit=0  prints temporal_cnn in custom cache
violawake-collect ...                  exit=1  prints pyaudio/[audio] requirement, no traceback
violawake-train ...                    exit=1  prints PyTorch/[training] requirement, no traceback
violawake-eval ...                     exit=1  prints scikit-learn/[training] requirement, no traceback
violawake-expand-corpus --list         exit=0  prints available corpora
violawake-contamination-check ...      exit=0  prints no contamination detected
violawake-generate ...                 exit=1  prints edge-tts/[generate] requirement, no traceback
```

Out-of-lane observed failure:

```text
violawake-streaming-eval --audio <missing wav> --model temporal_cnn
exit=1
Traceback ... ModelNotFoundError: OpenWakeWord is required for wake word detection.
```

## Example Evidence

Full log: `_diag/2026-06-03/audit_lane_06_final_oww_examples.log`

```text
$ .\.venv-audit\Scripts\python.exe -m pip install -e '.[oww]'
exit=0
Successfully installed ... openwakeword-0.6.0 ... violawake-0.2.6

$ .\.venv-audit\Scripts\python.exe -c "from openwakeword.utils import download_models; download_models()"
exit=0

$ .\.venv-audit\Scripts\python.exe examples\async_detection.py
exit=0
Listening for 'Viola'...
Done.

$ .\.venv-audit\Scripts\python.exe examples\basic_detection.py
exit=0
Listening for 'Viola'... (say it!)
Microphone unavailable: pyaudio is required for microphone features. Install with: pip install violawake[audio]
Detector initialized. Synthetic silence score=0.000
```

## Tests

```text
.\.venv-audit\Scripts\python.exe -m pytest -o addopts= tests\unit\test_cli.py -q
44 passed, 1 warning in 24.10s
```

The warning is expected in the minimal audit venv: `pytest-asyncio` is not installed, so pytest reports the repo's `asyncio_mode` config as unknown.

## Planned Gate

```yaml
# Planned gate (for orchestrator integration commit, not for this branch)
gate_id: installed-project-scripts-smoke
contract: Every published pyproject console script remains installed and its help path exits 0 after editable install.
detector: tests/unit/test_cli.py::TestInstalledProjectScripts
own_tests:
  - tests/unit/test_cli.py::TestInstalledProjectScripts::test_distribution_metadata_exposes_all_published_cli_scripts
  - tests/unit/test_cli.py::TestInstalledProjectScripts::test_installed_script_help_exits_zero
```

## Mandatory Self-Audit Gate

- I did not run the full `violawake-download-corpus` external download because it pulls a large LibriSpeech archive; I verified help for that command and `violawake-expand-corpus --list`, but not the full network/extract path.
- I did not fix `violawake-streaming-eval` because `docs/LANE_LEDGER.md` assigns that file to Lane 5; I captured the traceback as an out-of-lane finding.
- I did not edit README/API docs drift for `--architecture` because those docs are outside Lane 6 ownership in the ledger; the stale claim should be fixed by the docs/integration owner.
- I did not run a real microphone capture for `violawake-collect` because this Windows clean venv does not have `pyaudio` and microphone hardware is not a reliable audit dependency; the CLI now fails closed and documents `[audio]`.
- I did not run a real training job or real evaluation corpus under `[training]` because this lane's clean-install bar is CLI surface behavior; full model training/evaluation correctness is covered by Lane 4 and Lane 5.
