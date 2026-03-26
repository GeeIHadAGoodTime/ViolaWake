<!-- doc-meta
scope: Testing philosophy, test tier definitions, CI integration, coverage requirements
authority: LIVING — primary testing reference
code-paths: tests/, .github/workflows/ci.yml, pyproject.toml
staleness-signals: New test tier added, CI provider change, coverage target change
-->

# ViolaWake SDK — Test Strategy

**Version:** 0.1
**Last updated:** 2026-03-17

---

## 1. Philosophy

**Test behavior, not implementation.** Tests verify that `WakeDetector.process()` returns a score ≥ threshold when presented with a wake word — not that it internally calls `onnxruntime.InferenceSession.run()`. Public API contracts are stable; internal implementation can change without breaking tests.

**No hardware in CI.** Unit tests run without microphone hardware, model files, or GPU. Integration tests skip cleanly when model files are absent. Benchmark tests are opt-in. A developer should be able to run `pytest tests/unit/` immediately after `pip install -e ".[dev]"` with zero additional setup.

**Synthetic audio over real recordings.** Unit tests use programmatically generated audio (noise, tones, silence) from numpy. Real audio samples are used only in integration tests and are not committed to the repository.

**Accuracy claims require evidence.** The current Cohen's d claim (≥ 15.0 on the synthetic-negative benchmark) must be verifiable via a documented evaluation command, not just asserted. The `tests/benchmarks/bench_accuracy.py` test suite is the canonical evaluation that supports any accuracy claim in the README.

---

## 2. Test Tier Definitions

### Tier 1: Unit Tests (`tests/unit/`)

**What:** Pure Python logic with no external dependencies on models, hardware, or network.

**Rules:**
- No `onnxruntime.InferenceSession` calls (mock or skip)
- No `pyaudio.PyAudio()` calls (mock)
- No network access
- No file I/O outside `tmp_path` fixtures
- Must complete in ≤ 100ms per test, ≤ 30s total suite

**Coverage target:** ≥ 85% line coverage on `src/violawake_sdk/`

**What to test:**
- Decision policy logic (zero-input guard, cooldown, listening gate) — table-driven with synthetic scores
- Audio preprocessing functions (mel spectrogram, PCEN normalization, frame chunking) with numpy-generated signals
- VAD backends fallback logic (webrtcvad → silero → rms precedence)
- ModelCache path resolution, SHA-256 verification logic (with fake files)
- VoicePipeline state machine transitions (idle → listening → transcribing → responding)
- CLI argument parsing
- Error handling and exception types

**Run command:**
```bash
pytest tests/unit/ -v
```

---

### Tier 2: Integration Tests (`tests/integration/`)

**What:** Tests that require actual model files (`.onnx`). Verify that real models load and produce sensible outputs on synthetic input.

**Rules:**
- Skip cleanly with `pytest.mark.integration` if model not present:
  ```python
  pytestmark = pytest.mark.integration
  # Skips entire module if violawake_mlp_oww.onnx not in model cache
  ```
- Input is synthetic (numpy-generated), NOT real mic audio
- No network access during test execution (models must already be downloaded)
- Must complete in ≤ 30 min total

**Coverage target:** ≥ 60% line coverage on model-dependent paths

**What to test:**
- `WakeDetector` with `viola_mlp_oww.onnx`: score range [0.0, 1.0] on noise, score > 0.9 on known positive samples
- `TTSEngine` with `kokoro_v1_0.onnx`: synthesize short sentence → numpy array with correct dtype/shape
- `STTEngine` with `whisper_base`: transcribe 3s noise → empty or near-empty string
- `VADEngine` (all backends): returns float in [0.0, 1.0] per frame
- `VoicePipeline` integration (no mic): pipeline instantiates, callback registration works, clean shutdown

**Pre-condition setup:**
```bash
violawake-download --model viola_mlp_oww
violawake-download --model kokoro_v1_0
pytest tests/integration/ -v
```

---

### Tier 3: Benchmark Tests (`tests/benchmarks/`)

**What:** Performance and accuracy benchmarks. Not run in standard CI — run manually or in nightly builds.

**Rules:**
- Requires model files + hardware (at minimum a CPU)
- Uses `pytest-benchmark` for latency measurements
- Must write results to `benchmark-results/` directory
- May run for minutes (accuracy evaluation on full test set)

**What to benchmark:**

#### Latency benchmarks

```bash
pytest tests/benchmarks/bench_latency.py --benchmark-json=benchmark-results/latency.json
```

| Test | Metric | Target |
|------|--------|--------|
| `bench_wake_inference` | latency/frame (p50, p99) | ≤ 10ms, ≤ 15ms |
| `bench_vad_webrtc` | latency/frame | ≤ 1ms, ≤ 2ms |
| `bench_tts_sentence` | time-to-first-audio | ≤ 400ms (p50), ≤ 800ms (p99) |
| `bench_stt_3s` | transcription latency | ≤ 700ms (p50), ≤ 1500ms (p99) |

#### Accuracy benchmarks

```bash
pytest tests/benchmarks/bench_accuracy.py -v
```

| Test | Metric | Target | Methodology |
|------|--------|--------|-------------|
| `bench_wake_dprime` | Cohen's d (historical name) | ≥ 15.0 | Internal synthetic-negative test set: 500 positives, 2hr negatives; not a speech-negative d-prime benchmark |
| `bench_wake_far` | false accept rate/hr | ≤ 0.5 | 2hr continuous noise+music corpus |
| `bench_wake_frr` | false reject rate | ≤ 3% | 500 positive samples |
| `bench_stt_wer` | WER | ≤ 9% (base) | LibriSpeech test-clean subset (100 examples) |

These benchmarks support the numerical claims in `README.md` and `docs/PRD.md`. If a benchmark fails its target, the corresponding claim in docs must be updated.

---

### Tier 4: Hardware Tests (`tests/hardware/`)

**What:** Tests requiring a physical microphone. Not run in CI. Developer-run only.

**Rules:**
- Marked `pytest.mark.hardware`
- Skip if no audio input device detected
- Document expected hardware setup in test docstring

**What to test:**
- Real microphone capture: `stream_mic()` produces non-silent audio frames
- Wake word end-to-end: say "viola" near mic → `WakeDetector` triggers within 3 frames
- Full pipeline: wake → STT → TTS loop with real audio

---

## 3. Continuous Integration

### CI Matrix (`ci.yml`)

```
Platform:  ubuntu-22.04, windows-2022, macos-13
Python:    3.10, 3.11, 3.12
```

### CI Job: `lint`

```yaml
- run: ruff check .
- run: mypy src/violawake_sdk --strict
```

**Zero tolerance:** CI fails on any ruff or mypy error.

### CI Job: `test-unit`

```yaml
- run: pip install -e ".[dev]"
- run: pytest tests/unit/ -v --cov=violawake_sdk --cov-report=xml
- uses: Codecov/codecov-action
```

Runs on every push and PR. Must pass for PR merge.

### CI Job: `test-integration` (conditional)

Runs only on `main` branch and release tags. Downloads models from GitHub Releases before running.

```yaml
- run: pip install -e ".[all,dev]"
- run: violawake-download --model viola_mlp_oww --model kokoro_v1_0
- run: pytest tests/integration/ -v -m integration
```

### CI Job: `benchmark` (nightly)

Runs on schedule (nightly, 2am UTC) on `main` branch. Results committed to `benchmark-results/` via bot commit.

```yaml
- run: pytest tests/benchmarks/ --benchmark-json=benchmark-results/nightly.json
- run: python tools/benchmark_regression_check.py  # fails if >10% regression
```

---

## 4. Coverage Requirements

| Tier | Minimum Coverage | Enforced by |
|------|-----------------|-------------|
| Unit | 85% | `--cov-fail-under=85` in CI |
| Integration | 60% (model-dependent paths) | Manual review |
| Overall | 75% | Codecov PR gate |

**Excluded from coverage:**
- `src/violawake_sdk/tools/` (CLI scripts — tested via subprocess, not import)
- `tests/` itself
- `if TYPE_CHECKING:` blocks
- `if __name__ == "__main__":` blocks

---

## 5. Test Fixtures (`tests/conftest.py`)

### Audio fixtures

```python
@pytest.fixture
def silent_frame() -> bytes:
    """20ms of silence at 16kHz mono 16-bit PCM."""
    return (np.zeros(320, dtype=np.int16)).tobytes()

@pytest.fixture
def noise_frame() -> bytes:
    """20ms of white noise at 16kHz mono 16-bit PCM."""
    rng = np.random.default_rng(42)
    return (rng.integers(-1000, 1000, 320, dtype=np.int16)).tobytes()

@pytest.fixture
def tone_frame() -> bytes:
    """20ms of 440Hz sine wave at 16kHz mono 16-bit PCM."""
    t = np.linspace(0, 0.02, 320)
    signal = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    return signal.tobytes()

@pytest.fixture
def audio_3s() -> np.ndarray:
    """3 seconds of white noise as float32 array at 16kHz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(48000).astype(np.float32) * 0.1
```

### Mock fixtures

```python
@pytest.fixture
def mock_ort_session(mocker: MockerFixture) -> MagicMock:
    """Mock onnxruntime.InferenceSession for unit tests."""
    mock = mocker.patch("onnxruntime.InferenceSession")
    mock.return_value.run.return_value = [np.array([[0.05]])]
    return mock

@pytest.fixture
def mock_pyaudio(mocker: MockerFixture) -> MagicMock:
    """Mock pyaudio.PyAudio for unit tests."""
    return mocker.patch("pyaudio.PyAudio")
```

### Model cache fixtures

```python
@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Temporary directory as model cache root."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
```

---

## 6. Test Writing Guidelines

### Decision policy tests: table-driven

```python
@pytest.mark.parametrize("score,in_cooldown,is_playing,expected", [
    (0.90, False, False, True),   # normal detection
    (0.70, False, False, False),  # below threshold
    (0.90, True,  False, False),  # in cooldown window
    (0.90, False, True,  False),  # playback gate active
    (0.001, False, False, False), # zero-input guard (below rms)
])
def test_decision_policy(score, in_cooldown, is_playing, expected):
    policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=2.0)
    result = policy.evaluate(score=score, in_cooldown=in_cooldown, is_playing=is_playing)
    assert result == expected
```

### Exception test pattern

```python
def test_model_not_found_raises():
    with pytest.raises(ModelNotFoundError, match="viola_mlp_oww"):
        WakeDetector(model="nonexistent_model.onnx")
```

### Benchmark test pattern

```python
@pytest.mark.benchmark(min_rounds=100, warmup=True)
def test_wake_inference_latency(benchmark, loaded_wake_detector, noise_frame):
    result = benchmark(loaded_wake_detector.process, noise_frame)
    assert 0.0 <= result <= 1.0
```

---

## 7. What NOT to Test

- **ONNX Runtime internals:** We don't test that `ort.InferenceSession` loads a file correctly — that's ONNX Runtime's responsibility
- **PyAudio device enumeration:** Hardware-dependent, not meaningful in CI
- **Kokoro model quality:** "Does this TTS sound good?" is subjective and not testable via assertion
- **Training convergence:** "Does the model reach Cohen's d 15 on the synthetic benchmark after N epochs?" is a validation concern, not a unit test concern
- **Network reliability:** Download tests mock the network; we don't test GitHub Releases availability

---

## 8. Debugging Test Failures

### "Integration test skipped"
```
SKIP: Model viola_mlp_oww.onnx not found in cache. Run: violawake-download --model viola_mlp_oww
```
**Fix:** Run `violawake-download --model <model_name>` to populate the cache.

### "ruff error in CI"
Run locally: `ruff check . --fix` to auto-fix, then `ruff check .` to see remaining issues.

### "Coverage below threshold"
Add unit tests for the uncovered path. Check the HTML coverage report:
```bash
pytest tests/unit/ --cov=violawake_sdk --cov-report=html && open htmlcov/index.html
```

### "Benchmark regression detected"
```
FAIL: bench_wake_inference latency regressed by 23% (threshold: 10%)
      Baseline: 7.8ms → Current: 9.6ms
```
Profile the inference path: check for new Python overhead in `wake_detector.py`, verify ONNX Runtime version hasn't changed.
