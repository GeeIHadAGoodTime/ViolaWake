# ViolaWake SDK

**The open-source alternative to Porcupine.** A production-tested wake word engine with accessible training, ONNX inference, and a Python-first SDK.

[![PyPI version](https://badge.fury.io/py/violawake.svg)](https://badge.fury.io/py/violawake)
[![CI](https://github.com/GeeIHadAGoodTime/ViolaWake/actions/workflows/ci.yml/badge.svg)](https://github.com/GeeIHadAGoodTime/ViolaWake/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Why ViolaWake?

| | ViolaWake | Porcupine (Picovoice) | openWakeWord |
|---|---|---|---|
| **License** | Apache 2.0 | Proprietary (metered) | Apache 2.0 |
| **Training code open** | Yes | No (closed) | Yes |
| **Custom wake words** | Yes (training CLI) | Yes (paid Console) | Yes (fine-tune) |
| **Evaluation tooling** | `violawake-eval` (Cohen's d, EER, FAR/FRR, ROC AUC) | None published | Basic |
| **On-device** | Yes (ONNX) | Yes (proprietary C lib) | Yes (ONNX) |
| **Integrated TTS** | Yes (Kokoro-82M, optional extra) | No | No |
| **Speaker verification** | Yes (post-detection gate) | No | No |
| **Noise-adaptive threshold** | Yes (SNR-based) | No | No |
| **Power management** | Yes (duty cycling, battery-aware) | No | No |
| **Audio source abstraction** | Yes (mic, file, network, callback) | No | No |
| **Python SDK** | First-class | C wrapper | First-class |
| **Price at scale** | Free | Paid (free tier available) | Free |

**Our moat:** Open training code, transparent evaluation with reproducible benchmarks, production-hardened data augmentation (gain, time stretch, pitch shift, noise mixing), and a 4-gate decision policy that suppresses false positives during music playback. On a fair head-to-head benchmark against openWakeWord (same corpus, same pipeline, adversarial negatives for both systems), ViolaWake achieves **EER 5.49%** vs OWW's 8.24% — each system tested on its own best wake word. Running in production, not a demo.

> **A note on accuracy claims:** Our benchmark uses TTS-generated audio with adversarial confusables, not real-speaker recordings. Real-world accuracy depends on your deployment environment. We publish our benchmark scripts so you can reproduce and extend them. Run `violawake-eval` on your own test data.

---

## Quick Start

```bash
pip install "violawake[audio,download]"
violawake-download --model temporal_cnn
```

### Wake Word Detection (5 lines)

```python
from violawake_sdk import WakeDetector

detector = WakeDetector(model="temporal_cnn", threshold=0.80, confirm_count=3)

for audio_chunk in detector.stream_mic():  # 20ms chunks at 16kHz
    if detector.detect(audio_chunk):
        print("Wake word detected!")
        break
```

> `confirm_count=3` requires 3 consecutive above-threshold frames before firing, significantly reducing false accepts. Use `confirm_count=1` for lowest latency.

All major classes support context managers for automatic cleanup:

```python
with WakeDetector(model="temporal_cnn", threshold=0.80) as detector:
    for chunk in detector.stream_mic():
        if detector.detect(chunk):
            print("Detected!")
            break
# Resources automatically released on exit
```

### Threshold Tuning

The `threshold` parameter controls the trade-off between sensitivity and false positives:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.70 | Sensitive -- more detections, more false positives | Quiet rooms, close-mic setups |
| **0.80** | **Balanced (default)** -- recommended starting point | General-purpose, most environments |
| 0.85 | Conservative -- fewer false positives, may miss some wake words | Living rooms with TV/music |
| 0.90+ | Very conservative -- lowest false positive rate | Noisy environments, always-on kiosks |

Start at 0.80 and adjust based on your false accept rate. Use `violawake-streaming-eval` to measure FAPH (false accepts per hour) on representative audio from your deployment environment, or `violawake-eval` for clip-by-clip EER/FAR/FRR/ROC AUC.

### Text-to-Speech (Kokoro-82M)

```python
from violawake_sdk import TTSEngine

with TTSEngine() as tts:  # Downloads models on first run (~354MB total)
    audio = tts.synthesize("Hello from ViolaWake!")
    tts.play(audio)
```

### Voice Activity Detection

```python
from violawake_sdk import VADEngine

with VADEngine(backend="webrtc") as vad:  # or "silero", "rms"
    prob = vad.process_frame(audio_bytes)  # returns 0.0-1.0 speech probability
```

Three VAD backends are available:

| Backend | Engine | Latency | Accuracy | Dependencies |
|---------|--------|---------|----------|--------------|
| `webrtc` | WebRTC VAD | <1ms | Good | `webrtcvad` |
| `silero` | Silero VAD | ~2ms | Best | `torch` or `onnxruntime` |
| `rms` | RMS heuristic | <0.1ms | Basic | None (built-in) |
| `auto` | Best available | Varies | Varies | Tries webrtc -> silero -> rms |

### Full Pipeline (Wake -> STT -> TTS)

> Requires: `pip install "violawake[audio,stt,tts]"`

```python
from violawake_sdk import VoicePipeline

pipeline = VoicePipeline(
    wake_word="viola",
    stt_model="base",        # faster-whisper model size
    tts_voice="af_heart",    # Kokoro voice
)

@pipeline.on_command
def handle_command(text: str) -> str:
    print(f"Command: {text}")
    return f"You said: {text}"  # Returned string is spoken via TTS

pipeline.run()  # Blocks -- Ctrl+C to stop
```

The pipeline follows a 4-state machine: `IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> IDLE`. State transitions happen automatically based on wake word detection, VAD silence detection, and TTS completion.

---

## Audio Sources

ViolaWake defines an `AudioSource` protocol for pluggable audio input. Four implementations are included:

### MicrophoneSource (default)

```python
from violawake_sdk.audio_source import MicrophoneSource

source = MicrophoneSource(device_index=None)  # None = system default
source.start()
while True:
    frame = source.read_frame()  # 640 bytes (20ms at 16kHz, int16)
    if frame and detector.detect(frame):
        print("Detected!")
source.stop()
```

### FileSource (for testing and evaluation)

```python
from violawake_sdk.audio_source import FileSource

source = FileSource("test_audio.wav", loop=False)
source.start()
while (frame := source.read_frame()) is not None:
    if detector.detect(frame):
        print("Detected!")
source.stop()
```

Reads WAV files natively. FLAC/MP3 supported if `soundfile` is installed. Auto-warns on sample rate or channel mismatch (expects 16kHz mono int16).

### NetworkSource (for distributed systems)

```python
from violawake_sdk.audio_source import NetworkSource

# TCP streaming from a remote microphone
source = NetworkSource(host="0.0.0.0", port=9999, protocol="tcp")
source.start()  # Binds and accepts first connection
frame = source.read_frame()

# UDP streaming
source = NetworkSource(port=9999, protocol="udp", timeout=5.0)
```

> **Security note:** NetworkSource provides no authentication or encryption. Use only on trusted networks.

### CallbackSource (push model)

```python
from violawake_sdk.audio_source import CallbackSource

source = CallbackSource(timeout=1.0, max_queue_size=100)
source.start()

# Push audio from any thread (accepts bytes or numpy arrays)
source.push_audio(audio_bytes)
source.push_audio(numpy_float32_array)  # Auto-converts float32 -> int16

frame = source.read_frame()  # Blocks until data arrives or timeout
```

Ideal for integration with existing audio pipelines, WebSocket servers, or callback-based audio APIs. Drops oldest frames on queue overflow.

### Custom AudioSource

Implement the protocol for your own audio source:

```python
from violawake_sdk.audio_source import AudioSource

class MySource:
    def read_frame(self) -> bytes | None:
        """Return 640 bytes (320 int16 samples) or None if exhausted."""
        ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

assert isinstance(MySource(), AudioSource)  # Runtime-checkable protocol
```

---

## Async Detection

For asyncio-based applications, use `AsyncWakeDetector`:

```python
from violawake_sdk import AsyncWakeDetector

async with AsyncWakeDetector(model="temporal_cnn", threshold=0.80) as detector:
    result = await detector.detect(audio_frame)

    # Or stream from an async audio source
    async for detected in detector.stream(audio_source):
        if detected:
            print("Wake word!")
```

`AsyncWakeDetector` wraps `WakeDetector` in a `ThreadPoolExecutor(max_workers=1)` to avoid blocking the event loop during ONNX inference. All constructor parameters are forwarded to `WakeDetector`.

---

## Speaker Verification

Restrict wake word activation to enrolled speakers using post-detection speaker verification:

```python
from violawake_sdk.speaker import SpeakerVerificationHook, SpeakerProfile

# Create a verification hook
hook = SpeakerVerificationHook(threshold=0.65)

# Enroll a speaker (requires OWW backbone embeddings)
# Collect embeddings during a dedicated enrollment session
hook.enroll_speaker("alice", enrollment_embeddings)  # list of np.ndarray

# Wire into the detector
detector = WakeDetector(
    model="temporal_cnn",
    threshold=0.80,
    speaker_verify_fn=hook,  # Called after each detection
)

# Detection now requires both wake word match AND speaker match
for chunk in detector.stream_mic():
    if detector.detect(chunk):
        print("Wake word detected by enrolled speaker!")
```

### Persistence

```python
# Save enrolled speakers
hook.save("speakers.json")  # JSON metadata + .npz embeddings (no pickle)

# Load later
hook.load("speakers.json")
```

### Speaker Management

```python
hook.enroll_speaker("bob", embeddings)   # Returns enrollment count
hook.remove_speaker("bob")               # Returns True if found

# Verify a single embedding
result = hook.verify_speaker(embedding)
print(result.is_verified, result.speaker_id, result.similarity)
```

Thread-safe. Capped at 1000 embeddings per speaker for DoS protection on deserialization.

---

## Noise-Adaptive Detection

Automatically adjust the detection threshold based on ambient noise levels:

```python
from violawake_sdk import WakeDetector
from violawake_sdk.noise_profiler import NoiseProfiler

profiler = NoiseProfiler(
    base_threshold=0.80,
    noise_window_s=5.0,     # Rolling window for noise estimation
    min_threshold=0.60,      # Floor (never go below)
    max_threshold=0.95,      # Ceiling (never go above)
    snr_boost_db=6.0,        # High SNR -> lower threshold (easier detection)
    snr_penalty_db=3.0,      # Low SNR -> raise threshold (fewer false positives)
)

detector = WakeDetector(
    model="temporal_cnn",
    adaptive_threshold=True,
    noise_profiler=profiler,
)
```

The profiler estimates the noise floor as the 10th percentile of recent RMS values, then adjusts the threshold by up to +/-0.10 based on signal-to-noise ratio. In quiet rooms, the threshold drops for better sensitivity; in noisy environments, it rises to reduce false accepts.

```python
# Inspect the current noise profile
profile = profiler.get_profile()
print(f"Noise: {profile.noise_rms:.1f}, SNR: {profile.snr_db:.1f} dB, "
      f"Threshold: {profile.adjusted_threshold:.2f}")
```

---

## Power Management

Reduce CPU usage on battery-powered devices with intelligent frame skipping:

```python
from violawake_sdk import WakeDetector
from violawake_sdk.power_manager import PowerManager

pm = PowerManager(
    duty_cycle_n=1,          # Process every Nth frame when idle (1 = all)
    silence_rms=10.0,        # Skip inference below this RMS
    activity_threshold=0.3,  # Score above this triggers full-rate mode
    active_window_s=3.0,     # Stay in full-rate mode for 3s after activity
    battery_low_pct=20,      # Battery % below which power saving kicks in
    battery_multiplier=3,    # Multiply duty cycle by 3x on low battery
)

detector = WakeDetector(
    model="temporal_cnn",
    power_manager=pm,
)
```

Three power-saving strategies work together:
1. **Duty cycling** -- process every Nth frame when idle
2. **Silence skipping** -- skip inference entirely when input RMS is below threshold
3. **Battery-aware** -- multiplies duty cycle when on battery + low charge

Battery detection is cross-platform: tries `psutil`, falls back to Windows `ctypes` or Linux `/sys/class/power_supply/`.

```python
# Monitor power state
state = pm.get_state()
print(f"Battery: {state.battery_percent}%, Processed: {state.frames_processed}, "
      f"Skipped: {state.frames_skipped}, Rate: {state.effective_rate:.0%}")
```

---

## Multi-Model Ensemble

Run multiple models simultaneously and fuse their scores for higher accuracy:

```python
from violawake_sdk import WakeDetector, FusionStrategy

detector = WakeDetector(
    model="temporal_cnn",
    models=["temporal_cnn", "temporal_convgru"],
    fusion_strategy=FusionStrategy.AVERAGE,  # or MAX, VOTING, WEIGHTED_AVERAGE
    fusion_weights=[0.7, 0.3],  # For WEIGHTED_AVERAGE
)
```

| Strategy | Behavior | Best For |
|----------|----------|----------|
| `AVERAGE` | Mean of all model scores | General purpose |
| `MAX` | Highest score wins | Maximizing recall |
| `VOTING` | Fraction of models above threshold | Reducing false positives |
| `WEIGHTED_AVERAGE` | Weighted mean (requires `fusion_weights`) | Tuned deployments |

```python
# Access individual model scores
from violawake_sdk.ensemble import EnsembleScorer
scores = detector._ensemble.score_all(embedding)  # list[float] per model
```

---

## Confidence API

Track detection confidence beyond a simple boolean:

```python
from violawake_sdk import WakeDetector, ConfidenceResult, ConfidenceLevel

detector = WakeDetector(model="temporal_cnn", threshold=0.80, score_history_size=50)

# After processing frames...
result: ConfidenceResult = detector.get_confidence()
print(f"Score: {result.raw_score:.3f}")
print(f"Level: {result.confidence}")          # LOW, MEDIUM, HIGH, or CERTAIN
print(f"Confirm: {result.confirm_count}/{result.confirm_required}")
print(f"History: {result.score_history[-5:]}")  # Last 5 scores

# Access raw score history
scores = detector.last_scores  # tuple[float, ...]
```

Confidence levels are classified relative to the threshold:
- **LOW** -- below 75% of threshold
- **MEDIUM** -- 75-90% of threshold
- **HIGH** -- 90-100% of threshold
- **CERTAIN** -- above threshold with confirmation met

---

## Advanced Configuration

`DetectorConfig` bundles all advanced options into a single object:

```python
from violawake_sdk import DetectorConfig, WakeDetector, FusionStrategy
from violawake_sdk.noise_profiler import NoiseProfiler
from violawake_sdk.power_manager import PowerManager
from violawake_sdk.speaker import SpeakerVerificationHook

config = DetectorConfig(
    # Multi-model ensemble
    models=["temporal_cnn", "temporal_convgru"],
    fusion_strategy=FusionStrategy.AVERAGE,
    fusion_weights=None,

    # Noise-adaptive threshold
    adaptive_threshold=True,
    noise_profiler=NoiseProfiler(base_threshold=0.80),

    # Speaker verification (post-detection gate)
    speaker_verify_fn=SpeakerVerificationHook(threshold=0.65),

    # Power management
    power_manager=PowerManager(duty_cycle_n=2),

    # Confidence tracking
    confirm_count=3,
    score_history_size=50,
)

detector = config.build(model="temporal_cnn", threshold=0.80, cooldown_s=2.0)
```

Or pass options directly to the constructor:

```python
detector = WakeDetector(
    model="temporal_cnn",
    threshold=0.80,
    cooldown_s=2.0,
    backend="onnx",           # "onnx", "tflite", or "auto"
    providers=["CPUExecutionProvider"],  # ONNX Runtime execution providers
    confirm_count=3,
    adaptive_threshold=True,
    noise_profiler=NoiseProfiler(),
    power_manager=PowerManager(),
    speaker_verify_fn=hook,
    models=["temporal_cnn", "temporal_convgru"],
    fusion_strategy="average",
    score_history_size=50,
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VoicePipeline                            │
│                                                             │
│  AudioSource ──► [WakeDetector] ──► [VAD] ──► [STT] ──► cb│
│                                                             │
│  text ──► [TTS] ──► Speaker                                │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

| Module | Engine | Size | Latency |
|--------|--------|------|---------|
| Wake word | Temporal CNN on OWW embeddings (ONNX) | ~100 KB head (+OWW backbone via `openwakeword`) | ~8ms/frame |
| VAD | WebRTC VAD / Silero / RMS heuristic | <1 MB | <1ms/frame |
| STT | faster-whisper `base` | 145 MB | 0.5-2s |
| TTS | Kokoro-82M (ONNX) | 326 MB | 0.3-0.8s/sentence |

### Inference Backends

Two inference backends are supported:

| Backend | Runtime | Status |
|---------|---------|--------|
| `onnx` | ONNX Runtime | **Default** -- CPU/GPU via execution providers |
| `tflite` | TensorFlow Lite | Alternative for embedded/mobile targets |
| `auto` | Best available | Tries ONNX first, falls back to TFLite |

```python
from violawake_sdk.backends import get_backend

backend = get_backend("onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(backend.is_available())
```

---

## Training Your Own Wake Word

The training CLI lets you train a custom wake word model with ~200 positive samples:

```bash
# Collect positive samples (read prompts aloud)
violawake-collect --word "jarvis" --output data/jarvis/positives/ --count 200

# Train (auto-generates TTS positives, confusable negatives, and speech negatives)
violawake-train \
  --word "jarvis" \
  --positives data/jarvis/positives/ \
  --output models/jarvis.onnx \
  --epochs 50

# To disable augmentation, add --no-augment
# To use legacy MLP architecture, add --architecture mlp

# Evaluate (EER, FAR/FRR, ROC AUC)
violawake-eval \
  --model models/jarvis.onnx \
  --test-dir data/jarvis/test/ \
  --report
```

The `--test-dir` must contain `positives/` and `negatives/` subdirectories.

**Expected results:** EER < 10% (against the bundled synthetic negative corpus) with 200+ quality positive samples. Your real-world performance will depend on your deployment environment and negative speech corpus.

### Training Infrastructure

The training pipeline uses production techniques for class-imbalanced wake word detection:

- **FocalLoss** (gamma=2.0, alpha=0.75) -- down-weights easy negatives, focuses on hard examples
- **Label smoothing** (0.05) -- prevents overconfident predictions
- **EMA** (exponential moving average, decay=0.999) -- smooths weight updates
- **SWA** (stochastic weight averaging) -- averages weights across epochs for better generalization
- **AdamW** with cosine annealing learning rate schedule
- **Early stopping** on validation loss
- **80/20 group-aware train/val split** -- ensures same speaker's samples stay together

### Data Augmentation

Seven augmentation types are applied during training (configurable probabilities):

| Augmentation | Default Probability | Range |
|-------------|-------------------|-------|
| Gain | 80% | -6 to +6 dB |
| Time stretch | 50% | 0.9x - 1.1x speed |
| Pitch shift | 50% | +/-2 semitones |
| Additive noise | 70% | 5-20 dB SNR |
| Time shift | 50% | +/-10% of clip length |
| RIR convolution | 0% (opt-in) | Room impulse responses |
| SpecAugment | 0% (opt-in) | Frequency/time masking on spectrograms |

### Auto-Generated Training Data

When you run `violawake-train`, the CLI automatically generates:
- **TTS positives** -- 20 Edge TTS voices x 3 phrase variants (e.g., "jarvis", "hey jarvis", "ok jarvis") x 3 augmentation conditions
- **Confusable negatives** -- 16+ phonetic variants of the wake word generated via phonetic substitution tables (b/p, d/t, f/v, g/j, etc.)
- **Speech negatives** -- common English phrases that are not the wake word

### Proof: "Operator" Custom Wake Word (89 seconds, EER 7.2%)

To prove the training pipeline generalizes beyond "Viola," we trained a custom "operator" model from scratch -- zero manual data collection:

| | ViolaWake "viola" | ViolaWake "operator" | OWW "alexa" (pre-trained) |
|---|---|---|---|
| **EER** | **5.49%** | **7.2%** | 8.24% |
| **ROC AUC** | 0.988 | 0.984 | 0.956 |
| **Training time** | ~48s | **89s** | N/A (pre-trained) |
| **Architecture** | Temporal CNN | Temporal CNN | MLP on OWW embeddings |

The training CLI handled TTS sample generation (20 Edge TTS voices), confusable negative generation (16 phonetic variants), 10x augmentation, and Temporal CNN training end-to-end. OWW provides training notebooks but no pip-installable CLI tool.

Full methodology, corpus details, and reproducibility instructions: [`benchmark_v2/OPERATOR_BENCHMARK.md`](benchmark_v2/OPERATOR_BENCHMARK.md)

---

## CLI Tools Reference

ViolaWake ships 9 CLI tools, all available after `pip install "violawake[all]"`:

### Core Tools

| Command | Purpose |
|---------|---------|
| `violawake-download` | Download models from the registry |
| `violawake-train` | Train a custom wake word model |
| `violawake-eval` | Evaluate a model (EER, FAR/FRR, ROC AUC, Cohen's d) |
| `violawake-collect` | Record positive samples from microphone |

### Evaluation & Testing

| Command | Purpose |
|---------|---------|
| `violawake-streaming-eval` | Measure FAPH (false accepts per hour) on continuous audio |
| `violawake-test-confusables` | Test a model against phonetically similar words |
| `violawake-contamination-check` | Detect train/eval data overlap (filename, hash, embedding similarity) |

### Data & Corpus

| Command | Purpose |
|---------|---------|
| `violawake-generate` | Generate TTS positive samples and confusable negatives |
| `violawake-expand-corpus` | Download standard evaluation corpora (LibriSpeech, MUSAN) |

### Usage Examples

```bash
# Download a model
violawake-download --model temporal_cnn

# Generate training data without recording
violawake-generate --word "jarvis" --output data/jarvis/ \
  --count 200 --voices 20 --negatives --neg-count 300

# Streaming FAPH evaluation on a WAV file
violawake-streaming-eval --model models/jarvis.onnx --audio test_audio.wav

# Check for train/eval contamination
violawake-contamination-check \
  --train-dir data/jarvis/train/ \
  --eval-dir data/jarvis/test/ \
  --cosine-threshold 0.99

# Download evaluation corpora
violawake-expand-corpus --corpus librispeech-test-clean --output data/corpora/

# Test against phonetic confusables
violawake-test-confusables --model models/jarvis.onnx --word "jarvis"
```

---

## Models

Models are versioned and published to GitHub Releases. Use registry names without file extensions when passing `--model` or `WakeDetector(model=...)`. Download separately (too large for PyPI):

```bash
violawake-download --model temporal_cnn           # default, ~100 KB
violawake-download --model kokoro_v1_0             # TTS model, 326 MB
violawake-download --model kokoro_voices_v1_0      # TTS voices, 28 MB
```

| Model | Type | Size | EER* | Notes |
|-------|------|------|------|-------|
| `temporal_cnn.onnx` | Temporal CNN on OWW embeddings | ~100 KB | 5.49% | Production default -- best live recall + lowest FP |
| `temporal_convgru.onnx` | Temporal Conv-GRU on OWW embeddings | ~81 KB | -- | Reserve model |
| ~~`r3_10x_s42.onnx`~~ | MLP on OWW embeddings | ~34 KB | -- | **Deprecated** -- fails live mic test. Do not use. |
| `kokoro-v1.0.onnx` | Kokoro-82M TTS | ~326 MB | -- | Apache 2.0 (hosted by [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)) |

*EER (Equal Error Rate) from benchmark v2: 700 shared negatives (incl. adversarial confusables), 180 TTS positives, streaming inference. Lower is better. See `benchmark_v2/` for full methodology and scripts.

### Model Discovery

```python
from violawake_sdk import list_models, list_voices

# List available wake word models
for m in list_models():
    print(f"{m['name']:20s} {m['description']}")

# List available TTS voices
voices = list_voices()  # ['af_heart', 'af_bella', 'af_sarah', ...]
```

### Model Integrity

Downloads are verified via SHA-256 hash comparison against the model registry. The OpenWakeWord backbone files (mel-spectrogram + embedding models) are additionally verified at load time -- a hash mismatch produces a warning suggesting retraining, since backbone drift can silently degrade accuracy.

---

## Security

### Download Security

- All model downloads enforce HTTPS-only URLs
- SHA-256 integrity checks on every download
- Atomic writes prevent partial/corrupt model files
- Size validation (within 5% of declared size)

### Certificate Pinning

Optional TLS certificate pinning for model downloads:

```python
from violawake_sdk.models import download_model

# Download with certificate pinning enabled
path = download_model("temporal_cnn", use_pinning=True, strict=False)
```

The pinning system uses SPKI (Subject Public Key Info) SHA-256 hashes with TOFU (Trust On First Use) for GitHub and HuggingFace domains.

### OWW Backbone Integrity

The OpenWakeWord mel-spectrogram and embedding models are pinned by SHA-256 hash at training time and verified at inference time. If the hash changes (e.g., `openwakeword` package updated), a warning is logged:

```
WARNING: OWW backbone hash mismatch. Re-train your model if detection accuracy degrades.
```

### Safe Deserialization

Speaker verification profiles use JSON + `.npz` for persistence -- no pickle. Embedding counts are capped at 1000 per speaker to prevent DoS via crafted profile files.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIOLAWAKE_MODEL_DIR` | `~/.violawake/models/` | Directory for downloaded models |
| `VIOLAWAKE_NO_AUTO_DOWNLOAD` | unset | Set to `1`, `true`, or `yes` to disable automatic model downloads (raises `FileNotFoundError` instead) |

```bash
# Custom model directory
export VIOLAWAKE_MODEL_DIR=/opt/violawake/models

# Air-gapped deployment (pre-download models, disable auto-download)
export VIOLAWAKE_NO_AUTO_DOWNLOAD=1
```

---

## Thread Safety

All core classes are thread-safe:

- **`WakeDetector`** -- two-lock design (`_lock` for scores, `_backbone_lock` for OWW state) with documented lock ordering to prevent deadlocks
- **`SpeakerVerificationHook`** -- lock-guarded profile mutations, snapshot-based reads
- **`PowerManager`** -- lock-guarded frame counters and battery state
- **`VoicePipeline`** -- state machine transitions and worker management under locks
- **`CallbackSource`** -- thread-safe queue with `push_audio()` from any thread

Safe to share a `WakeDetector` across threads. For asyncio, use `AsyncWakeDetector` instead.

---

## Platform Support

| Platform | Wake Word | TTS | STT | Status |
|----------|-----------|-----|-----|--------|
| Windows 10/11 (x64) | ✅ | ✅ | ✅ | **Fully tested** |
| Linux (x64) | ✅ | ✅ | ✅ | CI-tested |
| macOS (arm64/x64) | ✅ | ✅ | ✅ | CI-tested (Intel), community (ARM) |
| Raspberry Pi 4 (ARM64) | ✅ | ⚠️ slow | ✅ | Supported |
| Browser/WASM | 🚧 | 🚧 | ❌ | Phase 2 (Q3 2026) |
| Android | ❌ | ❌ | ❌ | Phase 3 (2027) |
| iOS | ❌ | ❌ | ❌ | Phase 3 (2027) |

---

## Installation

**Minimum install (wake word + VAD only):**
```bash
pip install violawake
```

> **Note:** Both `import violawake` and `import violawake_sdk` work. The canonical import is `violawake_sdk` (e.g., `from violawake_sdk import WakeDetector`), but `from violawake import WakeDetector` is also supported for convenience.

**With microphone input and model downloading:**
```bash
pip install "violawake[audio,download]"
```

**With TTS:**
```bash
pip install "violawake[tts]"
```

**With STT:**
```bash
pip install "violawake[stt]"
```

**Full pipeline (all features):**
```bash
pip install "violawake[all]"
```

**Requirements:**
- Python 3.10+
- `onnxruntime >= 1.17` (CPU) or `onnxruntime-gpu` for GPU acceleration
- `pyaudio` for microphone input
- `numpy`, `scipy`
- `openwakeword >= 0.6` (optional `[oww]` extra -- provides the frozen mel/embedding backbone)

---

## Performance Benchmarks

Measured on i7-12700H, Windows 11, RTX 3060 (CPU inference):

| Operation | Latency (p50) | Latency (p99) |
|-----------|--------------|--------------|
| Wake word inference (20ms frame) | 7.8 ms | 12.1 ms |
| VAD (WebRTC, 20ms frame) | 0.4 ms | 0.8 ms |
| STT (Whisper base, 3s audio) | 680 ms | 1.2s |
| TTS first audio (Kokoro, 1 sentence) | 310 ms | 580 ms |

**Wake word accuracy** (benchmark v2 -- TTS corpus, 700 negatives incl. adversarial confusables):
- Temporal CNN model: **EER 5.49%**, ROC AUC 0.9877
- FAR @ FRR=5%: **5.43%** (vs OWW's 8.86% on its own best word)
- Live mic tested: 100% recall on direct speech, 0 false positives on podcast/music
- Real-world metrics depend on your deployment environment. Run `violawake-eval` (clip-by-clip) or `violawake-streaming-eval` (continuous FAPH) on your own test data.

---

## Debugging

Enable debug logging to see gate rejections, backbone output, score tracking, and detection decisions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from violawake_sdk import WakeDetector
detector = WakeDetector(model="temporal_cnn", threshold=0.80)
```

This produces output like:
- `Gate 1 reject: RMS 0.0 below floor 1.0` -- silence/DC offset filtered
- `Gate 3 reject: cooldown active (1.2s remaining)` -- too soon after last detection
- `Gate 4 reject: playback active` -- suppressed during music
- `Wake word detected! score=0.872` -- successful detection

Set `level=logging.INFO` for detections only (less verbose).

---

## Examples

The `examples/` directory contains runnable scripts:

| File | Description |
|------|-------------|
| `examples/basic_detection.py` | Minimal microphone wake word detection loop |
| `examples/async_detection.py` | Async wake word detection with AsyncWakeDetector |
| `examples/streaming_eval.py` | Evaluate false accepts per hour on a WAV file |

Run any example with:
```bash
python examples/basic_detection.py
```

---

## Comparison to openWakeWord

openWakeWord is the closest open-source alternative. ViolaWake differences:

- **Open, reproducible evaluation:** `violawake-eval` produces EER, FAR/FRR, ROC AUC on any model + test set. `violawake-streaming-eval` measures FAPH on continuous audio. Benchmark scripts in `benchmark_v2/` -- run them yourself.
- **Production-hardened decision policy:** 4-gate pipeline (zero-input guard, score threshold, cooldown, listening gate) plus optional multi-window confirmation -- suppresses false positives during music playback when `is_playing` state is wired up
- **Bundled pipeline:** ViolaWake ships integrated VAD + STT + TTS, not just the wake word component
- **Training infrastructure:** FocalLoss + EMA + SWA + augmentation pipeline (gain, stretch, pitch, noise, time shift; RIR and SpecAugment available opt-in) vs basic training in openWakeWord
- **Speaker verification:** Post-detection speaker gate with enrollment/persistence -- OWW has no speaker verification
- **Noise-adaptive threshold:** SNR-based threshold adjustment -- OWW uses static thresholds
- **Power management:** Duty cycling + silence skipping + battery-awareness -- OWW has no power management
- **Audio source abstraction:** Pluggable mic/file/network/callback sources -- OWW requires manual audio handling

---

## Migrating from openWakeWord

ViolaWake uses openWakeWord's mel-spectrogram embedding model as a frozen feature extractor backbone. If you have existing OWW training data, you can use it directly with ViolaWake's training CLI.

**Key differences from OWW:**
- **Decision policy:** ViolaWake adds a multi-gate pipeline (RMS floor, cooldown, playback suppression) on top of raw scores. OWW exposes raw sigmoid scores only.
- **Temporal models:** ViolaWake supports Temporal CNN and Conv-GRU heads that score across a sliding window of embeddings, not just a single frame. This reduces false positives on speech that partially matches the wake word.
- **Augmentation pipeline:** ViolaWake's training CLI applies gain, time stretch, pitch shift, noise mixing, and RIR convolution. SpecAugment is available for custom spectrogram-level pipelines via `AugmentationPipeline.augment_spectrogram()`. OWW's default training uses minimal augmentation.
- **Confidence API:** `detector.get_confidence()` and `detector.last_scores` provide structured confidence tracking that OWW does not offer.

**Using existing OWW training data:**
```bash
# Your OWW positive samples work as-is (16kHz WAV/FLAC)
violawake-train \
  --word "my_wake_word" \
  --positives path/to/oww_positives/ \
  --negatives path/to/oww_negatives/ \
  --output models/my_wake_word.onnx \
  --epochs 50
```

No format conversion is needed -- ViolaWake reads the same 16kHz mono WAV/FLAC files that OWW uses.

---

## API Reference

### Public Exports (`from violawake_sdk import ...`)

**Core Detection:**
- `WakeDetector` -- synchronous wake word detector
- `AsyncWakeDetector` -- async wrapper for asyncio
- `DetectorConfig` -- bundled advanced configuration
- `WakeDecisionPolicy` -- 4-gate decision pipeline
- `validate_audio_chunk` -- input validation utility

**Confidence & Scoring:**
- `ConfidenceResult` -- structured confidence snapshot
- `ConfidenceLevel` -- LOW/MEDIUM/HIGH/CERTAIN enum
- `FusionStrategy` -- ensemble fusion strategy enum

**Pipeline Components:**
- `VoicePipeline` -- full Wake -> VAD -> STT -> TTS orchestration
- `VADEngine` -- voice activity detection (3 backends)
- `TTSEngine` -- Kokoro-82M text-to-speech (optional `[tts]` extra)
- `STTEngine` -- faster-whisper speech-to-text (optional `[stt]` extra)

**Audio Sources** (`from violawake_sdk.audio_source import ...`):
- `AudioSource` -- runtime-checkable protocol
- `MicrophoneSource` -- PyAudio microphone capture
- `FileSource` -- WAV/FLAC file playback
- `NetworkSource` -- TCP/UDP raw PCM streaming
- `CallbackSource` -- push-model audio ingestion

**Advanced** (`from violawake_sdk.<module> import ...`):
- `NoiseProfiler`, `NoiseProfile` -- noise-adaptive threshold (`noise_profiler`)
- `PowerManager`, `PowerState` -- battery-aware power management (`power_manager`)
- `SpeakerVerificationHook`, `SpeakerProfile`, `SpeakerVerifyResult` -- speaker verification (`speaker`)
- `EnsembleScorer` -- multi-model scoring (`ensemble`)
- `ScoreTracker` -- score history tracking (`confidence`)

**Exceptions:**
- `ViolaWakeError` -- base exception
- `ModelNotFoundError` -- model not in registry or on disk
- `ModelLoadError` -- model file corrupt or incompatible
- `AudioCaptureError` -- microphone/audio source failure
- `PipelineError` -- voice pipeline state error

**Discovery:**
- `list_models()` -- available wake word models
- `list_voices()` -- available TTS voices
- `__version__` -- SDK version string

---

## Roadmap

**v1.0 (Q2 2026) -- Phase 1 MVP:**
- [x] Python SDK (Wake + VAD)
- [x] Kokoro TTS integration
- [x] faster-whisper STT integration
- [x] Full VoicePipeline class
- [x] Training CLI (9 tools)
- [x] PyPI release
- [x] Speaker verification
- [x] Noise-adaptive detection
- [x] Power management
- [x] Audio source abstraction
- [x] Multi-model ensemble
- [ ] Documentation site

**v1.1 (Q3 2026) -- Streaming + Web:**
- [ ] Streaming STT (faster-whisper generator mode)
- [ ] WASM build for ViolaWake
- [ ] JavaScript/Node SDK wrapper
- [ ] Custom wake word web Console (alpha)

**v2.0 (Q1 2027) -- Multi-platform:**
- [ ] Android SDK (ONNX Runtime Android)
- [ ] iOS SDK (ONNX Runtime iOS)
- [ ] DeepFilterNet noise suppression integration
- [ ] Speaker diarization (pyannote.audio)
- [ ] License/metering infrastructure

---

## Contributing

```bash
git clone https://github.com/GeeIHadAGoodTime/ViolaWake
cd ViolaWake
pip install -e ".[dev]"
pre-commit install
pytest tests/
```

See `CONTRIBUTING.md` for guidelines.

---

## License

Apache 2.0. Models trained on open datasets. See `LICENSE` for details.

ViolaWake uses OpenWakeWord as a frozen feature extractor backbone (also Apache 2.0). The classification heads (Temporal CNN, Conv-GRU) and training pipeline are original ViolaWake work.
