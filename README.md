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
| **On-device** | Yes (ONNX + TFLite) | Yes (proprietary C lib) | Yes (ONNX) |
| **Integrated TTS** | Yes (Kokoro-82M, streaming) | No | No |
| **Integrated STT** | Yes (faster-whisper, with segments) | No | No |
| **Speaker verification** | Yes (experimental, post-detection gate) | No | No |
| **Noise-adaptive threshold** | Yes (SNR-based) | No | No |
| **Power management** | Yes (duty cycling, battery-aware) | No | No |
| **Audio source abstraction** | Yes (mic, file, network, callback) | No | No |
| **Python SDK** | First-class | C wrapper | First-class |
| **Price at scale** | Free | Paid (free tier available) | Free |

**Our moat:** Open training code, transparent evaluation with reproducible benchmarks, production-hardened data augmentation (gain, time stretch, pitch shift, noise mixing, pink noise, synthetic RIR), and a 4-gate decision policy that suppresses false positives during music playback. On a fair head-to-head benchmark against openWakeWord (same corpus, same pipeline, adversarial negatives for both systems), ViolaWake achieves **EER 5.49%** vs OWW's 8.24% -- each system tested on its own best wake word. Running in production, not a demo.

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

### Raw Scores (for visualization and custom logic)

```python
# detect() returns boolean. process() returns the raw model score (0.0-1.0):
score = detector.process(audio_chunk)
print(f"Score: {score:.3f}")  # Use for plotting, custom thresholding, etc.
```

### Detection from Any AudioSource

```python
from violawake_sdk.audio_source import FileSource

source = FileSource("test.wav")
# from_source() ties an AudioSource to the detector and yields results
for detected in detector.from_source(source):
    if detected:
        print("Wake word found in file!")
```

---

## Installation

**Minimum install (wake word + VAD only):**
```bash
pip install violawake
```

> **Note:** Both `import violawake` and `import violawake_sdk` work. The canonical import is `violawake_sdk` (e.g., `from violawake_sdk import WakeDetector`), but `from violawake import WakeDetector` is also supported for convenience. The legacy `WakewordDetector` alias is available for backward compatibility.

### Extras

| Extra | Install | What it adds |
|-------|---------|-------------|
| `audio` | `pip install "violawake[audio]"` | Microphone capture (`pyaudio`, `soundfile`) |
| `download` | `pip install "violawake[download]"` | Model downloading with progress bars (`requests`, `tqdm`) |
| `tts` | `pip install "violawake[tts]"` | Kokoro-82M text-to-speech (`kokoro-onnx`, `sounddevice`) |
| `stt` | `pip install "violawake[stt]"` | faster-whisper speech-to-text |
| `vad` | `pip install "violawake[vad]"` | WebRTC VAD backend (`webrtcvad`) |
| `oww` | `pip install "violawake[oww]"` | OpenWakeWord backbone (skip if already installed) |
| `tflite` | `pip install "violawake[tflite]"` | TFLite inference backend (alternative to ONNX) |
| `training` | `pip install "violawake[training]"` | Full training pipeline (`torch`, `librosa`, `scikit-learn`, `edge-tts`, etc.) |
| `generate` | `pip install "violawake[generate]"` | TTS sample generation without torch (`edge-tts`, `pydub`) |
| `dev` | `pip install "violawake[dev]"` | Development tools (`pytest`, `ruff`, `mypy`, `pre-commit`) |
| `docs` | `pip install "violawake[docs]"` | API documentation generation (`pdoc`) |
| `all` | `pip install "violawake[all]"` | Everything above |

**Requirements:**
- Python 3.10+
- `onnxruntime >= 1.17` (CPU) or `onnxruntime-gpu` for GPU acceleration
- `numpy`, `scipy`

---

## Wake Word Detection

### Core Methods

```python
from violawake_sdk import WakeDetector

with WakeDetector(model="temporal_cnn", threshold=0.80, cooldown_s=2.0) as detector:
    # detect() — boolean detection with full 4-gate pipeline
    detected: bool = detector.detect(audio_chunk)
    detected = detector.detect(audio_chunk, is_playing=True)  # Gate 4: suppress during playback

    # process() — raw model score (0.0-1.0), bypasses decision gates
    score: float = detector.process(audio_chunk)

    # from_source() — detection loop from any AudioSource
    from violawake_sdk.audio_source import FileSource
    for detected in detector.from_source(FileSource("test.wav")):
        if detected:
            print("Detected!")

    # stream_mic() — built-in microphone streaming (requires [audio] extra)
    for chunk in detector.stream_mic():
        if detector.detect(chunk):
            break

    # reset_cooldown() — allow immediate re-detection after cooldown
    detector.reset_cooldown()

    # get_confidence() — structured confidence snapshot
    result = detector.get_confidence()
```

### Threshold Tuning

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.70 | Sensitive -- more detections, more false positives | Quiet rooms, close-mic setups |
| **0.80** | **Balanced (default)** -- recommended starting point | General-purpose, most environments |
| 0.85 | Conservative -- fewer false positives, may miss some wake words | Living rooms with TV/music |
| 0.90+ | Very conservative -- lowest false positive rate | Noisy environments, always-on kiosks |

Start at 0.80 and adjust based on your false accept rate. Use `violawake-streaming-eval` to measure FAPH (false accepts per hour) on representative audio from your deployment environment, or `violawake-eval` for clip-by-clip EER/FAR/FRR/ROC AUC.

---

## Voice Pipeline (Wake -> STT -> TTS)

> Requires: `pip install "violawake[audio,stt,tts]"`

```python
from violawake_sdk import VoicePipeline

pipeline = VoicePipeline(
    wake_word="viola",
    stt_model="base",         # faster-whisper model size
    tts_voice="af_heart",     # Kokoro voice
    threshold=0.80,           # Wake word threshold
    vad_backend="auto",       # "webrtc", "silero", "rms", or "auto"
    vad_threshold=0.4,        # VAD speech probability threshold
    enable_tts=True,          # Set False to disable TTS responses
    device_index=None,        # Microphone device (None = system default)
    on_wake=lambda: print("Wake!"),  # Callback fired on wake word detection
)

@pipeline.on_command
def handle_command(text: str) -> str:
    print(f"Command: {text}")
    return f"You said: {text}"  # Returned string is spoken via TTS

pipeline.run()  # Blocks -- Ctrl+C to stop
```

The pipeline follows a 4-state machine: `IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> IDLE`.

```python
# Programmatic TTS from within a handler
pipeline.speak("Processing your request...")  # Synthesize and play immediately
```

---

## Text-to-Speech (Kokoro-82M)

> Requires: `pip install "violawake[tts]"`

```python
from violawake_sdk import TTSEngine

with TTSEngine(speed=1.0) as tts:  # speed: 0.1-3.0 (default 1.0)
    # Batch synthesis
    audio = tts.synthesize("Hello from ViolaWake!")
    tts.play(audio)                # Blocking playback
    tts.play(audio, blocking=False)  # Non-blocking playback
    tts.play_async(audio)          # Alias for non-blocking

    # Streaming synthesis (sentence-by-sentence, low latency)
    for chunk in tts.synthesize_chunked("First sentence. Second sentence. Third."):
        tts.play(chunk)  # Play each sentence as it's synthesized
```

`synthesize_chunked()` splits text at sentence boundaries and yields audio progressively -- ideal for streaming LLM responses where you want playback to start before full synthesis completes.

```python
# Discover available voices
from violawake_sdk import list_voices
voices = list_voices()  # ['af_heart', 'af_bella', 'af_sarah', ...]
```

---

## Speech-to-Text (faster-whisper)

> Requires: `pip install "violawake[stt]"`

```python
from violawake_sdk import STTEngine

with STTEngine(model_size="base") as stt:
    # Simple transcription (returns text string)
    text = stt.transcribe(audio_numpy_array)

    # Full transcription with segments and metadata
    result = stt.transcribe_full(audio_numpy_array)
    print(result.text)             # Full text
    print(result.language)         # Detected language code
    print(result.language_prob)    # Language confidence
    for seg in result.segments:
        print(f"[{seg.start:.1f}-{seg.end:.1f}] {seg.text} (p={seg.no_speech_prob:.2f})")

    # Pre-warm the model (eager load, avoids cold-start latency)
    stt.prewarm()
```

### STT Model Profiles

| Model | Latency | WER | VRAM |
|-------|---------|-----|------|
| `tiny` | Fastest | Higher | ~1 GB |
| `base` | Fast | Good | ~1 GB |
| `small` | Medium | Better | ~2 GB |
| `medium` | Slow | Best | ~5 GB |

Language detection is cached with a configurable TTL to avoid repeated detection on consecutive utterances in the same language.

### File-Based Transcription

```python
from violawake_sdk.stt_engine import STTFileEngine, transcribe_wav_file

# Class-based
engine = STTFileEngine(model_size="base")
text = engine.transcribe("recording.wav")

# One-liner convenience function
text = transcribe_wav_file("recording.wav")
```

---

## Voice Activity Detection

```python
from violawake_sdk import VADEngine

with VADEngine(backend="webrtc") as vad:
    prob = vad.process_frame(audio_bytes)  # Returns 0.0-1.0 speech probability
    is_speech = vad.is_speech(audio_bytes, threshold=0.5)  # Boolean convenience

    # Check which backend was selected (useful with backend="auto")
    print(vad.backend_name)  # "webrtc", "silero", or "rms"
```

| Backend | Engine | Latency | Accuracy | Dependencies |
|---------|--------|---------|----------|--------------|
| `webrtc` | WebRTC VAD | <1ms | Good | `webrtcvad` (install via `[vad]` extra) |
| `silero` | Silero VAD | ~2ms | Best | `torch` or `onnxruntime` |
| `rms` | RMS heuristic | <0.1ms | Basic | None (built-in) |
| `auto` | Best available | Varies | Varies | Tries webrtc -> silero -> rms |

---

## Audio Sources

ViolaWake defines an `AudioSource` protocol for pluggable audio input. Four implementations are included:

### MicrophoneSource (default)

```python
from violawake_sdk.audio_source import MicrophoneSource

source = MicrophoneSource(device_index=None, sample_rate=16000, frame_samples=320)
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
```

Reads WAV files natively. FLAC and other formats supported if `soundfile` is installed. Auto-warns on sample rate or channel mismatch (expects 16kHz mono int16).

### NetworkSource (for distributed systems)

```python
from violawake_sdk.audio_source import NetworkSource

# TCP streaming from a remote microphone
source = NetworkSource(host="0.0.0.0", port=9999, protocol="tcp", timeout=5.0)

# UDP streaming
source = NetworkSource(port=9999, protocol="udp")
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
    score = await detector.process(audio_frame)  # Raw score (async)

    # Stream from an async audio source
    async for detected in detector.stream(audio_source):
        if detected:
            print("Wake word!")

    detector.reset_cooldown()  # Also available on async detector
```

`AsyncWakeDetector` wraps `WakeDetector` in a `ThreadPoolExecutor(max_workers=1)` to avoid blocking the event loop during ONNX inference. All constructor parameters are forwarded to `WakeDetector`.

---

## Speaker Verification (Experimental)

Restrict wake word activation to enrolled speakers using post-detection speaker verification. Verification accuracy has not been evaluated on standard speaker benchmarks -- validate on your own data before production use.

```python
from violawake_sdk.speaker import SpeakerVerificationHook, SpeakerProfile

hook = SpeakerVerificationHook(threshold=0.65)
hook.enroll_speaker("alice", enrollment_embeddings)  # list of np.ndarray

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

### Persistence and Management

```python
hook.save("speakers.json")          # JSON metadata + .npz embeddings (no pickle)
hook.load("speakers.json")
hook.enroll_speaker("bob", embeddings)  # Returns enrollment count
hook.remove_speaker("bob")             # Returns True if found

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

# Inspect current noise profile
profile = profiler.get_profile()
print(f"Noise: {profile.noise_rms:.1f}, SNR: {profile.snr_db:.1f} dB, "
      f"Threshold: {profile.adjusted_threshold:.2f}")
```

The profiler estimates the noise floor as the 10th percentile of recent RMS values, then adjusts the threshold by up to +/-0.10 based on signal-to-noise ratio.

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

detector = WakeDetector(model="temporal_cnn", power_manager=pm)

state = pm.get_state()
print(f"Battery: {state.battery_percent}%, Processed: {state.frames_processed}, "
      f"Skipped: {state.frames_skipped}, Rate: {state.effective_rate:.0%}")
```

Three strategies: duty cycling, silence skipping, battery-aware scaling. Battery detection is cross-platform: tries `psutil`, falls back to Windows `ctypes` or Linux `/sys/class/power_supply/`.

---

## Multi-Model Ensemble

Run multiple models simultaneously and fuse their scores:

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

---

## Confidence API

Track detection confidence beyond a simple boolean:

```python
from violawake_sdk import WakeDetector, ConfidenceResult, ConfidenceLevel

detector = WakeDetector(model="temporal_cnn", threshold=0.80, score_history_size=50)

result: ConfidenceResult = detector.get_confidence()
print(f"Score: {result.raw_score:.3f}")
print(f"Level: {result.confidence}")          # LOW, MEDIUM, HIGH, or CERTAIN
print(f"Confirm: {result.confirm_count}/{result.confirm_required}")
print(f"History: {result.score_history[-5:]}")  # Last 5 scores
scores = detector.last_scores  # tuple[float, ...]
```

Levels: **LOW** (below 75% of threshold), **MEDIUM** (75-90%), **HIGH** (90-100%), **CERTAIN** (above threshold with confirmation met).

---

## Advanced Configuration

`DetectorConfig` bundles all advanced options:

```python
from violawake_sdk import DetectorConfig, WakeDetector, FusionStrategy
from violawake_sdk.noise_profiler import NoiseProfiler
from violawake_sdk.power_manager import PowerManager
from violawake_sdk.speaker import SpeakerVerificationHook

config = DetectorConfig(
    models=["temporal_cnn", "temporal_convgru"],
    fusion_strategy=FusionStrategy.AVERAGE,
    fusion_weights=None,
    adaptive_threshold=True,
    noise_profiler=NoiseProfiler(base_threshold=0.80),
    speaker_verify_fn=SpeakerVerificationHook(threshold=0.65),
    power_manager=PowerManager(duty_cycle_n=2),
    confirm_count=3,
    score_history_size=50,
)

detector = config.build(model="temporal_cnn", threshold=0.80, cooldown_s=2.0)
```

Or pass all options directly to the `WakeDetector` constructor:

```python
detector = WakeDetector(
    model="temporal_cnn",
    threshold=0.80,
    cooldown_s=2.0,
    backend="onnx",           # "onnx", "tflite", or "auto"
    providers=["CPUExecutionProvider"],
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

| Module | Engine | Size | Latency |
|--------|--------|------|---------|
| Wake word | Temporal CNN on OWW embeddings (ONNX) | ~100 KB head (+OWW backbone via `openwakeword`) | ~8ms/frame |
| VAD | WebRTC VAD / Silero / RMS heuristic | <1 MB | <1ms/frame |
| STT | faster-whisper `base` | 145 MB | 0.5-2s |
| TTS | Kokoro-82M (ONNX) | 326 MB | 0.3-0.8s/sentence |

### Inference Backends

| Backend | Runtime | Status |
|---------|---------|--------|
| `onnx` | ONNX Runtime | **Default** -- CPU/GPU via execution providers |
| `tflite` | TensorFlow Lite | Alternative for embedded/mobile targets |
| `auto` | Best available | Tries ONNX first, falls back to TFLite |

```python
from violawake_sdk.backends import get_backend

backend = get_backend("onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
print(backend.is_available())

# TFLite backend supports num_threads for performance tuning
backend = get_backend("tflite")
session = backend.load("model.tflite", num_threads=4)
```

### ONNX-to-TFLite Conversion

Convert trained ONNX models to TFLite for embedded deployment:

```python
from violawake_sdk.backends.tflite_backend import convert_onnx_to_tflite

convert_onnx_to_tflite("model.onnx", "model.tflite")  # Optional int8 quantization
```

---

## Training Your Own Wake Word

> Requires: `pip install "violawake[training]"` (or `[generate]` for data generation without torch)

```bash
# Collect positive samples (read prompts aloud)
violawake-collect --word "jarvis" --output data/jarvis/positives/ --count 200

# Train (auto-generates TTS positives, confusable negatives, and speech negatives)
violawake-train \
  --word "jarvis" \
  --positives data/jarvis/positives/ \
  --output models/jarvis.onnx \
  --epochs 50 \
  --architecture temporal_cnn   # or "mlp" for legacy architecture

# Evaluate (EER, FAR/FRR, ROC AUC)
violawake-eval \
  --model models/jarvis.onnx \
  --test-dir data/jarvis/test/ \
  --report \
  --dump-scores scores.csv      # Per-file scores for debugging false rejects/accepts
```

### Model Architectures

Three architectures are available for training:

| Architecture | Class | Params | Best For |
|-------------|-------|--------|----------|
| `temporal_cnn` | `TemporalCNN` | ~25K | **Production default** -- best accuracy/speed tradeoff |
| `temporal_convgru` | `TemporalConvGRU` | ~18K | Smallest model, hybrid CNN+GRU |
| `mlp` | MLP on single embedding | ~4K | Legacy, fastest inference |

### Training Infrastructure

- **FocalLoss** (gamma=2.0, alpha=0.75) -- down-weights easy negatives, focuses on hard examples
- **Label smoothing** (0.05) -- prevents overconfident predictions
- **EMA** (exponential moving average, decay=0.999) -- smooths weight updates
- **SWA** (stochastic weight averaging) -- averages weights across epochs
- **Auto-select averaging** -- automatically compares raw, EMA, and SWA models on validation loss and picks the best
- **AdamW** with cosine annealing learning rate schedule
- **Early stopping** on validation loss
- **80/20 group-aware train/val split** -- ensures same speaker's samples stay together

### Data Augmentation

Eight augmentation types are applied during training (configurable probabilities):

| Augmentation | Default Probability | Range |
|-------------|-------------------|-------|
| Gain | 80% | -6 to +6 dB |
| Time stretch | 50% | 0.9x - 1.1x speed |
| Pitch shift | 50% | +/-2 semitones |
| Additive noise (white + pink) | 70% | 5-20 dB SNR; pink noise via Voss-McCartney algorithm |
| Time shift | 50% | +/-10% of clip length |
| RIR convolution | 0% (opt-in) | Room impulse responses; includes synthetic RIR generator |
| SpecAugment | 0% (opt-in) | Frequency/time masking on spectrograms |

Synthetic room impulse responses can be generated automatically when no real RIR files are available, using exponential decay modeling.

### Auto-Generated Training Data

`violawake-train` automatically generates:
- **TTS positives** -- 20 Edge TTS voices x 3 phrase variants x 3 augmentation conditions
- **Confusable negatives** -- 16+ phonetic variants via phonetic substitution tables (b/p, d/t, f/v, g/j, etc.)
- **Speech negatives** -- common English phrases that are not the wake word

### Proof: "Operator" Custom Wake Word (89 seconds, EER 7.2%)

| | ViolaWake "viola" | ViolaWake "operator" | OWW "alexa" (pre-trained) |
|---|---|---|---|
| **EER** | **5.49%** | **7.2%** | 8.24% |
| **ROC AUC** | 0.988 | 0.984 | 0.956 |
| **Training time** | ~48s | **89s** | N/A (pre-trained) |

Full methodology: [`benchmark_v2/OPERATOR_BENCHMARK.md`](benchmark_v2/OPERATOR_BENCHMARK.md)

### Programmatic Evaluation API

```python
from violawake_sdk.training.evaluate import (
    evaluate_onnx_model,
    compute_confusion_matrix,
    find_optimal_threshold,
)

# Full evaluation
results = evaluate_onnx_model("model.onnx", "test_dir/", threshold=0.80)
print(results["roc_auc"], results["cohens_d"], results["false_reject_rate"])

# Confusion matrix
cm = compute_confusion_matrix(scores, labels, threshold=0.80)
print(cm["precision"], cm["recall"], cm["f1"])

# Optimal threshold sweep
optimal = find_optimal_threshold(scores, labels)
print(optimal["threshold"], optimal["eer"])
```

---

## CLI Tools Reference

ViolaWake ships 9 CLI tools:

### Core Tools

| Command | Key Flags | Purpose |
|---------|-----------|---------|
| `violawake-download` | `--model NAME`, `--list` | Download models or list cached models |
| `violawake-train` | `--word`, `--positives`, `--output`, `--epochs`, `--architecture [temporal_cnn\|mlp]`, `--no-augment` | Train a custom wake word model |
| `violawake-eval` | `--model`, `--test-dir`, `--report`, `--dump-scores FILE` | Evaluate (EER, FAR/FRR, ROC AUC, Cohen's d, per-file CSV) |
| `violawake-collect` | `--word`, `--output`, `--count` | Record positive samples from microphone |

### Evaluation & Testing

| Command | Key Flags | Purpose |
|---------|-----------|---------|
| `violawake-streaming-eval` | `--model`, `--audio` | Measure FAPH on continuous audio |
| `violawake-test-confusables` | `--model`, `--word` | Test against phonetically similar words |
| `violawake-contamination-check` | `--train-dir`, `--eval-dir`, `--cosine-threshold` | Detect train/eval overlap (filename, hash, embedding) |

### Data & Corpus

| Command | Key Flags | Purpose |
|---------|-----------|---------|
| `violawake-generate` | `--word`, `--output`, `--count`, `--voices`, `--negatives`, `--neg-count` | Generate TTS positives and confusable negatives |
| `violawake-expand-corpus` | `--corpus [librispeech-test-clean\|librispeech-test-other\|musan-speech]`, `--output` | Download standard evaluation corpora |

### Usage Examples

```bash
# Download and list models
violawake-download --model temporal_cnn
violawake-download --list                     # Show cached models with sizes

# Generate training data without recording
violawake-generate --word "jarvis" --output data/ --count 200 --negatives --neg-count 300

# Train with specific architecture
violawake-train --word "jarvis" --positives data/positives/ --output model.onnx \
  --epochs 50 --architecture temporal_cnn

# Evaluate with per-file score dump
violawake-eval --model model.onnx --test-dir data/test/ --report --dump-scores scores.csv

# Streaming FAPH evaluation
violawake-streaming-eval --model model.onnx --audio test_audio.wav

# Check for train/eval contamination
violawake-contamination-check --train-dir data/train/ --eval-dir data/test/ --cosine-threshold 0.99
```

---

## Models

Models are versioned and published to GitHub Releases. Download separately (too large for PyPI):

```bash
violawake-download --model temporal_cnn           # default, ~100 KB
violawake-download --model kokoro_v1_0             # TTS model, 326 MB
violawake-download --model kokoro_voices_v1_0      # TTS voices, 28 MB
```

| Model | Type | Size | EER* | Notes |
|-------|------|------|------|-------|
| `temporal_cnn.onnx` | Temporal CNN on OWW embeddings | ~100 KB | 5.49% | Production default |
| `temporal_convgru.onnx` | Temporal Conv-GRU on OWW embeddings | ~81 KB | -- | Reserve model |
| ~~`r3_10x_s42.onnx`~~ | MLP on OWW embeddings | ~34 KB | -- | **Deprecated** |
| `kokoro-v1.0.onnx` | Kokoro-82M TTS | ~326 MB | -- | Apache 2.0 |

*EER from benchmark v2: 700 negatives (incl. adversarial confusables), 180 TTS positives, streaming inference. See `benchmark_v2/`.

### Model Discovery and Cache Management

```python
from violawake_sdk import list_models, list_voices
from violawake_sdk.models import list_cached_models, check_registry_integrity

# Discover available models
for m in list_models():
    print(f"{m['name']:20s} {m['description']}")

# List locally cached models with paths and sizes
for name, path, size_mb in list_cached_models():
    print(f"{name}: {path} ({size_mb:.1f} MB)")

# Validate registry integrity (for CI pipelines)
errors = check_registry_integrity(strict=True)
assert not errors, f"Registry issues: {errors}"

# List TTS voices
voices = list_voices()  # ['af_heart', 'af_bella', 'af_sarah', ...]
```

### Model Integrity

Downloads are verified via SHA-256 hash comparison. The OpenWakeWord backbone files are additionally verified at load time -- a hash mismatch warns about potential accuracy degradation.

---

## Security

### Download Security

- HTTPS-only URLs enforced
- SHA-256 integrity checks on every download
- Atomic writes prevent partial/corrupt files
- Size validation (within 5% of declared size)

### Certificate Pinning

Optional TLS certificate pinning for model downloads:

```python
from violawake_sdk.models import download_model
from violawake_sdk.security import (
    add_pins,             # Register custom certificate pins
    fetch_live_spki_pins, # Bootstrap pins from live server
    CertPinError,         # Catchable pinning violation exception
)

# Download with pinning
path = download_model("temporal_cnn", use_pinning=True)

# Add custom pins for self-hosted model repositories
pins = fetch_live_spki_pins("models.example.com")
add_pins("models.example.com", frozenset(pins))
```

TOFU (Trust On First Use) for GitHub and HuggingFace domains. Custom pins via `add_pins()` for self-hosted infrastructure.

### OWW Backbone Integrity

OpenWakeWord backbone files are pinned by SHA-256 at training time and verified at inference time. Hash mismatch logs a warning suggesting retraining.

### Safe Deserialization

Speaker profiles use JSON + `.npz` (no pickle). Embedding counts capped at 1000 per speaker for DoS protection.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIOLAWAKE_MODEL_DIR` | `~/.violawake/models/` | Directory for downloaded models |
| `VIOLAWAKE_NO_AUTO_DOWNLOAD` | unset | Set to `1`, `true`, or `yes` to disable auto-download |

```bash
export VIOLAWAKE_MODEL_DIR=/opt/violawake/models
export VIOLAWAKE_NO_AUTO_DOWNLOAD=1  # Air-gapped deployment
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
| Windows 10/11 (x64) | Yes | Yes | Yes | **Fully tested** |
| Linux (x64) | Yes | Yes | Yes | CI-tested |
| macOS (arm64/x64) | Yes | Yes | Yes | CI-tested |
| Raspberry Pi 4 (ARM64) | Yes | Slow | Yes | Supported |
| Browser/WASM | Planned | Planned | No | Phase 2 (Q3 2026) |

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
- Temporal CNN: **EER 5.49%**, ROC AUC 0.9877
- FAR @ FRR=5%: **5.43%** (vs OWW's 8.86%)
- Live mic tested: 100% recall on direct speech, 0 false positives on podcast/music
- Run `violawake-eval` or `violawake-streaming-eval` on your own data.

---

## Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from violawake_sdk import WakeDetector
detector = WakeDetector(model="temporal_cnn", threshold=0.80)
```

Output includes:
- `Gate 1 reject: RMS 0.0 below floor 1.0` -- silence/DC offset filtered
- `Gate 3 reject: cooldown active (1.2s remaining)` -- too soon after last detection
- `Gate 4 reject: playback active` -- suppressed during music
- `Wake word detected! score=0.872` -- successful detection

---

## Examples

| File | Description |
|------|-------------|
| `examples/basic_detection.py` | Minimal microphone wake word detection loop |
| `examples/async_detection.py` | Async wake word detection with AsyncWakeDetector |
| `examples/streaming_eval.py` | Evaluate false accepts per hour on a WAV file |

```bash
python examples/basic_detection.py
```

---

## Comparison to openWakeWord

- **Evaluation:** `violawake-eval` (EER, FAR/FRR, ROC AUC), `violawake-streaming-eval` (FAPH), `violawake-contamination-check`. OWW has basic evaluation.
- **Decision policy:** 4-gate pipeline + multi-window confirmation. OWW: raw sigmoid only.
- **Bundled pipeline:** Integrated VAD + STT + TTS with streaming synthesis. OWW: wake word only.
- **Training:** FocalLoss + EMA + SWA + auto-select + 8 augmentation types + synthetic RIR. OWW: basic training.
- **Speaker verification:** Post-detection speaker gate. OWW: none.
- **Noise-adaptive threshold:** SNR-based. OWW: static thresholds.
- **Power management:** Duty cycling + battery-awareness. OWW: none.
- **Audio sources:** Pluggable protocol with 4 implementations. OWW: manual audio handling.
- **Model conversion:** ONNX-to-TFLite converter included. OWW: ONNX only.

---

## Migrating from openWakeWord

```bash
# Your OWW positive samples work as-is (16kHz WAV/FLAC)
violawake-train \
  --word "my_wake_word" \
  --positives path/to/oww_positives/ \
  --negatives path/to/oww_negatives/ \
  --output models/my_wake_word.onnx \
  --epochs 50
```

No format conversion needed. ViolaWake reads the same 16kHz mono WAV/FLAC files as OWW. Key differences: multi-gate decision policy, temporal model heads, augmentation pipeline, confidence API (`detector.get_confidence()`, `detector.last_scores`).

---

## API Reference

### Top-Level Exports (`from violawake_sdk import ...`)

**Core Detection:**
- `WakeDetector` -- synchronous detector (`.detect()`, `.process()`, `.from_source()`, `.stream_mic()`, `.reset_cooldown()`, `.get_confidence()`, `.last_scores`)
- `AsyncWakeDetector` -- async wrapper (`.detect()`, `.process()`, `.stream()`, `.reset_cooldown()`)
- `DetectorConfig` -- bundled config (`.build()`)
- `WakeDecisionPolicy` -- 4-gate decision pipeline
- `validate_audio_chunk` -- input validation
- `WakewordDetector` -- backward-compat alias for `WakeDetector`

**Confidence & Scoring:**
- `ConfidenceResult` -- `.raw_score`, `.confidence`, `.confirm_count`, `.score_history`
- `ConfidenceLevel` -- `LOW`, `MEDIUM`, `HIGH`, `CERTAIN`
- `FusionStrategy` -- `AVERAGE`, `MAX`, `VOTING`, `WEIGHTED_AVERAGE`

**Advanced:**
- `NoiseProfiler` -- `.update()`, `.get_profile()`, `.reset()`
- `PowerManager` -- `.should_process()`, `.report_score()`, `.get_state()`

**Pipeline:**
- `VoicePipeline` -- `.run()`, `.stop()`, `.speak()`, `@on_command`
- `VADEngine` -- `.process_frame()`, `.is_speech()`, `.backend_name`, `.reset()`
- `TTSEngine` -- `.synthesize()`, `.synthesize_chunked()`, `.play()`, `.play_async()`
- `STTEngine` -- `.transcribe()`, `.transcribe_full()`, `.prewarm()`

**Exceptions:**
- `ViolaWakeError` -- base exception
- `ModelNotFoundError`, `ModelLoadError`, `AudioCaptureError`, `PipelineError`, `VADBackendError`

**Discovery:**
- `list_models()`, `list_voices()`, `__version__`

### Submodule Exports

| Module | Key Exports |
|--------|-------------|
| `violawake_sdk.audio_source` | `AudioSource`, `MicrophoneSource`, `FileSource`, `NetworkSource`, `CallbackSource` |
| `violawake_sdk.noise_profiler` | `NoiseProfiler`, `NoiseProfile` |
| `violawake_sdk.power_manager` | `PowerManager`, `PowerState` |
| `violawake_sdk.speaker` | `SpeakerVerificationHook`, `SpeakerProfile`, `SpeakerVerifyResult`, `CosineScorer` |
| `violawake_sdk.ensemble` | `EnsembleScorer`, `FusionStrategy`, `fuse_scores()` |
| `violawake_sdk.confidence` | `ScoreTracker`, `ConfidenceResult`, `ConfidenceLevel` |
| `violawake_sdk.models` | `ModelSpec`, `MODEL_REGISTRY`, `download_model()`, `get_model_path()`, `list_cached_models()`, `check_registry_integrity()` |
| `violawake_sdk.backends` | `get_backend()`, `InferenceBackend`, `BackendSession` |
| `violawake_sdk.stt_engine` | `STTFileEngine`, `transcribe_wav_file()` |
| `violawake_sdk.stt` | `TranscriptResult`, `TranscriptSegment`, `MODEL_PROFILES` |
| `violawake_sdk.security` | `add_pins()`, `fetch_live_spki_pins()`, `verify_certificate_pin()`, `CertPinError`, `PinSet` |
| `violawake_sdk.audio` | `load_audio()`, `normalize_audio()`, `compute_rms()`, `is_silent()` |
| `violawake_sdk.training.augment` | `AugmentConfig`, `AugmentationPipeline`, `generate_synthetic_rir()` |
| `violawake_sdk.training.evaluate` | `evaluate_onnx_model()`, `compute_confusion_matrix()`, `find_optimal_threshold()` |
| `violawake_sdk.training.losses` | `FocalLoss` |
| `violawake_sdk.training.weight_averaging` | `EMATracker`, `SWACollector`, `auto_select_averaging()` |
| `violawake_sdk.training.temporal_model` | `TemporalCNN`, `TemporalConvGRU`, `TemporalGRU`, `export_temporal_onnx()` |
| `violawake_sdk.backends.tflite_backend` | `convert_onnx_to_tflite()` |
| `violawake_sdk.tools.confusables` | Phonetic substitution tables and confusable word generation |

---

## Roadmap

**v1.0 (Q2 2026) -- Phase 1 MVP:**
- [x] Python SDK (Wake + VAD + STT + TTS)
- [x] Training CLI (9 tools, 3 architectures)
- [x] PyPI release
- [x] Speaker verification, noise-adaptive, power management
- [x] Audio source abstraction, multi-model ensemble
- [x] Streaming TTS, structured STT, ONNX-to-TFLite converter
- [ ] Documentation site

**v1.1 (Q3 2026) -- Streaming + Web:**
- [ ] Streaming STT (faster-whisper generator mode)
- [ ] WASM build for ViolaWake
- [ ] JavaScript/Node SDK wrapper

**v2.0 (Q1 2027) -- Multi-platform:**
- [ ] Android SDK (ONNX Runtime Android)
- [ ] iOS SDK (ONNX Runtime iOS)
- [ ] DeepFilterNet noise suppression
- [ ] Speaker diarization (pyannote.audio)

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
