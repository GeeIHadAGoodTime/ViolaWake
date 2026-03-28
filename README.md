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
| **Evaluation tooling** | `violawake-eval` (EER, FAR, FRR, ROC) | None published | Basic |
| **On-device** | Yes (ONNX) | Yes (proprietary C lib) | Yes (ONNX) |
| **Streaming TTS bundle** | Yes (Kokoro-82M) | Orca (proprietary) | No |
| **Python SDK** | First-class | C wrapper | First-class |
| **Price at scale** | Free | $6K+/year | Free |

**Our moat:** Open training code, transparent evaluation with reproducible benchmarks, production-hardened data augmentation (SpecAugment, RIR convolution, noise mixing), and a 4-gate decision policy that eliminates false positives during music playback. On a fair head-to-head benchmark against openWakeWord (same corpus, same pipeline, adversarial negatives for both systems), ViolaWake achieves **EER 5.49%** vs OWW's 8.24% — each system tested on its own best wake word. Running in production, not a demo.

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

detector = WakeDetector(model="temporal_cnn", threshold=0.80)

for audio_chunk in detector.stream_mic():  # 20ms chunks at 16kHz
    if detector.detect(audio_chunk):
        print("Wake word detected!")
        break
```

### Threshold Tuning

The `threshold` parameter controls the trade-off between sensitivity and false positives:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.70 | Sensitive -- more detections, more false positives | Quiet rooms, close-mic setups |
| **0.80** | **Balanced (default)** -- recommended starting point | General-purpose, most environments |
| 0.85 | Conservative -- fewer false positives, may miss some wake words | Living rooms with TV/music |
| 0.90+ | Very conservative -- lowest false positive rate | Noisy environments, always-on kiosks |

Start at 0.80 and adjust based on your false accept rate. Use `violawake-eval` to measure FAPH (false accepts per hour) on representative audio from your deployment environment.

### Text-to-Speech (Kokoro-82M)

```python
from violawake_sdk import TTSEngine

tts = TTSEngine()  # Downloads kokoro-v1.0.onnx + voices-v1.0.bin on first run (~354MB total)
audio = tts.synthesize("Hello from ViolaWake!")
tts.play(audio)
```

### Voice Activity Detection

```python
from violawake_sdk import VADEngine

vad = VADEngine(backend="webrtc")  # or "silero", "rms"
prob = vad.process_frame(audio_bytes)  # returns 0.0–1.0 speech probability
```

### Full Pipeline (Wake → STT → TTS)

> Requires: `pip install "violawake[audio,stt,tts]"`

```python
from violawake_sdk import VoicePipeline

pipeline = VoicePipeline(
    wake_word="viola",
    stt_model="base",        # faster-whisper model size
    tts_voice="af_heart",    # Kokoro voice
)

@pipeline.on_command
def handle_command(text: str) -> None:
    print(f"Command: {text}")
    pipeline.speak(f"You said: {text}")  # Or return a string to auto-speak

pipeline.run()  # Blocks — Ctrl+C to stop
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VoicePipeline                            │
│                                                             │
│  Mic ──► [WakeDetector] ──► [VAD] ──► [STT] ──► callback  │
│                                                             │
│  text ──► [TTS] ──► Speaker                                │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

| Module | Engine | Size | Latency |
|--------|--------|------|---------|
| Wake word | Temporal CNN on OWW embeddings (ONNX) | ~100 KB | ~8ms/frame |
| VAD | WebRTC VAD / Silero / RMS heuristic | <1 MB | <1ms/frame |
| STT | faster-whisper `base` | 145 MB | 0.5–2s |
| TTS | Kokoro-82M (ONNX) | 330 MB | 0.3–0.8s/sentence |

---

## Training Your Own Wake Word

The training CLI lets you train a custom wake word model with ~200 positive samples:

```bash
# Collect positive samples (read prompts aloud)
python -m violawake_sdk.tools.collect_samples --word "jarvis" --output data/jarvis/positives/ --count 200

# Train (auto-generates TTS positives, confusable negatives, and speech negatives)
violawake-train \
  --word "jarvis" \
  --positives data/jarvis/positives/ \
  --output models/jarvis.onnx \
  --epochs 50

# To disable augmentation, add --no-augment
# To use legacy MLP architecture, add --architecture mlp

# Evaluate (EER, FAR/FRR, ROC)
violawake-eval \
  --model models/jarvis.onnx \
  --test-dir data/jarvis/test/ \
  --report
```

The `--test-dir` must contain `positives/` and `negatives/` subdirectories.

**Expected results:** EER < 10% (against the bundled synthetic negative corpus) with 200+ quality positive samples. Your real-world performance will depend on your deployment environment and negative speech corpus.

---

## Models

Models are versioned and published to GitHub Releases. Use registry names without file extensions when passing `--model` or `WakeDetector(model=...)`. Download separately (too large for PyPI):

```bash
python -m violawake_sdk.tools.download_model --model temporal_cnn   # default, ~100 KB
python -m violawake_sdk.tools.download_model --model kokoro_v1_0    # TTS, 330 MB
```

| Model | Type | Size | EER* | Notes |
|-------|------|------|------|-------|
| `temporal_cnn.onnx` | Temporal CNN on OWW embeddings | ~100 KB | 5.49% | Production default — best live recall + lowest FP |
| `temporal_convgru.onnx` | Temporal Conv-GRU on OWW embeddings | ~81 KB | -- | Reserve model |
| ~~`r3_10x_s42.onnx`~~ | MLP on OWW embeddings | ~34 KB | -- | **Deprecated** — fails live mic test. Do not use. |
| `kokoro-v1.0.onnx` | Kokoro-82M TTS | ~326 MB | -- | Apache 2.0 (hosted by [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)) |

*EER (Equal Error Rate) from benchmark v2: 700 shared negatives (incl. adversarial confusables), 180 TTS positives, streaming inference. Lower is better. See `benchmark_v2/` for full methodology and scripts.

---

## Platform Support

| Platform | Wake Word | TTS | STT | Status |
|----------|-----------|-----|-----|--------|
| Windows 10/11 (x64) | ✅ | ✅ | ✅ | **Fully tested** |
| Linux (x64) | ✅ | ✅ | ✅ | CI-tested |
| macOS (arm64/x64) | ✅ | ✅ | ✅ | Community tested |
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
- `openwakeword` backbone assets are installed automatically with `violawake`

---

## Performance Benchmarks

Measured on i7-12700H, Windows 11, RTX 3060 (CPU inference):

| Operation | Latency (p50) | Latency (p99) |
|-----------|--------------|--------------|
| Wake word inference (20ms frame) | 7.8 ms | 12.1 ms |
| VAD (WebRTC, 20ms frame) | 0.4 ms | 0.8 ms |
| STT (Whisper base, 3s audio) | 680 ms | 1.2s |
| TTS first audio (Kokoro, 1 sentence) | 310 ms | 580 ms |

**Wake word accuracy** (benchmark v2 — TTS corpus, 700 negatives incl. adversarial confusables):
- Temporal CNN model: **EER 5.49%**, ROC AUC 0.9877
- FAR @ FRR=5%: **5.43%** (vs OWW's 8.86% on its own best word)
- Live mic tested: 100% recall on direct speech, 0 false positives on podcast/music
- Real-world metrics depend on your deployment environment. Run `violawake-eval` on your own test data.

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

- **Open, reproducible evaluation:** `violawake-eval` produces EER, FAR, FRR, and ROC curves on any model + test set. Benchmark scripts in `benchmark_v2/` — run them yourself.
- **Production-hardened decision policy:** 4-gate pipeline (zero-input guard, score threshold, cooldown, listening gate) plus optional multi-window confirmation and speaker verification — eliminates false positives during music playback
- **Bundled pipeline:** ViolaWake ships integrated VAD + STT + TTS, not just the wake word component
- **Training infrastructure:** FocalLoss + EMA + SWA + SpecAugment augmentation pipeline vs basic training in openWakeWord

---

## Migrating from openWakeWord

ViolaWake uses openWakeWord's mel-spectrogram embedding model as a frozen feature extractor backbone. If you have existing OWW training data, you can use it directly with ViolaWake's training CLI.

**Key differences from OWW:**
- **Decision policy:** ViolaWake adds a multi-gate pipeline (RMS floor, cooldown, playback suppression) on top of raw scores. OWW exposes raw sigmoid scores only.
- **Temporal models:** ViolaWake supports Temporal CNN and Conv-GRU heads that score across a sliding window of embeddings, not just a single frame. This reduces false positives on speech that partially matches the wake word.
- **Augmentation pipeline:** ViolaWake's training CLI applies SpecAugment, RIR convolution, and noise mixing before embedding extraction. OWW's default training uses minimal augmentation.
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

## Roadmap

**v1.0 (Q2 2026) — Phase 1 MVP:**
- [x] Python SDK (Wake + VAD)
- [x] Kokoro TTS integration
- [x] faster-whisper STT integration
- [x] Full VoicePipeline class
- [x] Training CLI
- [ ] PyPI release
- [ ] Documentation site

**v1.1 (Q3 2026) — Streaming + Web:**
- [ ] Streaming STT (faster-whisper generator mode)
- [ ] WASM build for ViolaWake
- [ ] JavaScript/Node SDK wrapper
- [ ] Custom wake word web Console (alpha)

**v2.0 (Q1 2027) — Multi-platform:**
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

The ViolaWake MLP model uses OpenWakeWord as a fixed feature extractor backbone (also Apache 2.0). The MLP classification head and training pipeline are original ViolaWake work.
