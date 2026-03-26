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
| **Evaluation tooling** | `violawake-eval` (Cohen's d, FAR, FRR) | None published | Basic |
| **On-device** | Yes (ONNX) | Yes (proprietary C lib) | Yes (ONNX) |
| **Streaming TTS bundle** | Yes (Kokoro-82M) | Orca (proprietary) | No |
| **Python SDK** | First-class | C wrapper | First-class |
| **Price at scale** | Free | $6K+/year | Free |

**Our moat:** Open training code, transparent evaluation (Cohen's d > 15 on our internal test set with synthetic negatives), production-hardened data augmentation (SpecAugment, RIR convolution, noise mixing), and a decision policy that eliminates false positives during music playback. This has been running in production -- not a demo.

> **A note on accuracy claims:** Our Cohen's d of 15.10 was measured against a synthetic noise negative corpus, not adversarial speech. Real-world accuracy depends on your negative corpus and deployment environment. We recommend evaluating with your own test data using `violawake-eval`.

---

## Quick Start

```bash
pip install violawake
```

### Wake Word Detection (5 lines)

```python
from violawake_sdk import WakeDetector

detector = WakeDetector(model="viola_mlp_oww.onnx", threshold=0.80)

for audio_chunk in detector.stream_mic():  # 20ms chunks at 16kHz
    if detector.process(audio_chunk):
        print("Wake word detected!")
        break
```

### Text-to-Speech (Kokoro-82M)

```python
from violawake_sdk import TTSEngine

tts = TTSEngine()  # Downloads kokoro-v1.0.onnx on first run (~330MB)
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

```python
from violawake_sdk import VoicePipeline

pipeline = VoicePipeline(
    wake_word="viola",
    stt_model="base",        # faster-whisper model size
    tts_voice="af_heart",    # Kokoro voice
)

@pipeline.on_command
def handle_command(text: str):
    print(f"Command: {text}")
    pipeline.speak(f"You said: {text}")

pipeline.run()  # Blocks — Ctrl+C to stop
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VoicePipeline                            │
│                                                             │
│  Mic ──► [VAD] ──► [WakeDetector] ──► [STT] ──► callback  │
│                                                             │
│  text ──► [TTS] ──► Speaker                                │
└─────────────────────────────────────────────────────────────┘
```

**Components:**

| Module | Engine | Size | Latency |
|--------|--------|------|---------|
| Wake word | MLP on OWW embeddings (ONNX) | 2.1 MB | ~8ms/frame |
| VAD | WebRTC VAD / Silero / RMS heuristic | <1 MB | <1ms/frame |
| STT | faster-whisper `base` | 145 MB | 0.5–2s |
| TTS | Kokoro-82M (ONNX) | 330 MB | 0.3–0.8s/sentence |

---

## Training Your Own Wake Word

The training CLI lets you train a custom wake word model with ~200 positive samples:

```bash
# Collect positive samples (read prompts aloud)
python -m violawake_sdk.tools.collect_samples --word "jarvis" --count 200

# Train (downloads negative sample corpus automatically)
python -m violawake_sdk.tools.train \
  --word "jarvis" \
  --positives data/jarvis/positives/ \
  --output models/jarvis_mlp.onnx \
  --epochs 50

# Evaluate (Cohen's d / separability metric)
python -m violawake_sdk.tools.evaluate \
  --model models/jarvis_mlp.onnx \
  --test-positives data/jarvis/test/ \
  --report
```

**Expected results:** Cohen's d > 10 (against the bundled synthetic negative corpus) with 200+ quality positive samples. Your real-world performance will depend on your deployment environment and negative speech corpus.

---

## Models

Models are versioned and published to GitHub Releases. Download separately (too large for PyPI):

```bash
python -m violawake_sdk.tools.download_model --model viola_mlp_oww  # default, 2.1 MB
python -m violawake_sdk.tools.download_model --model kokoro-v1.0    # TTS, 330 MB
```

| Model | Type | Size | Cohen's d* | Notes |
|-------|------|------|------------|-------|
| `viola_mlp_oww.onnx` | MLP on OWW embeddings | 2.1 MB | >15 | Production default |
| `viola_v4.onnx` | CNN (3-layer) | 1.8 MB | ~8 | Legacy -- kept for comparison |
| `kokoro-v1.0.onnx` | Kokoro-82M TTS | 330 MB | -- | Apache 2.0 |

*Cohen's d measured against synthetic noise negatives. Evaluate on your own data with `violawake-eval`.

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

**With TTS:**
```bash
pip install "violawake[tts]"
```

**With STT:**
```bash
pip install "violawake[stt]"
```

**Full pipeline:**
```bash
pip install "violawake[all]"
```

**Requirements:**
- Python 3.10+
- `onnxruntime >= 1.17` (CPU) or `onnxruntime-gpu` for GPU acceleration
- `pyaudio` for microphone input
- `numpy`, `scipy`

---

## Performance Benchmarks

Measured on i7-12700H, Windows 11, RTX 3060 (CPU inference):

| Operation | Latency (p50) | Latency (p99) |
|-----------|--------------|--------------|
| Wake word inference (20ms frame) | 7.8 ms | 12.1 ms |
| VAD (WebRTC, 20ms frame) | 0.4 ms | 0.8 ms |
| STT (Whisper base, 3s audio) | 680 ms | 1.2s |
| TTS first audio (Kokoro, 1 sentence) | 310 ms | 580 ms |

**Wake word accuracy** (internal test set, synthetic noise negatives):
- MLP OWW model: Cohen's d > 15 on synthetic noise negatives (FAR: 0.3/hr, FRR: <2% at default threshold)
- Threshold range: 0.70 (high sensitivity) to 0.85 (low false positives)
- Real-world metrics depend on your negative corpus. Run `violawake-eval` on your own test data.

---

## Comparison to openWakeWord

openWakeWord is the closest open-source alternative. ViolaWake differences:

- **Open, reproducible evaluation:** `violawake-eval` produces Cohen's d, FAR, FRR, and ROC curves on any model + test set. We publish our methodology and clearly note when a benchmark uses synthetic-only negatives.
- **Production-hardened decision policy:** 4-gate pipeline (zero-input guard, score threshold, cooldown, listening gate) eliminates false positives during music playback
- **Bundled pipeline:** ViolaWake ships integrated VAD + STT + TTS, not just the wake word component
- **Training infrastructure:** FocalLoss + EMA + SWA + SpecAugment augmentation pipeline vs basic training in openWakeWord

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
