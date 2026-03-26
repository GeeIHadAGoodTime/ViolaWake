<!-- doc-meta
scope: Product requirements — what we build, why, and how we measure success
authority: LIVING — primary product spec, maintained by PM/tech lead
code-paths: src/violawake_sdk/, docs/adr/, README.md
staleness-signals: Pivot in target market, major competitive shift, new funding/priorities
last-updated: 2026-03-17
-->

# ViolaWake SDK — Product Requirements Document

**Version:** 0.1
**Status:** Draft → Active
**Authors:** ViolaWake team
**Competitive context:** Based on `COMPETITIVE_AUDIT_REPORT.md`, `audit_viola_inventory.md`, `audit_gap_analysis.md` (Viola repo `.viola/agents/`)

---

## 1. Problem Statement

Voice AI tooling for developers is dominated by two unsatisfying options:

**Option A: Picovoice** — High accuracy, production-grade, but proprietary black-box models, expensive at scale ($6K+/year commercial), C-binary-wrapped Python, and no fine-tuning capability. Developers are locked in and can't adapt models to their domain (different accent, noisy environment, custom wake word variant).

**Option B: openWakeWord / open-source alternatives** — Free and inspectable, but with lower published/community benchmark numbers, no integrated pipeline (just wake detection, no STT or TTS bundle), and no production-hardened decision policy. Our current internal reference score is Cohen's d 15.10 on synthetic negatives; direct speech-negative comparison is still TBD.

**The gap:** No open-source voice SDK combines (1) genuinely competitive accuracy, (2) an accessible training pipeline that developers can use to fine-tune, and (3) a complete bundled pipeline (Wake→VAD→STT→TTS) that ships with a single `pip install`.

**Our position:** We have been running a production wake word + TTS pipeline inside the Viola assistant application. ViolaWake SDK extracts and packages this production-tested code into a standalone SDK that any Python developer can use.

---

## 2. Scope

### In scope — Phase 1 MVP (Q2 2026)

| Component | Description | Source |
|-----------|-------------|--------|
| **WakeDetector** | Wake word detection using MLP on OWW embeddings | Extracted from `violawake/engine.py` |
| **VADEngine** | Voice activity detection (WebRTC / Silero / RMS) | Extracted from `voice/wake_detector/` |
| **TTSEngine** | On-device TTS via Kokoro-82M ONNX | Extracted from `voice/synthesis/kokoro_engine.py` |
| **STTEngine** | Batch STT via faster-whisper | Extracted from `voice/transcription/whisper.py` |
| **VoicePipeline** | Bundled pipeline orchestrating all four | New |
| **Training CLI** | `violawake-train` — train a custom wake word model | Extracted from `violawake/training/trainer.py` |
| **Evaluation CLI** | `violawake-eval` — Cohen's d / FAR / FRR evaluation | Extracted from `violawake/training/evaluate_real.py` |
| **Model download** | `violawake-download` — fetch and cache models | New |
| **PyPI package** | `pip install violawake` | New |

### Out of scope — Phase 1

- Streaming STT (token-by-token ASR) — Phase 2
- WASM/browser build — Phase 2
- JavaScript/Node SDK — Phase 2
- Custom wake word web Console — Phase 2
- Speaker recognition / diarization — Phase 3
- DeepFilterNet noise suppression — Phase 2
- Mobile SDKs (Android/iOS) — Phase 3
- On-device LLM inference — Never (use Ollama/llama.cpp instead)
- Embedded/MCU targets (Cortex-M) — Never (Pi 4 is our minimum)

---

## 3. Target Users

**Primary: Python developer building a voice-enabled desktop or server application**
- Needs wake word detection that works out-of-the-box
- May want to customize the wake word for their product
- Does NOT want to pay Picovoice licensing fees
- Comfortable with Python but not necessarily ML expertise

**Secondary: ML researcher/engineer fine-tuning a wake word model**
- Has domain-specific positive samples (different accent, noisy environment)
- Wants to understand and modify the model architecture
- Needs the training pipeline to be transparent and reproducible

**Tertiary: Home automation / Raspberry Pi hobbyist**
- Building a smart home device on Pi 4+
- Wants offline-first, no cloud dependencies
- Budget-sensitive — free is a hard requirement

**Not targeting:** Mobile app developers (Phase 3), embedded engineers (never), enterprise customers without self-serve path (Phase 3).

---

## 4. Feature Catalog

### F1: Wake Word Detection

**Description:** Detect a specific wake word in a continuous audio stream from a microphone.

**API:**
```python
detector = WakeDetector(model="viola_mlp_oww.onnx", threshold=0.80)
score = detector.process(audio_chunk_16khz_mono)  # returns float 0.0–1.0
detected = score >= detector.threshold
```

**Acceptance criteria:**
- Cohen's d ≥ 15.0 on the internal synthetic-negative test set (MLP OWW model); add speech-negative benchmarking before making external accuracy claims
- Inference latency ≤ 15ms per 20ms frame on i7-12700H (CPU)
- False accept rate ≤ 0.5 events/hour at threshold=0.80
- False reject rate ≤ 3% at threshold=0.80
- Works on: Windows 10/11, Ubuntu 22.04, macOS 13+ (x64 and arm64)

**Decision policy (4-gate, from production Viola):**
1. Zero-input guard: skip if RMS < 1.0 (silence/dc-offset artifact)
2. Score threshold: skip if model score < threshold
3. Cooldown: ignore events within 2.0s of last detection
4. Listening gate: suppress during active playback (configurable)

**Supported models:**
- `viola_mlp_oww.onnx` — MLP on OWW 96-dim embeddings (default, Cohen's d 15.10 on synthetic negatives; speech-negative d-prime TBD)
- Custom models trained via `violawake-train` CLI

### F2: Voice Activity Detection (VAD)

**Description:** Detect speech presence in audio frames. Used to gate wake word scoring and determine command endpoint.

**API:**
```python
vad = VADEngine(backend="webrtc")  # or "silero", "rms"
prob = vad.process_frame(audio_20ms_bytes)  # float 0.0–1.0
is_speech = prob > 0.5
```

**Acceptance criteria:**
- WebRTC backend: ≤ 1ms latency per 20ms frame
- Silero backend: ≤ 5ms latency per 20ms frame
- RMS backend: ≤ 0.1ms latency (fallback when neither library available)
- Graceful degradation: falls back to RMS if webrtcvad/torch not installed

### F3: Text-to-Speech (Kokoro-82M)

**Description:** On-device neural TTS using the Kokoro-82M ONNX model (Apache 2.0 licensed).

**API:**
```python
tts = TTSEngine(voice="af_heart")
audio_pcm = tts.synthesize("Hello from ViolaWake!")  # returns np.ndarray
```

**Acceptance criteria:**
- Time-to-first-audio ≤ 400ms for single sentence (p50), ≤ 800ms (p99) on i7-12700H CPU
- Audio output: 24 kHz, mono, float32 (resampled to 16 kHz if requested)
- Thread-safe (multiple threads can synthesize concurrently)
- Sentence-chunked streaming: yields audio per sentence, not whole paragraph
- License: Kokoro model is Apache 2.0 — no royalties, no usage tracking

**Supported voices:** `af_heart` (default), `af_bella`, `af_sarah`, `am_adam`, `am_michael`, `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis`

### F4: Speech-to-Text (Whisper via faster-whisper)

**Description:** Batch speech-to-text transcription using Whisper models via faster-whisper (CTranslate2).

**API:**
```python
stt = STTEngine(model="base")  # base, small, medium, large-v3
text = stt.transcribe(audio_pcm)  # np.ndarray → str
segments = stt.transcribe_with_segments(audio_pcm)  # → list[Segment]
```

**Acceptance criteria:**
- WER ≤ 5% on LibriSpeech test-clean (base model)
- Prewarm on first call (model load), ≤ 200ms subsequent calls for 3s audio (base, CPU i7)
- Language detection with TTL caching (60s)
- No-speech detection: returns empty string + `no_speech_prob` > 0.6 for silence

**Model sizes and trade-offs:**

| Model | WER (test-clean) | VRAM | Latency (3s audio, CPU) |
|-------|-----------------|------|------------------------|
| `tiny` | ~14% | 75MB | ~120ms |
| `base` | ~9% | 145MB | ~380ms |
| `small` | ~7% | 465MB | ~850ms |
| `medium` | ~5% | 1.5GB | ~2.1s |
| `large-v3` | ~3% | 3.0GB | ~5s |

### F5: VoicePipeline

**Description:** Orchestrated pipeline combining Wake→VAD→STT→TTS with callback registration.

**API:**
```python
pipeline = VoicePipeline(
    wake_word="viola",
    stt_model="base",
    tts_voice="af_heart",
    threshold=0.80,
)

@pipeline.on_command
def handle(text: str) -> str | None:
    return f"You said: {text}"  # Return value spoken by TTS

pipeline.run()  # blocks
```

**Acceptance criteria:**
- Wake-to-STT latency ≤ 3s (end of speech to transcription, base model)
- TTS playback begins ≤ 500ms after callback returns (if response < 50 words)
- Clean shutdown on Ctrl+C (no zombie threads)
- Works headless (no display required)

### F6: Training CLI

**Description:** Train a custom wake word model from positive samples.

**Command:**
```bash
violawake-train \
  --word "jarvis" \
  --positives data/jarvis/positives/ \
  --output models/jarvis.onnx \
  --epochs 50 \
  --augment  # enable data augmentation
```

**Architecture:** MLP classifier on top of frozen OWW audio embeddings (same as production ViolaWake).

**Training infrastructure (from production Viola):**
- FocalLoss (handles class imbalance between positives and negatives)
- AdamW optimizer with cosine annealing LR schedule
- EMA (Exponential Moving Average) of weights
- Optional SWA (Stochastic Weight Averaging) for final model
- Data augmentation: time stretch, pitch shift, volume perturbation, SpecAugment, RIR convolution, noise mixing
- Checkpoint config embedding (prevents config-drift bug documented 2026-03-02)

**Acceptance criteria:**
- Cohen's d ≥ 10.0 on the synthetic-negative benchmark with 200+ quality positive samples
- Training completion in ≤ 30 min on i7-12700H CPU (50 epochs, 200 positives)
- Produces ONNX model loadable by `WakeDetector`
- Saves checkpoint config alongside model (prevents config drift)
- Progress reported to stdout with ETA

### F7: Evaluation CLI

**Description:** Evaluate wake word model accuracy using Cohen's d plus FAR/FRR.

**Command:**
```bash
violawake-eval \
  --model models/jarvis.onnx \
  --test-positives data/jarvis/test_positives/ \
  --report
```

**Output:**
```
Cohen's d: 15.10 (synthetic negatives)
False Accept Rate: 0.28/hr (at threshold=0.80)
False Reject Rate: 1.8% (at threshold=0.80)
ROC AUC: 0.998
```

---

## 5. Success Metrics

### Technical Metrics (MVP gate criteria — must pass before PyPI release)

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Wake word Cohen's d (synthetic benchmark) | ≥ 15.0 | `violawake-eval` on internal synthetic-negative test set |
| Wake inference latency | ≤ 15ms/frame (p99) | `pytest tests/benchmarks/bench_wake.py` |
| TTS first-audio latency | ≤ 400ms/sentence (p50) | `pytest tests/benchmarks/bench_tts.py` |
| STT WER (base model) | ≤ 9% | LibriSpeech test-clean |
| CI pass rate | 100% | GitHub Actions |
| Test coverage | ≥ 85% (unit), ≥ 60% (integration) | pytest-cov |
| Ruff lint violations | 0 | ruff check . |

### Product Metrics (post-launch, 90-day targets)

| Metric | Target | Source |
|--------|--------|--------|
| PyPI downloads | ≥ 500/week | PyPI stats |
| GitHub stars | ≥ 200 | GitHub |
| GitHub issues opened | ≥ 20 (signal of adoption) | GitHub |
| Issues closed within 7 days | ≥ 80% | GitHub |

### Competitive Position

| vs Porcupine | Target |
|---|---|
| Accuracy transparency | Publish Cohen's d on the synthetic benchmark and add speech-negative benchmarking before cross-vendor claims |
| Python SDK ergonomics | 5 lines to first detection (vs Porcupine's ~15) |
| Price at scale | $0 (vs $6K+/yr commercial Picovoice) |
| Training accessibility | Open CLI (vs Picovoice Console, invitation-only initially) |

---

## 6. Non-Requirements (Intentional Gaps)

These are explicitly out of scope and should not be implemented without a new ADR:

- **Streaming/real-time STT:** Adds significant complexity for marginal UX benefit in the voice-command use case. Batch Whisper is adequate. (See ADR-003 rationale.)
- **On-device LLM inference:** Ollama/llama.cpp does this better and for free. We integrate with it, not compete.
- **Speaker recognition:** Different product category. Phase 3 at earliest.
- **MCU/embedded targets:** Requires C library (contradicts ADR-003). Out of scope indefinitely.
- **Cloud APIs:** We are privacy-first, fully on-device. No cloud dependency in core SDK.
- **C library:** Python-first decision in ADR-003. Revisit at Phase 2+ only if Python adoption proves insufficient for embedded use cases.

---

## 7. Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| OWW embedding dependency limits "original model" claim | Medium | Medium | Disclose clearly in docs; MLP head + training pipeline is our original work |
| PyAudio installation friction on some platforms | High | Medium | Document fallback (sounddevice), add CI matrix |
| Kokoro model license change | Low | High | Track kokoro-onnx license, maintain Apache 2.0 attestation |
| Picovoice open-sources Porcupine training | Low | High | Our advantage shifts to production-hardening, pipeline bundle, and community |
| GPU ONNX inference complexity (CUDA setup) | Medium | Low | CPU-first; GPU is opt-in with clear docs |
| Windows microphone permission issues | High | Medium | Document Windows mic privacy settings in README |

---

## 8. Dependencies

| Dependency | Version | License | Risk |
|-----------|---------|---------|------|
| `onnxruntime` | ≥1.17 | MIT | Low |
| `numpy` | ≥1.24 | BSD-3 | Low |
| `pyaudio` | ≥0.2.14 | MIT | Medium (PortAudio native dep) |
| `scipy` | ≥1.11 | BSD-3 | Low |
| `kokoro-onnx` | ≥0.4 | Apache 2.0 | Medium (external package, track changes) |
| `faster-whisper` | ≥1.0 | MIT | Low |
| `webrtcvad` | ≥2.0.10 | BSD | Medium (C extension, platform builds) |
| OpenWakeWord (embedded backbone) | via ONNX | Apache 2.0 | Low (we use pre-computed embeddings) |

---

*Next review: After Phase 1 PyPI release. Update metrics targets based on initial adoption data.*
