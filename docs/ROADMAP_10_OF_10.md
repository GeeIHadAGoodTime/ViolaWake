# ViolaWake SDK — 10/10 Roadmap

**Goal:** Ship a Picovoice competitor (Option C: hybrid open-core + paid Console)
**Date:** 2026-03-25
**Based on:** 5 parallel research agents covering augmentation/corpus, browser recording, cloud infra, market analysis, and E2E testing

---

## Executive Summary

The market has a $0-to-$6,000 pricing gap. Picovoice charges $6K/yr for commercial use. openWakeWord is free but training breaks constantly. **We fill the gap**: free open-source SDK + paid web Console that trains custom wake words from 10 voice samples in under 5 minutes.

**Key competitive advantages we're building:**
1. Real voice samples (speaker-specific) vs Picovoice's text-only synthetic training
2. $0 SDK + affordable Console vs $6K/yr Picovoice
3. Open training pipeline vs Picovoice black box
4. Bundled Wake+VAD+STT+TTS pipeline vs piecemeal assembly
5. Cohen's d 15.10 on the current synthetic-negative benchmark, with real-world speech-negative d-prime still TBD, plus a production-hardened 4-gate decision policy

---

## Architecture Overview

```
                 ┌──────────────────────────────────┐
                 │     Console Website (React)       │
                 │  Record 10 samples → Train → Get  │
                 │  model + API key + SDK quickstart  │
                 └──────────┬───────────────────────┘
                            │
                 ┌──────────▼───────────────────────┐
                 │     FastAPI Backend               │
                 │  Auth │ Upload │ Jobs │ Delivery   │
                 └──┬────┬────┬────┬────────────────┘
                    │    │    │    │
        ┌───────────┘    │    │    └──────────┐
        ▼                ▼    ▼               ▼
   Supabase Auth    S3/R2   Modal.com GPU   Stripe
   (users, keys)   (models) (training)      (billing)
```

**Per-job economics:** ~$0.06 cost → charge via $29/mo plan for 20 models = healthy margins.

---

## Subsystem Roadmaps (Current → 10/10)

---

### 1. Training Pipeline (6/10 → 10/10) — THE CRITICAL PATH

**Current state:** Negatives are random noise. Augmentation flag is a no-op. No validation split. No early stopping.

**What 10/10 looks like:** Production training pipeline that produces strong separability scores on the synthetic benchmark and then validates against speech/background negatives from 10 user samples in <5 minutes on GPU.

#### 1a. Negative Corpus (Priority: P0)

| Corpus | Size | License | Purpose |
|--------|------|---------|---------|
| **MUSAN** | 109 hrs | Public domain | Music + speech + noise (start here) |
| **Common Voice EN** | ~2,000 hrs subset | CC0 | Speech negatives |
| **ACAV100M** (via OWW HuggingFace features) | 2,000 hrs pre-extracted | MIT | Noise/music/speech in the wild |
| **FSD50K** | 108 hrs | CC | Environmental sounds |
| **OpenSLR RIR** | Real room impulses | Free | Reverb augmentation |
| **BIRD** | 1M simulated RIRs | Free | Reverb augmentation at scale |

**Implementation:** Download MUSAN + Common Voice subset + OWW pre-computed features from HuggingFace. Store as pre-extracted OWW embeddings (not raw audio) to save training time. Total: ~4,000 hrs of diverse negatives.

**openWakeWord reference:** They use 31,000 hrs for production models. We start with 4,000 hrs (enough to move beyond synthetic-only evaluation) and scale up.

#### 1b. Data Augmentation (Priority: P0)

**Library:** `audiomentations` (MIT, 43 transforms, 2.2k stars, actively maintained)

**Augmentation chain for 10 user samples → 200+ augmented samples:**

```python
from audiomentations import Compose, OneOf, SomeOf

augment = Compose([
    # Volume variation (simulates distance/mic differences)
    Gain(min_gain_db=-6, max_gain_db=6, p=0.8),
    # Background noise (from MUSAN corpus)
    AddBackgroundNoise(sounds_path="corpus/musan/noise", min_snr_db=5, max_snr_db=20, p=0.7),
    # Room reverb (from RIR dataset)
    ApplyImpulseResponse(ir_path="corpus/rir/", p=0.5),
    # Speed variation (speaking rate differences)
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    # Pitch variation (vocal range differences)
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    # Codec degradation (phone/bluetooth quality)
    Mp3Compression(min_bitrate=32, max_bitrate=128, p=0.3),
    # SpecAugment equivalent (time masking in waveform domain)
    TimeMask(min_band_part=0.0, max_band_part=0.1, p=0.3),
])
```

Plus `torchaudio.transforms.FrequencyMasking` + `TimeMasking` on spectrograms.

#### 1c. Hard Negative Mining (Priority: P1)

**For any wake word, generate phonetically similar confusables:**

1. Convert wake word to phonemes via `g2p-en` (neural G2P)
2. Search CMU dictionary (134k words) for words within phoneme edit distance 2-3 via `pronouncing` library
3. Generate confusable phrases using LLM (e.g., for "viola" → "violent", "violet", "voila", "buy a lot", "via la")
4. Synthesize 50+ clips per confusable using Piper TTS (open source, multiple voices)
5. Add as hard negatives in training

**Libraries:** `pronouncing` (CMU dict wrapper), `g2p-en` (neural G2P), `piper-tts` (multi-voice synthesis)

#### 1d. Training Infrastructure Fixes (Priority: P0)

- [ ] Validation split (80/20) with early stopping (patience=10)
- [ ] Hyperparameter exposure (batch size, learning rate, hidden dims)
- [ ] `openwakeword` added to `[training]` deps in pyproject.toml
- [ ] Training progress callback (for Console SSE updates)
- [ ] Model metadata saved alongside ONNX (Cohen's d benchmark details, FAR, FRR, training config)
- [ ] Two-stage architecture option (primary detector + hard-negative verifier) per openWakeWord pattern

---

### 2. Core Wake Word Engine (8/10 → 10/10)

**Fixes needed:**

- [ ] Fix dead branch in `process()` line 196-198 (dtype check after forced conversion)
- [ ] Add multi-frame temporal pooling option (accumulate N frames of embeddings before MLP)
- [ ] Validate OWW backbone produces correct embeddings from single 20ms frames
- [ ] INT8 quantized model support (for edge deployment)
- [ ] GPU provider auto-detection (CUDA → DirectML → CPU fallback)

**No architectural changes needed** — the 2-stage OWW backbone → MLP head is proven.

---

### 3. Evaluation (8/10 → 10/10)

**Critical fix:** `evaluate_onnx_model()` uses mel features but MLP-on-OWW models need OWW embeddings.

- [ ] Fix eval to extract OWW embeddings (same path as train.py) for MLP-on-OWW models
- [ ] Auto-detect model architecture from config.json (CNN vs MLP-on-OWW) and use correct feature path
- [ ] Add threshold sweep (find optimal threshold automatically)
- [ ] Add confusion matrix output
- [ ] Add per-file score dump (CSV) for debugging false rejects
- [ ] Add ROC curve visualization (matplotlib, saved as PNG)
- [ ] Add comparison mode: evaluate two models side-by-side

---

### 4. VAD Engine (7/10 → 10/10)

- [ ] **Implement Silero VAD backend** — silero-vad (MIT) is a 2MB ONNX model, best open-source VAD. Wrap `silero_vad` package with ONNX Runtime inference
- [ ] Add `silero-vad` to `[vad]` optional deps
- [ ] Update auto-selection: WebRTC → Silero → RMS
- [ ] Add adaptive noise floor estimation (running average of silence RMS)
- [ ] Energy-based endpoint detection (for command boundary in pipeline)

---

### 5. TTS Engine (7/10 → 10/10)

- [ ] Replace naive sentence splitter with `pysbd` (Python Sentence Boundary Disambiguation) — handles "Dr. Smith", "3.14", URLs
- [ ] Add async `synthesize_async()` method for pipeline integration
- [ ] Add volume normalization on output (loudness normalization to -16 LUFS)
- [ ] Streaming playback: play chunk N while synthesizing chunk N+1
- [ ] Consider adding Piper TTS as alternative backend (MIT, more voices, lighter)

---

### 6. STT Engine (8/10 → 10/10)

- [ ] Fix `temperature=0.0` + `beam_size=5` conflict — use `temperature=(0.0, 0.2, 0.4)` for fallback decoding or remove beam settings with greedy
- [ ] Document thread-safety limitations (faster-whisper is NOT guaranteed thread-safe)
- [ ] Add mutex around `model.transcribe()` for safety
- [ ] Add hotword boosting option (bias toward wake-word-adjacent commands)
- [ ] Add basic streaming mode (chunked transcription with partial results)

---

### 7. Voice Pipeline (6/10 → 10/10)

- [ ] **Fix race condition:** line 184 `is_playing = state == _STATE_RESPONDING` when state is already confirmed `_STATE_IDLE` — always False
- [ ] Add pipeline event callbacks: `on_wake`, `on_listen_start`, `on_listen_end`, `on_transcribe_start`, `on_transcribe_end`, `on_response`
- [ ] Add state query API: `pipeline.state`, `pipeline.last_command`, `pipeline.last_score`
- [ ] Prevent overlapping STT calls (lock or queue)
- [ ] Add async pipeline variant (`AsyncVoicePipeline`)
- [ ] Use proper Protocol types for lazy STT/TTS instead of `object | None`
- [ ] Add configurable behavior for wake-during-transcription

---

### 8. Console Website (0/10 → 10/10) — NEW BUILD

#### 8a. Tech Stack

| Layer | Technology | License | Why |
|-------|-----------|---------|-----|
| **Frontend** | React + Vite + TypeScript | MIT | Fast, modern, component library ecosystem |
| **Recording** | RecordRTC (StereoAudioRecorder) | MIT | Direct 16kHz WAV output, cross-browser |
| **Waveform** | wavesurfer.js + Record plugin | BSD-3 | Live waveform during recording, post-recording visualization |
| **Quality** | Meyda.js | MIT | Real-time RMS meter, spectral flatness noise detection |
| **Client VAD** | @ricky0123/vad-web | MIT | Verify speech presence before upload |
| **Backend** | FastAPI | MIT | Python-native, async, SSE support |
| **Auth** | Supabase Auth | Apache 2.0 | $0 to 50K MAU, self-hostable, JWT + API keys |
| **Storage** | S3/R2 (Cloudflare R2) | N/A | $0 egress (R2), signed URLs for model delivery |
| **Training GPU** | Modal.com | N/A | ~$0.05/job, serverless, scale to zero |
| **Billing** | Stripe Billing | N/A | Usage-based metering, freemium support |
| **Progress** | Server-Sent Events (SSE) | N/A | Real-time training progress without WebSocket complexity |

#### 8b. Recording Flow (User Experience)

```
1. Sign up / Log in
2. Click "New Wake Word"
3. Enter wake word name (e.g., "Hey Jarvis")
4. System validates pronunciation feasibility
5. Recording UI appears:
   - Countdown timer (3... 2... 1...)
   - "Say 'Hey Jarvis' now!" with live waveform
   - Record 1.5s clip
   - Client-side quality gates:
     ├── Duration: 1.0-3.0s
     ├── Silence: RMS > 0.01 for >50% of clip
     ├── Clipping: <0.5% samples at +/-1.0
     ├── Noise: spectral flatness < 0.85
     └── Speech: VAD confirms speech detected
   - If quality fails: "Too quiet / Too noisy / No speech detected — try again"
   - If quality passes: show waveform, "Keep" or "Re-record"
   - Repeat 10x (varied prompts: "normal voice", "whisper", "from across the room")
6. Upload all 10 clips (REST multipart POST, ~48KB each = ~480KB total)
7. Training starts:
   - SSE stream shows real-time progress bar
   - "Augmenting samples... (10 → 200+)"
   - "Extracting embeddings..."
   - "Training model... epoch 12/50"
   - "Evaluating accuracy..."
   - ~2-5 minutes on Modal T4 GPU
8. Results page:
   - Cohen's d score with grade (Excellent/Good/Fair/Poor) plus a note about the benchmark corpus
   - False accept rate, false reject rate
   - "Download .onnx model" button
   - SDK quickstart code snippet (copy-paste ready)
   - API key for model management
9. If Cohen's d < 10 on the synthetic benchmark: "Your model could be better. Record 10 more samples and validate against harder negatives."
```

#### 8c. Client-Side Quality Gates (before upload)

```javascript
// RecordRTC setup
const recorder = new RecordRTC(stream, {
    type: 'audio',
    recorderType: StereoAudioRecorder,
    desiredSampRate: 16000,
    numberOfAudioChannels: 1,
});

// Quality checks using Meyda.js
const analyzer = Meyda.createMeydaAnalyzer({
    audioContext,
    source: micSource,
    featureExtractors: ['rms', 'spectralFlatness', 'zcr'],
    callback: features => {
        updateVolumeLevel(features.rms);
        if (features.spectralFlatness > 0.85) showNoiseWarning();
    },
});
```

#### 8d. Pages

| Page | Route | Purpose |
|------|-------|---------|
| Landing | `/` | Hero, competitive comparison, pricing, CTA |
| Sign Up / Login | `/auth` | Supabase Auth UI |
| Dashboard | `/dashboard` | List of user's wake word models, usage stats |
| New Wake Word | `/train/new` | Recording interface + training flow |
| Model Detail | `/models/:id` | Metrics, download, API key, SDK code snippet |
| Pricing | `/pricing` | Free / Developer / Business / Enterprise |
| Docs | `/docs` | API reference, SDK quickstart, migration guides |

---

### 9. Cloud Training Infrastructure (0/10 → 10/10) — NEW BUILD

#### 9a. Backend API (FastAPI)

```
POST /api/auth/register          → Supabase Auth
POST /api/auth/login             → Supabase Auth
POST /api/wake-words             → Create new wake word project
POST /api/wake-words/:id/samples → Upload audio samples (multipart)
POST /api/wake-words/:id/train   → Trigger training job
GET  /api/wake-words/:id/status  → SSE stream of training progress
GET  /api/wake-words/:id/model   → Signed download URL for .onnx
GET  /api/wake-words/:id/metrics → cohen_d_synthetic, FAR, FRR, ROC
GET  /api/keys                   → List API keys
POST /api/keys                   → Generate new API key
```

#### 9b. Training Job (Modal.com)

```python
import modal

app = modal.App("violawake-trainer")

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "torchaudio", "onnx", "onnxruntime",
        "numpy", "scipy", "audiomentations", "openwakeword",
        "scikit-learn", "piper-tts",
    )
)

# Negative corpus cached as Modal Volume (persistent, shared across jobs)
corpus_volume = modal.Volume.from_name("violawake-corpus", create_if_missing=True)

@app.function(
    gpu="T4",
    image=training_image,
    volumes={"/corpus": corpus_volume},
    timeout=600,
)
def train_wake_word(
    audio_samples: list[bytes],
    wake_phrase: str,
    callback_url: str,
) -> dict:
    """Train a wake word model from user audio samples.

    Returns: {"model_bytes": bytes, "metrics": dict}
    """
    # 1. Augment 10 samples → 200+ via audiomentations
    # 2. Extract OWW embeddings from augmented positives
    # 3. Load pre-computed negative embeddings from /corpus
    # 4. Generate hard negatives (phonetically similar) via Piper TTS
    # 5. Train MLP with FocalLoss + AdamW + cosine LR + EMA
    # 6. Evaluate Cohen's d on the held-out validation set and keep track of the negative corpus used
    # 7. Export to ONNX
    # 8. POST progress updates to callback_url
    ...
```

#### 9c. Model Delivery

1. Trained `.onnx` model uploaded to S3/R2
2. SHA-256 hash computed and stored in database
3. User gets signed download URL (expires in 1 hour)
4. SDK verifies SHA-256 after download
5. Model cached locally on user's device at `~/.violawake/models/`

---

### 10. Pricing (NEW)

| Tier | Price | Includes |
|------|-------|---------|
| **Free** | $0 | 3 wake word models, personal/non-commercial, community support |
| **Developer** | $29/month | 20 models/month, commercial use, email support |
| **Business** | $99/month | Unlimited models, priority training, custom tuning, Slack support |
| **Enterprise** | Custom | SLA, on-prem training, dedicated support, custom integrations |

**The gap we fill:** Picovoice Free = 3 users/month. Picovoice paid = $6K/yr minimum. Our $29/mo Developer tier serves the entire middle market.

---

### 11. Testing & CI/CD (6/10 → 10/10)

#### 11a. Missing Unit Tests

- [ ] `audio.py` — mel spectrogram, PCEN, normalization, pad/trim
- [ ] `pipeline.py` — state machine transitions, callback dispatch, concurrent safety
- [ ] `_constants.py` — `get_feature_config()` returns expected keys
- [ ] `models.py` — download logic, SHA verification, cache management (mocked network)
- [ ] `tools/train.py` — argument parsing, minimum sample validation
- [ ] `stt_engine.py` — language cache TTL, no-speech filtering

#### 11b. Console E2E Tests (Playwright)

**Setup:** Chrome with `--use-file-for-fake-audio-capture` to inject pre-recorded WAV files as mic input.

```javascript
// playwright.config.ts
{
  use: {
    launchOptions: {
      args: [
        '--use-fake-device-for-media-stream',
        '--use-fake-ui-for-media-stream',
        '--use-file-for-fake-audio-capture=fixtures/wake-word-sample.wav',
      ],
    },
  },
}
```

**Test scenarios:**

| Test | What it verifies |
|------|-----------------|
| `recording-flow.spec.ts` | User can record 10 clips, waveform displays, quality gates work |
| `upload-and-train.spec.ts` | Samples upload, training starts, progress bar updates via SSE |
| `model-download.spec.ts` | After training, user can download .onnx file |
| `auth-flow.spec.ts` | Sign up, log in, API key generation |
| `quality-rejection.spec.ts` | Silent audio is rejected, noisy audio shows warning |
| `pricing-flow.spec.ts` | Free tier limits enforced, upgrade flow works |

**Mock training backend:** FastAPI mock service returns pre-trained model instantly:

```python
# tests/mocks/training_mock.py
@app.post("/train")
async def mock_train(audio_files: list[UploadFile]):
    await asyncio.sleep(0.5)  # Simulate brief delay
    return {"model_id": "mock-001", "status": "complete", "metrics": {"cohen_d_synthetic": 15.1}}
```

#### 11c. Docker Compose for Full Stack Tests

```yaml
services:
  frontend:
    build: ./console/frontend
    ports: ["3000:3000"]
  backend:
    build: ./console/backend
    ports: ["8000:8000"]
    environment:
      TRAINING_MODE: mock  # Use mock trainer, no GPU needed
  e2e:
    build: ./tests/e2e
    command: npx playwright test
    depends_on: [frontend, backend]
```

#### 11d. CI/CD Additions

- [ ] Add Playwright E2E job to `ci.yml` (runs against Docker Compose stack)
- [ ] Create missing scripts: `tools/benchmark_regression_check.py`, `tools/fetch_release_models.py`
- [ ] Fix `pyproject.toml` wheel targets (remove nonexistent `src/wakeword`)
- [ ] Add `RELEASE_NOTES.md` template

---

### 12. Documentation & DX (9.5/10 → 10/10)

- [ ] Set up mkdocs-material for API reference docs (auto-generated from docstrings)
- [ ] Write migration guides: "From Snowboy to ViolaWake", "From Mycroft Precise to ViolaWake"
- [ ] Write Console quickstart tutorial
- [x] Fix all placeholder URLs (updated to `GeeIHadAGoodTime/ViolaWake`)
- [ ] Add real GitHub org + repo
- [ ] Fill in model SHA-256 hashes (currently PLACEHOLDER)

---

### 13. Packaging & Distribution (9/10 → 10/10)

- [ ] Fix `pyproject.toml` line 102: remove `src/wakeword` from wheel targets
- [ ] Add `openwakeword` to `[training]` optional deps
- [ ] Add `silero-vad` to `[vad]` optional deps
- [ ] Add `pysbd` to core deps (sentence boundary detection for TTS)
- [ ] Add `audiomentations` to `[training]` optional deps
- [ ] Create Docker image for cloud training: `violawake/trainer:latest`
- [ ] Test PyPI publish flow end-to-end (TestPyPI first)

---

## Open Source Projects to Leverage

| Project | License | What we take |
|---------|---------|-------------|
| **audiomentations** | MIT | Audio augmentation pipeline (43 transforms) |
| **openWakeWord** | Apache 2.0 | OWW backbone, training architecture reference, pre-computed negative features on HuggingFace |
| **Piper TTS** | MIT | Hard negative synthesis (multi-voice TTS for confusable words) |
| **silero-vad** | MIT | VAD backend (2MB ONNX model) |
| **RecordRTC** | MIT | Browser audio recording at 16kHz |
| **wavesurfer.js** | BSD-3 | Waveform visualization + recording plugin |
| **Meyda.js** | MIT | Client-side audio quality metrics |
| **pronouncing** | MIT | CMU dictionary for phoneme similarity search |
| **g2p-en** | MIT | Neural grapheme-to-phoneme conversion |
| **pysbd** | MIT | Sentence boundary detection (replaces naive TTS splitter) |
| **MUSAN corpus** | Public domain | Music + speech + noise for training negatives |
| **Common Voice** | CC0 | Speech negatives corpus |
| **BIRD RIR dataset** | Free | 1M room impulse responses for reverb augmentation |
| **Modal.com** | SaaS | Serverless GPU training ($0.05/job) |
| **Supabase** | Apache 2.0 | Auth + database + storage |

---

## Implementation Order

### Phase 1: Fix the SDK (make training actually work)
1. Training negatives (MUSAN + Common Voice subset download + embedding)
2. Augmentation pipeline (audiomentations integration)
3. Fix evaluation (OWW embedding path for MLP models)
4. Hard negative mining (pronouncing + g2p-en + Piper TTS)
5. Validation split + early stopping
6. Fix all bugs (pipeline race condition, process() dead branch, STT conflict)
7. Implement Silero VAD
8. Fix TTS sentence splitter
9. Complete unit test coverage to 85%

### Phase 2: Build the Console
10. FastAPI backend (auth, upload, job queue, model delivery)
11. React frontend (landing page, recording UI, dashboard)
12. Modal.com training integration
13. Supabase Auth integration
14. S3/R2 model storage + signed URLs
15. SSE training progress

### Phase 3: Polish & Ship
16. Playwright E2E tests (full recording → training → download flow)
17. Stripe billing integration
18. mkdocs API documentation site
19. Migration guides (Snowboy, Mycroft Precise)
20. PyPI publish (SDK)
21. Console deployment
22. Launch blog post + ProductHunt

---

## Key Market Insights (from research)

1. **The #1 pain point** across all wake word engines is training UX. Picovoice makes it easy but expensive. openWakeWord is free but breaks constantly. We solve both.

2. **$0 → $6,000 gap** is Picovoice's blind spot. Our $29/mo tier captures the entire indie/small company market.

3. **openWakeWord's Colab training breaks frequently** (their #1 GitHub issue is "dependency nightmare"). A web Console that just works is instant differentiation.

4. **Zero-shot training** (type text, get model) is the academic frontier (GE2E-KWS, 2024). We can add this as a Phase 2 feature — type wake word → Piper TTS generates synthetic samples → train automatically. No microphone needed.

5. **Browser/WASM** is the next frontier. Multiple independent projects are building this. We should target WASM inference in Phase 2.

6. **New competitor DaVoice** claims 99%+ accuracy with working web/WASM SDK. Worth monitoring closely.

7. **Picovoice uses zero user recordings** — pure transfer learning from text. Our advantage: real voice samples produce speaker-specific models with higher accuracy for the enrolled user.
