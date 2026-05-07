# ViolaWake SDK — 10/10 Roadmap

**Goal:** Ship a Picovoice competitor (Option C: hybrid open-core + paid Console)
**Date:** 2026-03-25 (created) | 2026-04-05 (updated — all phases complete)
**Based on:** 5 parallel research agents covering augmentation/corpus, browser recording, cloud infra, market analysis, and E2E testing

---

## Executive Summary

The market has a $0-to-$6,000 pricing gap. Picovoice charges $6K/yr for commercial use. openWakeWord is free but training breaks constantly. **We fill the gap**: free open-source SDK + paid web Console that trains custom wake words from 10 voice samples in under 5 minutes.

**Key competitive advantages we're building:**
1. Real voice samples (speaker-specific) vs Picovoice's text-only synthetic training
2. $0 SDK + affordable Console vs $6K/yr Picovoice
3. Open training pipeline vs Picovoice black box
4. Bundled Wake+VAD+STT+TTS pipeline vs piecemeal assembly
5. TemporalCNN d'=8.577 (EER=0.8%) on production eval, with real-world speech-negative d-prime still TBD, plus a production-hardened 4-gate decision policy

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

### 1. Training Pipeline — DONE (10/10)

**Status:** Complete. Trained "big chungus" model on 2026-04-05 with good results.

#### 1a. Negative Corpus — DONE
- 2 rounds of confusable negatives (phonetically similar words via edge-tts, 30+16 words x 10 voices)
- Speech negatives (100+ common phrases x 5 voices via edge-tts)
- Universal corpus auto-discovery (MUSAN speech/music/noise, LibriSpeech) from `corpus/`, `~/.violawake/corpus/`, or repo root
- User-provided negatives via `--negatives` directory

#### 1b. Data Augmentation — DONE
- `audiomentations` chain: Gain(-6..6dB, p=0.8), TimeStretch(0.9..1.1, p=0.5), PitchShift(-2..+2 semitones, p=0.5), Mp3Compression(32..128kbps, p=0.3), TimeMask(0..10%, p=0.3)
- SDK also has numpy-only augmentation pipeline (`training/augment.py`) with SpecAugment, RIR convolution, pink noise
- Auto-scales: 10 user samples → 210+ augmented clips before embedding extraction

#### 1c. Hard Negative Mining — DONE
- `_generate_confusable_negatives()`: 2 rounds (broad + tight phonetic variants)
- `_generate_speech_negatives()`: common phrases as general speech negatives
- TTS-generated via edge-tts (20 diverse voices)

#### 1d. Training Infrastructure — DONE
- [x] 80/20 group-aware validation split with early stopping (patience=15)
- [x] FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05) + AdamW + cosine annealing LR
- [x] EMA weight averaging with auto-selection (raw vs EMA vs SWA)
- [x] `openwakeword` in `[training]` deps
- [x] Training progress callback (wired to Console SSE)
- [x] Model metadata saved as `.config.json` (d', training config, quality gate results)
- [x] Post-training quality gate: speech FP rate, confusable FP rate, silence max score — grade F blocks ONNX export

---

### 2. Core Wake Word Engine — DONE (10/10)

- [x] TemporalCNN (9-frame windows of 96-dim OWW embeddings, ~12K params)
- [x] TemporalGRU and TemporalConvGRU alternatives available
- [x] OWW backbone produces correct frame embeddings
- [x] ONNX export with dynamic batch axis
- [x] Dead branch in `process()` fixed (adversarial audit round)
- [ ] INT8 quantized model support (future — not blocking launch)
- [ ] GPU provider auto-detection (future — CPU inference is <5ms)

---

### 3. Evaluation — DONE (10/10)

- [x] Auto-detect architecture from config.json or ONNX input shape (mlp_on_oww, temporal_oww, cnn)
- [x] OWW embedding extraction path for MLP and temporal models
- [x] Threshold sweep (find_optimal_threshold)
- [x] Confusion matrix output (compute_confusion_matrix)
- [x] Per-file score dump CSV (_dump_scores_csv)
- [x] d' (d-prime), ROC AUC, EER, FAR/FRR computation
- [ ] ROC curve PNG visualization (future)
- [ ] Side-by-side model comparison mode (future)

---

### 4. VAD Engine — DONE (10/10)

- [x] Silero VAD backend implemented
- [x] `silero-vad` in `[vad]` optional deps
- [x] Auto-selection: Silero → WebRTC → RMS fallback
- [x] WebRTC VAD backend
- [x] RMS energy-based fallback

---

### 5. TTS Engine — DONE (10/10)

- [x] Kokoro-82M on-device TTS (ONNX, MIT)
- [x] `pysbd` sentence boundary detection in core deps
- [x] Multiple voice support
- [ ] Volume normalization to -16 LUFS (future)
- [ ] Streaming chunk playback (future)

---

### 6. STT Engine — DONE (10/10)

- [x] faster-whisper integration
- [x] Language detection + caching
- [x] No-speech filtering
- [ ] Hotword boosting (future)
- [ ] Streaming partial results (future)

---

### 7. Voice Pipeline — DONE (10/10)

- [x] Wake → Listen → Transcribe → Respond state machine
- [x] Configurable components (detector, VAD, STT, TTS)
- [x] Pipeline event system
- [ ] Async pipeline variant (future)
- [ ] Wake-during-transcription handling (future)

---

### 8. Console Website — DONE (10/10)

**Deployed:** Frontend on Cloudflare Pages (violawake.com), Backend on Docker + Cloudflare Tunnel (api.violawake.com)

#### Tech Stack (as built)

| Layer | Technology | Status |
|-------|-----------|--------|
| **Frontend** | React + Vite + TypeScript | DONE |
| **Recording** | Web Audio API (ScriptProcessor → Float32 → WAV encoder) | DONE |
| **Quality** | Custom audioQuality.ts (RMS, clipping, SNR, duration) | DONE |
| **Backend** | FastAPI + SQLAlchemy + PostgreSQL | DONE |
| **Auth** | JWT (PyJWT) + bcrypt + email verification (Resend) | DONE |
| **Storage** | Local + S3 (boto3, presigned URLs for model delivery) | DONE |
| **Training** | On-server via ViolaWake SDK (`_train_temporal_cnn`) | DONE |
| **Billing** | Stripe Billing (checkout, webhooks, portal, usage) | CODE DONE, needs live keys |
| **Progress** | Server-Sent Events (SSE) via sse-starlette | DONE |

#### Recording Flow — DONE

- 3-2-1 countdown, live level meter, auto-stop at max duration
- 10-sample recording session with re-record and review
- Client-side quality gates: silence (RMS < 0.01 = error), clipping (>5% = warning), low SNR (<6dB = warning), duration (0.5-5s)
- Server-side quality gates: WAV validation, mono 16kHz conversion, RMS energy check
- Upload with per-user rate limiting

#### Pages — DONE

| Page | Route | Status |
|------|-------|--------|
| Landing | `/` | DONE |
| Auth | `/login`, `/register` | DONE |
| Dashboard | `/dashboard` | DONE |
| Record | `/record` | DONE |
| Training Progress | SSE stream | DONE |
| Model Performance | `/models/:id/performance` | DONE |
| Model Download | `/models/:id/download` | DONE |
| Pricing | `/pricing` | DONE |
| Billing | `/billing` | DONE |
| Teams | `/teams` | DONE |

#### Backend API — DONE

```
POST /api/auth/register         JWT auth + email verification
POST /api/auth/login            JWT auth + account lockout (5 fails → 15min)
POST /api/recordings/upload     WAV upload with quality gates
POST /api/training/start        Submit training job (quota-checked)
GET  /api/training/stream/:id   SSE progress stream
GET  /api/training/status/:id   Poll status
GET  /api/models                List user's models
GET  /api/models/:id/download   Download ONNX (presigned URL or direct)
GET  /api/models/:id/config     Training config + metrics
GET  /api/models/:id/performance  d', threshold, score distributions
POST /api/billing/checkout      Stripe Checkout Session
POST /api/billing/webhook       Stripe webhook handler
GET  /api/billing/subscription  Current tier + usage
POST /api/billing/portal        Stripe Billing Portal
GET  /api/billing/usage         Monthly usage vs limit
POST /api/teams/invite          Team invites (rate limited)
```

#### Security — DONE (hardened 2026-04-05)

- JWT via PyJWT (migrated from python-jose)
- Account lockout (5 failures → 15min lock)
- Rate limiting on all sensitive endpoints
- HSTS, X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
- 15MB body size limit, constant-time token comparison
- Non-root Docker (gosu entrypoint)
- Per-user SSE connection limits, per-user job queue limits
- Admin endpoints gated by token

---

### 9. Pricing — DONE (code complete)

| Tier | Price | Limit | Status |
|------|-------|-------|--------|
| **Free** | $0 | 3 models/month | ENFORCED |
| **Developer** | $29/month | 20 models/month | CODE DONE, needs Stripe Price ID |
| **Business** | $99/month | Unlimited | CODE DONE, needs Stripe Price ID |
| **Enterprise** | Custom | Custom | Future |

Quota enforcement works (check_training_quota dependency). Stripe checkout, webhooks, portal, subscription management — all built. Needs live Stripe keys + Price IDs configured.

---

### 10. Testing — DONE (9/10)

- [x] Unit tests: SDK + Console backend (adversarial audit rounds)
- [x] E2E tests: `test_browser_flow.py`, `test_api_flow.py` (Playwright)
- [x] CI pipeline: GitHub Actions (mypy, ruff, pytest, macOS + Linux)
- [ ] Playwright E2E in CI via Docker Compose (future)

---

### 11. Documentation — DONE (9/10)

- [x] ARCHITECTURE.md (594 lines)
- [x] Comprehensive API documentation
- [x] README with quickstart
- [x] CHANGELOG
- [x] Placeholder URLs fixed
- [ ] mkdocs-material hosted docs site (future)
- [ ] Migration guides from Snowboy/Mycroft (future)

---

### 12. Packaging & Distribution — DONE (10/10)

- [x] Published on PyPI: `pip install violawake` (v0.2.2)
- [x] `openwakeword` in `[training]` deps
- [x] `silero-vad` in `[vad]` deps
- [x] `pysbd` in core deps
- [x] `audiomentations` in `[training]` deps
- [x] Docker image for backend (`Dockerfile.backend`)
- [x] Docker Compose for full stack (PostgreSQL + backend + Cloudflare Tunnel)

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

## Implementation Status

### Phase 1: SDK — COMPLETE
All items done: training pipeline, augmentation, evaluation, negatives, validation split, early stopping, Silero VAD, TTS sentence splitting, unit tests.

### Phase 2: Console — COMPLETE
All items done: FastAPI backend, React frontend, recording UI, job queue, SSE progress, model download, S3 storage, auth (JWT, not Supabase — simpler for self-hosting).

### Phase 3: Polish & Ship — IN PROGRESS
- [x] Playwright E2E tests
- [x] Stripe billing code (checkout, webhooks, portal, usage, quotas)
- [x] PyPI publish (v0.2.2)
- [x] Console deployment (Cloudflare Pages + Docker + Cloudflare Tunnel)
- [x] Security hardening (20 findings fixed)
- [x] d' metric correction (was incorrectly labeled "Cohen's d")

---

## What's Left to Launch

**Configuration items (not code):**

1. **Stripe live keys** — set `VIOLAWAKE_STRIPE_SECRET_KEY`, `VIOLAWAKE_STRIPE_WEBHOOK_SECRET`, create Price IDs for developer ($29/mo) and business ($99/mo) tiers in Stripe Dashboard, set `VIOLAWAKE_STRIPE_PRICE_DEVELOPER` and `VIOLAWAKE_STRIPE_PRICE_BUSINESS`
2. **Resend API key** — set `VIOLAWAKE_RESEND_API_KEY` for email verification (without it, users auto-verify)
3. **Launch post** — Show HN draft exists, needs final review

---

## Key Market Insights (from research)

1. **The #1 pain point** across all wake word engines is training UX. Picovoice makes it easy but expensive. openWakeWord is free but breaks constantly. We solve both.

2. **$0 → $6,000 gap** is Picovoice's blind spot. Our $29/mo tier captures the entire indie/small company market.

3. **openWakeWord's Colab training breaks frequently** (their #1 GitHub issue is "dependency nightmare"). A web Console that just works is instant differentiation.

4. **Zero-shot training** (type text, get model) is the academic frontier (GE2E-KWS, 2024). We can add this as a Phase 2 feature — type wake word → Piper TTS generates synthetic samples → train automatically. No microphone needed.

5. **Browser/WASM** is the next frontier. Multiple independent projects are building this. We should target WASM inference in Phase 2.

6. **New competitor DaVoice** claims 99%+ accuracy with working web/WASM SDK. Worth monitoring closely.

7. **Picovoice uses zero user recordings** — pure transfer learning from text. Our advantage: real voice samples produce speaker-specific models with higher accuracy for the enrolled user.
