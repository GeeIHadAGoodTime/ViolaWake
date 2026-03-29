<!-- doc-meta
scope: Product requirements - what we build, why, and how we measure success
authority: LIVING - primary product spec, maintained by PM/tech lead
code-paths: src/violawake_sdk/, console/, README.md
staleness-signals: major product pivot, hosted architecture change, pricing model change
last-updated: 2026-03-28
-->

# ViolaWake Product Requirements Document

**Version:** 0.2
**Status:** Active
**Authors:** ViolaWake team

---

## 1. Product Statement

ViolaWake is an open-source Python voice SDK plus a shipped web Console for custom wake-word training.

The product exists in two layers:

1. **SDK layer:** wake detection, VAD, STT, TTS, training CLI, evaluation tooling
2. **Console layer:** browser-based recording, account auth, queued training, progress streaming, model download

The key correction in this PRD is scope: **the Console is not a Phase 2 idea. It is already delivered in-repo.** What remains for later phases is hosted hardening, team features, and managed infrastructure upgrades.

---

## 2. Problem

Developers building voice-enabled products still face an awkward choice:

- **Picovoice / Porcupine:** polished, proprietary, expensive for commercial use, and closed around training
- **openWakeWord and similar OSS options:** open and free, but less productized for end-to-end onboarding and managed training workflows

ViolaWake solves that by offering:

- an open SDK that developers can inspect and ship
- a real web Console for lower-friction training
- portable ONNX artifacts users can own directly

---

## 3. Current Product Reality

### 3.1 Delivered In Phase 1

| Component | Status | Notes |
|-----------|--------|-------|
| **WakeDetector** | Shipped | ONNX-based wake detection on top of OpenWakeWord embeddings |
| **VADEngine** | Shipped | WebRTC / Silero / RMS backends |
| **STTEngine** | Shipped | faster-whisper integration |
| **TTSEngine** | Shipped | Kokoro ONNX integration |
| **VoicePipeline** | Shipped | Wake -> VAD -> STT -> TTS orchestration |
| **Training CLI** | Shipped | Local custom wake-word training |
| **Evaluation CLI** | Shipped | Benchmark and threshold tooling |
| **Console Backend** | Shipped | FastAPI auth, upload, job queue, SSE, model management |
| **Console Frontend** | Shipped | React app for signup, recording, training, download |
| **Tier / quota scaffolding** | Shipped but partial | Free / Developer / Business usage limits exist; differentiated infra does not |

### 3.2 Delivered Architecture

The shipped Console currently uses:

- local email/password auth with bcrypt-hashed passwords
- local JWT access tokens
- SQLite by default
- local filesystem storage by default
- async queued CPU training jobs executed through background workers/threads

Optional integrations such as Stripe, Cloudflare R2, and external databases are not the default baseline and must not be described as if they are already the canonical deployment architecture.

### 3.3 Model Size Requirement

All product docs should use the same wake-word footprint numbers:

- **Wake head:** `102 KB`
- **Shared OpenWakeWord backbone:** `1.33 MB`
- **Total runtime footprint:** **`1.43 MB`**

This is the correct runtime footprint for the shipped default wake path.

---

## 4. Scope

### In Scope Now

- Python-first wake-word SDK
- Local training CLI and evaluation tooling
- Browser-based Console for custom wake-word creation
- Account registration/login for the Console
- Recording upload and model download flows
- Queued training with live progress updates
- Free/open inference and portable ONNX artifacts

### Out Of Scope For The Current Release

- Browser/WASM inference SDK
- JavaScript/Node SDK
- Android/iOS SDKs
- Team workspaces and shared project management
- Priority queues by paid tier
- GPU-backed managed training as the default path
- Automatic retention cleanup guarantees
- Enterprise SLAs and support operations
- MCU / Cortex-M targets

---

## 5. Requirements

### R1: Wake Detection

The SDK must provide reliable on-device wake detection with transparent evaluation tooling and portable ONNX artifacts.

### R2: Open Training Path

Users must be able to train custom wake words locally with open tooling, without depending on the hosted Console.

### R3: Shipped Console

The Console must support the complete browser flow:

1. create account
2. record/upload samples
3. start training
4. watch progress
5. download trained model

### R4: Honest Local-First Service Model

The Console requirements must match the shipped implementation:

- auth is local JWT auth unless a managed provider is explicitly configured
- storage is local by default unless object storage is explicitly configured
- training uses queued CPU workers unless managed GPU infrastructure is explicitly introduced

### R5: Trustworthy Claims

No product document should claim:

- Supabase auth as baseline reality
- Modal GPU training as baseline reality
- Cloudflare R2-only storage as baseline reality
- automatic recording deletion unless cleanup actually runs
- priority queues or team management unless those features are implemented

---

## 6. Success Metrics

### Technical

| Metric | Target |
|--------|--------|
| Wake inference latency | <= 15 ms / frame on target CPU |
| Console training completion | >= 95% successful jobs on valid input |
| Model download success | >= 99% for completed jobs |
| Queue observability | wait time and failure reasons visible in health/ops tooling |

### Product

| Metric | Target |
|--------|--------|
| Signup -> first training conversion | >= 50% |
| Training -> model download conversion | >= 70% |
| Free tier activation | meaningful weekly active training users |
| Paid conversion | measured only after hosted billing is truly live |

### Trust

| Metric | Target |
|--------|--------|
| Architecture contradictions across docs | 0 |
| Unimplemented feature promises in primary docs | 0 |
| Privacy/compliance overclaims in primary docs | 0 |

---

## 7. Competitive Position

| Dimension | ViolaWake requirement |
|-----------|-----------------------|
| **Price transparency** | Keep SDK free and explain any hosted pricing honestly |
| **Training accessibility** | Offer both open CLI and shipped Console workflows |
| **Portability** | Deliver standard ONNX model artifacts |
| **Trust** | Disclose benchmark limits and architecture truthfully |
| **Differentiation** | Compete on openness + productized workflow, not fake cloud sophistication |

---

## 8. Non-Goals

- Rebuilding Big Tech voice-assistant ecosystems
- Locking inference behind a metered API
- Claiming enterprise-grade hosted guarantees before the hosted stack is real
- Treating the SDK and Console as the same thing; the SDK must remain useful without the Console

---

## 9. Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Docs outrun implementation | High | Keep `Current Product Reality` and `Roadmap` separated everywhere |
| Hosted privacy claims exceed shipped behavior | High | Tie legal/business claims to actual storage + retention behavior |
| CPU queue becomes the bottleneck | Medium | Measure demand before adding GPU complexity |
| Competitive pressure from Picovoice | Medium | Win on open SDK + portable artifacts + honest product story |

---

## 10. Next Phase

The next phase is not "build a Console." The Console is already built.

The next phase is:

1. harden the hosted deployment story
2. implement retention cleanup and stronger privacy operations
3. add managed infrastructure only where usage justifies it
4. expand commercial features such as team management and differentiated queues after the baseline hosted service is stable
