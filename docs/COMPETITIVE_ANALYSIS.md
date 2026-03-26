<!-- doc-meta
scope: Competitive landscape — wake word engines, voice SDKs, and adjacent markets
authority: LIVING — updated when competitors ship major changes or pricing shifts
code-paths: docs/PRD.md, docs/adr/
staleness-signals: Picovoice pricing change, openWakeWord major release, new entrant with >500 GitHub stars
last-updated: 2026-03-18
-->

# ViolaWake SDK — Competitive Analysis

**Version:** 1.0
**Status:** Active
**Derived from:** COMPETITIVE_AUDIT_REPORT.md (Viola repo `.viola/agents/`), full codebase audit 2026-03-17

---

## 1. Competitive Landscape Overview

The wake word / voice SDK market has three distinct segments:

| Segment | Players | Our target? |
|---------|---------|-------------|
| **Proprietary commercial SDKs** | Picovoice, Amazon AVS, Sensory | Compete on open-core model |
| **Open-source community projects** | openWakeWord, Mycroft Precise (archived), Snowboy (archived) | Compete on production quality |
| **Big Tech embedded** | Google Assistant SDK, Amazon Alexa Voice Service | Not competing (ecosystem play) |

**Our positioning:** Fill the gap between "proprietary/expensive" and "open-source/unpolished." Production-tested accuracy + accessible training + Python-first SDK + open license.

---

## 2. Feature Matrix

> Accuracy metrics in this document are not apples-to-apples. ViolaWake's current `15.10` figure is Cohen's d measured against synthetic-only negatives, while competitor numbers below are published/community d-prime estimates or claims.
> Picovoice does not publish d-prime; comparison uses publicly available benchmark reports and community testing.

| Feature | **ViolaWake SDK** | Picovoice Porcupine | openWakeWord | Mycroft Precise | Snowboy |
|---------|:-----------------:|:-------------------:|:------------:|:---------------:|:-------:|
| **License** | Apache 2.0 | Proprietary (metered) | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Cost (commercial)** | Free | $6K+/yr Foundation | Free | Free (archived) | Free (archived) |
| **Accuracy disclosure** | **Cohen's d 15.10 on synthetic negatives** | No published d-prime; ~10–13 in third-party estimates | ~5–8 in community/public benchmarks | ~6–10 in historical reports | ~5–8 in historical reports |
| **FAR @ default threshold** | ≤0.5/hr | <1/hr (claimed) | ~1–3/hr (reported) | Variable | Variable |
| **FRR @ default threshold** | ≤3% | <5% (claimed) | ~5–15% | ~5–10% | ~5–15% |
| **Inference latency / 20ms frame** | ≤15ms | ≤20ms (Raspberry Pi 3) | ≤30ms | ≤25ms | ≤25ms |
| **Custom wake words** | ✅ Training CLI | ✅ Console (paid for commercial) | ✅ Fine-tuning | ✅ Training required | ❌ Limited |
| **Training code open-source** | ✅ Full pipeline | ❌ Closed | ✅ Yes | ✅ Yes | ❌ No |
| **Training without ML expertise** | ✅ 20+ samples, CLI | ⚠️ Console (simpler UX) | ⚠️ Complex setup | ❌ Requires ML expertise | N/A |
| **ONNX inference** | ✅ Yes | ❌ Proprietary C binary | ✅ Yes | ❌ TFLite/PyTorch | ❌ No |
| **Python SDK** | ✅ First-class | ⚠️ C-binary wrapper | ✅ Yes | ✅ Yes | ✅ Yes (unmaintained) |
| **Bundled VAD** | ✅ Yes (WebRTC/Silero/RMS) | ✅ Cobra (separate product) | ⚠️ Basic | ❌ Separate | ❌ No |
| **Bundled STT** | ✅ Whisper wrapper | ✅ Cheetah/Leopard (separate) | ❌ No | ❌ No | ❌ No |
| **Bundled TTS** | ✅ Kokoro-82M | ✅ Orca (separate, proprietary) | ❌ No | ❌ No | ❌ No |
| **Full voice pipeline (1 import)** | ✅ VoicePipeline class | ❌ Assemble yourself | ❌ No | ❌ No | ❌ No |
| **Windows** | ✅ Production | ✅ Yes | ⚠️ Tested | ⚠️ Partial | ❌ No (no binaries) |
| **Linux (x64)** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **macOS (Intel + ARM)** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Intel only | ⚠️ Intel only |
| **Raspberry Pi (ARM)** | ✅ Tested on Pi 4 | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Android** | ❌ Phase 3 | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **iOS** | ❌ Phase 3 | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **MCU/embedded** | ❌ Never | ✅ Yes (Cortex-M) | ❌ No | ❌ No | ❌ No |
| **Multi-language** | ❌ English only (Phase 2) | ✅ 9 languages | ⚠️ English primary | ❌ English only | ❌ English only |
| **Production-tested** | ✅ Viola production | ✅ Picovoice products | ❌ Community testing only | ❌ Archived 2022 | ❌ Archived 2020 |
| **Evaluation tool** | ✅ `violawake-eval` (Cohen's d, FAR, FRR) | ❌ No | ⚠️ Basic | ❌ No | ❌ No |
| **REST API** | ❌ Phase 2 | ✅ Yes | ❌ No | ❌ No | ❌ No |
| **Active maintenance** | ✅ Active | ✅ Active | ✅ Active | ❌ Archived | ❌ Archived |

---

## 3. Competitor Deep Dives

### 3.1 Picovoice Porcupine

**Summary:** The production gold standard for proprietary wake word detection. Best-in-class platform coverage, proven at scale, developer-friendly Console. The dominant player we are directly competing against.

**Strengths:**
- 12+ platform SDKs (Python, iOS, Android, Web, Raspberry Pi, MCUs, Rust, Go, C, .NET)
- 9 languages with pre-trained wake words
- Web Console for custom wake word training (no ML expertise required)
- Well-documented with extensive examples
- Proven at scale in commercial products

**Weaknesses:**
- **Pricing.** $6K+/year Foundation Plan for commercial use. Metered model for custom training. Personal/dev use free, but commercial requires purchase.
- **Black-box models.** Cannot inspect, fine-tune, or adapt models to domain-specific needs (accents, environments). You are locked into what Picovoice trains.
- **Python SDK is a wrapper.** Python code calls a C binary. You can't debug the inference, contribute to it, or use Python ML tooling (numpy, pytorch) alongside it.
- **No standalone training pipeline.** Training goes through their Console only — not reproducible locally, not open to community improvement.
- **Accuracy not published.** Picovoice does not publish d-prime or equivalent metrics. Third-party benchmarks suggest ~10–13 d-prime range.
- **C-binary dependency.** Makes packaging, debugging, and customization harder for Python-first developers.

**Our gaps vs Porcupine:**
- Platform coverage (mobile, MCU, 9 languages) — we are Python+desktop+Pi only
- Console UX (no code required for training) — our training requires CLI
- Brand recognition and ecosystem

**Our advantages over Porcupine:**
- $0 vs $6K+/year commercial
- Open training pipeline (inspectable, improvable, auditable)
- Transparent internal benchmark disclosure: Cohen's d 15.10 on synthetic negatives; not directly comparable to Porcupine's unpublished speech-negative performance
- First-class Python (not C-binary wrapper)
- Complete bundled pipeline (Wake+VAD+STT+TTS in one `pip install`)
- ONNX inference (portable, debuggable, integratable with Python ML tooling)

**Takeaway for V1:** Porcupine is the "too expensive" option that sends developers looking for alternatives. We win by being the open-core alternative they find.

---

### 3.2 openWakeWord

**Summary:** The closest open-source competitor. Apache-2.0, ONNX-based, Python, growing community traction. Our approach is architecturally similar (MLP on OWW embeddings) but with more production-hardening and a stronger internal synthetic-negative benchmark score.

**Strengths:**
- Apache-2.0, fully open-source
- ONNX inference (same as us)
- Community of developers actively using and contributing
- Multiple pre-trained models
- Decent fine-tuning documentation

**Weaknesses:**
- **Accuracy ceiling appears lower in published/community benchmarks.** openWakeWord is commonly reported around ~5–8 d-prime, while our current internal number is Cohen's d 15.10 on synthetic negatives and is not directly comparable.
- **No bundled pipeline.** Provides wake detection only. User must integrate VAD, STT, TTS separately.
- **No standalone evaluation tool.** No standardized way to evaluate custom model quality.
- **Complex fine-tuning.** The fine-tuning workflow requires ML expertise and isn't streamlined for "I have 20 samples of my wake word."
- **No decision policy.** Raw score output without production-hardened false-accept mitigation (cooldown, RMS gate, listening gate).
- **Windows support fragile.** Community reports installation issues on Windows.
- **No active productization.** Maintained as a research/community project, not a production SDK.

**Our gaps vs openWakeWord:**
- Community size (they have more GitHub stars and community familiarity)
- Number of pre-trained wake word models
- Open training approach (they also have this; we need to match)

**Our advantages over openWakeWord:**
- Transparent benchmark methodology and a strong internal synthetic-negative score (Cohen's d 15.10); direct speech-negative comparison still needs to be published
- Production-hardened decision policy (4-gate system from Viola production)
- Bundled pipeline (Wake+VAD+STT+TTS)
- Streamlined training CLI (20 samples → model, no ML expertise)
- Evaluation tool (`violawake-eval`) that reports Cohen's d, FAR, FRR, and ROC data
- Windows production-tested

**Takeaway for V1:** openWakeWord users who hit training UX or production-hardening walls are our primary acquisition channel. We should publish a speech-negative benchmark before claiming an accuracy win.

---

### 3.3 Mycroft Precise (Archived)

**Summary:** The wake word engine from the Mycroft AI open-source voice assistant. Actively maintained until Mycroft AI shutdown (2022). Archived but still referenced as an option by home automation communities.

**Strengths (historical):**
- Fully open-source (Apache 2.0)
- Designed specifically for Python voice assistants
- Worked well integrated into Mycroft's pipeline
- Home Assistant community used it extensively

**Weaknesses:**
- **Archived and unmaintained since 2022.** No security patches, no bug fixes, no platform updates.
- **TFLite/PyTorch models.** Not ONNX-portable. Harder to deploy across platforms.
- **Lower accuracy than current approaches.** Pre-dates modern embedding-based approaches like OWW.
- **No training CLI for non-ML users.** Required understanding of the ML training process.
- **No bundled pipeline.** Detection only.

**Our position vs Mycroft Precise:**
- Precise users (typically home automation hobbyists) are natural ViolaWake targets — they want open-source, Python, on-device, and Precise no longer works on modern Python versions.
- We offer a drop-in spiritual successor with better accuracy, active maintenance, and a bundled pipeline.

**Takeaway for V1:** Marketing to Home Assistant / ex-Mycroft users is a clear acquisition path. The "Mycroft Precise alternative" angle is worth a blog post.

---

### 3.4 Snowboy (Archived)

**Summary:** KITT.AI's wake word engine, widely used 2016–2020. Acquired by Baidu and eventually shut down in December 2020. Still referenced in older tutorials and Stack Overflow answers.

**Strengths (historical):**
- Extremely low latency (C library)
- Simple Python API
- Worked on Raspberry Pi
- Had a training service (300 voice samples → model)

**Weaknesses:**
- **Shut down December 2020.** Training service gone. Models may expire. No maintenance.
- **Proprietary model format.** Models were tied to their service; no offline training.
- **No open training.** Training required their server-side service.
- **Platform binary distribution.** Pre-compiled binaries for specific platforms only.
- **Lower accuracy** than modern embedding-based approaches.

**Our position vs Snowboy:**
- Snowboy users are stranded and actively looking for alternatives. Multiple GitHub issues on archived Snowboy repo request "what should I use instead?"
- We are the answer. Open-source, active, better accuracy, trainable offline.

**Takeaway for V1:** The "Snowboy alternative" search query is an acquisition channel. Explicit migration guide from Snowboy to ViolaWake is worth building.

---

### 3.5 Amazon Alexa Voice Service / Google Assistant SDK / Apple Siri SDK

**Summary:** Big Tech embedded wake word solutions. Designed for devices joining their respective voice assistant ecosystems. Not general-purpose SDKs.

**Strengths:**
- Massive investment, industry-leading accuracy for their wake words ("Alexa", "Hey Google", "Hey Siri")
- Deeply integrated with cloud ecosystems (shopping, smart home, calendar, etc.)

**Weaknesses / Why developers avoid them:**
- **Ecosystem lock-in.** You're building an Alexa device, a Google device, or an Apple device. Not your own product.
- **Custom wake words not supported.** You cannot teach "Alexa" to respond to "Jarvis" — you join their ecosystem.
- **Requires cloud connectivity.** No offline mode; all audio sent to Amazon/Google/Apple servers.
- **Privacy concerns.** Significant regulatory scrutiny. Many users/developers explicitly avoid these for privacy-sensitive applications.
- **Commercial licensing restrictions.** Strict certification requirements for consumer products.
- **Data collection.** All voice data potentially used for model improvement.

**Our position vs Big Tech:**
- Completely different value proposition. We serve developers who want **their own wake word** in **their own product** with **no cloud dependency**.
- Privacy-first positioning directly addresses the reason developers avoid Big Tech solutions.
- ViolaWake is what you use when you don't want to give Alexa/Google/Apple access to your users' home audio.

**Takeaway for V1:** The positioning "no cloud, no Big Tech, your wake word, your data" resonates strongly with IoT developers and privacy-conscious applications.

---

### 3.6 Sensory TrulyHandsfree

**Summary:** Commercial embedded wake word engine used in commercial IoT products. Less developer-facing than Picovoice.

**Strengths:**
- Very low power (MCU-optimized)
- Proven in consumer products (used by various OEMs)
- High accuracy for single wake word use cases

**Weaknesses:**
- Enterprise licensing only. No self-serve.
- Not developer-accessible (no PyPI, no GitHub public SDK)
- Proprietary, closed training
- Primarily for MCU/embedded (not Python developer use case)

**Takeaway for V1:** Not a direct competitor for Python SDK market. If ViolaWake ever adds C library + MCU support, Sensory becomes relevant.

---

## 4. Market Gaps We Exploit

### Gap 1: Open-core wake word with accessible training (CRITICAL)

**The problem:** Porcupine's training requires their Console (paid at commercial scale). openWakeWord's training requires ML expertise. No solution exists that is: (a) open-source, (b) high accuracy, (c) easy for non-ML developers to use.

**Our answer:** `violawake-train --word "jarvis" --positives data/ --output models/jarvis.onnx` — 20 samples, one CLI command.

### Gap 2: Bundled voice pipeline (HIGH VALUE)

**The problem:** Every existing solution provides detection only. Developers must manually integrate VAD, STT, TTS, and wire together callbacks. This is the 20-line boilerplate every voice developer writes from scratch.

**Our answer:** `VoicePipeline(wake_word="viola", stt_model="base", tts_voice="af_heart")` — complete pipeline in one class.

### Gap 3: Production accuracy metric with evaluation tool (DIFFERENTIATOR)

**The problem:** Picovoice doesn't publish accuracy metrics. openWakeWord uses informal benchmarks. Developers can't compare wake word engines on a common metric, or evaluate their custom-trained models objectively.

**Our answer:** `violawake-eval` produces Cohen's d, FAR, FRR, and ROC AUC on any model + test set. We publish our current internal score (Cohen's d 15.10 on synthetic negatives) and note that speech-negative d-prime is still TBD.

### Gap 4: Stranded users from deprecated projects (ACQUISITION CHANNEL)

**The problem:** Snowboy (shut down 2020), Mycroft Precise (archived 2022) — thousands of developers are using dead projects that don't work on modern Python/platforms.

**Our answer:** Migration guides from Snowboy and Mycroft Precise to ViolaWake. These developers are already sold on "open source, Python, offline" — they just need a maintained alternative.

---

## 5. V1 Feature Set — Competitive Justification

The following features are included in V1 because competitive analysis shows they are either:
- **Table stakes** (every competitor has them, we must match)
- **Differentiators** (we can win on these, and they fill a real gap)

| Feature | Competitive Justification | Priority |
|---------|--------------------------|----------|
| `WakeDetector` with a strong published benchmark and transparent methodology | Honest benchmark publication is required before we claim an accuracy win | **P0** |
| `violawake-train` CLI (20 samples → ONNX) | Fills gap: open training accessible to non-ML devs | **P0** |
| `violawake-eval` with Cohen's d / FAR / FRR metrics | Unique: no competitor provides this for custom models | **P0** |
| `VADEngine` (WebRTC/Silero/RMS) | Table stakes: Porcupine has Cobra, we bundle ours | **P1** |
| `STTEngine` (Whisper wrapper) | Bundled pipeline gap; Porcupine charges separately for Cheetah | **P1** |
| `TTSEngine` (Kokoro-82M) | Unique: Apache-2.0, on-device, Porcupine charges for Orca | **P1** |
| `VoicePipeline` (bundled Wake→VAD→STT→TTS) | Unique: no competitor provides full pipeline in one object | **P1** |
| Windows + Linux + macOS support | Table stakes: must match Porcupine's desktop coverage | **P1** |
| Raspberry Pi (ARM) | Table stakes: critical for IoT/home automation market | **P1** |
| 4-gate decision policy | Production-hardened: openWakeWord has none | **P1** |
| Python 3.10+ first-class | Table stakes for Python developer market | **P0** |
| Apache 2.0 license | Required to compete with Porcupine's commercial pricing | **P0** |
| PyPI package | Table stakes: `pip install violawake` | **P0** |

**Explicitly out of V1 (justified by competitive landscape):**

| Feature | Justification for deferral |
|---------|---------------------------|
| Android/iOS SDKs | Big investment; Porcupine-level platform coverage is not needed to win Python market |
| MCU/embedded targets | Sensory territory; requires C library (ADR-003 says no) |
| Multi-language | Porcupine has 9 languages; we start with English and win on quality |
| Streaming STT | Cheetah architecture; batch Whisper is sufficient for voice command use case |
| Speaker recognition | Eagle territory; different use case (transcription/call center) |
| Web/WASM | High value but Phase 2; JavaScript SDK needs separate architecture work |
| REST API | Phase 2; cloud deployment model, different from embedded SDK value prop |

---

## 6. Pricing & Business Model Comparison

| Solution | Free Tier | Commercial | Enterprise |
|----------|-----------|------------|------------|
| **ViolaWake SDK** | Apache 2.0, unlimited | $0 (open core) | TBD — support/Console planned |
| Porcupine | Non-commercial only | $6K+/yr Foundation | Custom |
| openWakeWord | Apache 2.0, unlimited | $0 | N/A |
| Mycroft Precise | Apache 2.0, unlimited | $0 (archived) | N/A |
| Snowboy | Archived | Archived | N/A |
| Amazon Alexa Voice | AVS ecosystem only | Royalties + certification | Custom |
| Sensory | Not public | Enterprise licensing | Custom |

**Our model:** Open core — SDK is free forever (Apache 2.0). Revenue comes from:
1. **Custom wake word Console** (web UI for training, Phase 2) — charge per trained model or monthly subscription
2. **Enterprise support** (SLAs, integration help, custom training services)
3. **Managed cloud training** (users without GPU can submit samples → trained model returned)

This directly attacks Porcupine's model: they charge for what we give away for free, and the commercial value we charge for (Console, enterprise support) is transparent and optional.

---

## 7. Summary: Where We Win, Where We Lag

### We win clearly:
1. **Benchmark transparency** — our current Cohen's d 15.10 synthetic-negative result is documented and reproducible, but we still need speech-negative benchmarking before making cross-vendor accuracy claims.
2. **Price** — $0 commercial use beats Porcupine's $6K+/year
3. **Open training** — full training pipeline, inspectable, reproducible, open-source
4. **Bundled pipeline** — no competitor ships Wake+VAD+STT+TTS in one package
5. **Evaluation tooling** — Cohen's d / FAR / FRR eval CLI is unique in this space
6. **Production-tested** — running in Viola production, not a demo

### We lag currently:
1. **Platform coverage** — Porcupine has 12+ platforms; we have Python desktop + Pi
2. **Language support** — Porcupine has 9 languages; we have English only
3. **Brand/community** — We're new; openWakeWord has community traction
4. **Mobile** — No Android/iOS in V1 (Phase 3)
5. **Console UX** — Porcupine's web Console is polished; our training is CLI-only in V1
6. **Pre-trained models** — openWakeWord ships many pre-trained words; we ship one

### The bet:
Python developers who want a wake word SDK, don't want to pay Picovoice prices, and need better accuracy than openWakeWord are our core V1 user. We win that user. Everything else is Phase 2+.

---

*Next review: After V1 PyPI release. Update pricing data if Picovoice announces pricing changes.*
