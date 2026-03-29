<!-- doc-meta
scope: Architecture decision — licensing and business model
authority: ADR — immutable once accepted
code-paths: LICENSE, pyproject.toml, README.md
staleness-signals: VC funding changes strategy, revenue model proves unworkable, major competitor goes fully open-source
-->

# ADR-004: Open-Core Licensing Strategy

**Status:** Accepted
**Date:** 2026-03-17
**Authors:** ViolaWake team
**Supersedes:** N/A

---

## Context

We need a licensing model that:
1. Allows the developer community to freely use, modify, and redistribute the SDK
2. Differentiates from Picovoice's fully proprietary model
3. Maintains a viable path to commercial revenue without undermining open-source adoption
4. Is compatible with all dependencies (OWW, ONNX Runtime, Kokoro — all Apache 2.0)

The core tension: **Free enough to win developer trust and adoption. Differentiated enough to support a business.**

---

## Decision

**Apache 2.0 for all SDK code, models, and training pipeline. Commercial differentiation comes from the hosted managed Console service, not from reclassifying the SDK.**

This is the "open core" model:
- **Open source (Apache 2.0):** Everything that runs on the developer's machine — inference code, training code, training data pipeline, pre-trained models, documentation
- **Commercial:** The hosted web Console service that provides managed training, storage, and account/billing convenience

---

## Rationale

### Option A: Fully Open Source (MIT/Apache, no commercial layer) — Rejected

**Pros:**
- Maximum adoption
- No license friction
- Easy to integrate into commercial products

**Cons:**
- No revenue path
- We are a funded development team, not an academic lab — sustainability matters
- Fully free and fully open means we compete only on quality, not on services
- Hard to fund ongoing maintenance, model improvements, and documentation

**Why rejected:** Sustainability requires a revenue path. A fully free SDK with no commercial layer means we depend on donations or VC funding to maintain quality. openWakeWord is essentially this model — great community traction but limited resources for production-hardening.

### Option B: Proprietary (Picovoice model) — Rejected

**Pros:**
- Clear monetization
- Can charge for all use

**Cons:**
- Direct contradiction of our differentiation story (we are the "open alternative to Picovoice")
- Black-box models prevent fine-tuning — one of our core selling points
- Community won't form around a proprietary product
- Developer advocates won't write tutorials about proprietary tools

**Why rejected:** This is what we're competing against. Adopting the same model makes us "another Picovoice" with no structural advantage.

### Option C: SSPL / Commons Clause (rejected)

MongoDB (SSPL) and others add restrictions on SaaS use of open source software.

**Pros:**
- Prevents competitors from wrapping our SDK in a cloud service and competing with us
- Some protection of commercial interests

**Cons:**
- SSPL is not OSI-approved — widely rejected by enterprise legal teams
- Creates friction with developers who worry about license compliance
- The developer community correctly views Commons Clause as "not really open source"
- Likely to backfire: drives developers to alternatives that are truly Apache/MIT

**Why rejected:** License friction is adoption friction. The developer community's trust is worth more than protection from a hypothetical SaaS competitor.

### Option D: Open Core (Apache 2.0 core + proprietary Console) — Chosen

This is the model used by HashiCorp (Terraform), Elastic (pre-2021), Confluent (Kafka), and many successful infrastructure companies.

**What's Apache 2.0 (free forever):**
- All inference code
- Training pipeline (trainer.py, augmentation, evaluation)
- Pre-trained models (`viola_mlp_oww.onnx`, `kokoro-v1.0.onnx`)
- VoicePipeline, WakeDetector, TTSEngine, STTEngine, VADEngine
- All documentation
- All negative training data and data collection tooling
- The Cohen's d / FAR / FRR evaluation framework
- All GitHub-hosted tooling

**What's commercial (hosted SaaS layer):**
- The managed training Console service: upload your positive samples via web UI, receive a trained ONNX model. No Python environment needed.
- Model hosting and version management
- Support SLAs for enterprise users
- The "high-speed lane" for model training (GPU cluster vs local CPU)

**Why the Console specifically?**
- The Console solves the real friction point: developers want custom wake words without managing Python ML environments
- It's a service, not a feature — Apache 2.0 training code + managed training service coexist without conflict
- This mirrors Picovoice's Console, which is their primary lock-in mechanism — we undercut them by making the Console cheaper while making the underlying tech open

---

## Implementation

**License file:** Apache 2.0, placed at `LICENSE`. This covers all files in the repository unless explicitly excluded.

**Attribution requirements (Apache 2.0 notices):**
```
Copyright 2026 ViolaWake Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
...
```

**Third-party licenses:**
- OpenWakeWord backbone: Apache 2.0 — compatible, requires attribution (in NOTICE file)
- Kokoro-82M model: Apache 2.0 — compatible, requires attribution
- ONNX Runtime: MIT — compatible, no restrictions
- faster-whisper: MIT — compatible
- webrtcvad: BSD-3 — compatible

**No CLAs for Phase 1.** Contributor License Agreements add friction. For Phase 1, contributors accept that their contributions are Apache 2.0 via the repository's license. If we need CLA later (for commercial licensing of contributed code), we'll add it then.

---

## Commercial Boundaries

**Will NEVER be commercial:**
- Core inference code
- Training pipeline
- Pre-trained ViolaWake models
- The evaluation framework

**May be commercial as hosted offerings:**
- Managed training Console service (the training itself still uses open-source code)
- Enterprise support tiers
- Custom model training services

**Explicitly NOT prohibited by Apache 2.0:**
- Companies using ViolaWake SDK in commercial products (permitted)
- SaaS products built on top of ViolaWake (permitted — not using SSPL/Commons Clause)
- Forking ViolaWake and shipping proprietary improvements (permitted, with attribution)

---

## Consequences

**Positive:**
- Developer trust: Apache 2.0 is recognized and trusted by enterprise legal teams
- No friction for commercial adoption — our target users (Python devs) can use it in commercial products without asking legal
- Community formation: truly open tools attract contributors, tutorials, and integration guides
- Compatible with all dependencies
- Competitive positioning: directly attacks Picovoice's lock-in on model training

**Negative:**
- No protection if a well-resourced company forks and builds a better managed service
- Revenue depends on turning the shipped Console into a credible hosted managed service
- Apache 2.0 attribution requirements must be tracked for all downstream model files

**Risks:**
- "Open core creep" — the temptation to move features from open source to commercial erodes developer trust. We must be disciplined: the local training CLI stays open forever.
- The Console may not generate sufficient revenue. If so, fallback is consulting/custom-model-training services or enterprise support contracts.

---

## Revisit Criteria

This decision should be revisited if:
- A major company forks ViolaWake and operates a direct competitor managed Console at scale (action: add a CLA and negotiate cross-licensing; consider SSPL for Console-adjacent code)
- Apache 2.0 proves incompatible with a major distribution channel (unlikely)
- Business model requires moving training CLI to commercial (must NOT happen — this is our core differentiation)
