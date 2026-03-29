# ViolaWake Business Plan

> **Document Type:** Living strategic plan
> **Last Updated:** 2026-03-28
> **Status:** V2 - aligned to shipped product

---

## 1. Executive Summary

ViolaWake is an **open-source wake word SDK** with a **shipped web Console** for collecting recordings, training custom models, and downloading ONNX artifacts. The important business truth is that the Console already exists in-repo today, but its **reference deployment is local-first**, not fully cloud-managed.

**Current product reality:**
- **SDK ships now**: wake detection, VAD, STT, TTS, evaluation tooling, and local training CLI
- **Console ships now**: React frontend + FastAPI backend + real recording/upload/training/download flows
- **Current backend stack**: local email/password auth with bcrypt + JWT, SQLite by default, local filesystem storage by default, CPU training jobs executed through the app's async queue/background workers
- **Not current reality**: Supabase auth, Modal GPU training, Cloudflare R2-only storage, or support-tier operations

**The business thesis:** keep detection free and open, then monetize convenience around managed training, hosted infrastructure, and commercial support once the hosted Console is hardened.

**One-line positioning:** *Train a custom wake word from your browser, keep the SDK free, and stay in control of the model file you ship.*

---

## 2. Current State

### 2.1 What Ships Today

| Component | Status | What exists now |
|-----------|--------|-----------------|
| **ViolaWake SDK** | Shipped | Python SDK for wake detection, VAD, STT, TTS, evaluation, and model download |
| **Training CLI** | Shipped | Local CLI for collecting samples, training custom wake words, and evaluating models |
| **Console Frontend** | Shipped | React/Vite app for account creation, recording, training progress, and model download |
| **Console Backend** | Shipped | FastAPI API for auth, recording upload, queued training jobs, SSE progress, model management |
| **Quota/Billing Code Paths** | Partial | Free / Developer / Business tier limits and Stripe checkout routes exist in code, but all current training runs through the same local CPU queue |
| **Hosted SaaS Hardening** | Not shipped | No production-only managed stack should be described as baseline reality yet |

### 2.2 Current Architecture

The current Console is a **working product**, not a Phase 2 mockup. Its reference implementation is:

```
Browser (React/Vite)
        |
        v
FastAPI backend
  - email/password auth
  - bcrypt password hashing
  - local JWT access tokens
  - SSE training progress
        |
        +--> SQLite database (default)
        +--> local disk storage for recordings/models (default)
        +--> async job queue + background CPU training workers
```

Important distinction: the backend includes optional hooks for Stripe billing, Cloudflare R2 storage, and external databases, but those are **integration paths**, not the current default architecture.

### 2.3 Model Footprint

ViolaWake should describe wake-word size precisely:

- **Wake head:** `102 KB` (`temporal_cnn`)
- **Shared OpenWakeWord backbone:** `1.33 MB`
- **Total runtime wake-word footprint:** **`1.43 MB`**

That is the correct current runtime story. Older `2.1 MB` language referred to a deprecated unreleased `viola_mlp_oww` artifact and should not be used in customer-facing business docs.

### 2.4 Current Commercial State

The SDK is genuinely free and Apache 2.0 licensed today. The Console is shipped and usable today. The monetization layer is **partially implemented**:

- The backend already models **Free / Developer / Business** subscription tiers
- Monthly training limits are enforced in code as **3 / 20 / unlimited**
- Stripe checkout/webhook routes exist, but billing only works when Stripe is configured
- Higher-tier users (developer, business) get priority queue placement for training jobs
- There is **no differentiated support operation** yet

This means the correct business framing is:

1. **Current product:** free SDK + working Console + local-first deployment story
2. **Near-term monetization:** hosted managed Console with honest limits and billing
3. **Later expansion:** enterprise support, team features, and managed infrastructure upgrades

---

## 3. Compliance And Trust Status

### 3.1 What We Can Honestly Claim Today

- The **SDK runs on-device** and does not require cloud inference
- The Console supports **manual deletion** of recordings and models through the application/API
- Passwords are hashed with **bcrypt**
- API auth uses **local JWT tokens**
- The default reference deployment stores recordings and models on **local disk**

### 3.2 What We Cannot Claim Yet

The privacy/legal copy elsewhere in the product was written against a **target architecture**, not the current shipped baseline. The business plan should be explicit about that gap:

> **Compliance status:** the current Console implementation uses local storage and local-first infrastructure by default. Any wording about Cloudflare R2 encryption, Modal-based training, or analytics pipelines describes a target hosted architecture, not the reference deployment that ships today. Automatic retention cleanup is now active.

### 3.3 Risk Posture

Current gaps to close before strong hosted-service privacy claims:

- **Automated retention cleanup:** shipped — recordings deleted after 90 days, models after 365 days
- **Hosted storage guarantees:** not baseline, because local filesystem storage is still the default
- **Analytics disclosures:** no product analytics story should be promised unless instrumentation is actually enabled
- **Hosted compliance review:** still needed before presenting the Console as a production SaaS with legal-grade guarantees

This honesty is strategically useful. Developers tolerate early infrastructure constraints. They do not tolerate inaccurate privacy claims.

---

## 4. Market Positioning

### 4.1 The Gap We Target

The wake-word market still has a clear pricing and usability gap:

- **Picovoice / Porcupine:** polished and proven, but expensive and closed
- **openWakeWord:** open and free, but training UX and production polish remain pain points
- **ViolaWake:** open SDK, shipped Console, inspectable training path, and direct ownership of the final ONNX model

The strongest current positioning is not "we already have hyperscale SaaS infrastructure." It is:

1. **The Console exists now**
2. **The SDK is open now**
3. **The model artifact is portable now**
4. **The hosted business layer is being built on top of a working product, not a slide deck**

### 4.2 Customer Segments

| Segment | Pain Point | Current fit |
|---------|------------|-------------|
| **Indie developers** | Picovoice pricing is too high for experimentation | Strong fit now |
| **Startups building voice UX** | Need custom wake words without a bespoke ML team | Strong fit now |
| **Open-source / self-hosting users** | Want inspectable tooling and local control | Strong fit now |
| **Small commercial teams** | Want a hosted easy-button without enterprise pricing | Emerging fit; needs hosted hardening |
| **Enterprise / fleet deployments** | Need SLAs, team controls, compliance guarantees | Roadmap, not current |

### 4.3 Why The Shipped Console Matters

The business story improves materially now that the Console is real:

- We are no longer selling only a CLI workflow
- We can demonstrate the full browser-based training loop end to end
- We can price against managed convenience instead of pricing against a hypothetical future UI
- We can collect real feedback on conversion, training quality, and onboarding friction before investing in heavier cloud infrastructure

---

## 5. Current State Vs Roadmap

| Area | Current state | Roadmap |
|------|---------------|---------|
| **Auth** | Local email/password + bcrypt + JWT | Supabase or another managed auth provider if hosted scale justifies it |
| **Database** | SQLite by default | Managed Postgres for production multi-instance deployments |
| **Storage** | Local disk by default | R2 or another object store for durable hosted storage |
| **Training** | CPU jobs through local async queue/background workers | Optional GPU-backed managed training after demand is proven |
| **Queueing** | Priority queues shipped: free=0, developer=5, business=10 | Per-user rate limits and burst capacity controls |
| **Accounts** | Single-user accounts + team management (shipped) | Org-level admin, SSO, and SCIM provisioning later |
| **Support** | Community / direct founder support | Formal support tiers and SLAs later |
| **Retention** | Automatic cleanup active: recordings 90 days, models 365 days | Configurable per-tier retention policies |
| **Hosting** | Local/self-hosted reference flow | Fully managed hosted Console |

---

## 6. Roadmap

### 6.1 Next 90 Days

- Align all customer-facing copy to the current local-first Console architecture
- Ship a clean hosted beta using the existing Console flows
- Validate real conversion from free quota to paid quota
- [x] Retention cleanup shipped — privacy language now accurate
- [x] Priority queues shipped — higher tiers get faster queue placement
- Prove benchmark quality on stronger real-world negative sets

### 6.2 3-6 Months

- Move hosted storage from local disk to object storage
- Add managed production database for hosted deployments
- Introduce queue observability and honest capacity controls
- Decide whether GPU training is economically justified for paid tiers

### 6.3 6-12 Months

- Team workspace UI (the backend API is shipped; frontend controls are next)
- Priority queues once differential infrastructure actually exists
- Enterprise support packaging and SLAs
- Hosted compliance posture strong enough to make stricter privacy claims

---

## 7. Revenue Strategy

### 7.1 Near-Term Monetization

The most defensible near-term model is:

- **Free SDK forever**
- **Free Console quota for evaluation**
- **Paid hosted convenience**, not paid inference licensing

That keeps us aligned with developer expectations while preserving a clean upgrade path.

### 7.2 Honest Pricing Narrative

Current truthful positioning:

- The repository already supports **free**, **developer**, and **business** quota concepts
- Those tiers currently differ primarily by **usage entitlement**, not by distinct infrastructure classes
- Claims about **GPU acceleration**, **priority queues**, **team admin**, or **premium support operations** belong in roadmap copy until shipped

### 7.3 12-Month Goal

The business target remains reasonable: prove that a free/open SDK plus a paid convenience layer can produce sustainable recurring revenue without undermining developer trust. The near-term success metric is not vanity MRR at any cost. It is **conversion without overclaiming**.

---

## 8. Risks

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| **Narrative drift** | Docs and product pages can outrun the code | Keep `Current State` and `Roadmap` separate everywhere |
| **Privacy overclaim** | Legal/trust damage if target architecture is described as current | Treat hosted privacy language as roadmap until implemented |
| **Hosted infra cost creep** | GPU/storage upgrades can arrive before demand exists | Validate paid demand on the CPU queue first |
| **Accuracy skepticism** | Developers will challenge synthetic-only benchmark framing | Publish stronger speech-negative benchmarking |
| **Competitive response** | Picovoice/openWakeWord can close the UX gap | Win on openness, portability, and honest developer experience |

---

## 9. Metrics To Track

### Product

- Console signup to first-training conversion
- First model trained per new account
- Training completion rate
- Model download rate after completed training
- Free-to-paid conversion once hosted billing is live

### Technical

- Queue wait time on the shared CPU backend
- Training completion time
- Failure rate by tier
- Storage growth and retention backlog
- Benchmark quality on real-world negative sets

### Trust

- Number of docs/pages with architecture mismatches
- Time-to-correct for narrative drift
- Privacy/compliance TODO count

---

## 10. Strategic Bottom Line

ViolaWake no longer needs to pretend the Console is a future idea. It is already built. The real business challenge is now narrower and more credible:

1. Tell the truth about the shipped local-first Console
2. Use that real product to validate paid demand
3. Add managed infrastructure only where usage justifies it
4. Preserve trust by keeping privacy, pricing, and architecture claims tied to implementation reality

That is a stronger business narrative than the previous version because it is both more ambitious **and** more believable.
