<!-- doc-meta
scope: Competitive landscape - wake word engines, voice SDKs, and adjacent markets
authority: LIVING - updated when competitors ship major changes or pricing shifts
code-paths: docs/PRD.md, console/, src/violawake_sdk/
staleness-signals: Picovoice pricing changes, openWakeWord major workflow changes, ViolaWake hosted architecture changes
last-updated: 2026-03-28
-->

# ViolaWake Competitive Analysis

**Version:** 1.1
**Status:** Active

---

## 1. Market Position

The wake-word market still breaks into three practical buckets:

| Segment | Main players | ViolaWake position |
|---------|--------------|-------------------|
| **Commercial proprietary** | Picovoice, Sensory | Compete on openness and price |
| **Open-source tooling** | openWakeWord, archived Mycroft Precise, archived Snowboy | Compete on productization and managed workflow |
| **Big-tech ecosystem SDKs** | Alexa / Google / Apple stacks | Not direct competitors |

ViolaWake now sits in a clearer position than earlier docs suggested:

- **The SDK is real**
- **The Console is real**
- **The hosted infrastructure story is still maturing**

That means the right comparison is no longer "CLI-only OSS project vs polished vendors." It is "open SDK plus shipped Console MVP vs closed managed platforms."

---

## 2. Current ViolaWake Offer

### 2.1 What We Have Today

- Open-source Python SDK for wake detection, VAD, STT, and TTS
- Local training and evaluation CLI
- Shipped React/FastAPI Console for signup, recording, queued training, and model download
- Local-first backend architecture: JWT auth, SQLite by default, local disk by default, CPU training queue by default

### 2.2 What We Do Not Have Yet

- Cloud-hardened managed auth as the baseline
- GPU-backed paid training lanes
- Enterprise support/SLA operations

This distinction matters competitively because it changes the product story from "future web UI" to "real Console with early-stage infrastructure."

---

## 3. Feature Matrix

| Feature | **ViolaWake** | Picovoice Porcupine | openWakeWord | Mycroft Precise | Snowboy |
|---------|:-------------:|:-------------------:|:------------:|:---------------:|:-------:|
| **License** | Apache 2.0 SDK | Proprietary | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Commercial SDK use** | Free | Paid | Free | Archived | Archived |
| **Training code open** | Yes | No | Yes | Yes | No |
| **Web Console** | **Yes, shipped MVP** | Yes | No broadly adopted managed Console | No | No |
| **Training without ML setup** | **Yes: shipped Console + CLI** | Yes | Partial / more manual | No | N/A |
| **REST API** | **Yes, shipped for Console backend** | Yes | No | No | No |
| **Portable ONNX output** | Yes | No | Yes | Partial | No |
| **Bundled VAD/STT/TTS** | Yes | Separate products | No | No | No |
| **Current default backend story** | Local-first/self-hostable | Managed commercial platform | OSS toolkit | Archived | Archived |
| **Team features** | Yes (create, invite, roles, model sharing) | Yes / enterprise | No | No | No |
| **Priority infra tiers** | Yes (4-tier system) | Yes | No | No | No |

---

## 4. Competitor Deep Dives

### 4.1 Picovoice Porcupine

**What they do well**

- Best-in-class commercial polish
- Wide platform coverage
- Mature managed Console
- Strong developer onboarding

**Why developers still look elsewhere**

- Closed training stack
- Proprietary artifacts
- Commercial pricing and vendor lock-in

**Our current position vs Picovoice**

- We now have a real Console, which removes the old "CLI-only" weakness
- Their hosted offering is still much more mature than ours
- Our advantage remains openness, portability, and a lower-cost business model

**Competitive takeaway**

ViolaWake should position against Picovoice as the **open, portable, developer-controlled alternative with a real browser workflow**, not as a hypothetical future Console.

### 4.2 openWakeWord

**What they do well**

- Strong OSS credibility
- ONNX-based workflow
- Community adoption
- Useful baseline for DIY training

**Where ViolaWake is stronger**

- Shipped Console for a full browser workflow
- More productized path from samples to downloaded model
- Broader bundled voice stack around wake-word use cases

**Where openWakeWord is still strong**

- OSS mindshare
- Simpler expectation set: it does not pretend to be a managed SaaS

**Competitive takeaway**

openWakeWord is still the closest OSS peer, but ViolaWake now competes as **open-source plus productized onboarding**, not merely as another training CLI.

### 4.3 Archived Alternatives

Mycroft Precise and Snowboy still matter as search and migration channels, but they are no longer feature competitors. Their importance is strategic:

- they left orphaned users behind
- those users already want open, local, and inspectable tooling
- ViolaWake can be their modern landing spot

---

## 5. Where ViolaWake Wins

### 5.1 Open SDK Plus Real Console

This is the biggest narrative correction.

ViolaWake is no longer forced into the weakest possible position of "great training code, but you still need the CLI." The product now includes a working browser-based recording and training flow, which materially improves its competitiveness against both Picovoice and ad-hoc OSS workflows.

### 5.2 Portable Ownership

The exported model is still an ONNX artifact the user can keep. That remains a differentiator against closed managed platforms.

### 5.3 Honest Architecture Story

There is an underappreciated advantage in being explicit:

- the Console exists today
- it currently runs on local-first infrastructure
- managed cloud hardening is still roadmap work

That is more credible than pretending hosted sophistication that is not yet the default.

---

## 6. Where ViolaWake Still Lags

| Area | Current gap |
|------|-------------|
| **Hosted maturity** | Picovoice is much more production-hardened as a managed platform |
| **Platform coverage** | ViolaWake remains primarily Python/desktop/Pi oriented |
| **Infra differentiation** | No dedicated GPU training lanes yet |
| **Support operations** | No enterprise-grade support motion yet |

This is the correct gap analysis now. The gap is **not** "no Console." The gap is "Console exists, but the managed service layer is early."

---

## 7. Pricing And Business Comparison

| Solution | Free SDK / OSS use | Managed training story | Current business reality |
|----------|--------------------|------------------------|--------------------------|
| **ViolaWake** | Yes | Shipped Console MVP | Quotas exist; infra is still shared/local-first |
| Picovoice | Limited free use, then paid | Mature managed Console | Fully commercialized |
| openWakeWord | Yes | Mostly DIY | Community / OSS |

Important nuance for ViolaWake:

- The product can honestly talk about a **Console business**
- It cannot honestly talk about **different infra classes per paid tier** yet (priority queues exist but GPU lanes do not)
- Pricing copy should center on **convenience, quotas, and priority queues**, not on GPU lanes or support promises that do not exist

---

## 8. Strategic Implications

### 8.1 Product Strategy

ViolaWake should keep pushing the hybrid position:

1. open SDK
2. portable ONNX output
3. shipped browser-based training flow
4. gradual move toward a managed hosted service

### 8.2 Messaging Strategy

The strongest message is now:

> **ViolaWake is the open alternative with a real Console, not just another wake-word training script.**

### 8.3 Execution Strategy

The next competitive unlock is not "invent a Console." It is:

- harden hosting
- add retention/compliance hygiene
- validate quota-based monetization
- add premium team/infra features only after usage proves they are needed

---

## 9. Bottom Line

Earlier docs described ViolaWake as if it were still in the gap between a CLI project and a future hosted product. That is outdated.

The accurate competitive story is:

- **Console shipped**
- **SDK shipped**
- **managed SaaS maturity still in progress**

That is a much stronger position than the previous narrative because it is both more competitive and more believable.
