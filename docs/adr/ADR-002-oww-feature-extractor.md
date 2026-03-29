<!-- doc-meta
scope: Architecture decision — wake word feature extraction backbone
authority: ADR — immutable once accepted
code-paths: src/violawake_sdk/wake_detector.py, src/violawake_sdk/audio.py
staleness-signals: OWW license change, new dominant audio embedding model, our CNN v4 significantly outperforms OWW-MLP
-->

# ADR-002: Use OpenWakeWord Embeddings as Feature Extractor Backbone

**Status:** Accepted
**Date:** 2026-03-17
**Authors:** ViolaWake team
**Supersedes:** N/A

---

## Context

The ViolaWake wake word detector has two architectural layers:
1. **Feature extraction:** Convert raw audio (16kHz PCM) → feature vector
2. **Classification:** Feature vector → wake word probability score

For feature extraction, we have three options: train a custom feature extractor from scratch, use mel spectrograms with hand-crafted features (PCEN), or use a pre-trained audio embedding model as a frozen backbone.

The original decision compared **two model families**:
- `viola_v1–v4.onnx`: Custom CNN (3 conv layers, 28K params) trained on mel+PCEN features — in-house
- `viola_mlp_oww.onnx`: Early MLP head on frozen OpenWakeWord (OWW) 96-dim audio embeddings

The architectural decision remains current even though the shipped default wake head later moved to `temporal_cnn`. The core choice is unchanged: **ViolaWake uses a frozen OWW backbone plus a ViolaWake-owned wake head.**

On our current synthetic-negative benchmark, the MLP+OWW approach achieves **Cohen's d 15.10** vs **Cohen's d 3.07** for the CNN model. This is a large internal separability improvement, but it is not the same as a real-world speech-negative d-prime comparison. This ADR documents why we adopt OWW as the standard backbone for the SDK.

---

## Decision

**Use OpenWakeWord (OWW) as the fixed audio feature extractor backbone for wake word detection. Train only the MLP classification head, not the feature extractor.**

The OWW backbone is frozen at inference time. The ViolaWake SDK uses the OWW feature extractor plus a separate ViolaWake wake head. Both are required for detection.

---

## Rationale

### Option A: From-scratch CNN on mel+PCEN features (rejected as primary)

This is the `viola_v1–v4` approach.

**Pros:**
- 100% in-house — no external backbone dependency
- Smaller model (28K params vs OWW's 4M backbone)
- No license dependencies beyond our own

**Cons:**
- Cohen's d 3.07 vs 15.10 on the synthetic-negative benchmark — a large internal gap
- Requires large negative dataset to train competitive feature extractor
- Requires significant audio augmentation expertise to avoid overfitting
- We validated this approach through v4 iterations and it plateaued around Cohen's d ~8 on the same synthetic-negative benchmark before OWW integration

**Why rejected:** The internal benchmark gap is large enough that the CNN path is clearly the weaker current option. We should not frame the `15.10` number as directly competitive with Porcupine until we have speech-negative benchmarking, but it is still strong evidence that the OWW backbone is the better internal baseline. The SDK's primary value proposition is accuracy, so the feature extractor decision directly determines whether we have a compelling product.

**Kept as secondary:** `viola_v4.onnx` (CNN, Cohen's d 8.2 on the same synthetic-negative benchmark) is retained in the model registry as a "lightweight" option for heavily resource-constrained deployments (Pi Zero, etc.) that cannot afford OWW's 4M param feature extractor. But it is NOT the recommended/default path.

### Option B: Custom Wav2Vec2 or HuBERT backbone (rejected)

Large pre-trained speech models (Wav2Vec2, HuBERT) produce high-quality audio embeddings.

**Pros:**
- Potentially even higher accuracy than OWW
- Well-understood and widely used in research

**Cons:**
- Model size: Wav2Vec2-base is 360MB+. The current shared OWW backbone runtime asset is about 1.33 MB. Wav2Vec2 is not appropriate for an SDK targeting Pi-class devices.
- License: Wav2Vec2 is Apache 2.0 (acceptable), but inference overhead is prohibitive
- Overkill for binary wake word classification — we don't need full speech understanding, just keyword fingerprint

### Option C: OpenWakeWord backbone (chosen)

OpenWakeWord is itself a pre-trained audio embedding model designed specifically for wake word detection. It produces 96-dimensional audio embeddings at 8Hz (one embedding per 125ms frame).

**Pros:**
- Specifically designed for wake word detection — the embeddings encode acoustically-relevant features for short-duration keyword detection
- Small: the current shared OWW backbone runtime asset is about 1.33 MB
- Apache 2.0 license — compatible with our own Apache 2.0 SDK
- We have production validation of the architecture in Viola, and the current reference model scores Cohen's d 15.10 on the synthetic-negative benchmark
- Training overhead is minimal: we only train the small MLP head (~50K params) instead of the full model
- Custom wake words can be trained with fewer positive samples because OWW backbone already understands general audio structure

**Cons:**
- Dependency: our "original model" claim is partial — the backbone is OWW, the MLP head is ours
- License risk: OWW changing its license would break ours (mitigated: Apache 2.0 is irrevocable)
- OWW's embedding quality ceiling: if OWW's embeddings fail to capture the distinctive acoustic features of a particular wake word, we can't compensate in the MLP alone (mitigated: works for "Viola" + all tested custom words so far)
- Inference pipeline: two model passes per frame (OWW embedding → MLP score) vs one for CNN

**Why chosen:** The 5x accuracy improvement over our best in-house CNN directly determines product viability. We have production evidence this works. The dependency is stable (Apache 2.0) and the model is small enough for target platforms.

---

## Implementation

The inference pipeline per 20ms audio frame:

```python
# Step 1: Extract OWW audio embeddings (runs every 8 OWW frames = 1s)
oww_input = preprocess_audio_frame(raw_pcm_16khz)  # mel-scale features
oww_embedding = oww_session.run(None, {"input": oww_input})[0]  # (1, 96)

# Step 2: MLP classification
score = mlp_session.run(None, {"embedding": oww_embedding})[0][0]  # float

# Step 3: Decision policy
if score >= threshold and not in_cooldown and not is_playing:
    trigger_wake_word_callback()
```

**Two runtime model assets are loaded:**
- `oww_backbone` (`1.33 MB`) — OWW feature extractor (frozen)
- ViolaWake wake head — current default `temporal_cnn` is `102 KB`, for a combined runtime footprint of **1.43 MB**

---

## Disclosure

We disclose the OWW dependency clearly in:
1. `README.md` — "The ViolaWake MLP model uses OpenWakeWord as a fixed feature extractor backbone (Apache 2.0). The MLP classification head and training pipeline are original ViolaWake work."
2. `LICENSE` — Attribution of OWW backbone
3. `docs/PRD.md` — Technical dependency table

We do NOT claim the feature extractor is original ViolaWake work. Our original contributions are:
- The MLP classification head architecture and training
- The training pipeline (FocalLoss, EMA, SWA, augmentation)
- The 4-gate decision policy (zero-input guard, score threshold, cooldown, listening gate)
- The Cohen's d / FAR / FRR evaluation infrastructure
- The SDK packaging and API design

---

## Consequences

**Positive:**
- The current reference model has a strong internal synthetic-negative score (Cohen's d 15.10), though direct comparison to Porcupine still requires speech-negative benchmarking
- MLP training runs on CPU, doesn't require GPU for custom wake word development
- OWW backbone handles audio preprocessing — less custom code to maintain

**Negative:**
- Marketing must accurately describe the OWW dependency
- Custom wake word training requires OWW's data format for positive samples
- If OWW is abandoned or goes unmaintained, we'd need to train our own feature extractor (fallback: CNN path is maintained)

---

## Revisit Criteria

This decision should be revisited if:
- OWW changes its license to non-Apache terms (action: switch to custom CNN or Wav2Vec2-tiny)
- A new backbone (Wav2Vec2-tiny, EfficientAudio, etc.) achieves better benchmark separation at comparable size
- Our CNN model (via continued training) reaches Cohen's d ≥ 13.0 on the synthetic-negative benchmark and also holds up on speech-negative evaluation
- A Phase 2 WASM build reveals that OWW's ONNX doesn't compile to WASM cleanly (action: CNN path for browser build)
