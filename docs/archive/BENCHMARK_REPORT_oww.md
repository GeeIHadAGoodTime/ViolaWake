> **ARCHIVED 2026-04-05.** Historical MLP-era benchmark. Production model is now `temporal_cnn` (d'=8.577). See `docs/PROVEN_TRAINING_RECIPE.md`.

# Wake Word Detection Benchmark: ViolaWake vs OpenWakeWord

## Setup

- **ViolaWake**: `viola_mlp_oww.onnx` (mean-pool MLP on OWW embeddings) detecting **"viola"**
  - Positives: 272 TTS-generated samples (trained phrases: viola, hey viola, ok viola)
  - Negatives: 330 samples (adversarial confusables + common speech + noise)
- **OpenWakeWord**: pre-trained `alexa` model detecting **"alexa"**
  - Positives: 180 TTS-generated samples (alexa, hey alexa, ok alexa)
  - Negatives: 546 samples (same corpus as ViolaWake)

**Methodology**: Same 20 Edge TTS voices (en-US, en-GB, en-AU, en-IN, en-ZA, en-IE, en-CA),
same augmentations (clean + noisy + reverb), same negative corpus, same metrics.
Each system evaluated on its own best wake word -- this is NOT about detecting 'viola' with OWW.

## Results at Threshold 0.50

| Metric | ViolaWake (viola) | OWW (alexa) |
|--------|-------------------|-------------|
| **Cohen's d** | **4.14** | **6.14** |
| ROC AUC | 0.9871 | 0.9840 |
| EER | 0.0631 | 0.0212 |
| FAR | 3.9% (13/330) | 2.0% (11/546) |
| FRR | 8.8% (24/272) | 2.8% (5/180) |
| Precision | 0.9502 | 0.9409 |
| Recall | 0.9118 | 0.9722 |
| F1 | 0.9306 | 0.9563 |

## Multi-Threshold Results

| Threshold | VW FAR | VW FRR | VW F1 | OWW FAR | OWW FRR | OWW F1 |
|-----------|--------|--------|-------|---------|---------|--------|
| 0.30 | 6.1% | 7.0% | 0.928 | 2.0% | 2.2% | 0.959 |
| 0.50 | 3.9% | 8.8% | 0.931 | 2.0% | 2.8% | 0.956 |
| 0.70 | 3.3% | 11.8% | 0.918 | 1.8% | 6.1% | 0.942 |
| 0.80 | 1.8% | 15.8% | 0.903 | 1.8% | 12.8% | 0.905 |
| 0.90 | 0.3% | 26.1% | 0.848 | 1.8% | 16.7% | 0.882 |

## Score Distributions

| Statistic | ViolaWake (viola) | OWW (alexa) |
|-----------|-------------------|-------------|
| Pos mean | 0.8784 | 0.9291 |
| Pos std | 0.2311 | 0.1590 |
| Pos min | 0.0003 | 0.0092 |
| Pos max | 1.0000 | 1.0000 |
| Neg mean | 0.0497 | 0.0205 |
| Neg std | 0.1641 | 0.1363 |
| Neg min | 0.0000 | 0.0000 |
| Neg max | 0.9437 | 1.0000 |

## Bootstrap Confidence Intervals (10,000 resamples)

| System | Cohen's d | 95% CI |
|--------|-----------|--------|
| ViolaWake (viola) | 4.14 | [3.55, 4.98] |
| OWW (alexa) | 6.14 | [5.00, 8.11] |

## Per-Phrase Breakdown: OWW (alexa)

| Phrase | N | Mean Score | Pass @ 0.50 |
|--------|---|-----------|-------------|
| alexa | 60 | 0.9987 | 60/60 (100.0%) |
| hey alexa | 60 | 0.9215 | 58/60 (96.7%) |
| ok alexa | 60 | 0.8672 | 57/60 (95.0%) |

## Per-Phrase Breakdown: ViolaWake (viola)

| Phrase | N | Mean Score | Pass @ 0.50 |
|--------|---|-----------|-------------|
| viola | 103 | 0.7894 | 86/103 (83.5%) |
| hey viola | 85 | 0.9283 | 83/85 (97.6%) |
| ok viola | 84 | 0.9370 | 79/84 (94.0%) |

## Analysis

**OpenWakeWord wins on separability.** OWW's Cohen's d (6.14) is 1.5x higher than ViolaWake's (4.14).

ROC AUC: ViolaWake 0.9871 vs OWW 0.9840.
EER: ViolaWake 0.0631 vs OWW 0.0212.

### Context

- OWW's "alexa" model was trained by Amazon/David Scripka on a large corpus of real speech.
- ViolaWake's "viola" model is a custom MLP trained on OWW's embedding features with TTS-generated data.
- Both are evaluated here on TTS-generated audio only (no real recordings).
- The negative corpus contains phonetically adversarial words for ViolaWake (vanilla, villa, violet, etc.)
  but NOT adversarial words for OWW (e.g., no 'alexis', 'election', 'electric'). This gives OWW a slight
  advantage on FAR since the negatives weren't designed to trick it.
