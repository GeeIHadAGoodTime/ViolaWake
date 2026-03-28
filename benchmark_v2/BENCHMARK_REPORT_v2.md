## ViolaWake vs OpenWakeWord -- Corrected Benchmark v2

### Methodology
- Shared negative corpus: 700 files
  - adversarial_alexa: 105 files
  - adversarial_viola: 105 files
  - noise: 20 files
  - speech: 200 files
  - speech_existing: 270 files
- Matched positives: 180 viola, 180 alexa
- Same 20 Edge TTS voices, same 3 augmentations (clean, noisy, reverb)
- Streaming inference: 1280-sample chunks (80ms at 16kHz), max-score per file
- Primary metrics: EER, FAR@FRR

### Results

| Metric | ViolaWake (viola) | OWW (alexa) |
|--------|-------------------|-------------|
| EER | 5.49% | 8.24% |
| ROC AUC | 0.9877 | 0.9555 |
| FAR @ FRR=1% | 10.00% | 15.00% |
| FAR @ FRR=3% | 6.71% | 12.00% |
| FAR @ FRR=5% | 5.43% | 8.86% |
| FAR @ FRR=10% | 4.14% | 8.14% |
| FRR @ FAR=0.1% | 61.11% | 89.44% |
| FRR @ FAR=0.5% | 45.00% | 89.44% |
| FRR @ FAR=1.0% | 28.33% | 89.44% |
| FRR @ FAR=5.0% | 5.56% | 40.56% |

### Score Distributions

| Statistic | ViolaWake (viola) | OWW (alexa) |
|-----------|-------------------|-------------|
| Pos mean +/- std | 0.9009 +/- 0.0645 | 0.9044 +/- 0.2123 |
| Pos median [IQR] | 0.9190 [0.8844-0.9432] | 0.9954 [0.9530-1.0000] |
| Pos range | [0.6120, 0.9785] | [0.0013, 1.0000] |
| Neg mean +/- std | 0.1755 +/- 0.2470 | 0.0878 +/- 0.2658 |
| Neg median [IQR] | 0.0565 [0.0076-0.2408] | 0.0000 [0.0000-0.0003] |
| Neg range | [0.0000, 0.9512] | [0.0000, 1.0000] |

### Per-Phrase Breakdown

| Phrase | VW Score (mean +/- std) | OWW Score (mean +/- std) |
|--------|------------------------|--------------------------|
| Standalone word | 0.9270 +/- 0.0359 (n=60) | 0.9984 +/- 0.0124 (n=60) |
| "hey [word]" | 0.8905 +/- 0.0689 (n=60) | 0.9099 +/- 0.1873 (n=60) |
| "ok [word]" | 0.8851 +/- 0.0734 (n=60) | 0.8050 +/- 0.2849 (n=60) |

### Adversarial Resistance

How each system scores on the OTHER system's adversarial words:

| Adversarial Set | VW mean score | OWW mean score |
|-----------------|---------------|----------------|
| Viola-confusables (n=105) | 0.2609 +/- 0.2550 | 0.0006 +/- 0.0045 |
| Alexa-confusables (n=105) | 0.0611 +/- 0.0646 | 0.5585 +/- 0.4464 |

### Analysis

**ViolaWake has lower EER** (5.49% vs 8.24%), indicating better overall discrimination.

ViolaWake has higher AUC (0.9877 vs 0.9555).

### Context

- OWW's 'alexa' model: pre-trained by David Scripka on large real-speech corpus
- ViolaWake's 'viola' model: temporal CNN on OWW embeddings, TTS-trained
- Both evaluated on TTS audio only (no real recordings in this benchmark)
- Adversarial negatives included for BOTH systems (v1 only had viola adversarials)
- Negatives do NOT contain either actual wake word
