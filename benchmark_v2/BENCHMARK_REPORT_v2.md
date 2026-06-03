## ViolaWake vs OpenWakeWord -- Corrected Benchmark v2

### Reproduction

- Script: `python benchmark_v2/reproduce_claims.py --benchmark-dir benchmark_v2`
- Model: `temporal_cnn` version `0.1.0`, SHA-256 `9c0b12c68593cfdb3d320a3b34667913b18d63e89eb01247d6332d7839ac9efe`
- Shared negative score corpus: 700 files
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
| ROC AUC | 0.9877 | 0.9574 |
| FAR @ FRR=1% | 10.00% | 15.00% |
| FAR @ FRR=3% | 6.71% | 12.00% |
| FAR @ FRR=5% | 5.43% | 8.86% |
| FAR @ FRR=10% | 4.14% | 8.14% |
| FRR @ FAR=0.1% | 61.11% | 75.56% |
| FRR @ FAR=0.5% | 45.00% | 75.56% |
| FRR @ FAR=1.0% | 28.33% | 75.56% |
| FRR @ FAR=5.0% | 5.56% | 40.56% |

### Per-Category FAR/FRR

Per-category FAR is computed at the global threshold selected for the target FRR.

| System | Negative category | N | FAR @ FRR=5% | FAR @ FRR=10% |
|--------|-------------------|---:|-------------:|--------------:|
| ViolaWake | adversarial_alexa | 105 | 0.00% | 0.00% |
| ViolaWake | adversarial_viola | 105 | 7.62% | 4.76% |
| ViolaWake | noise | 20 | 0.00% | 0.00% |
| ViolaWake | speech | 200 | 13.50% | 10.50% |
| ViolaWake | speech_existing | 270 | 1.11% | 1.11% |
| OpenWakeWord | adversarial_alexa | 105 | 56.19% | 53.33% |
| OpenWakeWord | adversarial_viola | 105 | 0.00% | 0.00% |
| OpenWakeWord | noise | 20 | 10.00% | 5.00% |
| OpenWakeWord | speech | 200 | 0.50% | 0.00% |
| OpenWakeWord | speech_existing | 270 | 0.00% | 0.00% |

### Analysis

**ViolaWake has lower EER** (5.49% vs 8.24%), indicating better overall discrimination.

ViolaWake has higher AUC (0.9877 vs 0.9574).

### Context

- OWW's `alexa` model: pre-trained by David Scripka on a larger real-speech corpus
- ViolaWake's `viola` model: temporal CNN on OWW embeddings, TTS-trained
- Both evaluated on TTS audio only (no real recordings in this benchmark)
- Adversarial negatives included for both systems
- Negatives do not contain either actual wake word
