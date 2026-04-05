> **ARCHIVED 2026-04-05.** Clean eval results for MLP models (d'=4.14). Production `temporal_cnn` achieves d'=8.577.

# ViolaWake Clean Eval Set Results

**Date**: 2026-03-26
**Eval set**: `eval_clean/` (generated via `tools/build_clean_eval_set.py`)
**Zero training overlap**: All samples are freshly synthesized TTS -- no audio from any training corpus.

## Executive Summary

The prior claimed d-prime of 11.56 was measured on a contaminated evaluation set (samples that overlapped with or were very similar to training data). On a clean eval set with zero training overlap, the production mean-pool model achieves **d-prime 4.14** on trained phrases -- approximately **2.8x lower** than claimed. The model still works (91% recall at threshold 0.50, AUC 0.87), but the published metrics were significantly inflated by data contamination.

## Eval Set Composition

| Category | Count | Description |
|----------|-------|-------------|
| Positives (trained phrases) | 272 | "viola", "hey viola", "ok viola" across 18+ TTS voices + reverb/noisy augmentations |
| Positives (untrained phrases) | 84 | "viola wake up", "viola please" -- phrases the model was NOT trained on |
| Total positives | 356 | 20 edge-tts voices (US, GB, AU, IN, ZA, IE, CA) + 1 pyttsx3 voice |
| Negatives (confusable words) | 320 | "vanilla", "villa", "violet", "vinyl", etc. + common speech + silence/noise |
| Total negatives | 330 | 10 TTS voices + procedurally generated noise |

**Voices**: 20 Edge TTS voices spanning en-US, en-GB, en-AU, en-IN, en-ZA, en-IE, en-CA accents, plus 1 Windows SAPI5 voice (David Desktop). All voices are neural TTS, providing diverse speaker characteristics.

## Full Results

### Mean-Pool Model (Production: `viola_mlp_oww.onnx`)

| Metric | All Phrases | Trained Phrases Only | Prior Claim |
|--------|------------|---------------------|-------------|
| **D-prime (Cohen's d)** | **2.07** | **4.14** | **11.56** |
| 95% CI | [1.83, 2.35] | [3.55, 4.98] | -- |
| Standard Error | 0.131 | 0.363 | -- |
| AUC | 0.868 | -- | 0.9996 |
| FRR @ 0.50 | 28.1% | 8.8% | 0% |
| FP @ 0.50 | 13/330 | 13/330 | 6/302 |
| Pos mean score | 0.694 | 0.878 | -- |
| Neg mean score | 0.050 | 0.050 | -- |
| Optimal threshold | 0.39 | -- | 0.63 |

### Maxpool Model (`viola_mlp_oww_maxpool.onnx`)

| Metric | All Phrases | Trained Phrases Only | Prior Claim |
|--------|------------|---------------------|-------------|
| **D-prime (Cohen's d)** | **1.62** | **2.70** | **3.07** |
| 95% CI | [1.45, 1.81] | [2.43, 3.04] | -- |
| Standard Error | 0.091 | 0.156 | -- |
| AUC | 0.853 | -- | -- |
| FRR @ 0.50 | 84.0% | 79.0% | -- |
| FP @ 0.50 | 0/330 | 0/330 | -- |
| Pos mean score | 0.304 | 0.394 | -- |
| Neg mean score | 0.036 | 0.036 | -- |
| Optimal threshold | 0.15 | -- | -- |

## Comparison: Clean vs Contaminated Measurements

| Metric | Clean Eval (Mean-Pool, trained) | Prior Contaminated Eval | Ratio |
|--------|--------------------------------|------------------------|-------|
| D-prime | 4.14 | 11.56 | 2.8x inflation |
| AUC | 0.87 | 0.9996 | -- |
| FRR @ 0.50 | 8.8% | 0% | -- |
| N positives | 272 | 113 | -- |
| N negatives | 330 | 302 | -- |

The d-prime of 11.56 in the prior eval was inflated because the evaluation set contained samples that were closely related to training data (same speakers, same recording conditions, or near-duplicate audio). On genuinely unseen TTS voices, the separability drops substantially.

## Per-Phrase Breakdown (Mean-Pool Model)

| Phrase | N | Mean Score | Pass @ 0.50 | Status |
|--------|---|-----------|-------------|--------|
| ok viola | 84 | 0.937 | 79/84 (94.0%) | Trained -- best performance |
| hey viola | 84 | 0.928 | 82/84 (97.6%) | Trained -- strong |
| viola (standalone) | 102 | 0.788 | 85/102 (83.3%) | Trained -- moderate |
| viola please | 18 | 0.440 | 8/18 (44.4%) | NOT trained |
| viola wake up | 66 | 0.005 | 0/66 (0.0%) | NOT trained -- completely fails |

The model performs best on "hey viola" and "ok viola" (carrier phrase + wake word). Standalone "viola" is harder -- 17% of samples are false rejects. Extended phrases the model was not trained on ("viola wake up", "viola please") perform poorly or not at all, which is expected.

## Per-Voice Breakdown (Mean-Pool, Trained Phrases)

Best voices (by mean score on edge-tts samples with 12 clips each):
| Voice | Mean | Pass % |
|-------|------|--------|
| en-US-BrianNeural | 0.749 | 75% |
| en-AU-WilliamNeural | 0.748 | 75% |
| en-US-EmmaNeural | 0.742 | 75% |
| en-GB-SoniaNeural | 0.733 | 75% |
| en-US-AndrewNeural | 0.724 | 75% |
| en-CA-ClaraNeural | 0.723 | 75% |

Weakest voice:
| Voice | Mean | Pass % |
|-------|------|--------|
| en-IE-EmilyNeural | 0.231 | 25% |

The Irish English accent (Emily) is the hardest for the model -- likely because training data had no Irish English speakers. The model generalizes best to American and British accents.

## Most Confusable Negative Words (Mean-Pool)

| Word | Mean Score | Max Score | False Positives @ 0.50 |
|------|-----------|----------|----------------------|
| vanilla | 0.494 | 0.856 | 4 |
| villa | 0.425 | 0.899 | 3 |
| vienna | 0.170 | 0.740 | 1 |
| vinyl | 0.162 | 0.820 | 1 |
| vital | 0.142 | 0.770 | 1 |
| viper | 0.132 | 0.944 | 1 |
| vocal | 0.097 | 0.755 | 1 |
| video | 0.096 | 0.766 | 1 |

"Vanilla" and "villa" are the most confusable words -- their phonetic overlap with "viola" (v-i-l-a pattern) causes consistent false positives. "Viper" has the single highest false positive score (0.944), meaning it nearly perfectly mimics "viola" in at least one voice.

Non-speech negatives (silence, noise, hum, clicks) all score near zero -- the model correctly ignores non-speech audio.

## Threshold Sweep (Trained Phrases, Mean-Pool)

| Threshold | FRR | FP | Precision | Recall | F1 |
|-----------|-----|-----|-----------|--------|-----|
| 0.30 | 7.0% | 20 | 0.927 | 0.930 | 0.928 |
| 0.40 | 7.7% | 14 | 0.947 | 0.923 | 0.935 |
| **0.50** | **8.8%** | **13** | **0.950** | **0.912** | **0.931** |
| 0.60 | 9.6% | 13 | 0.950 | 0.904 | 0.927 |
| 0.70 | 11.8% | 11 | 0.956 | 0.882 | 0.918 |
| **0.80** | **15.8%** | **6** | **0.974** | **0.842** | **0.903** |
| 0.90 | 26.1% | 1 | 0.995 | 0.739 | 0.848 |

The current production threshold of 0.80 (set after a false-positive flood) gives:
- 15.8% FRR (roughly 1 in 6 wake word attempts rejected)
- 6 false positives per 330 negatives
- Good precision (0.974) but moderate recall (0.842)

## Bootstrap Confidence Intervals

All confidence intervals computed via 10,000 bootstrap resamples (seed=42).

| Model | D-prime | 95% CI | SE |
|-------|---------|--------|-----|
| Mean-Pool (all phrases) | 2.07 | [1.83, 2.35] | 0.131 |
| Mean-Pool (trained phrases) | 4.14 | [3.55, 4.98] | 0.363 |
| Maxpool (all phrases) | 1.62 | [1.45, 1.81] | 0.091 |
| Maxpool (trained phrases) | 2.70 | [2.43, 3.04] | 0.156 |

The prior claim of 11.56 falls **far** outside the 95% CI of the clean measurement (upper bound 4.98). This is statistically conclusive evidence that the prior evaluation was inflated.

## Honest Assessment

### What the clean eval tells us

1. **The model works** -- d-prime 4.14 on trained phrases is genuinely useful. The mean-pool production model achieves 91% recall at threshold 0.50 with only 13 false positives out of 330 adversarial negatives. This is a functional wake word detector.

2. **The prior claims were inflated by ~2.8x** -- d-prime 11.56 was measured on a contaminated evaluation set. The true separability on unseen data is d-prime 4.14. The AUC of 0.9996 was also inflated (clean: 0.87).

3. **The mean-pool model is significantly better than maxpool** -- d-prime 4.14 vs 2.70. The maxpool model also has a much lower optimal threshold (0.15), making it nearly unusable at the default threshold of 0.50 (79% FRR).

4. **Accent generalization is weak** -- Irish English (en-IE-EmilyNeural) has only 25% pass rate. The model was trained predominantly on American English voices.

5. **Phonetic confusability is a real problem** -- "vanilla" and "villa" are genuine false positive risks. A production deployment should either raise the threshold to 0.80+ or add a second-pass verifier.

### What should change in public-facing claims

- The README, Show HN draft, and CLAUDE.md all cite d-prime 11.56. This should be updated to **4.14** (with a note about the eval methodology).
- The AUC of 0.9996 should be updated to **0.87** or removed.
- The "FRR 0% at optimal threshold" claim should be removed -- clean eval shows 8.8% FRR at 0.50.
- The claim "100% recall, 95% precision" should be updated to reflect the clean eval numbers.

### Caveats about this eval

- This eval uses TTS voices only (no real human speech). TTS has different characteristics than real microphone recordings. The true production d-prime on real speech may differ in either direction.
- The eval set is still synthetic -- a real-world eval would include background noise, far-field recordings, and diverse real speakers.
- The "viola wake up" and "viola please" phrases were NOT part of the model's training vocabulary. Their inclusion in the "all phrases" d-prime (2.07) unfairly penalizes the model. The "trained phrases" d-prime (4.14) is the fair comparison.

## Files

| File | Description |
|------|-------------|
| `scores_meanpool.csv` | Per-file scores for mean-pool model |
| `scores_maxpool.csv` | Per-file scores for maxpool model |
| `results_meanpool.json` | Raw results dict (JSON) |
| `results_maxpool.json` | Raw results dict (JSON) |
| `run_eval.py` | Evaluation runner script |
| `analyze_scores.py` | Detailed per-file analysis |
| `analyze_final.py` | Final analysis with trained/untrained split |
| `MANIFEST.md` | Eval set generation manifest |
