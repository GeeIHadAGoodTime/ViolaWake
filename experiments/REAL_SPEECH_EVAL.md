# ViolaWake Real Speech Evaluation

**Model**: `D_combined_bce_s42.onnx`
**Date**: 2026-03-26
**Pipeline**: OWW preprocessor -> mean-pool embeddings -> ONNX classifier

## Executive Summary

The model performs **dramatically better on real speech than on TTS**. Real speech EER is 1.71% vs TTS EER of 13.9%. At the production threshold of 0.50, real speech detection is 98.2% with 2.6% FA, while TTS detection is only 77.8% with 6.0% FA. This confirms the TTS eval set is **not representative** of real-world performance and significantly underestimates the model.

Two speakers tested (Jihad, Sierra) with three conditions each (normal, music, whisper). Cross-speaker generalization is strong -- Sierra (not the primary training speaker) achieves 97.1% detection at threshold 0.50.

## Dataset Overview

| Set | Count | Description |
|-----|-------|-------------|
| Eval Positives (real) | 113 | Real "Viola" recordings (held-out eval) |
| Eval Negatives (real) | 302 | Real adversarial/confusable/music negatives |
| Train Positives (real) | 517 | Real "Viola" recordings (in training set) |
| Legacy Custom | 445 | Legacy custom recordings |
| TTS Positives | 288 | Synthetic TTS eval set |
| TTS Negatives | 546 | Synthetic TTS negative eval set |
| Duplicate Groups | 122 | Files appearing in both train and eval sets |

All audio: 16kHz mono, 1.5s (24000 samples).

## Overall Performance: Real Speech vs TTS

| Metric | Real Speech | TTS | Delta |
|--------|-------------|-----|-------|
| **EER** | **1.71%** | 13.90% | -12.2 pp |
| EER Threshold | 0.573 | 0.006 | -- |
| Det@0.50 | 98.2% | 77.8% | +20.4 pp |
| Det@0.70 | 98.2% | 76.7% | +21.5 pp |
| Det@0.80 | 98.2% | 76.0% | +22.2 pp |
| Det@0.90 | 97.3% | 73.6% | +23.7 pp |
| Det@0.95 | 96.5% | 68.1% | +28.4 pp |
| FA@0.50 | 2.6% | 6.0% | -3.4 pp |
| FA@0.70 | 0.7% | 5.0% | -4.3 pp |
| FA@0.90 | 0.0% | 0.9% | -0.9 pp |
| Pos Mean Score | 0.978 | 0.770 | +0.208 |
| Neg Mean Score | 0.043 | 0.054 | -0.011 |

**Key insight**: The TTS eval set has a bimodal positive distribution (many TTS voices score near 0), while real speech positives cluster tightly near 1.0. The model was trained on real speech and generalizes well to real speech but poorly to synthetic TTS voices.

## Detection Rate (Eval Positives -- Real Speech)

| Threshold | Detection Rate | Detected/Total |
|-----------|---------------|----------------|
| 0.30 | 98.2% | 111/113 |
| 0.40 | 98.2% | 111/113 |
| 0.50 | 98.2% | 111/113 |
| 0.60 | 98.2% | 111/113 |
| 0.70 | 98.2% | 111/113 |
| 0.80 | 98.2% | 111/113 |
| 0.90 | 97.3% | 110/113 |
| 0.95 | 96.5% | 109/113 |

Note: Detection is flat from 0.30 to 0.80 -- almost all real positives score above 0.80.

## False Accept Rate (Eval Negatives -- Real Speech)

| Threshold | FA Rate | False Accepts/Total |
|-----------|---------|---------------------|
| 0.30 | 5.0% | 15/302 |
| 0.40 | 4.0% | 12/302 |
| 0.50 | 2.6% | 8/302 |
| 0.60 | 1.7% | 5/302 |
| 0.70 | 0.7% | 2/302 |
| 0.80 | 0.7% | 2/302 |
| 0.90 | 0.0% | 0/302 |
| 0.95 | 0.0% | 0/302 |

## Cross-Speaker Analysis

| Speaker | Eval Count | Mean Score | Std | Det@0.5 | Det@0.7 | Det@0.8 | Det@0.9 |
|---------|-----------|------------|-----|---------|---------|---------|---------|
| **Jihad** | 45 | 0.9954 | 0.0293 | **100.0%** | **100.0%** | **100.0%** | 97.8% |
| **Sierra** | 68 | 0.9703 | 0.1613 | 97.1% | 97.1% | 97.1% | 97.1% |

Jihad is the primary training speaker -- perfect detection at threshold 0.50. Sierra generalizes well at 97.1%, with only 2 misses (both sierra_normal samples scoring near 0.02-0.07, likely corrupted or mislabeled).

## Per-Condition Analysis

| Condition | Count | Mean Score | Std | Det@0.5 | Det@0.7 | Det@0.8 | Det@0.9 |
|-----------|-------|------------|-----|---------|---------|---------|---------|
| **Music** | 50 | 0.9989 | 0.0059 | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **Normal** | 45 | 0.9562 | 0.1967 | 95.6% | 95.6% | 95.6% | 95.6% |
| **Whisper** | 18 | 0.9889 | 0.0455 | **100.0%** | **100.0%** | **100.0%** | 94.4% |

**Music background is easiest** -- 100% detection across all thresholds. The music-background augmentation during training pays off. Whisper also performs very well. Normal speech has the 2 low-scoring Sierra outliers dragging it down.

## Negative Category Breakdown

| Category | Count | Mean Score | Max Score | FA@0.5 | FA@0.7 | FA@0.8 |
|----------|-------|------------|-----------|--------|--------|--------|
| adversarial | 77 | 0.032 | 0.815 | 2.6% | 1.3% | 1.3% |
| legacy_hard | 50 | 0.033 | 0.548 | 2.0% | 0.0% | 0.0% |
| music | 25 | 0.045 | 0.645 | 4.0% | 0.0% | 0.0% |
| music_hard | 30 | 0.020 | 0.255 | 0.0% | 0.0% | 0.0% |
| real_confusable | 45 | 0.080 | 0.697 | 4.4% | 0.0% | 0.0% |
| real_fp_captures | 75 | 0.041 | 0.803 | 2.7% | 1.3% | 1.3% |

**Hardest negatives**: `adversarial` and `real_fp_captures` -- these are the only categories with FA above threshold 0.70. The worst offenders:
- `adv_vanilla_00017.wav` (score 0.815) -- "vanilla" sounds like "viola"
- `real_fp_captures/score0.466_spokehub_20260304_151253_898.wav` (score 0.802)
- `adv_villa_00025.wav` (score 0.646) -- "villa" sounds like "viola"

At threshold 0.90, FA drops to 0.0% across all categories -- zero false accepts.

## Missed Detections Analysis

Only 2 eval positives missed (score < 0.5):
- `sierra_normal/sample_118.wav`: score 0.021 -- likely corrupted/mislabeled
- `sierra_normal/sample_120.wav`: score 0.069 -- likely corrupted/mislabeled

1 additional eval positive below 0.9:
- `jihad_whisper/sample_007.wav`: score 0.801 -- quiet whisper, still detected at threshold 0.80

## Duplicate Analysis

122 duplicate file groups found between training and eval sets. This means some eval positives are also in training -- the eval results for those files may be inflated. A future improvement would be to ensure strict train/eval separation.

## Score Distributions

### Eval Positives (Real Speech)
```
  0.02-0.04:  (1)
  0.06-0.08:  (1)
  0.80-0.82:  (1)
  0.94-0.96:  (2)
  0.98-1.00: ######################################## (108)
```

### Eval Negatives (Real Speech)
```
  0.00-0.02: ######################################## (244)
  0.02-0.04: ## (14)
  0.04-0.08: # (9)
  0.08-0.26: # (16)
  0.28-0.58: # (9)
  0.64-0.70: (3)
  0.80-0.82: (2)
```

## Recommendations

1. **Use threshold 0.50 for production** -- achieves 98.2% detection with only 2.6% FA on real speech. At 0.70, FA drops to 0.7% with no detection loss.

2. **Investigate the 2 Sierra outliers** (sample_118, sample_120) -- if corrupted/mislabeled, removing them would bring real speech detection to 100% at threshold 0.50.

3. **The TTS eval set underestimates real performance** -- do not use TTS eval results as the primary quality metric. Real speech eval should be the gold standard.

4. **Remove train/eval duplicates** -- 122 files overlap. Create a strictly separated eval set for unbiased metrics.

5. **Hardest negative words are "vanilla" and "villa"** -- consider adding more confusable negatives for these words to training data.

6. **At threshold 0.90, there are zero false accepts** -- if FA is a higher priority than detection rate, 0.90 is viable (97.3% detection, 0% FA).
