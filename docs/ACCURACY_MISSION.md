# ViolaWake Accuracy Mission — Living Document

> **PURPOSE:** This document drives the ViolaWake accuracy improvement campaign.
> It survives context compaction. Read it FIRST after any break.
> **Last updated:** 2026-04-05 — **SUPERSEDED BY TEMPORAL CNN.** Production model is now `temporal_cnn` (TemporalCNN(96,9), d'=8.577, EER 0.8%, AUC 0.9993). The MLP-era accuracy campaign below is historical. The temporal CNN uses 9-frame sliding windows over OWW embeddings instead of mean-pooling, resolving the "feature extractor ceiling" and eliminating the FAPH crisis. See `docs/PROVEN_TRAINING_RECIPE.md` for the canonical pipeline.

---

## Recovery Prompt (copy-paste after compaction)

> You are improving ViolaWake's wake word detection (wake word is **"Viola"**, NOT "Hey Viola"). Project: `J:\CLAUDE\PROJECTS\Wakeword`.
> Read `ACCURACY_MISSION.md` for full context. Current state:
>
> **WHERE WE ARE — TEMPORAL CNN ERA:**
> 1. **Production model: `temporal_cnn.onnx`** (TemporalCNN(96,9), 25,409 params, d'=8.577, EER 0.8%)
> 2. **8-phase training pipeline proven reproducible** — "big chungus" from 20 recordings → Grade A
> 3. **SDK inference path FIXED** — now uses real OWW 2-model pipeline (melspectrogram + embedding_model)
> 4. **Multi-window confirmation in SDK** — 3-of-3 consecutive above threshold
> 5. **CLI/Console/standalone use identical pipeline** — verified 2026-04-05
> 6. **Old MLP (r3_10x_s42) DEPRECATED** — failed live mic test (max score 0.50)
>
> **DEAD ENDS (do not revisit):**
> - 864-dim temporal concat: 4x worse FAPH (11.84 vs 2.96). Distribution shift.
> - Pure focal loss: destroys TP. Only "hardonly" variant works.
> - oww_backbone.onnx: never existed. OWW uses melspectrogram.onnx + embedding_model.onnx internally.
> - Real speech eval: 100% contaminated (training duplicates). Use held-out pos_backup instead.
>
> **STILL OPEN:**
> - Real mic TP validation (TTS eval is 57% but that's non-English artifacts)
> - Statistical confidence: 2 triggers in 5.4h → Poisson 95% CI [0.045, 1.34]
> - 2 surviving sustained FAs from speakers 5639 and 908 (phonetic confusables)
> - PyPI packaging and release
> - FAPH spot-check comparing SDK direct pipeline vs embed_clips (running)
>
> **KEY FILES:**
> - `experiments/models/j5_temporal/temporal_cnn.onnx` — production model (d'=8.577)
> - `experiments/train_temporal_j5.py` — experiment that produced the proven model
> - `experiments/j5_temporal_results.json` — full metrics across 4 architectures x 3 seeds
> - `docs/PROVEN_TRAINING_RECIPE.md` — canonical training pipeline documentation
> - `src/violawake_sdk/wake_detector.py` — SDK (rewritten 2026-03-27)
> - `experiments/META_ANALYSIS.md` — 30 patterns from MLP era (historical)
>
> **PRINCIPLES:**
> 1. Wake word is "Viola" (one word, no prefix)
> 2. PROPOSE-CHALLENGE-SYNTHESIZE before any >30min compute plan
> 3. Mine existing data before downloading new corpora
> 4. Measurement must match production (debounce, buffer boundaries)
> 5. Always check TP rate after mining — not all seeds survive
> 6. Report debounced FAPH as primary metric, raw as secondary
> 7. See `experiments/META_ANALYSIS.md` for 30 documented patterns

---

## Goal

### Primary Metrics (what actually matters for shipping)

| Metric | Baseline (D_bce) | **Production (r3_10x_s42 + 3-of-3)** | Target | Status |
|--------|------------------|--------------------------------------|--------|--------|
| **FAPH (test-clean, 5.4h)** | ~105 @0.5 raw | **0.37 @0.80 (2 triggers in 5.4h)** | <1/hour | **ACHIEVED** |
| **Held-out TP @0.80** | 99.1% | **96.9%** (pos_backup, 1067 samples) | >95% | **ACHIEVED** |
| **Real speech detection** | ~~98.2%~~ INVALID | TBD (need real mic test) | >95% | OPEN |
| **Confusable rejection (TTS)** | voila=0.59 | TBD | all <0.3 | DEPRIORITIZED |
| **Confusable rejection (real)** | max=0.697, 0% FA @0.7 | TBD | all <0.3 | CLOSE |

### Secondary Metrics (diagnostic)

| Metric | Value | Notes |
|--------|-------|-------|
| TTS trained_eer | 2.35% | Trained phrases on TTS eval — good |
| TTS all_eer | 13.14% | Includes untrained "viola wake up" — misleading |
| Real speech AUC | ~0.99 | Excellent separation |
| Duplicate train/eval groups | 122 | Potential data leak — investigate |

**Ship criteria:** FAPH < 1/hour AND real detection > 95% AND real confusable max < 0.3.

### CRITICAL: Real Speech Eval Set is 100% Contaminated (2026-03-26T22:30)

**ALL 415 eval files (113 pos + 302 neg) are byte-identical copies of training files.** Cosine similarity = 1.0000 for every pair. The eval files in `violawake_data/eval_real/` are copies of `violawake_data/positives/` and `violawake_data/negatives/`. Different paths, same audio.

**Impact:**
- 1.71% EER: **INVALID** — model tested on training data
- 98.2% detection: **INVALID** — ditto
- TP eval on faph_hardened_s43: **INVALID** — used same contaminated set
- All per-speaker, per-condition breakdowns: **INVALID**

**What IS still valid:**
- FAPH numbers (LibriSpeech — no overlap with training)
- TTS eval (eval_clean/ — only 1.8% overlap, mostly clean)
- Mining loop effectiveness (measured on FAPH, not eval set)

**What we need:**
- ~~Record NEW "Viola" wake word audio from real speakers NOT in training set~~
- ~~Or: hold out a portion of existing training positives that were NEVER used for training~~

**RESOLVED (22:45):** Found 2,236 clean held-out embeddings in `pos_backup` (1,067) and `pos_excluded` (1,169) — not used in D_combined training, deduplicated against training set (cosine >0.99 removed).

**Held-out detection rates (pos_backup clean, 1,067 embeddings, 355 files):**

| Threshold | Baseline (D_bce_s42) | faph_hardened_s43 | Delta |
|-----------|---------------------|-------------------|-------|
| 0.50 | 99.5% | 99.2% | -0.3% |
| 0.80 | 99.1% | **96.0%** | -3.1% |
| 0.90 | 98.2% | **92.6%** | **-5.6%** |
| 0.95 | 96.4% | **85.5%** | **-10.9%** |

**CRITICAL IMPLICATION:** At threshold 0.9, detection is only 92.4-92.6% — BELOW 95% target. The model can maintain >95% detection only at threshold **≤0.80**. At 0.80, R1 debounced FAPH = 6.31. The real gap to <1 FAPH while maintaining >95% detection is **6.3x.**

**ROUND 2 HELD-OUT RESULTS (22:50):**

| Threshold | Baseline | R1 (faph_hard_s43) | **R2 (round2_best)** |
|-----------|---------|--------------------|--------------------|
| 0.50 | 99.5% | 99.2% | **98.4%** |
| 0.80 | 99.1% | 96.0% | **96.4%** |
| 0.90 | 98.2% | 92.6% | **92.4%** |
| 0.95 | 96.4% | 85.5% | **87.9%** |

R2 matches R1 at the operating point (0.80: 96.4% vs 96.0%). The 2x positive weight protected detection.

**ROUND 2 FAPH RESULTS (23:15) — NO IMPROVEMENT:**

| Threshold | R1 Deb FAPH | R2 Deb FAPH | Change |
|-----------|------------|------------|--------|
| 0.50 | 10.39 | 12.81 | +23% worse |
| 0.80 | 6.31 | **6.12** | -3% (noise) |
| 0.90 | 4.64 | 4.45 | -4% (noise) |
| 0.95 | 2.23 | 2.23 | 0% |

**Adding 3,978 ACAV hard negatives did NOT reduce dev-clean FAPH.** Cross-domain mining (YouTube → audiobook speech) has limited transfer. The dev-clean false alarms are from different speech patterns than ACAV. Some triggers shifted but total count unchanged.

**New pattern: domain-specific mining only helps within that domain.** To reduce dev-clean FAPH, must mine FROM dev-clean (or similar audiobook speech). ACAV mining would help FAPH on YouTube-like audio but not on LibriSpeech.

See `experiments/duplicate_audit_report.json` and `experiments/eval_heldout_check.py` for details.

### CRITICAL FINDING (2026-03-26T18:00): FAPH is catastrophic

**FAPH at threshold 0.5: ~260/hour on LibriSpeech test-clean.**

- 35/40 speakers (87.5%) trigger the detector above 0.5
- Median speaker max score: 0.92 — HALF of speakers trigger above 0.9
- Total: ~132 triggers in ~0.5h audio across all 40 speakers
- Even at threshold 0.95: many speakers still trigger (max scores near 1.0)
- This is NOT a confusable word problem — random English speech triggers the model
- Worst speakers: 908 (15 triggers), 1188 (13), 8224 (12), 3729 (9)
- The model detects "viola" well but doesn't reject non-viola speech well enough

**Root cause hypothesis:** The model's negative training set is predominantly TTS. Real human speech has different embedding distributions than TTS speech. The model learned to distinguish "viola TTS" from "not-viola TTS" but that boundary doesn't generalize to "not-viola real speech".

**UPDATE (18:30):** Tested ALL 40 speakers, then compared ALL trained models including G_acav_bal (100K ACAV real speech negatives, balanced 5:1):

| Model | FAPH@0.5 | FAPH@0.7 | FAPH@0.9 | FAPH@0.95 | Detection@0.7 |
|-------|----------|----------|----------|-----------|---------------|
| D_bce (baseline) | ~42 | ~26 | ~16 | ~12 | 98.2% |
| G_acav_bal (best) | ~34 | ~22 | ~16 | ~10 | 96.5% |

(FAPH measured with 2s debounce on ~0.5h LibriSpeech, 200 files, 40 speakers)

**Even the best model produces ~10 FAPH at threshold 0.95** with 2s debounce. This is 10x above target.

**Root cause confirmed:** The OWW 96-dim embedding space doesn't separate "viola" from arbitrary speech well enough. Some speech phoneme combinations produce embeddings indistinguishable from "viola". This is a **feature extractor ceiling**, not a classifier problem.

**FAPH BREAKTHROUGH (19:00) — then DATA LEAKAGE CONFIRMED (19:30):**

**Round 1 results (test-clean — LEAKED):**

| Model | FAPH@0.5 | FAPH@0.7 | FAPH@0.9 | Det@0.85 (real) | Max LS score |
|-------|----------|----------|----------|-----------------|--------------|
| D_bce (baseline) | 162.1 | 113.3 | 61.4 | 97.3% | 1.00 |
| **faph_hard_s43** | **0** | **0** | **0** | **98.2%** | **0.85** |

**Held-out validation (dev-clean — 2,703 files, 5.39h, different speakers):**

| Model | FAPH@0.5 | FAPH@0.7 | FAPH@0.9 | FAPH@0.95 | Reduction |
|-------|----------|----------|----------|-----------|-----------|
| D_bce (baseline) | 105.1 | 72.0 | 39.7 | 26.0 | — |
| **faph_hard_s43** | **19.3** | **13.9** | **6.7** | **3.3** | **82-87%** |

**Conclusions:**
1. 0 FAPH on test-clean was memorization of the 629 mined windows
2. Real improvement on held-out data: **88% FAPH reduction** (162→19.3 at 0.5)
3. Still **6.7 FAPH at 0.9** raw — but see debounce correction below
4. **The iterative mining approach works but needs more rounds**
5. Feature extractor upgrade is NOT the next step — mining has headroom

**MEASUREMENT CORRECTION (20:30):** faph_test.py had NO debounce. Production uses 2s debounce. Adjacent overlapping windows (100ms step) counted as separate triggers. **ACTUAL DEBOUNCED RESULTS (22:00, full 5.39h):** Debounce reduces raw by ~46% (not 3-4x as estimated — per-file reset limits effect). s43 debounced: **10.39@0.5, 4.64@0.9, 2.23@0.95**. Saved to `experiments/faph_devclean_s43_debounced.json`.

**ACAV GOLDMINE (20:30):** Scored 100K existing ACAV embeddings through round-1 model. Found **3,978 hard negatives** (score > 0.3) — 6x more than the 629 from LibriSpeech. 655 score >0.9. These are diverse real speech (YouTube), not clean read speech. **Already extracted, zero download time.** This was discovered by a challenger agent questioning "why aren't you using existing data?"

**Method:** Mined 629 hard negatives (score > 0.3) from LibriSpeech test-clean, added to training with 10x weight, retrained D_combined + confusable_v2 data. Total training time: 43 seconds.

**ENSEMBLE RESULTS (21:00):** Multi-seed comparison on 500-file dev-clean sample (0.99h) with debounce:

| Model | Deb FAPH@0.5 | Deb FAPH@0.9 | Deb FAPH@0.95 |
|-------|-------------|-------------|--------------|
| s42 | 14.1 | 8.1 | 7.1 |
| s43 | 11.1 | 6.0 | 5.0 |
| s44 | 25.2 | 3.0 | 1.0 |
| **ensemble** | **11.1** | **2.0** | **1.0** |

Ensemble at 0.95 = 1.0 FAPH. BUT: 0.99h is statistically meaningless (95% CI: [0.025, 5.57]). Need full corpus run for real numbers. Seed lottery is severe — s44 alone hits 1.0 at 0.95 but is 25.2 at 0.5. **UPDATE: s44 FAILS TP safety floor (92.9% @0.9) — must exclude from ensemble. Revised ensemble = s42+s43 only (2-seed average).**

**TP EVAL ON HARDENED MODEL (21:45) — GATE PASSED:**

| Model | Det@0.5 | Det@0.8 | Det@0.9 | Det@0.95 | Mean Score |
|-------|---------|---------|---------|----------|------------|
| Baseline (D_bce_s42) | 98.2% | 98.2% | 97.3% | 96.5% | 0.9803 |
| **faph_hardened_s43** | **99.1%** | **98.2%** | **98.2%** | **97.3%** | **0.9875** |
| faph_hardened_s42 | 98.2% | 97.3% | 97.3% | 96.5% | 0.9823 |
| faph_hardened_s44 | 97.3% | 94.7% | 92.9% | 92.0% | 0.9608 |

s43 IMPROVED over baseline at every threshold. s44 fails 95% floor — exclude from ensemble. Mining did NOT degrade detection. Round 2 is safe to proceed (for s42/s43 seeds).

**CHALLENGER v2 FINDINGS (21:30):** Second adversarial review found CRITICAL gaps:
1. **SDK inference path ≠ measurement path** — Production SDK WakeDetector uses `oww_backbone.onnx` with 20ms frames, entirely different from `embed_clips` 1.5s clips used in all training/eval. Our numbers don't apply to the SDK.
2. **No TP eval on hardened model** — faph_hardened_s43 has NEVER been tested on positive detection. 10x-weighted negatives may have degraded wake word sensitivity. Running eval now.
3. **Mean-pooling ceiling hypothesis** — The "OWW ceiling" may actually be a mean-pool ceiling. The 9×96=864-dim temporal embedding preserves phoneme order that mean-pooling destroys. Running experiment now.
4. **Statistical invalidity** — Cannot claim <1 FAPH with useful precision on 5 hours. Need 50-100h.
5. **122 duplicate groups** — STILL unaudited. 10-minute task blocking confidence in all TP metrics.

**Top dev-clean false alarms (next mining targets):**
- 0.9993: speaker 2803, file 154328-0007 @ 3.8s
- 0.9989: speaker 777, file 126732-0066 @ 6.8s
- 0.9951: speaker 1993, file 147966-0002 @ 0.0s
- 0.9925: speaker 6313, file 66125-0011 @ 2.7s

**Next steps (revised after challenger debate — see META_ANALYSIS Pattern 15):**
1. **Get debounced FAPH numbers** — Running now. Reveals actual gap to <1 FAPH (estimated ~2-3, not 6.7)
2. **Ensemble scoring test** — Average s42+s43+s44 outputs for free improvement. Running now.
3. **Round 2 training** — Combine ALL hard negative sources:
   - 629 test-clean hard negs (round 1, already in cache)
   - Dev-clean hard negs at threshold 0.3, deduplicated (cluster overlapping windows, keep peak)
   - **3,978 ACAV hard negs** (score >0.3, already extracted — the goldmine)
   - Retrain 3 seeds with combined data
4. **Validate on test-clean** — Valid because round 2 adds dev-clean + ACAV negatives (test-clean negatives from round 1 were already in the model)
5. **Multi-seed FAPH** — Compare s42/s43/s44 to check seed lottery. Running now.
6. **122 duplicate audit** — Still needed but doesn't block mining loop
7. **If still >1 FAPH**: Mine from Common Voice (diverse accents) or LibriSpeech train-clean-100 (100h)

---

## Real Speech Eval Results (2026-03-26)

**Model:** `D_combined_bce_s42.onnx` | **Pooling:** mean | **113 eval positives, 302 eval negatives**

### Detection Rates (eval positives — real recordings)

| Threshold | Detection Rate | Details |
|-----------|---------------|---------|
| 0.30 | 98.2% | 111/113 |
| 0.50 | 98.2% | 111/113 |
| 0.70 | 98.2% | 111/113 |
| 0.80 | 98.2% | 111/113 |
| 0.90 | 97.3% | 110/113 |
| 0.95 | 96.5% | 109/113 |

### False Accept Rates (eval negatives — real adversarial/confusable)

| Threshold | FA Rate | FAs |
|-----------|---------|-----|
| 0.50 | 2.6% | 8/302 |
| 0.70 | 0.7% | 2/302 |
| 0.80 | 0.7% | 2/302 |
| 0.90 | 0.0% | 0/302 |

### Cross-Speaker Breakdown

| Speaker | Eval Files | Mean Score | Det@0.5 | Det@0.8 | Det@0.9 |
|---------|-----------|------------|---------|---------|---------|
| **Jihad** | 45 | 0.995 | 100% | 100% | 97.8% |
| **Sierra** | 68 | 0.970 | 97.1% | 97.1% | 97.1% |

Sierra's min score = 0.021 (2 files scoring very low — likely corrupted or mislabeled).

### Per-Condition Breakdown

| Condition | Files | Mean | Det@0.5 | Det@0.8 | Det@0.9 |
|-----------|-------|------|---------|---------|---------|
| **Music background** | 50 | 0.999 | 100% | 100% | 100% |
| **Normal** | 45 | 0.956 | 95.6% | 95.6% | 95.6% |
| **Whisper** | 18 | 0.989 | 100% | 100% | 94.4% |

### Per-Negative-Category Breakdown

| Category | Files | Mean | Max | FA@0.5 | FA@0.7 | FA@0.8 |
|----------|-------|------|-----|--------|--------|--------|
| adversarial | 77 | 0.032 | 0.815 | 2.6% | 1.3% | 1.3% |
| legacy_hard | 50 | 0.033 | 0.548 | 2.0% | 0% | 0% |
| music | 25 | 0.045 | 0.645 | 4.0% | 0% | 0% |
| music_hard | 30 | 0.020 | 0.255 | 0% | 0% | 0% |
| real_confusable | 45 | 0.080 | 0.697 | 4.4% | 0% | 0% |
| real_fp_captures | 75 | 0.041 | 0.802 | 2.7% | 1.3% | 1.3% |

**Key insight:** Real confusable words max out at 0.697 — much lower than TTS confusables (voila=0.97). The TTS eval overstates the confusable problem. At threshold 0.7, real confusables produce **zero** false alarms.

---

## Eval Set (SACRED — Never Change Mid-Campaign)

### TTS Eval
- **Location:** `eval_clean/`
- **288 positives + 546 negatives** (TTS-generated, 22 diverse voices)
- Zero training overlap

### Real Speech Eval
- **Location:** `violawake_data/eval_real/`
- **113 positives** (Jihad 45 + Sierra 68, conditions: music/normal/whisper)
- **302 negatives** (adversarial 77, legacy_hard 50, music 25+30, real_confusable 45, real_fp_captures 75)
- **WARNING:** 122 duplicate file groups between train and eval — potential data leak to audit

---

## Experiment Log

| # | Name | What Changed | all_EER | trained_EER | Real EER | Date |
|---|------|-------------|---------|-------------|----------|------|
| 0 | BASELINE (production) | Nothing | 16.6% | — | — | 2026-03-26 |
| B | + confusable negatives | 325 TTS confusables | 15.0% | — | — | 2026-03-26 |
| C | + diverse positives | 117 diverse TTS voices | 14.6% | — | — | 2026-03-26 |
| **D** | **B + C combined** | **Confusable negs + diverse pos** | **13.2%** | **2.35%** | — | **2026-03-26** |
| D_bce | D + BCE loss | BCE instead of focal loss | 13.14% | 2.35% | **1.71%** | 2026-03-26 |
| F | + MUSAN | D + MUSAN speech/music/noise | completed | — | — | 2026-03-26 |
| H | + Confusable v2 | D + 494→1,151 phoneme-mined negatives | completed | — | — | 2026-03-26 |
| G | + ACAV100M | D + 100K pre-computed negatives | completed | — | — | 2026-03-26 |
| I | Full corpus | D + MUSAN + ACAV + confusable_v2 | completed | — | — | 2026-03-26 |
| **HRD** | **Hardened** | **Weighted 5x/10x + wide + two-head** | **RUNNING** | — | — | **2026-03-26** |

**Key learnings:**
1. Targeted data beats kitchen-sink (D > E)
2. **Real speech EER (1.71%) is 7.6x better than TTS all_EER (13.14%)** — TTS eval is much harder than reality
3. MLP architecture + OWW embeddings ceiling is ~13% on TTS eval but ~2% on real speech
4. Production pooling bug (max→mean) was causing silent degradation
5. 64-32 MLP architecture is optimal — wider/deeper don't help
6. Confusable problem is severe on TTS eval but mild on real speech (max 0.697 vs 0.97)

---

## Bugs Found & Fixed

| Bug | Impact | Fix | Date |
|-----|--------|-----|------|
| **Production pooling mismatch** | Model trained with mean-pool, production ran max-pool | `engine.py:395`: `.max(axis=1)` → `.mean(axis=1)` | 2026-03-26 |
| **Embedding cache CRC** | `incremental_extract.py` accessed stale data after overwrite | Capture count before `savez_compressed` | 2026-03-26 |

---

## In-Flight Work & Completed Tasks

### Completed (2026-03-26)
1. **FAPH baseline measured** — 162 FAPH at 0.5 on test-clean (catastrophic)
2. **Hard negative mining round 1** — 629 windows from test-clean, retrained with 10x weight
3. **Data leakage validated** — dev-clean shows 19.3 FAPH at 0.5 (not 0), confirming leakage but genuine 88% improvement
4. **Hardened training complete** — 12 models (4 variants × 3 seeds). Best: wide_5x_s44 for TTS confusables
5. **Real speech eval complete** — 1.71% EER, 98.2% detection (2 speakers)
6. **Confusable mega generation** — 4,824 TTS confusable files generated (not yet extracted)
7. **META_ANALYSIS.md** — 12 patterns documented with challenger audit

### Running Now (2026-03-26T21:30)
1. **Debounced FAPH on full dev-clean** — faph_hardened_s43 with 2s debounce, 44% complete (~1200/2703 files)
2. **TP eval on hardened model** — Running real_speech_eval on faph_hardened_s42/s43/s44 to verify mining didn't kill detection
3. **864-dim temporal embedding experiment** — Testing whether concatenated (9×96) input beats mean-pooled (96-dim)

### Completed Since Last Update
- **Multi-seed + ensemble FAPH** — 500-file sample. Ensemble = 2.0 FAPH@0.9, 1.0@0.95 (debounced). See ensemble results above.
- **Challenger v2** — Found SDK mismatch, TP gap, mean-pool ceiling hypothesis, statistical invalidity. See findings above.

### Next Steps (Priority Order — Updated After Challenger v2)
1. ~~**GATE: TP eval on hardened model**~~ — ~~PASSED~~ **INVALIDATED.** Duplicate audit found ALL 415 eval files are training duplicates (cosine sim 1.0). The 98.2% detection was the model scoring its own training data. **Need fresh held-out recordings to validate TP.**
2. ~~**GATE: 864-dim experiment**~~ — **PARTIALLY CONFIRMED.** 256→128→64 MLP on 864-dim concat gets 24.5% EER improvement (0.0145 vs 0.0192) and 33% better confusable rejection. Real but moderate — not enough to change strategy from data→architecture. Worth incorporating but mining remains primary approach.
3. **122 duplicate audit** — 10 minutes. Blocking confidence in ALL TP metrics. No more deferring.
4. **ROUND 2 training** — Combine 629 test-clean + dev-clean hard negs + 3,978 ACAV hard negs → retrain 3 seeds → validate on test-clean. ONLY after gates 1-2 pass.
5. **Statistical confidence corpus** — LibriSpeech train-clean-100 (100h) for sub-1 FAPH validation at 95% CI.
6. **SDK inference path resolution** — Either fix SDK to use `embed_clips` with circular buffer, or characterize `oww_backbone.onnx` to confirm compatibility.
7. **Diverse speaker eval** — 10+ speakers for robust detection validation.
8. **Domain diversity** — Test FAPH on Common Voice (accented) and conversational speech.
9. **Sierra outliers** — Investigate the 2 files scoring 0.02.

---

## Current Data Inventory

### Embedding Cache: 139,140 entries

| Source Tag | Count | Label | Used in D_combined? |
|------------|-------|-------|---------------------|
| pos_main | 12,402 | positive | Yes |
| pos_diverse | 1,053 | positive | Yes |
| pos_backup | 4,131 | positive | No |
| pos_eval_real | 1,017 | positive | No |
| pos_excluded | 2,439 | positive | No |
| neg_main | 10,175 | negative | Yes |
| neg_confusable | 325 | negative | Yes |
| **neg_confusable_v2** | **1,151** | **negative** | **Hardened training** |
| neg_acav100m | 100,000 | negative | G, I experiments |
| neg_musan_speech | 426 | negative | F experiment |
| neg_musan_music | 660 | negative | F experiment |
| neg_musan_noise | 930 | negative | F experiment |
| neg_backup | 816 | negative | No |
| neg_downloads | 2,620 | negative | No |
| neg_eval_real | 302 | negative | No |
| neg_noise | 57 | negative | No |
| neg_music | 7 | negative | No |
| **neg_librispeech_hard** | **629** | **negative** | **faph_hardened models** |
| **TOTAL** | **139,140** | — | — |

### Trained Models (experiments/models/)

| Model | Data | Loss | Architecture | Status |
|-------|------|------|-------------|--------|
| D_combined_bce_s42.onnx | D_combined | BCE | 64-32 | Baseline |
| D_combined_bce_s43/44 | D_combined | BCE | 64-32 | Seed variants |
| H_confusable_v2_bce_s42/43/44 | D + confusable_v2 | BCE | 64-32 | Completed |
| G_acav_bal5_bce_s42/43/44 | D + ACAV100M (balanced) | BCE | 64-32 | Completed |
| I_full_corpus_bal3_bce_s42/43/44 | Full corpus (balanced) | BCE | 64-32 | Completed |
| hardened_weighted5x_s42/43/44 | D + confusable_v2 | Weighted BCE (5x) | 64-32 | Completed |
| hardened_weighted10x_s42/43/44 | D + confusable_v2 | Weighted BCE (10x) | 64-32 | Completed |
| hardened_wide5x_s42/43/44 | D + confusable_v2 | Weighted BCE (5x) | 128-64 | Completed |
| hardened_twohead_s42/43/44 | D + confusable_v2 | BCE + reject | 64-32 (2-head) | Completed |
| **faph_hardened_s42/43/44** | **D + confusable_v2 + 629 LS hard negs (10x)** | **Weighted BCE** | **64-32** | **CURRENT BEST (s43)** |

---

## Architecture Reference

### Current Model (D_combined_bce)
```
OWW embed_clips (frozen) → (1, 9, 96) → mean-pool → 96-dim embedding
    ↓
MLP: Linear(96, 64) → ReLU → Dropout(0.3)
     Linear(64, 32) → ReLU → Dropout(0.2)
     Linear(32, 1)  → Sigmoid
    ↓
Score [0, 1] → threshold → wake/no-wake
```

Training: BCE loss, AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR, EMA(0.999), early stopping (patience=12)

### Two-Head Variant (hardened_twohead)
```
OWW → 96-dim → Shared backbone (64→32)
                ├── Detection head → Sigmoid → detect_score
                └── Rejection head → Sigmoid → reject_score
                Combined: detect_score × (1 - reject_score)
```

Loss: BCE_detect + 2.0 × BCE_reject. Rejection head trained on confusable label.

---

## Key Scripts

```bash
# Run specific experiment from config
python experiments/run_all_experiments.py --only D_combined --seeds 3

# Incremental embedding extraction (add new data sources)
python experiments/incremental_extract.py neg_confusable_v2

# FAPH test on LibriSpeech
python experiments/faph_test.py --top-k 30

# Real speech eval (all Jihad/Sierra recordings)
python experiments/real_speech_eval.py

# Hardened model training (weighted + two-head)
python experiments/train_hardened.py --seeds 3

# BCE variant training
python experiments/train_bce_variant.py D_combined --seeds 3

# Balanced training (for high neg:pos ratios like ACAV)
python experiments/train_balanced.py I_full_corpus --ratio 3 --seeds 3

# Generate confusable TTS negatives
python experiments/generate_viola_contexts.py
```

---

## Principles

1. **Empirical only.** Every technique is tested against eval. No shipping based on theory.
2. **Real speech is truth.** TTS eval is useful for development, but real speech eval is the ship gate.
3. **FAPH over EER.** EER is a diagnostic metric. FAPH is the user-facing metric.
4. **Iterate rapidly when there are obvious issues.** If a 2-minute quick test shows FAPH is ~100/hr, DON'T run a 60-minute full test. Fix the problem first, then measure. Quick-test → identify problem → fix → quick-test again → commit to full measurement only when quick tests look promising.
5. **One variable at a time.** Each experiment changes one thing from champion.
6. **Eval set is sacred.** Never train on eval data. Never change eval set mid-campaign.
7. **Cache everything.** Embedding extraction is the bottleneck. Cache and reuse.
8. **Delegate.** Use subagents for generation, extraction, training. Main agent directs.

---

## Questions Answered

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| Q1 | Does real speech match TTS eval? | **Real is much better** — 1.71% vs 13.14% EER | `real_speech_eval.json` |
| Q2 | Does the model work on Jihad's voice? | **Yes — 100% detection at 0.8** | 45 eval files, mean=0.995 |
| Q3 | Does the model work on Sierra? | **Yes — 97.1% detection at 0.8** | 68 eval files, mean=0.970 |
| Q4 | Are real confusables as bad as TTS? | **No — max 0.697 vs 0.97 on TTS** | real_confusable category |
| Q5 | Is whisper detection reliable? | **Yes — 100% at 0.8, 94.4% at 0.9** | 18 whisper files |
| Q6 | Does music background hurt detection? | **No — 100% at all thresholds** | 50 music-bg files |
| Q7 | Is 64→32 the optimal MLP architecture? | **Yes** — wider/deeper don't improve | Architecture sweep |
| Q8 | Does median pooling help? | **Modest — 12.5% vs 13.2% on TTS** | `exp_f_policy_on_d.json` |
| Q9 | Does two-stage verifier help? | **No — 14.0% vs 13.2%** | `exp_two_stage_results.json` |
| Q10 | Is production pooling correct? | **Was WRONG — max instead of mean. FIXED.** | `engine.py:395` |

## Questions Answered (continued)

| # | Question | Answer | Evidence |
|---|----------|--------|----------|
| Q11 | What is actual FAPH on clean speech? | **162 FAPH at 0.5 (baseline), 19.3 (round 1 mined) on held-out dev-clean** | `faph_results.json`, `faph_devclean_s43.json` |
| Q12 | Does weighted confusable training reduce TTS FAs? | **Yes, 5x is sweet spot** — voila 0.59→0.35, vanilla 0.53→0.14 | `hardened_model_results.json` |
| Q13 | Does two-head reject architecture work? | **No** — worse EER (13.9% vs 13.1% weighted_5x), marginal FA improvement | `hardened_model_results.json` |
| Q16 | Does hard negative mining generalize to held-out data? | **Yes, 88% FAPH reduction on dev-clean** but not 100% (data leakage on test-clean was real) | `faph_devclean_s43.json` |
| Q17 | Is iterative mining the path to <1 FAPH? | **Likely yes** — one round gave 88% reduction; 2-3 more rounds needed | Pattern 11 in META_ANALYSIS |

## Questions Still Open

| # | Question | How We'll Answer | Status |
|---|----------|-----------------|--------|
| Q14 | Are train/eval sets contaminated? | Audit 122 duplicate groups | **ANSWERED: YES — 100% contaminated. All 415 eval files are byte-identical copies of training files. 1.71% EER and 98.2% detection are INVALID. See `experiments/duplicate_audit_report.json`** |
| Q15 | What are Sierra's 2 outlier files? | Inspect files scoring 0.02 | TODO |
| Q18 | Does round 2 mining further reduce FAPH? | Mine, retrain, validate on test-clean | Blocked on Q22, Q23 |
| Q19 | Does multi-seed FAPH show consistent results? | Run s42, s44 on dev-clean | **ANSWERED** — massive variance, see ensemble results |
| Q20 | Does FAPH hold on non-LibriSpeech speech? | Test on Common Voice or conversational speech | TODO |
| Q21 | Does batch scoring match production frame-by-frame? | Compare outputs on same audio | TODO |
| Q22 | Did hard negative mining degrade TP detection? | Run real_speech_eval on faph_hardened_s43 | **ANSWERED: YES on held-out data. s43 drops to 92.6%@0.9 (from baseline 98.2%). >95% only achievable at ≤0.80 threshold. Earlier "PASSED" result was on contaminated eval.** |
| Q23 | Does 864-dim temporal embedding beat mean-pooled 96-dim? | Train MLP on concatenated (9×96) input | **ANSWERED: YES (24.5% EER improvement with wide MLP), but moderate — doesn't change strategy** |
| Q24 | What is the debounced FAPH on full dev-clean? | faph_test.py with debounce on all 2,703 files | **ANSWERED: 10.39@0.5, 4.64@0.9, 2.23@0.95 (s43 debounced, 5.39h)** |
| Q25 | Does ensemble diversity survive Round 2 training? | Measure pairwise seed agreement after Round 2 | TODO |
| Q26 | Is `oww_backbone.onnx` compatible with our MLP? | Download, characterize input/output shape | TODO |
| Q27 | Does group-aware split leak through augmentations? | Check if source_idx is per-file or per-augmentation | TODO |

---

## File Map

```
J:\CLAUDE\PROJECTS\Wakeword\
├── ACCURACY_MISSION.md          ← THIS FILE (living doc)
├── PROGRESS.md                  ← Console/SDK progress (Gates 1-5)
├── experiments/
│   ├── run_all_experiments.py   ← Main test harness
│   ├── faph_test.py             ← FAPH measurement on LibriSpeech
│   ├── real_speech_eval.py      ← Real recording eval (Jihad/Sierra)
│   ├── train_hardened.py        ← Weighted + two-head training
│   ├── train_bce_variant.py     ← BCE loss training
│   ├── train_balanced.py        ← Class-balanced training
│   ├── incremental_extract.py   ← Add new data sources to cache
│   ├── mine_confusables.py      ← Phoneme-based confusable mining
│   ├── embedding_cache.npz      ← 138K cached embeddings (96-dim)
│   ├── real_speech_eval.json    ← Real speech results (COMPLETE)
│   ├── faph_results.json        ← FAPH results (PENDING)
│   ├── hardened_model_results.json ← Hardened training results (PENDING)
│   ├── models/                  ← 26+ trained ONNX models
│   │   ├── D_combined_bce_s42.onnx  ← CURRENT CHAMPION
│   │   └── hardened_*.onnx          ← IN TRAINING
│   ├── training_data/
│   │   ├── confusable_negatives/    ← 325 original confusables
│   │   ├── confusable_negatives_v2/ ← 1,151 phoneme-mined confusables
│   │   ├── diverse_positives/       ← 1,053 diverse TTS positives
│   │   └── viola_contexts/          ← 182 "viola + command" clips (not needed)
│   ├── STREAMING_VS_CLIP_ANALYSIS.md ← Production vs eval methodology analysis
│   └── RED_TEAM_REPORT.md           ← Self-audit challenging our conclusions
├── eval_clean/                  ← SACRED TTS eval set (288 pos + 546 neg)
├── corpus/
│   ├── librispeech/             ← LibriSpeech test-clean (~5.4h)
│   ├── musan/                   ← MUSAN speech/music/noise (2,016 files)
│   └── acav100m_embeddings.npz  ← 100K pre-computed OWW embeddings
└── src/violawake_sdk/           ← SDK source code
```
