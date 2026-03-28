# ViolaWake Accuracy Campaign — Meta Analysis & Pattern Report

**Date:** 2026-03-26
**Session:** Accuracy improvement campaign day 1

## Pattern 1: TTS vs Real Speech — The Overstated Problem

### Finding
TTS eval overstates the confusable problem by **10x**:

| Metric | TTS Eval | Real Speech Eval | Gap Factor |
|--------|----------|------------------|------------|
| EER | 13.14% | 1.71% | **7.7x worse on TTS** |
| voila max score | 0.97 | 0.697 | 1.4x |
| hola max score | 0.93 | — | N/A (not in real eval) |
| vanilla max score | 0.91 | — | N/A |
| Detection @0.5 | ~80% | 98.2% | — |

### Why this matters
We spent significant effort optimizing for TTS confusable rejection (hardened models, weighted loss, two-head architecture) when the real-world confusable problem is already mild. At threshold 0.7, real confusables produce **zero false alarms** on the baseline model.

### Root cause
TTS voices are more "crisp" and phonetically precise than real speech. A TTS voice saying "voila" produces a cleaner phonetic signature that's closer to "viola" in embedding space than a real human saying "voila" (with accent, mumbling, background noise, etc).

### Implication
**The ship blocker is FAPH on clean speech (random false triggers), not confusable words.** The early 10-file FAPH sample showed 2 triggers at 0.5 in just 70 seconds of audio — that would extrapolate to ~100+ FAPH.

---

## Pattern 2: Hardened Training — Diminishing Returns with Side Effects

### Finding
Weighted confusable training reduces TTS confusable scores significantly but introduces new real-speech regressions:

| Variant | voila (TTS avg) | random_speech_009 (real) | Net FA change @0.5 |
|---------|-----------------|--------------------------|---------------------|
| baseline | 0.73-0.97 | 0.259 | 8 FAs |
| weighted_5x_s43 | 0.28 | not tested | 5 FAs |
| wide_5x (best) | 0.27 | **0.993** (from 0.259!) | 5 FAs |

### Why this matters
The wide_5x model reduced voila from 0.97 → 0.27 (great!) but random_speech_009.wav went from 0.26 → 0.99 (catastrophic regression on one file). Wider networks memorize TTS confusable patterns but generalize worse to unseen real speech.

### Root cause
The training set is TTS-dominated. When you upweight confusable negatives (which are all TTS), the model learns TTS-specific rejection patterns. Real speech has different acoustic characteristics, and the model's new decision boundary can accidentally include real speech patterns.

### Implication
**Standard 64-32 architecture with 5x weighting is safer than wide 128-64.** The wide model has a lower trained_EER (1.9% vs 2.7%) but worse generalization to real negatives. The simpler model is more robust.

---

## Pattern 3: The Three-Metric Hierarchy

### Finding
Three metrics capture different truths:

| Level | Metric | What it measures | Current |
|-------|--------|-----------------|---------|
| **Operational** | FAPH on clean speech | Real-world user annoyance | ~100+ est. (measuring) |
| **Held-out real** | Real speech EER | True detection quality | **1.71%** |
| **Development** | TTS eval EER | Model improvement signal | 13.14% |

### Why this matters
We initially chased TTS all_EER (13.14% → target <5%), then pivoted to FAPH after realizing EER doesn't map to user experience. The real speech eval (1.71% EER) showed the model is already excellent on actual wake word detection. The remaining problem is **false wakes on non-wake speech**, which TTS eval partially captures through confusables but FAPH on continuous speech captures directly.

### Implication
The correct development cycle is:
1. Train and iterate using TTS eval (fast, reproducible)
2. Validate on real speech eval (truth check)
3. Gate shipping on FAPH (user experience metric)

---

## Pattern 4: Data Scaling — Non-Monotonic Returns

### Finding
Adding more negative data does NOT always help:

| Experiment | Negatives | EER | Notes |
|-----------|-----------|-----|-------|
| D_combined | 10.5K | **13.2%** | Best controlled experiment |
| E_maxdata | 14.0K | 14.1% | More data, worse result |
| G_acav (100K) | 110K+ | worse than D | Class imbalance problem |
| F_musan | 12.5K | completed | Pending analysis |

### Why this matters
Naive data scaling (kitchen sink) hurts because:
- Class imbalance: 100K negatives vs 13K positives overwhelms the positive signal
- Distribution shift: ACAV100M/MUSAN speech patterns differ from TTS training distribution
- Label noise: More data = more ambiguous boundary cases

Targeted data (confusable negatives specifically designed to be hard) beats random large-scale data.

### Implication
**Quality of negatives matters more than quantity.** The 1,151 phoneme-mined confusable negatives had more impact than 100K random ACAV negatives. For FAPH reduction, we need negatives specifically from the failure modes (speech that scores high).

---

## Pattern 5: The Threshold Regime

### Finding
Model behavior changes dramatically across threshold regimes:

| Threshold | Detection | FA Rate (real) | FAPH (est.) | Regime |
|-----------|-----------|---------------|-------------|--------|
| 0.30 | 98.2% | 5.0% | Very high | Unusable |
| 0.50 | 98.2% | 2.6% | High | Prototype |
| 0.70 | 98.2% | 0.7% | Moderate | Reasonable |
| 0.80 | 98.2% | 0.7% | Moderate | Good |
| 0.90 | 97.3% | 0.0% | Low | Conservative |
| 0.95 | 96.5% | 0.0% | Very low | Very conservative |

### Why this matters
At threshold 0.90, real speech has **zero false alarms** with only 1% detection loss. But 0.90 on TTS eval loses 10% of positives. This suggests the operating threshold should be set based on FAPH measurement, not TTS eval.

### Implication
**Threshold 0.80-0.90 is likely the operating range.** Need FAPH data to confirm exact value. If FAPH at 0.80 is already <1/hour, we can ship without model changes.

---

## Pattern 6: Production Bugs as Silent Killers

### Finding
The mean→max pooling bug (`engine.py:395`) was silently degrading production detection quality without any observable error. The model was trained with mean-pool but deployed with max-pool. Max-pool amplifies the strongest frame embedding, which changes the distribution the MLP sees versus what it was trained on.

### Why this matters
This type of bug is invisible in:
- Unit tests (which test individual components, not the full pipeline)
- Manual testing (detection still "works" but is subtly worse)
- Monitoring (no error logs, no crashes)

It's only visible through systematic eval against a held-out set with known ground truth.

### Implication
**Every pipeline change needs end-to-end eval**, not just component tests. The pooling mismatch existed for weeks. A regression test that runs the production pipeline through the eval set after each deployment would have caught it immediately.

---

## Pattern 7: The Seed Lottery — Noise in Small Models

### Finding
Across 3 random seeds, results vary more than expected:

| Metric | weighted_5x | Range |
|--------|-------------|-------|
| all_EER | 13.1% ± 0.1% | 12.9-13.3% |
| trained_EER | 2.7% ± 0.5% | 2.2-3.4% |
| voila mean | 0.35 ± 0.06 | 0.28-0.43 |

The trained_EER ranges from 2.2% to 3.4% — a **55% relative variation** just from seed. Individual confusable word scores vary even more (voila: 0.28 to 0.43).

### Why this matters
Any single-seed result is unreliable for confusable-specific metrics. A model that scores voila at 0.28 on one seed might score 0.43 on another. This means:
- Never declare a technique "works" based on 1 seed
- Report mean ± std across 3+ seeds
- Best-seed cherry-picking gives misleading results

### Implication
**Always run 3+ seeds. Report ranges, not best-of.** The "best model" is seed-dependent. The hardened_best was picked by trained_EER, which varied from 1.9% to 3.4% across variants — the 1.9% wide_5x_s42 might just be lucky.

---

## Summary: What We Know and Don't Know

### Know (hard evidence)
1. Real speech detection is excellent: 98.2% at threshold 0.8, EER 1.71%
2. TTS eval overstates confusable problem by ~10x
3. Confusable-weighted training reduces TTS FAs but can regress real speech
4. Data quality > data quantity for negatives
5. Production pooling bug was silently degrading quality (fixed)
6. 64-32 architecture is optimal — wider doesn't generalize better

### Don't Know (pending)
1. **FAPH on clean speech** — this is THE critical unknown. In progress.
2. **FAPH at different thresholds** — is 0.80 sufficient, or do we need 0.90+?
3. **What triggers false alarms in LibriSpeech?** — top-30 list pending from FAPH test
4. **Optimal threshold for shipping** — depends on FAPH/detection tradeoff curve

### Risks
1. ~~If FAPH at 0.80 is >10/hour, we need fundamental model improvement~~ **CONFIRMED: FAPH is ~18-24/hour at 0.80 even with best model. Fundamental improvement required.**
2. The 122 train/eval duplicate groups could mean our real speech EER is optimistically biased
3. ~~The 10-file early FAPH sample suggests FAPH could be very high~~ **CONFIRMED: full 200-file scan shows ~34-42 FAPH at 0.5**

---

## Pattern 8: The Feature Extractor Ceiling (CRITICAL)

### Finding
**ALL models produce >10 FAPH regardless of threshold, training data, or architecture.** The problem is in the OWW 96-dim embedding, not the MLP classifier.

Evidence:
- D_bce (13K negs, TTS-only): ~42 FAPH at 0.5
- G_acav_bal (100K real speech negs): ~34 FAPH at 0.5 — only 19% improvement despite 7.5x more data
- Even at threshold 0.95 with 2s debounce: ~10 FAPH
- 35/40 LibriSpeech speakers (87.5%) trigger above 0.5 at some point
- Random utterances like "BROTHER HICKEY" and "GOOD NIGHT HUSBAND" score >0.95

### Why this matters
Adding more training data gives diminishing returns (100K ACAV only reduced FAPH by ~20%). The 96-dim embedding simply doesn't contain enough information to reliably separate "viola" from all possible speech. Some phoneme combinations map to the same region of 96-dim space regardless of what word was said.

### Root cause
The OWW embedding model was trained for general-purpose wake word detection across many words. Its 96-dim representation must encode enough information for ALL possible wake words, which means no individual wake word gets high-resolution representation. A 768-dim model like HuBERT, trained on speech representation, would have much higher resolution.

### Implication
**Phase 3 (feature extractor upgrade) is no longer optional — it's the critical path to <1 FAPH.** Phases 1-2 (data + architecture) have been exhausted with limited returns. The path forward is:
1. HuBERT/wav2vec2 768-dim embeddings → new MLP
2. Or: frame-level OWW backbone (not the aggregated embedding) → different architecture
3. Or: fundamentally different approach (e.g., full end-to-end model trained from spectrograms)

---

## Pattern 9: Rapid Iteration Principle

### Finding
The user correctly identified that running a 60-minute full FAPH test was wasteful when a 2-minute quick test already showed the problem (2 triggers in 70s). This saved ~58 minutes and led to faster diagnosis.

### When to use rapid iteration
- Quick test reveals an obvious problem → fix it before committing to full measurement
- An assumption is trivially testable → test it in <5 minutes before building on it
- Multiple models exist → do a quick comparison before deep-diving one model

### When to commit to long runs
- Quick tests look promising and you need precise numbers
- You've iterated to a likely solution and need validation
- The test is inherently long (e.g., 24h stability test)

---

## Pattern 10: Hard Negative Mining — The Kill Shot

### Finding
Mining 629 hard negatives from LibriSpeech test-clean (windows scoring > 0.3 on baseline model) and retraining with 10x weight reduced FAPH from ~42 to ~0 at threshold 0.9, while IMPROVING detection from 97.3% to 98.2%.

### Why this works
The model's failure mode was specific: certain speech phoneme combinations produced viola-like embeddings. By explicitly showing the model these specific failure cases (weighted 10x), the model learns to reject them without losing sensitivity to actual "viola" speech. This is essentially "the model tells you what it's confused about, you teach it those are negatives."

### Why it's better than bulk data
- 100K ACAV100M negatives: ~20% FAPH reduction (most negatives are easy, wasted capacity)
- 629 targeted hard negatives: ~85-100% FAPH reduction (every negative teaches something new)

### Caveat
**CRITICAL: Data leakage CONFIRMED.** The hard negatives were mined from the same test-clean split used for FAPH measurement. The 0 FAPH result was memorization — see Pattern 11.

### Implication
Hard negative mining is a repeatable process:
1. Score a new speech corpus through the model
2. Collect windows scoring above threshold
3. Add as weighted negatives
4. Retrain
5. Repeat on a different corpus

Each iteration teaches the model about a new failure mode. **One round is not enough.** See Pattern 11 for held-out validation and the path forward.

---

## Pattern 11: Data Leakage Confirmed — But Real Improvement Is Genuine

### Finding (2026-03-26T19:30)
Held-out validation on LibriSpeech dev-clean (2,703 files, 5.39h, completely separate speakers from test-clean):

| Model | Corpus | FAPH@0.5 | FAPH@0.7 | FAPH@0.9 | FAPH@0.95 |
|-------|--------|----------|----------|----------|-----------|
| D_bce (baseline) | test-clean | 162.1 | 113.3 | 61.4 | 42.9 |
| D_bce (baseline) | dev-clean | 105.1 | 72.0 | 39.7 | 26.0 |
| faph_hard_s43 | test-clean (LEAKED) | **0** | **0** | **0** | **0** |
| faph_hard_s43 | **dev-clean (HELD-OUT)** | **19.3** | **13.9** | **6.7** | **3.3** |

Reduction on dev-clean (apples-to-apples): **82% at 0.5, 83% at 0.9, 87% at 0.95**

### What this proves
1. **Data leakage was real** — 0 FAPH on test-clean was memorization of the 629 mined windows
2. **But the improvement is genuine** — 88% FAPH reduction on held-out data (162→19.3 at 0.5)
3. **One round of mining is not enough** — 6.7 FAPH at 0.9 is still 6x above the <1 FAPH target
4. **The approach scales** — each mining round patches specific failure modes on the mined corpus

### Resolution of Pattern 8 vs Pattern 10 contradiction
Both are partially true:
- Pattern 8 (ceiling): The OWW 96-dim embedding DOES have a ceiling — you can't reach <1 FAPH with a single round of mining
- Pattern 10 (mining works): But iterative mining DOES reduce FAPH substantially (88% per round on held-out data)
- **Prediction:** 2-3 more rounds of iterative mining across diverse corpora may close the gap to <1 FAPH, OR the ceiling will become apparent as diminishing returns set in around 1-3 FAPH

### Top false alarm triggers on dev-clean
| Score | File | Time |
|-------|------|------|
| 0.9993 | 2803-154328-0007 | 3.8s |
| 0.9989 | 777-126732-0066 | 6.8s |
| 0.9951 | 1993-147966-0002 | 0.0s |
| 0.9925 | 6313-66125-0011 | 2.7s |
| 0.9924 | 1919-142785-0007 | 13.7s |

These are the next mining targets — each represents a phoneme pattern the model hasn't learned to reject yet.

### Implication
**The iteration loop is validated. Execute round 2:**
1. Mine hard negatives from dev-clean (the 104 triggers at 0.5)
2. Add to training cache
3. Retrain (3 seeds)
4. Validate on test-clean (now held-out)
5. If still >1 FAPH, acquire more corpora and repeat

---

## Pattern 12: Challenger Audit — Validated Weaknesses

### Key findings from two independent challenger agents (2026-03-26)

**CRITICAL issues confirmed:**
1. **122 train/eval duplicate groups still unaudited** — could invalidate 1.71% EER and 98.2% detection
2. **2-speaker real eval is not cross-speaker validation** — need 10+ diverse speakers
3. **Single-seed FAPH test** — s42 and s44 not yet FAPH-tested, seed lottery risk
4. **5.4h insufficient for sub-1 FAPH confidence** — Poisson upper bound for k=0 in 5.4h is 0.55 FAPH at 95% CI; need 50-100h for precision

**HIGH issues identified:**
5. **LibriSpeech monoculture** — all FAPH measurement on clean read speech; no data on conversational, accented, or noisy speech
6. **Production pipeline divergence** — batch clip scoring vs frame-by-frame 20ms may give different results
7. **faph_test.py debounce inconsistency** — code has debounce but earlier inline tests may not have matched

**Challenger predictions vs reality:**
- Predicted dev-clean would show 10+ FAPH → Actual: 6.68 at 0.9 (partially correct)
- Predicted Pattern 10 was "just data leakage" → Partially wrong: real improvement exists (88% on held-out)
- Predicted Pattern 8/10 contradiction → Correct: resolved as "both partially true" (see Pattern 11)

### What the challengers validated as sound:
- Pattern 1 (TTS overstates) — directionally reliable
- Pattern 4 (quality > quantity) — supported by data
- Pattern 6 (pooling bug) — confirmed real
- Pattern 7 (seed lottery) — undermines several single-seed claims
- Pattern 9 (rapid iteration) — good methodology

---

## Pattern 13: The ACAV Goldmine — Existing Data You Already Have

### Finding (2026-03-26T20:30)
Scoring 100K existing ACAV embeddings through the round-1 model revealed **3,978 hard negatives** (score > 0.3) — **6x more** than the 629 mined from LibriSpeech test-clean. These come from diverse real speech (not read audiobook speech).

| Score threshold | Count | % of 100K |
|----------------|-------|-----------|
| >0.1 | 9,091 | 9.1% |
| >0.2 | 5,819 | 5.8% |
| >0.3 | 3,978 | 4.0% |
| >0.5 | 2,823 | 2.8% |
| >0.7 | 1,808 | 1.8% |
| >0.9 | 655 | 0.7% |

### Why this matters
1. **6x more hard negatives than round 1** — 3,978 vs 629, with no new data collection
2. **Different distribution** — ACAV is YouTube audio (conversational, varied accents, background noise), not LibriSpeech (clean read speech). This directly addresses the "LibriSpeech monoculture" concern from Pattern 12.
3. **Already extracted** — These embeddings were in the cache the whole time. Scoring 100K embeddings through the MLP took 3 minutes. Zero download, zero OWW extraction.
4. **655 scoring >0.9** — These are embeddings the round-1 model is nearly certain are "viola" but aren't. Each one represents a distinct failure mode in diverse speech.

### Why we missed this
The original plan (mine more from LibriSpeech) was tunnel-visioned on the corpus that revealed the problem. We had 100K diverse embeddings sitting in the cache, never scored through the improved model. It took a challenger agent questioning "why aren't you using the data you already have?" to surface this.

### Root cause of the miss
**Recency bias in problem-solving.** The FAPH crisis was discovered on LibriSpeech, so all mining efforts focused on LibriSpeech. The ACAV embeddings were added in experiment G (for bulk training) and had been mentally categorized as "bulk data that doesn't help much" (Pattern 4). But Pattern 4's finding — that 100K random ACAV negatives gave only 20% improvement — doesn't apply here. These aren't random ACAV negatives. They're the **hardest 4%** of ACAV, surgically selected by the round-1 model. Quality > quantity applies in their favor.

### Implication
**Before mining new corpora, always score existing cached data through the current model.** The cache may contain undiscovered hard negatives from previous experiments. This is a 3-minute operation that should happen at the start of every mining round, not as an afterthought.

---

## Pattern 14: The Measurement Flaw — Debounce Changes Everything

### Finding (2026-03-26T20:30)
The `faph_test.py` script had **zero debounce** — every overlapping 100ms-step window above threshold counted as a separate trigger. Production uses 2s debounce. Analysis of the top-20 triggers showed 20 windows from only 10-14 unique speech events (1.4-2.0x inflation). The full inflation factor across all triggers is estimated at ~3-4x.

| Metric | Raw (no debounce) | Estimated debounced | Impact |
|--------|-------------------|---------------------|--------|
| Baseline FAPH@0.5 (dev-clean) | 105.1 | ~25-35 | 3-4x |
| faph_hard_s43 FAPH@0.5 (dev-clean) | 19.3 | ~5-8 | 2.5-4x |
| faph_hard_s43 FAPH@0.9 (dev-clean) | 6.7 | ~2-3 | 2-3x |

### Why this matters
The perceived gap to <1 FAPH was 6.7x (raw 6.7 vs target 1.0). With debounce, the actual gap may be only **2-3x**. This fundamentally changes:
- How many mining rounds are needed (maybe 1-2, not 3-4)
- Whether ensemble scoring alone might close the gap
- The urgency of a feature extractor upgrade (less urgent if gap is 2x not 7x)

### Root cause
The ACCURACY_MISSION.md referenced "2s debounce" in early inline FAPH numbers (~42 at 0.5), but when the formal `faph_test.py` was written, debounce was omitted. Two measurement systems coexisted — the inline scripts with debounce and the formal script without. Nobody noticed because the formal script was never compared back to the inline numbers on the same data.

### Fix applied
Added 2s per-file debounce to `faph_test.py`. Now reports both raw and debounced FAPH for all thresholds. Running validation to confirm actual debounced numbers.

### Implication
**Every measurement script must match production behavior exactly.** Debounce, buffer boundaries, inference interval — any mismatch inflates or deflates the metric. The measurement IS the ground truth; if it's wrong, every decision built on it is wrong.

---

## Pattern 15: Propose-Challenge-Synthesize — The Meta-Workflow

### Finding
The most impactful discoveries in this campaign came not from executing plans, but from **challenging them before execution**. The Pattern 13 (ACAV goldmine) and Pattern 14 (debounce flaw) findings were invisible to the main agent until a challenger agent forced re-examination.

### The workflow that produced these discoveries

```
1. PROPOSE  — Main agent designs a plan based on current understanding
   Example: "Mine dev-clean hard negatives → retrain → validate on test-other"

2. QUESTION (Socratic) — Before launching, ask first-principles questions:
   - "What assumption am I making?"
   - "What data do I already have that I'm not using?"
   - "Is my measurement actually measuring what I think?"
   - "What would a skeptic say about this plan?"

3. CHALLENGE (Adversarial) — Launch a challenger agent with:
   - Full context of the plan AND the data
   - Explicit instructions to find flaws, not confirm
   - Specific challenge areas (methodology, alternatives, blind spots)
   - Requirement to cite evidence, not just theorize

4. EVALUATE — Main agent reviews challenger findings against evidence:
   - Which challenges are supported by data?
   - Which are speculative?
   - What new information did the challenger surface?

5. SYNTHESIZE — Merge the best of both into a revised plan:
   - Accept challenges with evidence
   - Reject challenges without evidence (but note them)
   - The revised plan is usually DIFFERENT from both the original and the challenger's proposal

6. EXECUTE — Run the revised plan with measurements that can validate or invalidate
```

### Why this works
- **Tunnel vision is the default.** The main agent builds a mental model and optimizes within it. The challenger has no sunk cost in that model.
- **The challenger asks "why NOT" instead of "how."** The main agent asks "how do I mine dev-clean?" The challenger asks "why are you mining dev-clean when you have 100K ACAV embeddings?"
- **Cheap to run, expensive to skip.** A challenger agent takes ~5 minutes and catches errors that would waste hours of compute.

### When to use this workflow
- **Before any plan that involves >30 minutes of compute or new data acquisition**
- **When you've been working on the same problem for >3 rounds** (tunnel vision risk increases)
- **When the plan feels "obvious"** — obvious plans are most likely to have unexamined assumptions
- **NOT for routine operations** (running a test, extracting embeddings, etc.)

### Evidence from this campaign
| Discovery | Found by | Would main agent have found it? | Time saved |
|-----------|----------|--------------------------------|------------|
| ACAV goldmine (3,978 hard negs) | Challenger | No — plan was to download test-other | ~2h download + 40min FAPH |
| Debounce measurement flaw | Challenger | Possibly — but not before running 40min test | ~40min on wrong numbers |
| Ensemble scoring opportunity | Challenger | No — not considered | Models already exist, zero training cost |
| Dev-clean window deduplication | Challenger | Possibly — but would have fed duplicates | Cleaner training data |

### Anti-patterns (what NOT to do)
1. **Don't challenge every decision** — only plans with significant time/compute cost
2. **Don't accept all challenges uncritically** — evaluate each against evidence
3. **Don't skip the Socratic step** — your own questioning often surfaces 50% of what a challenger would find
4. **Don't launch a challenger without giving it access to the data** — a challenger without evidence is just speculation

### The meta-principle
**The fastest path to the right answer is not always the shortest path. Sometimes the fastest path includes a 5-minute detour to question whether you're solving the right problem.**

---

## Pattern 16: Multi-Seed Ensemble — Free FAPH Reduction (2026-03-26T21:00)

### Finding
Averaging sigmoid outputs from 3 seeds (s42, s43, s44) of faph_hardened on a 500-file dev-clean sample (0.99 hours) with 2s debounce:

| Model | Deb FAPH@0.5 | Deb FAPH@0.7 | Deb FAPH@0.9 | Deb FAPH@0.95 |
|-------|-------------|-------------|-------------|--------------|
| s42 | 14.1 | 13.1 | 8.1 | 7.1 |
| s43 | 11.1 | 11.1 | 6.0 | 5.0 |
| s44 | 25.2 | 12.1 | 3.0 | 1.0 |
| **ensemble** | **11.1** | **7.1** | **2.0** | **1.0** |

### Why this matters
1. **Seed lottery is severe** — s44 at 0.5 is 2.3x worse than s43 (25.2 vs 11.1), but at 0.95 is 5x BETTER (1.0 vs 5.0). No single seed dominates all thresholds.
2. **Ensemble smooths the tail** — at 0.9, ensemble (2.0) beats every individual seed except s44 (3.0). At 0.95, ensemble matches s44's lucky best (1.0) while being robust.
3. **Nearly at target** — 1.0 FAPH at 0.95 on this sample, before Round 2 training.

### Caveats
- **0.99 hours is too short** — 1 trigger in 1 hour → 95% CI is [0.025, 5.57] FAPH. Cannot distinguish 1 from 5 FAPH.
- **TP rate unmeasured on ensemble** — threshold 0.95 may kill detection. MUST verify.
- **3x inference cost** — always-on detector running 3 models is expensive. May need to pick best single seed for production and use ensemble only for validation.

### Implication
Ensemble is a free improvement for offline evaluation. For production, either accept 3x cost or use it to select the best operating point, then deploy best single seed at that point.

---

## Pattern 17: Challenger v2 — Critical Gaps (2026-03-26T21:30)

A second, deeper challenger agent identified these issues. Unlike Challenger v1 (Pattern 12), this agent read the actual SDK source code and training scripts.

### CRITICAL: SDK Uses Different Inference Path Than All Measurements

The production SDK `WakeDetector.process()` (at `src/violawake_sdk/wake_detector.py:186-218`) feeds **20ms frames** through `oww_backbone.onnx` — a separate ONNX model that does NOT exist locally and has never been characterized. Meanwhile, ALL training, evaluation, and FAPH measurement uses `embed_clips()` on **1.5s clips** → 9 temporal frames → mean-pool → 96-dim.

These are completely different inference paths. `embed_clips` on 320 samples (20ms) crashes. The two paths produce different embeddings from different amounts of audio. **Every number in this campaign is valid only for the `embed_clips` pathway.**

The `STREAMING_VS_CLIP_ANALYSIS.md` acknowledges this but notes "the SDK WakeDetector is NOT what production Viola uses." This means:
- Production Viola (the app) uses `embed_clips` → our numbers apply
- The SDK (for third-party developers) uses a different path → our numbers DON'T apply
- The SDK path needs to be either fixed to match or documented as incompatible

### CRITICAL: No True Positive Rate on Hardened Model

faph_hardened_s43 has NEVER been tested on positive detection. The real_speech_eval.json was run on D_combined_bce_s42 (baseline), not the hardened model. Adding 629 hard negatives with 10x weight shifts the decision boundary. The effective negative mass is 17,941 vs 13,455 positives (1.33:1 ratio). Round 2 adds ~5,000 more hard negatives at 10x weight — could cause catastrophic TP regression.

**Action:** Running real_speech_eval on faph_hardened_s43 now. MUST complete before Round 2.

### CRITICAL: Per-File Debounce Underestimates Production FAPH

`faph_test.py` resets debounce at each file boundary (line 181). LibriSpeech files average ~7.4s. Production audio is continuous — no file boundaries. The per-file reset creates ~5-15% optimistic bias. Not a strategic issue but erodes measurement trust.

### HIGH: Mean-Pooling Destroys Temporal Information (Potential Architectural Win)

The 9 OWW temporal frames (9×96 = 864-dim) contain temporal ORDER information. Mean-pooling discards this entirely. A clip with "viola" at the start produces the same mean-pooled embedding as completely different speech that happens to average similarly.

The 655 ACAV embeddings scoring >0.9 are indistinguishable from actual "viola" clips in mean-pooled 96-dim space. But in the full (9,96) temporal representation, they may be clearly different — the temporal pattern of random speech differs from the temporal pattern of "viola."

**Hypothesis:** The "OWW ceiling" (Pattern 8) may actually be a "mean-pool ceiling." A 1-hour experiment training on 864-dim concatenated embeddings could test this.

**Action:** Running 864-dim experiment now.

### HIGH: Statistical Invalidity of Sub-1 FAPH Claims

For Poisson process with k=1 in T=0.99h, 95% CI = [0.025, 5.57] FAPH. Even on full 5.39h dev-clean, 18 triggers gives CI [2.06, 5.29] FAPH. Cannot claim <1 FAPH with useful precision on 5 hours. Need 50-100 hours of negative speech for reliable sub-1 claims.

### HIGH: 122 Duplicate Groups — Still Unaudited

Every model quality decision rests on the 1.71% EER and 98.2% detection numbers. If train/eval overlap inflates these, the entire campaign narrative changes. This is a 10-minute scripting task that has been deferred since day 1.

### MEDIUM: Ensemble Diversity May Not Survive Round 2

After Round 2 with ~5,000 hard negatives at 10x weight, the strong constraint signal may cause all 3 seeds to converge to nearly identical solutions. Ensemble benefit vanishes. Measure pairwise seed agreement after Round 2.

### MEDIUM: Mono-Domain Mining Creates Whack-a-Mole

LibriSpeech (read English) + ACAV (YouTube) does not cover: conversational speech, children, accented English, TV/radio, music with vocals. Each new domain will reveal new failure modes.

### MEDIUM: Group-Aware Split May Leak Through Augmentations

If `source_idx` in embedding cache is assigned per-augmentation rather than per-source-file, augmented copies of the same source could appear in both train and val, making validation loss optimistically biased.

---

## Pattern 18: TP Eval After Mining — Detection Preserved (2026-03-26T21:45)

### Finding
Hard negative mining did NOT degrade true positive detection for 2 of 3 seeds:

| Model | Det@0.8 | Det@0.9 | Det@0.95 | Mean Score |
|-------|---------|---------|----------|------------|
| Baseline (D_bce_s42) | 98.2% | 97.3% | 96.5% | 0.9803 |
| **faph_hardened_s43** | **98.2%** | **98.2%** | **97.3%** | **0.9875** |
| faph_hardened_s42 | 97.3% | 97.3% | 96.5% | 0.9823 |
| faph_hardened_s44 | 94.7% | 92.9% | 92.0% | 0.9608 |

### Why this matters — INVALIDATED BY PATTERN 21
The original TP eval showed s43 "improved" to 98.2%@0.9. **This was tested on contaminated data (all eval files were training duplicates).**

**Corrected results (Pattern 22, held-out pos_backup):** s43 drops to **92.6%@0.9** (from baseline 98.2%). Mining DOES degrade detection at high thresholds. >95% detection requires threshold ≤0.80.

### Revised Implication
Round 2 mining must be balanced — cannot just add more hard negatives at 10x weight without monitoring the detection/FAPH tradeoff. The operating point is constrained: threshold ≤0.80 for >95% detection, where FAPH = 6.31 (s43 debounced). The gap to <1 FAPH is 6.3x, not 2.2x as previously thought.

---

## Pattern 19: Temporal Embeddings — Moderate Win (2026-03-26T22:00)

### Finding
Training on concatenated 864-dim (9×96) temporal embeddings vs 96-dim mean-pooled:

| Model | EER | AUC | Confusable Mean |
|-------|-----|-----|-----------------|
| meanpool_96 (64→32→1) | 0.0192 ± 0.0043 | 0.9975 | 0.0237 |
| concat_864 (128→64→1) | 0.0209 ± 0.0025 | 0.9970 | 0.0231 |
| **concat_864_wide (256→128→64→1)** | **0.0145 ± 0.0032** | **0.9984** | **0.0159** |

### Key insights
1. **Temporal info helps** — 24.5% relative EER improvement with adequate model capacity
2. **Architecture must match input** — naive small MLP on 864-dim is WORSE than baseline. Need 256→128→64.
3. **Confusable rejection improves 33%** — temporal patterns help disambiguate similar-sounding words
4. **Only 13K of 25K samples extracted** (missing audio files) — results may improve with full data
5. **ACAV 100K excluded** — no temporal data for pre-computed embeddings

### Resolution of Pattern 8 (OWW ceiling) + challenger hypothesis
The "OWW ceiling" is PARTLY a mean-pool ceiling. Temporal info adds ~25% improvement. But it's not the dominant factor — the remaining FAPH reduction must come from data (mining) not architecture alone. Both approaches compound.

### Implication
**Strategy remains data-driven (mining) as primary, but temporal embeddings are worth incorporating in Round 2.**

---

## Pattern 22: Held-Out Eval Reveals Hidden TP Degradation (2026-03-26T22:45)

### Finding
After Pattern 21 (total eval contamination), we discovered 2,236 genuinely held-out positive embeddings in `pos_backup` (1,067 clean) and `pos_excluded` (1,169 clean) — not used in training, deduplicated by cosine similarity >0.99.

Detection on `pos_backup` clean (the cleaner set, 355 unique files):

| Threshold | Baseline (D_bce_s42) | faph_hardened_s43 | Delta |
|-----------|---------------------|-------------------|-------|
| 0.50 | 99.5% | 99.2% | -0.3% |
| 0.80 | 99.1% | **96.0%** | **-3.1%** |
| 0.90 | 98.2% | **92.6%** | **-5.6%** |
| 0.95 | 96.4% | **85.5%** | **-10.9%** |

### Why this matters
1. **Hard negative mining DOES degrade detection** — the contaminated eval hid a 5.6% drop at 0.9 and 10.9% at 0.95
2. **The operating point is constrained** — >95% detection requires threshold ≤0.80, where FAPH = 6.31 debounced
3. **The real gap to <1 FAPH is 6.3x** — not 2.2x as calculated at threshold 0.95
4. **Every "TP preserved" claim from Pattern 18 was wrong** — built on contaminated data
5. **The FAPH-vs-detection tradeoff is real and severe** — 10x hard negative weighting pushes the boundary too aggressively

### Resolution
This finding reframes the entire campaign. The question is no longer "can we reach <1 FAPH at 0.95?" but "can we reach <1 FAPH at 0.80 while keeping detection >95%?" This is a much harder problem.

Possible approaches:
- **Lower hard negative weight** (5x instead of 10x) to preserve detection
- **Focal loss** instead of weighted BCE — focuses on hard examples without as much boundary shift
- **864-dim temporal input** — may provide better separation, allowing lower threshold
- **More positive augmentation** — increase positive mass to counterbalance hard negatives
- **Threshold-aware training** — optimize for detection@0.80 specifically

---

## Pattern 23: Cross-Domain Mining Has Limited Transfer (2026-03-26T23:15)

### Finding
Round 2 added 3,978 ACAV hard negatives (YouTube audio, score >0.3) with 10x weight and 2x positive protection. The FAPH result on dev-clean:

| Threshold | R1 Deb FAPH | R2 Deb FAPH | Change |
|-----------|------------|------------|--------|
| 0.50 | 10.39 | 12.81 | +23% worse |
| 0.80 | 6.31 | 6.12 | -3% (noise) |
| 0.90 | 4.64 | 4.45 | -4% (noise) |
| 0.95 | 2.23 | 2.23 | 0% |

**3,978 diverse hard negatives produced zero meaningful FAPH improvement on dev-clean.**

### Why this happened
1. **Domain mismatch:** ACAV = YouTube clips (music, conversation, varied acoustics). Dev-clean = clean audiobook read speech. The false alarm patterns are in different regions of embedding space.
2. **Different failure modes:** The top dev-clean triggers shifted (speaker 2803 dropped from 0.999→0.982, new speaker 2412 emerged at 0.999) but the total trigger count stayed the same. The model traded some failures for others.
3. **Round 1 worked because domain matched:** The 629 test-clean hard negatives were ALSO LibriSpeech audiobook speech. Same domain → same failure modes → direct fix.

### Resolution of Pattern 4 (quality > quantity)
Pattern 4 was partially wrong. It said "targeted data beats kitchen-sink." This is true, but targeting must be **domain-matched**. ACAV hard negatives are targeted (high-scoring) but wrong-domain (YouTube vs audiobook). They're targeted in score space but not in acoustic space.

### Implication
**To reduce FAPH on a given corpus, mine from THAT corpus or acoustically similar corpora.** The mining loop works, but each round must mine from the validation corpus itself (or same domain). This means:
- To reduce dev-clean FAPH → mine dev-clean hard negatives
- To reduce real-world FAPH → mine from real-world-like audio (conversational, noisy)
- ACAV mining helps generalization breadth but not depth on any specific domain

**Next step:** Mine the 56 debounced dev-clean triggers directly, retrain, validate on test-clean.

---

## Pattern 21: Total Eval Contamination — 100% Train/Eval Overlap (2026-03-26T22:30)

### Finding
The duplicate audit revealed that **ALL 415 real speech eval files are byte-identical copies of training files.** Every single eval file (113 positives, 302 negatives) exists in both `violawake_data/eval_real/` and the training directories (`violawake_data/positives/`, `violawake_data/negatives/`). Cosine similarity = 1.0000 for every pair.

### What this invalidates
- **1.71% Real speech EER** — model scored its own training data
- **98.2% detection @0.8** — ditto
- **All per-speaker breakdowns** (Jihad 100% @0.8, Sierra 97.1%) — ditto
- **All per-condition breakdowns** (music bg, whisper, normal) — ditto
- **TP eval on faph_hardened** (Pattern 18) — used same contaminated set
- Every decision based on "real speech is truth" (Principle 2) — built on invalid data

### What remains valid
- **FAPH on LibriSpeech** — completely separate corpus, no training overlap
- **TTS eval (eval_clean/)** — 98.2% clean (only 15/834 files overlap)
- **Mining loop methodology** — validated by FAPH reduction, not eval set
- **Architecture findings** — validated by TTS eval EER

### How this happened
The eval files were copied from training directories into a separate eval directory. The embedding extraction pipeline processed both copies, tagging them differently (`pos_main` vs `pos_eval_real`). Nobody checked for content overlap because the file paths were different.

### Why it wasn't caught earlier
1. The "122 duplicate groups" warning was based on MD5 hashing but was marked TODO and deferred
2. Both challengers flagged it but it was always deprioritized vs FAPH work
3. The eval metrics looked "too good" (1.71% EER) but we attributed this to "real speech is easier than TTS" (Pattern 1)
4. Pattern 1's conclusion ("TTS overstates the problem by 10x") was partially driven by comparing invalid real metrics to valid TTS metrics

### Re-evaluation of Pattern 1
Pattern 1 claimed real speech EER is 7.7x better than TTS EER (1.71% vs 13.14%). With the real speech eval invalidated, we don't actually know how real speech compares. The TTS eval is the only trustworthy benchmark. The model may or may not work well on real speech — **we simply don't know.**

### Implication
**The entire "real speech is truth" pillar of this campaign was built on contaminated data.** The model might still work excellently on real speech (the TTS eval suggests good discrimination), but we have zero evidence for this claim.

**Immediate action needed:**
1. Create a genuinely held-out eval set — either new recordings or hold out training positives that are EXCLUDED from all future training
2. Re-run all eval metrics on the clean set
3. Until then, gate shipping on TTS eval + FAPH only Could extract temporal embeddings for the full training set and train a 256→128→64 MLP on 864-dim input with all hard negatives. The 24.5% EER improvement would compound with mining's 88% FAPH reduction.

Caveat: FAPH test needed on temporal model to confirm EER improvement translates to FAPH reduction. These are different metrics on different data.

---

## Pattern 20: Definitive Debounced FAPH (2026-03-26T22:00)

### Finding
Full dev-clean (5.39h, 2,703 files, 154,852 windows) with 2s per-file debounce:

| Threshold | Raw FAPH | **Debounced FAPH** | Debounced Triggers |
|-----------|----------|-------------------|-------------------|
| 0.50 | 19.30 | **10.39** | 56 |
| 0.70 | 13.92 | **8.17** | 44 |
| 0.90 | 6.68 | **4.64** | 25 |
| 0.95 | 3.34 | **2.23** | 12 |

### Key insights
1. **Debounce reduces by ~46%, not 3-4x** — the per-file reset (Pattern 17, CRITICAL 3) limits the effect. With continuous audio (no file boundaries), reduction would be larger.
2. **The real gap to <1 FAPH is 2.23x at threshold 0.95** for single seed s43. With s42+s43 ensemble, estimated ~1-1.5 FAPH.
3. **25 debounced triggers at 0.9 in 5.39h** — enough for statistical significance (95% CI: [3.0, 6.8] FAPH). The 2.23 @0.95 has wider CI: [1.2, 3.9].
4. **Round 2 mining needs ~60% reduction at 0.95** to reach <1 FAPH. Round 1 gave 88% on raw. With 6x more diverse hard negatives (ACAV goldmine), this is achievable.

### Top false alarm events (debounced = unique speech events)
All 12 triggers at 0.95 from only 10 unique speakers. Top 3: speaker 2803 (0.999), 777 (0.999), 1993 (0.995). These are the Round 2 mining targets.

### Implication
**The path to <1 FAPH is clear and achievable in 1-2 more mining rounds.** The combination of Round 2 mining + ensemble + threshold 0.95 should reach the target. Statistical validation on 50-100h still needed for production confidence.

---

## Pattern 24: Focal Loss — Hybrid "HardOnly" Wins, Pure Focal Destroys TP

### Finding
Trained 4 focal loss variants × 2 seeds (8 models). Results on 1,067 held-out positives:

| Model | @0.80 | @0.85 | @0.90 | @0.95 |
|---|---|---|---|---|
| D_combined_bce (baseline) | 99.1% | 99.0% | 98.2% | 96.4% |
| R1 (weighted BCE) | 96.0% | 95.1% | 92.6% | 85.5% |
| R2 best (weighted BCE) | 97.8% | 96.7% | 94.5% | 89.6% |
| **focal_g2_hardonly_s42** | **98.3%** | **97.9%** | **95.7%** | **90.0%** |
| focal_g2 (pure) | 76.6% | 62.2% | 44.1% | 21.1% |
| focal_g3 (aggressive) | 57.8% | 41.2% | 23.4% | 6.6% |
| focal_g2_a (alpha-weighted) | 85.8% | 72.8% | 54.4% | 27.9% |

### Key Insight
**Only the "hardonly" variant works** — focal loss on hard negatives, standard BCE on everything else. Pure focal (g2, g3) and alpha-weighted (g2_a) catastrophically destroy TP because they over-focus on hard examples at the expense of maintaining positive confidence.

### Why hardonly works
Standard BCE maintains the positive class boundary (sigmoid outputs near 1.0 for positives). Focal loss on hard negatives focuses gradient on the hardest false alarms without distorting the positive class distribution. It's the best of both worlds.

### FAPH — 500-file estimate (NOISY, needs full validation)
focal_g2_hardonly_s42: ~4.03 @0.80, ~1.01 @0.90 (500 files, 0.99h)
**WARNING**: Poisson 95% CI for 1 trigger in 1h = [0.025, 5.57] FAPH. This number is NOT statistically meaningful. Full 5.4h run launched on test-clean.

---

## Pattern 25: R3 Same-Domain Mining — Contamination Warning

### Finding
Mined 90 hard negatives from dev-clean through round2_best (score >0.3, 2s dedup). Trained 3 variants × 2 seeds.

R3 TP (held-out): r3_10x_s42 = 96.9%@0.80, 93.1%@0.90. Marginally better than R2 (+0.5% at 0.80).

### CRITICAL: Test-Set Contamination Risk
Mining hard negatives FROM dev-clean and measuring FAPH ON dev-clean = optimizing for the eval metric. Pattern 11 already showed this produces 0 FAPH through memorization (not generalization).

**Validation approach**: FAPH measured on test-clean (5.4h, never used for mining). If FAPH drops on test-clean, the improvement is real. If it only drops on dev-clean, it's memorization. Test-clean FAPH runs launched.

### The 90 hard negatives ARE the false alarms
Top scorers: 2412-153954-0016 (0.999), 777-126732-0066 (0.998), 2803-154328-0007 (0.982) — identical to the top false alarms in the FAPH test. Training on them will memorize these specific acoustic patterns.

---

## Pattern 26: Challenger v4 Synthesis — Agreed Truths

### Methodology
Adversarial challenger agent reviewed ALL experimental evidence files. Main agent challenged back. Convergence on:

### Agreed Truths
1. **500-file FAPH estimates are statistical noise.** Only full 5.4h runs are trustworthy. Poisson CI makes point estimates meaningless on <2h.
2. **focal_g2_hardonly_s42 has the best TP of any hardened model** (95.7%@0.90 vs R1's 92.6% and R2's 94.5%). Mechanism: focal on hard negs preserves positive class boundary.
3. **R3 must validate on held-out corpus** (test-clean), not the mined corpus (dev-clean).
4. **Multi-window confirmation had a counting bug** — reset() zeroed cumulative counts per file. Re-running with fix. Temporal analysis valid: 57-80% of false alarm events are transient (1-window).
5. **864-dim temporal embeddings are an unexplored high-potential path** — 24.5% EER improvement but FAPH never measured. This is the biggest blind spot.
6. **TTS-only positive evaluation** limits confidence in real-world TP rates. No valid human recordings exist.
7. **LibriSpeech monoculture** — all FAPH measured on clean audiobook speech. Real-world may differ significantly.
8. **Independence assumption for combining techniques is INVALID** — cannot multiply focal + multi-window + mining reductions.

### Disagreements
- **Confidence level**: Challenger: 3/10. Main: 5/10. Challenger more pessimistic about generalization to real-world; Main sees proven 88% FAPH reduction per mining round as evidence the loop works.
- **Multi-window TP impact**: Challenger cited 40%→17% drop from 30-file test. Main: that test used non-English TTS (eval_fresh), not held-out English TTS. Invalid input data. Real impact unknown until re-run.
- **Multi-window physics**: Challenger claims short utterances might be rejected. Main: 400ms utterance at 100ms step = 4 consecutive windows — 2-of-3 should easily pass.

### Recommended Action (ranked by expected impact)
1. **Full-corpus FAPH for focal_g2_hardonly on test-clean** (RUNNING)
2. **Full-corpus FAPH for r3_10x on test-clean** (RUNNING)
3. **Multi-window re-run with bug fix** (RUNNING)
4. **864-dim temporal model FAPH test** (NOT YET STARTED — should do)
5. **Acquire train-clean-100 (100h)** for statistical validation
6. **Collect real human recordings** for TP validation

---

## Pattern 27: Multi-Window Confirmation — MASSIVE FAPH Reduction (DEFINITIVE)

### Finding
Ran 6 N-of-M configurations on full 5.39h dev-clean with faph_hardened_s43. Bug-fixed run (cumulative counts preserved across files).

| Config | @0.80 FAPH | @0.90 FAPH | @0.95 FAPH | Reduction |
|---|---|---|---|---|
| 1-of-1 (baseline) | 6.31 | 4.64 | 2.23 | — |
| 2-of-2 | 1.86 | 0.93 | 0.74 | 70-80% |
| 2-of-3 | 2.41 | 1.11 | 0.74 | 62-76% |
| **3-of-3** | **0.93** | **0.74** | **0.37** | **83-85%** |
| 3-of-5 | 0.93 | 0.74 | 0.37 | 83-85% |

**3-of-3 @ 0.80 = 0.93 FAPH — FIRST CONFIG BELOW 1 FAPH TARGET.**

### Temporal Analysis (top 50 false alarms)
- TRANSIENT (1 window spike): 25/50 (50%) — eliminated by any N≥2
- BRIEF (2 consecutive windows): 8/50 (16%) — eliminated by 3-of-3
- SUSTAINED (3+ consecutive): 17/50 (34%) — survives even 3-of-3

The sustained false alarms are real confusables (speaker 2803: 4 consecutive windows >0.80, speaker 6313: 3 consecutive, speaker 1272: 4 consecutive). These are the hard cases that training must fix.

### Statistical note
5 triggers in 5.39h = 0.93 FAPH. Poisson 95% CI: [0.30, 2.17]. Not yet statistically proven below 1.0 (upper CI = 2.17). Need 50-100h corpus OR the 5 surviving triggers to be eliminated by a better model.

### TP impact
Positive eval on 30 eval_fresh files (non-English TTS) showed 56.7% for all multi-window configs vs 86.7% baseline. BUT eval_fresh is known unreliable (non-English voices, Pattern from prior work). Real English "Viola" produces 4-6 consecutive high windows — 3-of-3 should pass easily. Need audio-level TP test on held-out English positives.

### Key insight
Multi-window 3-of-3 is the single highest-impact technique discovered. 83-85% FAPH reduction is MORE than any single training round (R1: 88% on test-clean mining, but that was memorization on the mining corpus). Combined with focal_g2_hardonly (better model), this could push well below 0.5 FAPH.

### Implication
**The operating point 3-of-3 @ threshold 0.80 is the production recommendation.** It hits <1 FAPH on the OLD model. With focal_g2_hardonly (better TP preservation) and the 5 surviving sustained false alarms targeted by R3 mining, the combined system should reach <0.5 FAPH.


## Pattern 28: Test-Clean FAPH — Uncontaminated Model Comparison (DEFINITIVE)

### Finding
Three models tested on full 5.40h test-clean (never used for mining or training). This is the ONLY uncontaminated FAPH comparison.

| Model | @0.80 FAPH | @0.90 FAPH | @0.95 FAPH | Training Method |
|---|---|---|---|---|
| focal_g2_hardonly_s42 | 3.33 | 2.41 | 1.11 | Focal loss (hard negs only) |
| round3_s42 | 3.70 | 2.04 | 1.11 | Dev-clean mining, basic |
| **r3_10x_s42** | **2.96** | **1.85** | **0.93** | Dev-clean mining, 10x weight |

### Key findings
1. **r3_10x_s42 wins at all thresholds** — best uncontaminated FAPH of any model tested
2. **r3_10x already hits 0.93 FAPH @0.95** without multi-window — below target!
3. R3 same-domain mining DOES generalize: 41-53% FAPH reduction on held-out corpus (contradicts contamination concern from Pattern 25)
4. focal_g2_hardonly is middle-pack — better at @0.80 than round3 but worse at @0.90
5. All three models converge at @0.95 (1.11 FAPH for focal/round3, 0.93 for r3_10x)

### Why contamination fear was overblown
Dev-clean hard negatives are phonetically confusable with "Viola" — they're not random noise that only appears in dev-clean. The model learns to suppress these phonetic patterns, which transfer to any English speech corpus. Same-domain mining ≠ memorization when the hard negatives are phonetically motivated.

### Statistical note
r3_10x @0.95: 5 triggers in 5.40h = 0.93 FAPH. Poisson 95% CI: [0.30, 2.17]. Point estimate below 1.0 but upper CI is 2.17. Statistical significance requires longer test or fewer triggers.

### Next step
Running r3_10x + 3-of-3 multi-window on test-clean. If the 85% multi-window reduction applies to r3_10x the same way it did to faph_hardened_s43, projected FAPH @0.80 would be ~0.44 (2.96 × 0.15). But this assumes independence — must measure directly.


## Pattern 29: 864-dim Temporal Concat — CATASTROPHIC FAPH Failure

### Finding
The 864-dim temporal concatenation model (concat_864_wide_s42) was the challenger's #1 recommended experiment — "biggest blind spot." It looked great on cached embedding evaluation (best EER=0.0145, best d'=5.755). But on full FAPH test-clean:

| Model | @0.80 FAPH | @0.90 FAPH | @0.95 FAPH |
|---|---|---|---|
| **concat_864_wide_s42** | **11.84** | **8.88** | **7.59** |
| r3_10x_s42 (96-dim) | 2.96 | 1.85 | 0.93 |
| focal_g2_hardonly (96-dim) | 3.33 | 2.41 | 1.11 |

The 864-dim model is **4x WORSE** than mean-pooled 96-dim models at every threshold.

### Root cause
**Distribution shift between cached embeddings and sliding-window inference.**

The temporal cache (embedding_cache_temporal.npz) contains embeddings extracted from pre-segmented audio clips — each clip is a carefully bounded segment (center-cropped to 1.5s). The sliding window in FAPH testing produces clips starting at arbitrary positions, crossing speech/silence boundaries, catching partial words. The 864-dim model learned temporal patterns specific to clean, centered clips. Mean-pooling is robust to this because it averages away temporal ordering — the 96-dim embedding is similar regardless of alignment. Concatenation preserves the exact frame ordering, which is DIFFERENT in sliding-window vs pre-segmented data.

### Key lesson
**Embedding-level metrics (EER, d-prime, AUC) do NOT predict FAPH.** The 864-dim model had the best EER of any architecture tested, but the worst FAPH by far. You MUST run full sliding-window FAPH tests before drawing conclusions. Held-out embedding eval is necessary but not sufficient.

### Implication
864-dim temporal information is a dead end for this pipeline. Mean-pooling is not just simpler — it's fundamentally more robust to the alignment variability inherent in sliding-window wake word detection. If temporal information is to be exploited, it must be done within the sliding window itself (e.g., attention over frames), not by simple concatenation.


## Pattern 30: PRODUCTION CONFIG — r3_10x + 3-of-3 = 0.37 FAPH (DEFINITIVE)

### Finding
Combined the best model (r3_10x_s42) with multi-window confirmation on full 5.40h test-clean (uncontaminated). This is the definitive production configuration result.

| Config | @0.80 FAPH | @0.90 FAPH | @0.95 FAPH |
|---|---|---|---|
| 1-of-1 (baseline) | 2.96 | 1.85 | 0.93 |
| 2-of-2 | 0.74 | 0.37 | 0.19 |
| 2-of-3 | 0.74 | 0.37 | 0.19 |
| **3-of-3** | **0.37** | **0.37** | **0.19** |
| 3-of-5 | 0.37 | 0.37 | 0.19 |

### The production recommendation

**Model: r3_10x_s42.onnx, Threshold: 0.80, Confirmation: 3-of-3, FAPH: 0.37**

This means: 2 false triggers in 5.4 hours of continuous English speech. In a typical home environment (much less continuous speech), real-world FAPH would be dramatically lower.

Alternative: **2-of-2 @ 0.80 = 0.74 FAPH** — simpler implementation (2-element deque), still well below 1.0 target.

### Temporal analysis
At @0.80, r3_10x has 16 false alarm events:
- TRANSIENT (1 window): 12/16 (75%) — eliminated by any N≥2
- BRIEF (2 windows): 2/16 (12.5%) — eliminated by 3-of-3
- SUSTAINED (3+ windows): 2/16 (12.5%) — survives all multi-window configs

The 2 surviving sustained FAs are:
1. Speaker 5639 (file 5639-40744-0022.flac) — 4 consecutive windows >0.80 (score peaks 0.9978)
2. Speaker 908 (file 908-31957-0010.flac) — 3 consecutive windows >0.80 (score peaks 0.9914)

These are genuine phonetic confusables that no amount of multi-window can fix. They need targeted model training (e.g., mine these specific speakers as R4 hard negatives) OR a higher threshold.

### Statistical confidence
2 triggers in 5.40h: Poisson 95% CI for FAPH = [0.045, 1.34]. Upper bound 1.34 — still reasonably close to 1.0 but not yet proven below 1.0 with 95% confidence. However, the 2-of-2 result (4 triggers, CI [0.20, 1.89]) and the consistent pattern across thresholds give high practical confidence.

### Positive eval caveat
TP on 30 TTS files drops from 70% (1-of-1) to 57% (multi-window). BUT: 13 non-detecting files are non-English TTS (Italian, Spanish, German, French, Portuguese) that produce only 1 scoring window. The 7 English files that DO produce multiple high-scoring windows all trigger reliably at every threshold. Real English "Viola" in continuous speech will produce 3-6 overlapping high windows — multi-window should have minimal TP impact for the target use case.

### What this means for ViolaWake v1.0
The FAPH crisis is resolved. The production pipeline is:
1. OWW embedding extraction (96-dim mean-pooled)
2. r3_10x_s42 MLP classifier (96→64→32→1)
3. 3-of-3 multi-window confirmation at threshold 0.80
4. 2-second debounce

Expected performance: **<0.5 FAPH** on English speech, **>95% TP** on held-out English positives (confirmed by held-out embedding eval: 96.9% @0.80). Production TP validation with real mic input is the remaining gap.

---

## Pattern 31: Synthetic FAPH Champion Fails Live Microphone Test

### Finding
Live head-to-head comparison (2026-03-27, ~4 minutes, real microphone, real music, real podcast) reveals that r3_10x_s42 — the synthetic FAPH champion (0.37 FAPH on test-clean) — has **0% live recall**. Max score 0.501, never crosses 0.80 threshold from real microphone input.

| Model | Live Recall | Max Score (speaking) | FP (music) | FP (podcast) | Noise Floor (talking avg) |
|-------|------------|---------------------|------------|-------------|--------------------------|
| **temporal_cnn** | **100%** | 0.990 | 0 | 0 | 0.106 |
| viola_mlp_oww | 100% | 1.000 | 0 | 2 | 0.025 |
| temporal_convgru | 100% | 0.989 | 0 | 1 | 0.059 |
| r3_10x_s42 | **0%** | 0.501 | 0 | 0 | 0.003 |

### Why this matters
r3_10x_s42 was selected as production champion based on synthetic evaluation (LibriSpeech test-clean FAPH = 0.37, held-out TP = 96.9%). But it completely fails to detect the wake word from a real microphone. This means:
1. Synthetic FAPH evaluation is necessary but NOT sufficient for production readiness
2. Held-out embedding TP doesn't predict live microphone recall
3. The model likely overfit to the specific acoustic characteristics of cached embeddings vs live streaming embeddings

### Root cause
r3_10x_s42 was trained and evaluated on cached embeddings extracted via `embed_clips()` (batch mode). Live inference uses streaming frame-by-frame extraction with sliding windows. Even though the SDK normalization was verified (cosine=1.000 on single clips), the frame alignment and windowing differences in continuous streaming produce subtly different mean-pooled vectors that never reach the model's learned decision boundary.

### Implication
**Every model MUST pass live microphone validation before shipping.** The evaluation hierarchy is now:
1. Synthetic eval (EER, Cohen's d) — development signal
2. Streaming FAPH on LibriSpeech — false positive gate
3. **Live microphone test** — final ship gate (NEW, mandatory)

temporal_cnn is the new production recommendation: 100% live recall, 0 FP on music, 0 FP on podcast speech. Higher noise floor (peak 0.774 during talking, avg 0.106) mitigated by 3-of-3 multiwindow confirmation.

### Streaming FAPH Validation (temporal_cnn, test-clean 5.40h)

| Confirm | @0.80 | @0.85 | @0.90 | @0.95 |
|---------|-------|-------|-------|-------|
| 1-of-1 | 7.40 | 3.33 | 1.11 | 0.19 |
| 2-of-2 | 3.15 | 0.93 | 0.56 | 0.19 |
| **3-of-3** | **0.93** | **0.37** | **0.19** | **0.19** |

At @0.85 + 3-of-3, temporal_cnn achieves 0.37 FAPH — matching r3_10x_s42's best synthetic score while having 100% live recall. Score distribution: mean=0.011, p99=0.21, p99.9=0.67, max=0.97. 5 triggers in 5.40h at production config (@0.80, 3-of-3).
