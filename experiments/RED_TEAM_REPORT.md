# Red Team Challenge Report: ViolaWake Accuracy Campaign

**Date**: 2026-03-26
**Reviewer**: Red Team Challenger (adversarial audit)
**Scope**: All 7 claims from the ViolaWake accuracy improvement experiments
**Methodology**: Independent file review, data verification, code tracing, numerical checks

---

## Overall Confidence Assessment

**Reaching <5% EER on the current eval set is UNLIKELY without fundamental changes.**

The current best is 13.14% EER (BCE) on an eval set that is 100% TTS-synthesized. The experiments have explored loss functions, data augmentation, two-stage detection, and architecture variants -- all yielding results in the 11.9%-14.3% range. The bottleneck is not methodology noise; it is a combination of (a) the 96-dim OWW embedding's limited representational capacity and (b) a training corpus dominated by a single phrase ("viola" standalone) with no carrier-phrase diversity.

Getting to <5% EER will require either a fundamentally richer feature extractor or a dramatically different training data composition. Incremental tuning of the existing MLP + OWW pipeline is approaching its ceiling.

---

## Claim 1: "BCE loss gives 13.14% +/- 0.21% EER vs focal's 13.56% +/- 0.10%"

**Verdict: CHALLENGED -- the comparison is methodologically inconsistent**

### Evidence

1. **The 11.9% loss sweep number and the 13.14% verification number use different evaluation pipelines.** The loss sweep (`exp_loss_sweep.py`) evaluates PyTorch models in-memory via `evaluate_model_torch()`, which runs `find_optimal_threshold()` on all positives. The BCE verification (`exp_verify_bce.py`) exports to ONNX, then evaluates via the SDK's `evaluate_onnx_model()`, then runs `find_optimal_threshold()` excluding "viola_wake_up" and "viola_please" for the `trained_eer` metric.

2. **Critical: EMA is applied inconsistently.** The loss sweep (`exp_loss_sweep.py` lines 68-103) restores `best_state` but never applies EMA weights. The BCE verification (`exp_verify_bce.py` lines 143-148) applies EMA before ONNX export. EMA smoothing can shift decision boundaries significantly on a small MLP. This means the 11.9% BCE result in the loss sweep was measured on non-EMA weights, while the 13.14% ONNX result was measured with EMA. The 2 percentage point gap is at least partly an artifact of this difference, not ONNX quantization.

3. **The focal baseline (13.56%) cited in the comparison is from `all_results.json`, which uses the main harness (`run_all_experiments.py`).** That harness uses FocalLoss with `gamma=2.0, alpha=0.75, label_smoothing=0.05`. But the loss sweep's focal variants range from 12.5% to 16.2%. The "default" focal config (`focal_g2_a75_ls05`) got 14.05% in the sweep, not 13.56%. So the focal baseline being compared against varies depending on which file you look at.

4. **The claim is likely directionally correct** -- BCE does appear to slightly outperform focal loss on this data. But the magnitude (0.42% absolute improvement) is within the noise range given the methodological inconsistencies.

### Specific Numbers

| Source | BCE EER | Focal EER | Delta | EMA Applied? |
|--------|---------|-----------|-------|-------------|
| Loss sweep (PyTorch, no EMA) | 11.9% | 14.05% | -2.15% | No |
| BCE verification (ONNX, EMA) | 13.14% +/- 0.21% | -- | -- | Yes |
| all_results.json (ONNX, focal+EMA) | -- | 13.56% +/- 0.10% | -- | Yes |
| Apples-to-apples (both ONNX+EMA) | 13.14% | 13.56% | -0.42% | Both Yes |

### Recommendation

Re-run the loss sweep with ONNX export + EMA for every variant. The current loss sweep numbers are not comparable to the verification numbers. Until then, the apples-to-apples delta is 0.42%, which is statistically marginal given only 3 seeds and overlapping confidence intervals.

---

## Claim 2: "Architecture doesn't matter -- data is the bottleneck"

**Verdict: PARTIALLY CHALLENGED -- the architecture search was too narrow**

### Evidence

1. **Experiment A (`exp_a_results.json`) tests per-frame scoring policies, not architectures.** It evaluates the existing production ONNX model under different decision policies (mean, max, median, top3_mean, consecutive_2, consecutive_3, moving_avg_3). All policies yield EER in the 18.7-23.3% range. This does NOT test different architectures -- it tests different aggregation strategies on the SAME model.

2. **The architecture variants in `experiment_config.json` (lines 72-93) are all MLPs with the same structure** -- just different widths/depths:
   - default: [64, 32]
   - wide: [128, 64]
   - deep: [128, 64, 32]
   - narrow: [32, 16]

   These are all shallow feed-forward networks with ReLU + Dropout. No attention mechanism, no 1D convolution, no temporal processing. The "architecture exploration" is actually just "MLP size exploration."

3. **The `all_results.json` only contains one result (D_combined with default architecture).** There are no results showing wide/deep/narrow variants. The claim that "architecture doesn't matter" appears to be based on the observation that D_combined's EER is similar across data variations -- but the architecture dimension was never properly explored.

4. **The OWW embedding is 96-dimensional.** The MLP sits on top of a fixed 96-dim input. On such a low-dimensional input, the claim that a [64,32] vs [128,64] MLP "doesn't matter" is plausible -- but this does not mean architecture fundamentally doesn't matter. It means the feature extractor is the bottleneck, which is a different claim.

### Recommendation

The claim should be restated as: "MLP width/depth doesn't matter on 96-dim OWW embeddings." This is likely true. The real architecture question is whether replacing the OWW embedding backbone (96-dim) with a richer one (768-dim HuBERT/wav2vec2 or 192-dim ECAPA) would improve performance. The `feature_extractor_results.json` explores this (see Claim 6), but no end-to-end training + evaluation on alternative embeddings has been completed.

---

## Claim 3: "Two-stage detection doesn't beat single-stage"

**Verdict: CHALLENGED -- the implementation has a fundamental flaw in its EER calculation**

### Evidence

1. **The two-stage EER calculation is wrong.** `exp_two_stage.py` line 219 computes `eer_approx = (frr + far) / 2`. This is NOT EER. EER is the operating point where FRR equals FAR, not the average of arbitrary FRR and FAR at a grid point. The grid search finds the combo with the lowest `(frr + far) / 2`, which is a different optimization target.

2. **The "combined EER" (line 229-233) uses `min(primary_score, verifier_score)` as the combined metric.** This is a reasonable approach but discards information. A proper two-stage system would use the primary as a gate (hard threshold) and then evaluate the verifier's EER on samples that pass stage 1. Using `min()` collapses two independent scores into one, which loses the cascaded nature.

3. **The verifier was trained on only 325 confusable negatives.** The verifier sees positives + confusable negatives only (`neg_confusable` tag: 325 samples). This is an extremely small negative set for a verifier that needs to generalize. The verifier achieves pos_mean=0.9377, neg_mean=0.3162 -- meaning it scores confusable negatives at 0.32, far from 0.0. This suggests the verifier barely learned the discrimination task.

4. **The result: combined_eer=0.1395 vs single_stage_eer=0.1317, delta=+0.0078.** Two-stage is *worse*. But given the flawed EER calculation and the underpowered verifier, this does not constitute evidence that two-stage detection is inherently inferior. It constitutes evidence that THIS particular implementation doesn't work.

5. **Critical observation from the grid search:** The best grid result (`primary_threshold=0.45, verifier_threshold=0.8`) yields FRR=21.88%, FAR=4.4%. These are far from equal. A properly computed EER would likely be different from the reported 13.14% "eer_approx."

### Recommendation

Redesign the two-stage experiment:
- Train the verifier on a larger and more diverse negative set
- Compute proper EER by sweeping thresholds on the verifier CONDITIONED on primary passing
- Consider a learned fusion (e.g., logistic regression on [primary_score, verifier_score]) instead of min()
- Use at least 5 seeds for the verifier given its tiny training set

---

## Claim 4: "ACAV100M features are compatible with our pipeline"

**Verdict: VERIFIED with caveats**

### Evidence

1. **Dimensionality matches.** Our ACAV embeddings: `(100000, 96)` float32. Cache embeddings: `(N, 96)` float32. Same 96-dim space.

2. **Our ACAV embeddings are correctly derived from the OWW source.** The original OWW ACAV file is `(5,625,000, 16, 96)` float16 -- 5.6M clips, 16 frames each, 96 dims. Our file takes the first 100,000 clips and applies `mean(axis=1)` (averaging across 16 frames), exactly matching what `_embed()` does in `run_all_experiments.py` line 112: `embeddings.mean(axis=1)[0]`. First-row verification confirms: `OWW[0].mean(axis=0)` matches `Our[0]` to within float16->float32 conversion tolerance.

3. **Distribution statistics are compatible:**
   - Cache positive norms: mean=138.93, std=7.70
   - Cache negative norms: mean=138.93, std=7.70 (approximately)
   - ACAV norms: mean=139.78, std=7.56
   - Per-dimension means are in the same range

4. **Caveat: We only use 100K of 5.6M available clips (1.78%).** This is a subsample. The G_acav experiment is marked `pending_data` and has not been run. The compatibility claim is verified, but the actual impact of ACAV data on training has NOT been tested.

5. **Caveat: The mean-across-frames operation may lose temporal information.** The OWW ACAV has 16 frames per clip. By averaging, we collapse temporal structure. For wake word detection, the temporal pattern (syllable ordering) may be important. This is a potential reason why the 96-dim embeddings have limited discriminative power between "viola" and "vanilla."

### Recommendation

Run experiment G_acav to actually test ACAV impact. Also consider: instead of mean-pooling 16 frames into a single 96-dim vector, concatenate or use a temporal model (1D conv or LSTM) over the 16x96 frame sequence. This would give the classifier access to temporal ordering information.

---

## Claim 5: "The eval set is comprehensive enough"

**Verdict: REFUTED -- the eval set has critical gaps**

### Evidence

1. **The eval set is 100% TTS-generated.** Zero real human speech recordings. The MANIFEST.md confirms this: "All positives are freshly synthesized via TTS." LibriSpeech was supposed to be included for negatives but was skipped ("--skip-librispeech or torchaudio unavailable"). The RESULTS.md acknowledges this caveat.

2. **The eval set size is small.** 288 positives and 546 negatives = 834 total samples. For comparison, Picovoice's published benchmarks use thousands of real-speech recordings. With 288 positives, a single misclassification shifts EER by ~0.35%. This makes the measurement noisy.

3. **Speaker diversity is limited.** 23 positive voices (22 Edge TTS + 1 SAPI5), 10 negative voices. All are synthetic neural TTS. Kokoro voices failed entirely (36 errors noted in MANIFEST.md). Real-world wake word systems face thousands of unique speakers with accents, ages, genders, and speaking styles that TTS cannot simulate.

4. **No far-field or noisy real-world conditions.** The eval only has synthetic reverb (RT60=0.3s, 70/30 dry/wet) and synthetic pink noise (20dB SNR). Real-world conditions include reverberation from large rooms, concurrent music playback, TV speech, multi-speaker environments, and fan noise -- none of which are represented.

5. **The eval set already found a real weakness:** Irish English (en-IE-EmilyNeural) achieves only 25% pass rate. This single voice accounts for a disproportionate share of false rejects. With more diverse real speakers, there would likely be MORE accents that fail similarly.

6. **"viola_wake_up" scores 0.0% across all seeds** (per-phrase analysis from BCE score CSVs: mean score ~0.1, 0 passes out of 72 attempts per seed). This is included in the all_eer calculation but was never in the training vocabulary. The eval set's inclusion of untrained phrases without clearly separating them in the primary metric inflates the reported EER.

7. **Positive data in the eval is phrase-imbalanced:** 72 "hey_viola" + 72 "ok_viola" + 72 "viola" + 72 "viola_wake_up" = 288. The untrained "viola_wake_up" constitutes 25% of all positives. Since it scores near 0.0, it single-handedly inflates all_eer by approximately 25% * miss_rate.

### Recommendation

- Create a real-speech evaluation subset (even 50 real recordings would be valuable)
- Drop "viola_wake_up" from the standard EER reporting or report it separately
- Add LibriSpeech negative clips (fix the torchaudio dependency)
- Include recorded noise conditions (music playback, TV, fan noise)
- Report trained-phrase EER as the PRIMARY metric, all-phrase EER as secondary

---

## Claim 6: "HuBERT gives better feature separation"

**Verdict: CHALLENGED -- tested only on a single pair, not on the full eval set**

### Evidence

1. **The feature extractor test (`feature_extractor_results.json`) tests each backend on exactly TWO files:** one positive (`en-AU-NatashaNeural_hey_viola.wav`) and one negative (`en-AU-NatashaNeural_hey_violet.wav`). Cosine similarity results:
   - OWW: 0.966 (very similar)
   - HuBERT: 0.934 (slightly more separated)
   - wav2vec2: 0.695 (much more separated)
   - ECAPA: 0.748 (more separated)

2. **A single-pair test is not a feature separation evaluation.** You cannot draw conclusions about feature space quality from cosine similarity of one positive-negative pair. The claim should be based on computing EER or d-prime across the full eval set with each feature extractor. This was never done.

3. **Latency comparison is valid but incomplete:**
   - OWW: 79ms mean
   - HuBERT: 92ms mean (+13ms, +16%)
   - wav2vec2: 81ms mean (+2ms, +3%)
   - ECAPA: 98ms mean (+19ms, +24%)

   These are ALL within acceptable real-time bounds. The 13ms HuBERT overhead is NOT a disqualifier for wake word detection, which operates on ~1.5s audio clips with a latency budget of hundreds of milliseconds.

4. **Memory costs are dramatically different:**
   - OWW: 398 MB
   - HuBERT: 1125 MB (+727 MB)
   - wav2vec2: 1571 MB (+1173 MB)
   - ECAPA: 1694 MB (+1296 MB)

   For a desktop application (which ViolaWake targets), 1-2 GB is likely acceptable. For embedded/mobile, it would be prohibitive.

5. **The most promising extractor (wav2vec2, cosine sim 0.695) was never followed up on.** If the single-pair test is directionally meaningful, wav2vec2 provides the best separation AND has near-OWW latency (81ms vs 79ms). No experiment trains an MLP on wav2vec2 embeddings.

### Recommendation

Run a proper feature comparison:
- Extract embeddings from the full eval set using each extractor
- Train MLP classifiers on each embedding type using the D_combined data split
- Report EER, AUC, and d-prime for each
- Pay special attention to wav2vec2 (768-dim, 81ms, cosine sim 0.695)

---

## Claim 7: "The embedding cache is correct"

**Verdict: VERIFIED with one significant concern**

### Evidence

1. **No train/eval data leakage.** Independent verification confirms zero filename overlap between training files and eval files. The eval set is entirely freshly synthesized TTS.

2. **Correct label distribution:**
   - Total: 35,344 embeddings (21,042 positive, 14,302 negative)
   - Class imbalance: 1.47:1 (positives outnumber negatives)
   - This is unusual -- typically negatives should dominate. The cause is 8x augmentation on positives only (line 195-205 of `run_all_experiments.py`): 2,338 original positives become 21,042 after augmentation.

3. **Augmentation ratio confirmed:** 2,338 original positive files, 18,704 augmented = exactly 8.0x augmentation as configured.

4. **Tag/source breakdown is consistent with config:**
   - pos_main: 12,402 (1,378 base * ~9 with augmentation)
   - pos_diverse: 1,053 (117 base * ~9)
   - neg_main: 10,175 (no augmentation)
   - neg_confusable: 325 (no augmentation)

5. **SIGNIFICANT CONCERN: Training data phrase composition.** The pos_main training data contains:
   - 693 "viola" standalone files
   - 685 "other" files (legacy recordings named `viola_NNNNN.wav`, `sample_NNN.wav`, `vee-ola_NNNNN.wav`)
   - **ZERO** "hey_viola" files (these exist only in `_excluded/`)
   - **ZERO** "ok_viola" files
   - **ZERO** "viola_wake_up" files

   The pos_diverse adds 117 files with "hey_viola", "ok_viola", and "viola" variants from 13 new voices. But the vast majority of training positives (12,402 out of 13,455, i.e., 92%) are standalone "viola" only.

   Yet the eval set tests all four phrases equally (72 each). The model achieves 99%+ on "hey_viola" and "ok_viola" despite having almost no training examples, which suggests the OWW embedding collapses these phrases to a similar representation. But "viola_wake_up" fails completely (mean score 0.1-0.16), suggesting it occupies a different embedding region.

### Recommendation

The cache itself is technically correct. The concern is with what the cache reveals about training data composition:
- Add "hey_viola" and "ok_viola" training data if the eval set will test them
- Either add "viola_wake_up" training data or remove it from the eval set
- Consider a dedicated experiment testing the effect of carrier-phrase diversity in training data

---

## Cross-Cutting Issues

### Issue A: The "all_eer" metric is misleading

Every experiment reports `all_eer` as the headline metric. This includes "viola_wake_up" which scores near 0.0 (25% of positives). The `trained_eer` (which excludes untrained phrases) is consistently ~10 percentage points better:
- BCE: all_eer=13.14%, trained_eer=1.78%
- D_combined: all_eer=13.43%, trained_eer=1.79%

The headline 13% EER is artificially inflated by including a phrase the model was never trained on. If "viola_wake_up" is removed from the eval set, the current model already achieves ~1.8% EER on trained phrases. **This is already below the <5% EER target.**

However, this 1.8% is measured on only ~216 trained-phrase positives vs 546 negatives on a TTS-only eval set. The real-world trained-phrase EER is unknown.

### Issue B: No architecture experiments were actually completed

The config defines 4 architecture variants. The experiment results contain only `default` architecture runs. No wide, deep, or narrow variants were trained and evaluated. The claim that "architecture doesn't matter" is unfounded.

### Issue C: Promising experiments were never run

- G_acav (ACAV100M negatives): marked `pending_data` but data exists and is compatible
- H_confusable_v2 (phoneme-mined confusables): pending
- I_full_corpus (everything): pending
- wav2vec2 end-to-end training: never attempted despite promising single-pair results
- Architecture variants (wide/deep/narrow): never run

### Issue D: Reproducibility gap

The loss sweep and verification scripts use different evaluation methods (PyTorch in-memory vs ONNX export), different EMA handling, and report metrics from different code paths. Any comparison across experiments needs to specify which evaluation pipeline was used.

---

## Summary Table

| # | Claim | Verdict | Key Issue |
|---|-------|---------|-----------|
| 1 | BCE beats focal by 0.42% | **CHALLENGED** | Evaluation methodology differs (EMA inconsistency, PyTorch vs ONNX). Direction likely correct, magnitude unreliable. |
| 2 | Architecture doesn't matter | **CHALLENGED** | Only MLP size was tested, not fundamentally different architectures. Feature extractor is likely the real bottleneck. |
| 3 | Two-stage doesn't help | **CHALLENGED** | Flawed EER calculation (`(frr+far)/2` is not EER), underpowered verifier (325 negatives), poor fusion method. |
| 4 | ACAV100M is compatible | **VERIFIED** | Dimensions match (96), derivation verified (mean across frames from OWW source). Impact on training untested. |
| 5 | Eval set is comprehensive | **REFUTED** | 100% TTS, no real speech, small size, includes 25% untrained phrases that inflate EER. |
| 6 | HuBERT separates better | **CHALLENGED** | Tested on exactly 1 positive-negative pair. wav2vec2 looks more promising. No full-set evaluation done. |
| 7 | Embedding cache is correct | **VERIFIED** | No leakage, correct augmentation. But reveals 92% of training positives are standalone "viola" only. |

---

## Concrete Recommendations (Priority Order)

1. **Fix the eval metric immediately.** Report `trained_eer` (excluding "viola_wake_up") as the primary metric. The current model may already meet <5% EER on trained phrases. Verify by adding more diverse real-speech eval data.

2. **Run the pending experiments.** G_acav, architecture variants (wide/deep/narrow), and a wav2vec2-based pipeline are all ready to execute with existing infrastructure.

3. **Make loss sweep and verification methodologically identical.** Export to ONNX with EMA in all experiments, evaluate with the same SDK function.

4. **Fix the two-stage EER calculation.** Replace `(frr+far)/2` with proper ROC-based EER computation. Retrain verifier on larger negative set.

5. **Add real speech to the eval set.** Even 50-100 real recordings of "viola", "hey viola", "ok viola" from diverse speakers would dramatically improve eval validity.

6. **Train on carrier phrases.** Add "hey_viola" and "ok_viola" to the training corpus (not just pos_diverse's 117 files, but a proper set). If the product needs "viola_wake_up", add it to training.

7. **Explore wav2vec2 embeddings end-to-end.** The single-pair cosine similarity (0.695 vs OWW's 0.966) suggests dramatically better discrimination. Train a classifier on wav2vec2 768-dim embeddings and measure proper EER.
