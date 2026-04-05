> **ARCHIVED 2026-04-05.** Historical analysis of MLP mean-pooling vs streaming. The temporal CNN uses 9-frame sliding windows, resolving this architectural concern.

# Streaming vs Clip Evaluation: Validity Analysis

**Date**: 2026-03-26
**Author**: Claude (investigation agent)
**Question**: Is the offline single-clip evaluation methodology valid, or does it test something that doesn't happen in production?

## TL;DR

**The single-clip eval methodology IS valid.** Production Viola uses the same `embed_clips` pipeline on a 1.5s sliding buffer -- not frame-by-frame streaming. However, the eval has a **pooling mismatch** (mean vs max) with production, and the "viola wake up" failure is genuine because the entire 1.5s clip contaminates all 9 embedding frames.

The hypothesis that "streaming would catch 'viola' before 'wake up' arrives" is **incorrect** -- production does not do frame-level streaming for the MLP model.

## Finding 1: Production uses clip-level processing, NOT frame-by-frame streaming

### The SDK `WakeDetector` (frame-level) is NOT what production Viola uses

The ViolaWake SDK at `src/violawake_sdk/wake_detector.py:186-218` defines a `WakeDetector.process()` method that feeds 20ms frames (320 samples) to a separate `oww_backbone.onnx` model. This IS frame-level streaming.

**BUT** the `oww_backbone.onnx` model is not available locally (not downloaded), and more importantly, **production Viola does not use the SDK's `WakeDetector` class**. Production Viola uses `violawake.engine.ViolaWake`.

### Production Viola uses `embed_clips` on a 1.5s sliding buffer

The actual production inference path is:

1. **Audio accumulation**: `violawake_listener.py:1285` creates a circular buffer:
   ```python
   audio_buffer = np.zeros(CLIP_SAMPLES, dtype=np.float32)  # 24000 samples = 1.5s
   ```

2. **Sliding window**: Every N chunks (~100ms at infer_interval=2), the full 1.5s buffer is passed to `engine.process_audio()` (`violawake_listener.py:1896`).

3. **Embedding extraction**: `engine.py:392-395` uses `embed_clips` on the full 1.5s clip:
   ```python
   embeddings = self._oww_preprocessor.embed_clips(audio_int16.reshape(1, -1), ncpu=1)
   embedding = embeddings.max(axis=1)[0].astype(np.float32)
   ```

4. **MLP classification**: The pooled 96-dim embedding is fed to the MLP.

**Code references**:
- Buffer creation: `NOVVIOLA/voice/wake_detector/violawake_listener.py:1285`
- Buffer fill (circular): `violawake_listener.py:1831-1840`
- Inference call: `violawake_listener.py:1896`
- `embed_clips` + max pool: `NOVVIOLA/violawake/engine.py:392-395`

### What `embed_clips` actually does internally

`openwakeword/utils.py:358-385` (`AudioFeatures.embed_clips`):
1. Computes melspectrogram of the full 1.5s clip (produces ~147 mel frames)
2. Slides a 76-frame window with step=8 across the melspec
3. Each window produces one 96-dim embedding via the embedding model
4. For a 1.5s clip: returns shape `(1, 9, 96)` -- **9 time-position embeddings**

The 9 embeddings cover these approximate time ranges (76 mel frames = ~760ms window, 8-frame step = ~80ms):
| Frame | Mel range | Approx time range |
|-------|-----------|-------------------|
| 0 | 0-76 | 0ms - 760ms |
| 1 | 8-84 | 80ms - 840ms |
| 2 | 16-92 | 160ms - 920ms |
| 3 | 24-100 | 240ms - 1000ms |
| 4 | 32-108 | 320ms - 1080ms |
| 5 | 40-116 | 400ms - 1160ms |
| 6 | 48-124 | 480ms - 1240ms |
| 7 | 56-132 | 560ms - 1320ms |
| 8 | 64-140 | 640ms - 1400ms |

**Code reference**: `openwakeword/utils.py:225-236` (`_get_embeddings` method)

## Finding 2: Eval pipeline matches production -- with one exception

### What eval does

`src/violawake_sdk/training/evaluate.py:340-358` (`_build_oww_scorer`):
1. Loads audio file
2. `center_crop` to CLIP_SAMPLES (24000)
3. Converts to int16
4. `embed_clips` on the full 1.5s clip -- returns (1, 9, 96)
5. **`mean` pool across time axis** (axis=1) -- produces (1, 96)
6. MLP inference

### What training does

`src/violawake_sdk/tools/train.py:317-330` (`_audio_to_embedding`):
Identical to eval: `center_crop` -> int16 -> `embed_clips` -> **mean pool** -> 96-dim

### What production does

`NOVVIOLA/violawake/engine.py:381-395`:
`center_crop` -> int16 -> `embed_clips` -> **MAX pool** -> 96-dim

### The pooling mismatch

| Pipeline | Pooling | Code location |
|----------|---------|---------------|
| Training | `mean(axis=1)` | `tools/train.py:330` |
| Evaluation | `mean(axis=1)` | `training/evaluate.py:356` |
| Production | `max(axis=1)` | `NOVVIOLA/violawake/engine.py:395` |

Training and eval are consistent with each other (both use mean-pool). But production uses max-pool. This means eval metrics accurately reflect the model trained with mean-pool, but the production deployment uses a different pooling strategy.

**Note**: The production config comment at `violawake/config.py:138` says "Mean-pool OWW 96-dim model achieved d-prime 15.10" and the default model is `viola_mlp_oww.onnx`, but the engine code at line 395 uses `.max(axis=1)`. This appears to be a bug -- the model was trained with mean-pool but production runs with max-pool.

## Finding 3: "Viola wake up" single-clip eval IS valid (not unrealistic)

### Why the hypothesis is wrong

The hypothesis was: "In production streaming mode, the detector would catch 'viola' in the first ~300ms window before 'wake up' arrives."

This is incorrect because:

1. **Production is not frame-by-frame streaming.** It processes a 1.5s sliding buffer via `embed_clips`, exactly like eval.

2. **All 9 embedding frames see "wake up" contamination.** For a "viola wake up" clip that is exactly 1.5s (confirmed: `en-AU-NatashaNeural_viola_wake_up.wav` is 24000 samples = 1.500s), `center_crop` returns the entire clip unchanged. The 9 embedding windows span 0-1400ms, and "wake up" occupies roughly 500-1200ms. Even frame 0 (0-760ms) overlaps with the start of "wake up".

3. **There IS a sliding window in production**, but it slides at ~80-100ms intervals, replacing the oldest audio in the circular buffer. When someone says "Viola, wake up", there is a brief moment where the buffer contains mostly "Viola" before "wake up" fills in. **This IS a valid detection opportunity that the single-clip eval misses.**

### But the eval clips are center-cropped to exactly 1.5s

The TTS clips are generated at exactly 1.5s (`CLIP_SAMPLES`). When `center_crop` receives a 1.5s clip, it returns it unchanged -- no cropping happens. The eval processes the entire utterance as a single blob.

In production, the sliding buffer approach means:
- At time T+0.4s: buffer = [old audio | "viola"] -- model might detect
- At time T+0.8s: buffer = [old audio | "viola wake"] -- detection window closing
- At time T+1.2s: buffer = ["viola wake up" | silence] -- contaminated, likely miss

So there IS a ~200-400ms window in production where the buffer contains mostly "viola" without "wake up". The single-clip eval does not capture this transient opportunity.

### Quantifying the gap

However, this window is narrow (200-400ms) and depends on:
- The production infer_interval (every 2 chunks = ~100ms for MLP models)
- Whether the "viola" portion alone fills enough of the 1.5s buffer
- Whether the surrounding old audio is silence or speech

In practice, "viola wake up" would need the user to pause between "viola" and "wake up" for the sliding buffer to catch it reliably. For continuous speech ("violawakeup"), the window is essentially zero.

## Finding 4: Which EER metric to track

| Metric | Value | What it measures |
|--------|-------|-----------------|
| `trained_eer` | 2.35% | EER on phrases the model was trained on ("viola", "hey viola", "ok viola") |
| `all_eer` | 13.14% | EER including untrained phrases ("viola wake up", "viola please") |

### Recommendation: Track `trained_eer` (2.35%)

**`trained_eer` is the correct metric** because:

1. The model was deliberately trained on specific phrases. Evaluating it on untrained phrases measures out-of-vocabulary generalization, not detection quality.

2. "viola wake up" scores 0.005 mean (essentially zero) because the model has never seen this phrase pattern. Including it in EER inflates the error rate with failures that are expected by design.

3. The production wake word is "viola", "hey viola", or "ok viola" -- these are the trigger phrases users are told to say. "viola wake up" is not a documented trigger phrase.

4. `all_eer` is useful as a generalization diagnostic but should not be the primary quality metric.

### Caveat

If the product intent is for "viola" to be detected regardless of what follows (e.g., "viola, play music", "viola, what time is it"), then the model has a real limitation: it fails on "viola + additional words" when the additional words occupy enough of the 1.5s window. This is a training data gap, not an eval methodology problem.

## Summary of findings

| Question | Answer | Evidence |
|----------|--------|----------|
| Does production use streaming? | **No** -- uses 1.5s sliding buffer with `embed_clips` | `violawake_listener.py:1285,1896`, `engine.py:392` |
| Does eval match production? | **Mostly** -- same `embed_clips` path, but **mean vs max pool mismatch** | eval: `evaluate.py:356`, prod: `engine.py:395` |
| Is "viola wake up" eval valid? | **Mostly yes** -- the single-clip eval slightly overstates the failure (production sliding buffer gives a brief detection window), but the clip still genuinely fails | Clip is 1.500s, CLIP_SAMPLES is 24000, no cropping occurs |
| Which EER to track? | **trained_eer (2.35%)** | Model is designed for specific phrases; untrained phrase failure is expected |
| Is there a bug? | **Yes: production uses max-pool but model was trained with mean-pool** | `engine.py:395` vs `train.py:330` |

## Recommendations

1. **Fix the pooling mismatch**: Either change production to use mean-pool (matching training), or retrain with max-pool. The production config comment already says the mean-pool model is the default, so changing `engine.py:395` from `.max(axis=1)` to `.mean(axis=1)` is likely the right fix.

2. **Track `trained_eer`** as the primary metric. Report `all_eer` as supplementary.

3. **Do NOT retrain the model to detect "viola wake up"** unless this is an intentional product requirement. The current model correctly detects the trained trigger phrases.

4. **Consider a streaming eval mode** that simulates the production sliding buffer for more realistic "viola + trailing words" evaluation. This would process clips in 100ms chunks, sliding through a 1.5s buffer, and report the peak score across all windows.
