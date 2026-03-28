<!-- doc-meta
scope: Technical architecture of the ViolaWake wake-word detection pipeline
authority: LIVING
code-paths: src/violawake_sdk/wake_detector.py, src/violawake_sdk/oww_backbone.py, src/violawake_sdk/training/temporal_model.py, src/violawake_sdk/confidence.py, src/violawake_sdk/vad.py, src/violawake_sdk/models.py, src/violawake_sdk/ensemble.py
staleness-signals: backbone asset format changes, new default head architecture, decision-policy changes, model distribution changes, VAD backend reorder, ensemble input-shape changes
-->
# ViolaWake SDK Architecture
This document explains the runtime wake-word detection pipeline used by the ViolaWake SDK.
It is written for contributors, integrators, and power users who need to understand where scores come from, how buffering and gating work, and how models are loaded and combined.
The core design is intentionally split into two layers:
1. A frozen OpenWakeWord backbone that converts audio into embeddings.
2. A ViolaWake wake head that scores those embeddings.
The decision policy then turns scores into a boolean detection event.
Primary implementation refs: `src/violawake_sdk/wake_detector.py:33`; `src/violawake_sdk/wake_detector.py:201`; `src/violawake_sdk/wake_detector.py:440`; `src/violawake_sdk/oww_backbone.py:103`; `src/violawake_sdk/oww_backbone.py:131`; `src/violawake_sdk/training/temporal_model.py:49`; `src/violawake_sdk/training/temporal_model.py:173`; `src/violawake_sdk/confidence.py:47`; `src/violawake_sdk/vad.py:263`; `src/violawake_sdk/models.py:44`; `src/violawake_sdk/ensemble.py:26`
---
## 1. Pipeline Overview
At a high level, the SDK does not classify raw audio directly. It classifies OpenWakeWord embeddings.
```text
16 kHz mono PCM
    |
    v
+--------------------+
| Audio validation   |
| 20 ms frames       |
| 320 samples/frame  |
+--------------------+
    |
    v
+--------------------+
| OWW mel frontend   |
| melspectrogram.*   |
| 32 mel bins        |
+--------------------+
    |
    v
+--------------------+
| OWW embedding net  |
| embedding_model.*  |
+--------------------+
    |
    v
+--------------------+
| 96-d embedding     |
| one vector/window  |
+--------------------+
    |
    +------------------------------+
    |                              |
    v                              v
+--------------------+     +----------------------+
| MLP legacy head    |     | Temporal head        |
| single embedding   |     | seq of embeddings    |
+--------------------+     +----------------------+
    |                              |
    +---------------+--------------+
                    |
                    v
+------------------------------------------+
| Wake score in [0, 1]                     |
| optional score history / confidence API  |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
| 4-Gate decision policy                   |
| 1. Zero-input guard                      |
| 2. Score threshold                       |
| 3. Cooldown                              |
| 4. Listening gate                        |
| + optional confirm_count multi-window    |
+------------------------------------------+
                    |
                    v
            Detection event
```
The public `WakeDetector.detect()` path processes audio in this order:
1. Validate the audio frame.
2. Compute raw-frame RMS on int16-scale PCM.
3. Normalize model input if needed.
4. Ask the OWW backbone for the latest embedding.
5. Run either a single-embedding head or a temporal head.
6. Update score history.
7. Apply multi-window confirmation if enabled.
8. Apply the 4-gate policy.
9. Emit `True` only if every enabled stage accepts the event.
Runtime entry points: `WakeDetector.process()` returns a score only; `WakeDetector.detect()` returns a boolean event; `WakewordDetector` is a lazy-loading compatibility wrapper.
Refs: Frame constants: `src/violawake_sdk/wake_detector.py:33`; Scoring path: `src/violawake_sdk/wake_detector.py:594`; Detection path: `src/violawake_sdk/wake_detector.py:669`
### 1.1 Runtime Cadence
The SDK accepts 20 ms input frames at 16 kHz. That is 320 samples per call.
The OWW backbone does not emit a new embedding every 20 ms. It accumulates audio until it has 1,280 samples, then advances the streaming mel/embedding pipeline.
```text
Input frames:
frame 1   frame 2   frame 3   frame 4
20 ms     20 ms     20 ms     20 ms
320 smp   320 smp   320 smp   320 smp
   \         |         |         /
    \        |         |        /
     +-------+---------+-------+
               1280 samples
                  80 ms
                    |
                    v
         OWW mel update + embedding step
```
This matters for two reasons: on frames that do not produce a new embedding, the detector reuses the previous score; temporal models only advance when a fresh embedding arrives.
Refs: OWW chunk size: `src/violawake_sdk/oww_backbone.py:19`; Streaming push: `src/violawake_sdk/oww_backbone.py:159`; Score reuse logic: `src/violawake_sdk/wake_detector.py:629`
### 1.2 Data Contract Between Layers
The contract between the backbone and the wake heads is fixed:
- Input to the head is not raw waveform.
- Input to the head is not mel frames.
- Input to the head is a `float32` embedding vector of size `96`, or a temporal stack of such vectors.
This separation is the main architectural boundary in the SDK.
Implications:
- Wake heads can be retrained or swapped without reimplementing the audio frontend.
- Different inference backends can share the same backbone contract.
- The decision policy only needs a scalar score, so model architecture changes are isolated from event policy.
Refs: Embedding dimension: `src/violawake_sdk/oww_backbone.py:18`; Embedding return path: `src/violawake_sdk/oww_backbone.py:221`; Temporal-vs-MLP runtime detection: `src/violawake_sdk/wake_detector.py:446`
---
## 2. Audio Ingress and Mel Spectrogram Frontend
### 2.1 Input Format
The detector is configured for:
- sample rate: `16_000` Hz
- channel count: mono
- streaming frame size: `20 ms`
- streaming frame samples: `320`
Refs: `src/violawake_sdk/wake_detector.py:33`; `src/violawake_sdk/vad.py:29`
The public validator accepts:
- `bytes` containing little-endian `int16` PCM
- `numpy.int16`
- `numpy.float32`
- `numpy.float64`
It rejects:
- empty input
- wrong dimensionality
- unsupported dtypes
- extremely large chunks
Non-finite floats are replaced with `0.0` before further processing.
Refs: `src/violawake_sdk/wake_detector.py:145`
### 2.2 Raw Audio Scale vs Model Scale
Two scales are used at runtime, and contributors should keep them separate:
1. Raw int16-style PCM scale for RMS checks and VAD-style energy logic.
2. Normalized float scale for model inference when input arrives as int16.
`WakeDetector.detect()` computes RMS before normalization, specifically because `rms_floor=1.0` is calibrated on int16-like magnitudes.
The code comments explicitly call out the consequence:
- speech RMS is expected in roughly the `500-5000` range
- silence is expected in roughly the `0-5` range
- normalizing first would make the floor effectively unreachable
Refs: RMS note: `src/violawake_sdk/wake_detector.py:684`; Int16 normalization helper: `src/violawake_sdk/wake_detector.py:579`; Model-audio preparation: `src/violawake_sdk/wake_detector.py:587`
### 2.3 OWW Streaming Audio Buffering
The OWW backbone owns the audio-to-feature frontend. It maintains:
- a raw-sample ring buffer
- a mel spectrogram buffer
- a remainder buffer for partial chunks
- the last emitted embedding
The raw ring buffer stores up to 10 seconds of int16 audio. The mel buffer stores up to roughly 10 seconds of mel history.
The backbone converts any accepted input format back to int16-style PCM before feeding the OWW frontend. That includes scaling `[-1, 1]` floats back to int16 if needed.
Refs: ring buffer: `src/violawake_sdk/oww_backbone.py:27`; reset state: `src/violawake_sdk/oww_backbone.py:149`; int16 coercion: `src/violawake_sdk/oww_backbone.py:226`
### 2.4 Mel Spectrogram Generation
Mel spectrogram inference is performed by the OWW asset `melspectrogram.onnx` or `melspectrogram.tflite`, depending on backend selection.
The streaming path is:
1. Pull a raw window from the ring buffer.
2. Run the OWW mel frontend session.
3. Squeeze the output to a 2-D mel array.
4. Normalize it using OWW's expected log-mel transform.
5. Append it to the rolling mel buffer.
Refs: path resolution: `src/violawake_sdk/oww_backbone.py:103`; mel session load: `src/violawake_sdk/oww_backbone.py:134`; mel inference: `src/violawake_sdk/oww_backbone.py:213`
### 2.5 Mel Feature Shape
The wake detector constants declare:
- `MEL_BINS = 32`
- `MEL_FRAMES_PER_EMBEDDING = 76`
- `MEL_STRIDE = 8`
This means the embedding model sees a sliding mel window of:
- `76` frames
- `32` mel bins
and consecutive embeddings advance by:
- `8` mel frames
Refs: detector constants: `src/violawake_sdk/wake_detector.py:43`; backbone constants: `src/violawake_sdk/oww_backbone.py:16`
### 2.6 OWW Normalization
The OWW mel output is normalized with this exact transform:
```text
normalized_spec = spec / 10.0 + 2.0
```
This is not a ViolaWake-specific heuristic. It is part of matching OpenWakeWord's expected input range for its embedding network.
If you retrain, replace, or export the backbone, this normalization must stay consistent with the embedding model.
Refs: `src/violawake_sdk/oww_backbone.py:217`
### 2.7 Frontend Context Window
When updating the streaming mel spectrogram, the backbone pulls:
```text
n_samples + 3 * 160 samples
```
of raw context.
That extra context comes from `_OWW_MELSPEC_CONTEXT_SAMPLES = 160 * 3`. This gives the mel frontend enough overlap around chunk boundaries to match OWW's streaming assumptions.
Refs: context constant: `src/violawake_sdk/oww_backbone.py:20`; streaming mel update: `src/violawake_sdk/oww_backbone.py:208`
---
## 3. OpenWakeWord Backbone
### 3.1 What the Backbone Is
The ViolaWake SDK uses OpenWakeWord as a frozen, pre-trained feature extractor.
The backbone is composed of two model assets:
- `melspectrogram.onnx` or `.tflite`
- `embedding_model.onnx` or `.tflite`
These are resolved from the installed `openwakeword` package, not from ViolaWake's model cache.
Refs: path resolution: `src/violawake_sdk/oww_backbone.py:103`; package-managed registry entry: `src/violawake_sdk/models.py:64`; package-managed guard: `src/violawake_sdk/models.py:124`
### 3.2 Frozen Feature Extractor Role
ViolaWake's architecture treats OWW as immutable at inference time.
The backbone:
- handles streaming audio buffering
- computes the mel frontend
- runs the embedding network
- returns the latest 96-dim vector
ViolaWake-specific models then operate on those embeddings.
The wake heads do not see:
- raw waveform tensors
- spectrogram tensors
- OWW internal states
They see only embedding vectors or short embedding sequences.
Refs: backbone class: `src/violawake_sdk/oww_backbone.py:131`; shared backbone creation: `src/violawake_sdk/wake_detector.py:497`; embedding accessor: `src/violawake_sdk/wake_detector.py:565`
### 3.3 Embedding Shape and Emission
The embedding model output is reshaped to a 96-element `float32` vector.
```text
(OWW mel window 76 x 32)
        |
        v
embedding_model
        |
        v
96-dim vector
```
At runtime, `OpenWakeWordBackbone.push_audio()` returns:
- `produced_embedding = True` when a new embedding was emitted this step
- the newest embedding if one exists
- otherwise the last known embedding
This design lets the detector continue returning a stable score between embedding updates.
Refs: embedding reshape: `src/violawake_sdk/oww_backbone.py:221`; push contract: `src/violawake_sdk/oww_backbone.py:159`
### 3.4 Streaming Windowing Logic
When enough samples have accumulated, the backbone:
1. updates the mel spectrogram buffer
2. computes how many OWW chunks were accumulated
3. walks the mel buffer newest-first
4. slices `76` mel frames per embedding window
5. emits embeddings for complete windows only
The reverse iteration is important. It maps chunk indices to negative mel offsets so the most recent chunk corresponds to the tail of the mel buffer.
Refs: chunk loop: `src/violawake_sdk/oww_backbone.py:175`; newest-first offset logic: `src/violawake_sdk/oww_backbone.py:179`
### 3.5 Why ViolaWake Models Operate on Embeddings
This split exists for engineering reasons, not only modeling reasons.
Benefits: much smaller Viola-specific heads; simpler retraining workflow; easier model registry and download story; cleaner separation between audio frontend and wake classifier; consistent interface across ONNX and TFLite backends.
Cost: two-stage inference instead of a single monolithic model; dependency on the installed `openwakeword` package for backbone assets.
The decision rationale is also captured in:
- `docs/adr/ADR-002-oww-feature-extractor.md`
---
## 4. Wake Word Heads
### 4.1 Head Families
ViolaWake currently supports three conceptual head families over OWW embeddings:
1. `temporal_cnn`
2. `temporal_convgru`
3. MLP-style single-embedding heads
At runtime, the detector infers whether a loaded model is temporal by inspecting the model input rank:
- rank `3` means temporal input
- rank `2` means single embedding input
Refs: runtime shape check: `src/violawake_sdk/wake_detector.py:446`
### 4.2 Architecture Summary
```text
Head family          Input shape            Core idea
-------------------  --------------------   -------------------------------
MLP legacy           (1, 96)                Score one embedding at a time
temporal_cnn         (1, 9, 96)             1D conv over embedding sequence
temporal_convgru     (1, 9, 96)             conv frontend + GRU sequence model
```
The production alias `viola` currently resolves to `temporal_cnn`.
Refs: alias map: `src/violawake_sdk/wake_detector.py:39`; registry default: `src/violawake_sdk/models.py:45`; alias entry: `src/violawake_sdk/models.py:120`
### 4.3 `temporal_cnn`
`temporal_cnn` is the production default.
It operates on a 9-frame sequence of 96-dim embeddings. In the training module, the architecture is:
```text
Conv1d(96 -> 64, k=3) + BN + ReLU + Dropout
Conv1d(64 -> 32, k=3) + BN + ReLU
AdaptiveMaxPool1d(1)
Linear(32 -> 16) + ReLU + Dropout
Linear(16 -> 1) + Sigmoid
```
Parameter count in the current implementation is approximately `25,409`, which is why it is typically described as a ~25K-parameter head.
Why it is the default:
- small enough for edge deployment
- preserves short-range temporal order
- best current live recall / false-positive trade-off in the model registry
Refs: class definition: `src/violawake_sdk/training/temporal_model.py:49`; registry description: `src/violawake_sdk/models.py:45`
### 4.4 `temporal_convgru`
`temporal_convgru` is the reserve temporal model.
Its training architecture is:
```text
Conv1d(96 -> 48, k=3) + BN + ReLU + Dropout
GRU(48 -> 24)
Linear(24 -> 16) + ReLU + Dropout
Linear(16 -> 1) + Sigmoid
```
This combines:
- local pattern extraction from the conv layer
- sequence-state modeling from the GRU
It is currently described in the registry as the reserve model rather than the production default.
Refs: class definition: `src/violawake_sdk/training/temporal_model.py:173`; registry entry: `src/violawake_sdk/models.py:103`
### 4.5 MLP Legacy Heads
The legacy path scores a single 96-d embedding at a time.
In runtime terms, that means:
- reshape embedding to `(1, 96)`
- run the head once
- treat the result as the wake score
This path remains useful for backward compatibility and testing, but current user-facing documentation should prefer `temporal_cnn`.
Legacy registry examples:
- `r3_10x_s42`
- `viola_mlp_oww`
Refs: single-embedding path: `src/violawake_sdk/wake_detector.py:650`; registry entries: `src/violawake_sdk/models.py:56`; registry entries: `src/violawake_sdk/models.py:111`
### 4.6 Temporal Buffering at Runtime
Temporal heads do not run on a single fresh embedding. They run on a deque of recent embeddings.
The runtime logic is:
1. append each new 96-d embedding to a fixed-length deque
2. wait until the deque length reaches the model's required sequence length
3. stack the deque into shape `(1, seq_len, 96)`
4. run the temporal head
Until the deque is full, the score is forced to `0.0`.
For the default temporal path, that means the detector needs `9` emitted embeddings before the first real temporal score is available.
Refs: default sequence length: `src/violawake_sdk/wake_detector.py:52`; temporal deque setup: `src/violawake_sdk/wake_detector.py:448`; temporal inference path: `src/violawake_sdk/wake_detector.py:635`
### 4.7 Score Reuse Between Embedding Updates
Because embeddings are emitted more slowly than input frames, the detector reuses `self._last_score` when no new embedding is produced.
This avoids introducing artificial zeros between backbone updates.
Behavior by case:
- no embedding yet: return previous score
- fresh embedding + temporal head not ready: return `0.0`
- fresh embedding + temporal head ready: run temporal model
- no fresh embedding after warm-up: reuse previous score
Refs: `src/violawake_sdk/wake_detector.py:629`
---
## 5. Confidence Scoring and Score History
Confidence scoring is not a gate. It is an introspection layer over the raw score stream.
`ScoreTracker` stores recent scores in a fixed-size deque and exposes:
- recent history
- latest raw score
- a coarse confidence class
Confidence levels are:
- `LOW`
- `MEDIUM`
- `HIGH`
- `CERTAIN`
Classification logic:
- `CERTAIN` if `raw >= threshold` and confirmation is already satisfied
- `HIGH` if `raw >= 0.90 * threshold`
- `MEDIUM` if `raw >= 0.75 * threshold`
- otherwise `LOW`
This API is useful for:
- debugging near-threshold behavior
- tuning thresholds
- showing richer UI feedback than a single boolean event
Refs: `src/violawake_sdk/confidence.py:18`; `src/violawake_sdk/confidence.py:47`; detector integration: `src/violawake_sdk/wake_detector.py:658`; public confidence API: `src/violawake_sdk/wake_detector.py:776`
---
## 6. 4-Gate Decision Policy
### 6.1 Gate Order
The core policy lives in `WakeDecisionPolicy.evaluate()`.
```text
raw frame
   |
   v
score available?
   |
   v
Gate 1: RMS floor
   |
   v
Gate 2: score threshold
   |
   v
Gate 3: cooldown
   |
   v
Gate 4: listening gate
   |
   v
trigger event
```
The policy itself is intentionally simple. Optional features such as confirmation, adaptive thresholding, speaker verification, and power management are layered around it by `WakeDetector.detect()`.
Refs: policy class: `src/violawake_sdk/wake_detector.py:201`; policy call site: `src/violawake_sdk/wake_detector.py:734`
### 6.2 Gate 1: Zero-Input Guard
Gate 1 rejects audio when frame RMS is below `rms_floor`, which defaults to `1.0`.
Purpose:
- suppress silent buffers
- suppress degenerate near-zero input
- avoid spurious detections on dead-air or DC-offset-like artifacts
Important implementation detail:
- RMS is computed on raw int16-scale PCM before normalization.
Refs: gate description: `src/violawake_sdk/wake_detector.py:207`; gate implementation: `src/violawake_sdk/wake_detector.py:234`; RMS computation: `src/violawake_sdk/wake_detector.py:684`
### 6.3 Gate 2: Score Threshold
Gate 2 rejects scores below the current threshold.
The threshold is normally fixed, but can be updated dynamically by the optional noise profiler before policy evaluation.
Default tuning guidance in the detector docstring is:
- `0.70`: sensitive
- `0.80`: balanced
- `0.85`: conservative
- `0.90+`: very conservative
Refs: threshold field: `src/violawake_sdk/wake_detector.py:215`; threshold check: `src/violawake_sdk/wake_detector.py:237`; adaptive threshold update: `src/violawake_sdk/wake_detector.py:701`
### 6.4 Gate 3: Cooldown
Gate 3 prevents repeated triggers within `cooldown_s` of the last accepted detection.
The default cooldown is `2.0` seconds.
Implementation notes:
- timing uses `time.monotonic()`
- the gate stores only the timestamp of the last accepted detection
- a helper exists to reset cooldown for tests or controlled flows
Refs: default cooldown: `src/violawake_sdk/wake_detector.py:38`; cooldown check: `src/violawake_sdk/wake_detector.py:239`; reset helper: `src/violawake_sdk/wake_detector.py:253`
### 6.5 Gate 4: Listening Gate
Gate 4 suppresses detections while playback is active.
This is controlled by the `is_playing` argument passed into `WakeDetector.detect()`.
Purpose:
- reduce self-triggering from the assistant's own TTS or audio playback
- enforce a simple half-duplex listening model when desired
Refs: gate description: `src/violawake_sdk/wake_detector.py:210`; gate implementation: `src/violawake_sdk/wake_detector.py:246`
### 6.6 Optional Multi-Window Confirmation
Before the 4-gate policy is evaluated, `WakeDetector.detect()` can require `confirm_count` consecutive above-threshold scores.
This is implemented outside `WakeDecisionPolicy`, but it is part of the effective decision pipeline.
Logic:
1. if score >= threshold, increment the confirmation counter
2. else, reset the counter to zero
3. only call `WakeDecisionPolicy.evaluate()` once the counter reaches the configured requirement
4. after an accepted detection, reset the confirmation counter
This behaves like a short temporal debounce over the score stream.
Refs: config field: `src/violawake_sdk/wake_detector.py:111`; detector field: `src/violawake_sdk/wake_detector.py:412`; confirmation logic: `src/violawake_sdk/wake_detector.py:725`
### 6.7 Effective Decision Stack
Putting the pieces together, the effective runtime decision stack is:
```text
audio frame
  -> validation
  -> RMS measurement
  -> optional power-manager skip
  -> optional adaptive-threshold update
  -> score generation
  -> optional confirm_count window
  -> Gate 1: zero-input guard
  -> Gate 2: threshold
  -> Gate 3: cooldown
  -> Gate 4: listening gate
  -> optional speaker verification
  -> detection event
```
This explains why "score above threshold" is necessary but not sufficient for a final `True`.
Refs: full detect flow: `src/violawake_sdk/wake_detector.py:669`
---
## 7. Model Distribution and Caching
### 7.1 Registry Model
ViolaWake models are declared in `MODEL_REGISTRY`. Each entry carries:
- `name`
- `url`
- `sha256`
- `size_bytes`
- `description`
- `version`
Refs: spec type: `src/violawake_sdk/models.py:27`; registry table: `src/violawake_sdk/models.py:44`
### 7.2 Distribution Channel
The SDK distributes ViolaWake head models via GitHub Releases, not via the PyPI wheel.
The main wake-head registry entries currently point at release assets such as:
- `temporal_cnn.onnx`
- `temporal_convgru.onnx`
- legacy MLP artifacts
Reasoning from the implementation:
- keep the Python package small
- allow updating model artifacts independently
- keep model metadata centralized in the registry
Refs: module docstring: `src/violawake_sdk/models.py:1`; release URLs: `src/violawake_sdk/models.py:45`; release URLs: `src/violawake_sdk/models.py:103`
### 7.3 Cache Location
By default, downloaded models are cached under:
```text
~/.violawake/models/
```
This path can be overridden with:
```text
VIOLAWAKE_MODEL_DIR
```
Refs: default cache dir: `src/violawake_sdk/models.py:23`; override handling: `src/violawake_sdk/models.py:167`
### 7.4 Auto-Download on First Use
`WakeDetector` resolves named models through `get_model_path()`.
If the file is not already cached and auto-download is enabled, the SDK downloads it automatically on first use.
Operational path:
1. user asks for a named model such as `temporal_cnn`
2. detector calls `_resolve_model_path()`
3. `_resolve_model_path()` calls `get_model_path()`
4. `get_model_path()` triggers `_auto_download_model()` if missing
5. the downloaded file is atomically moved into the cache
Refs: detector resolution: `src/violawake_sdk/wake_detector.py:536`; cache lookup: `src/violawake_sdk/models.py:324`; auto-download path: `src/violawake_sdk/models.py:193`
### 7.5 SHA-256 Verification
Downloaded models are verified with SHA-256 hashes from the registry.
There are two distinct paths:
- lightweight auto-download path
- explicit `download_model()` path
Both verify integrity unless the registry entry still has a placeholder hash.
The explicit download path is stricter:
- it rejects placeholder hashes unless `skip_verify=True`
- it also supports optional certificate pinning
Refs: auto-download verify: `src/violawake_sdk/models.py:298`; explicit download verify: `src/violawake_sdk/models.py:542`; SHA helper: `src/violawake_sdk/models.py:549`
### 7.6 Size Validation and Atomic Writes
The model downloader also performs:
- file size validation against `size_bytes`
- atomic temp-file writes before rename
- cleanup of partial downloads on failure
These measures reduce the risk of leaving a corrupt model in the cache.
Refs: size tolerance: `src/violawake_sdk/models.py:126`; auto-download temp file: `src/violawake_sdk/models.py:258`; auto-download size validation: `src/violawake_sdk/models.py:307`; explicit download temp file: `src/violawake_sdk/models.py:489`; explicit download size validation: `src/violawake_sdk/models.py:532`
### 7.7 OpenWakeWord Backbone Exception
The OWW backbone is intentionally different from ViolaWake head models.
It is registered for metadata purposes, but it is package-managed rather than downloaded into the ViolaWake cache.
That means:
- `oww_backbone` is not fetched from GitHub Releases by `violawake-download`
- the actual backbone files are discovered inside the installed `openwakeword` package
- missing backbone assets are treated as an installation problem, not a cache miss
Refs: package-managed set: `src/violawake_sdk/models.py:124`; cache guard: `src/violawake_sdk/models.py:349`; OWW path discovery: `src/violawake_sdk/oww_backbone.py:103`
---
## 8. VAD Engine
### 8.1 Role
`VADEngine` is an auxiliary subsystem for speech activity detection. It is not one of the four gates in `WakeDecisionPolicy`.
Instead, it is an optional component that callers can use upstream or alongside the wake detector.
Refs: module overview: `src/violawake_sdk/vad.py:1`
### 8.2 Backend Fallback Chain
The backend selection order for `backend="auto"` is:
```text
WebRTC -> Silero -> RMS
```
This is a hard-coded fallback chain.
Refs: enum values: `src/violawake_sdk/vad.py:34`; fallback logic: `src/violawake_sdk/vad.py:263`
### 8.3 WebRTC Backend
WebRTC VAD is the preferred backend.
Characteristics:
- binary speech / non-speech output
- low dependency overhead
- strict frame-size requirements
- uses 10 ms, 20 ms, or 30 ms int16 PCM frames at 16 kHz
Refs: class: `src/violawake_sdk/vad.py:82`; valid frame sizes: `src/violawake_sdk/vad.py:100`
### 8.4 Silero Backend
Silero is the second fallback.
Characteristics:
- probabilistic score
- heavier dependency surface
- loaded lazily through `torch.hub`
- short wake-detector frames are zero-padded to 512 samples because Silero does not accept 320-sample windows directly
Refs: class: `src/violawake_sdk/vad.py:136`; torch.hub load: `src/violawake_sdk/vad.py:171`; zero-padding logic: `src/violawake_sdk/vad.py:201`
### 8.5 RMS Heuristic Backend
The final fallback is a pure energy heuristic.
Characteristics:
- zero external dependencies
- computes frame RMS on int16 PCM
- returns `1.0` above `speech_threshold`
- returns `0.0` below `silence_threshold`
- linearly interpolates in the ambiguous zone
Refs: class: `src/violawake_sdk/vad.py:223`; interpolation logic: `src/violawake_sdk/vad.py:251`
### 8.6 Practical Relationship to Wake Detection
In the current architecture, VAD and wake detection are separate concerns:
- VAD estimates whether there is speech-like activity
- wake detection estimates whether the speech content matches the wake word
Keeping them separate avoids coupling the core detector policy to a specific VAD backend or probability scale.
---
## 9. Ensemble Scoring
### 9.1 Purpose
`EnsembleScorer` is an experimental multi-model fusion layer.
It lets `WakeDetector` combine multiple model sessions into a single fused score when extra models are provided.
Refs: module overview: `src/violawake_sdk/ensemble.py:1`; detector integration: `src/violawake_sdk/wake_detector.py:416`; model loading into ensemble: `src/violawake_sdk/wake_detector.py:466`
### 9.2 Fusion Strategies
Supported strategies are:
- `AVERAGE`
- `MAX`
- `VOTING`
- `WEIGHTED_AVERAGE`
Refs: enum: `src/violawake_sdk/ensemble.py:26`
### 9.3 Strategy Semantics
```text
AVERAGE
    fused = mean(scores)
MAX
    fused = max(scores)
VOTING
    each model votes yes if score >= voting_threshold
    fused = yes_votes / total_models
WEIGHTED_AVERAGE
    fused = dot(scores, weights)
    if weights do not sum to 1.0 exactly,
    they are normalized with a warning
```
Refs: fusion function: `src/violawake_sdk/ensemble.py:35`
### 9.4 Runtime Behavior
Each registered model session is run independently.
The scorer:
1. reshapes the input embedding to `(1, -1)`
2. executes each model
3. extracts the scalar output
4. clamps each individual score to `[0.0, 1.0]`
5. fuses the score list with the selected strategy
Inference failures are not fatal to the whole ensemble. A failed model contributes `0.0` and emits a warning.
Refs: score execution: `src/violawake_sdk/ensemble.py:150`; fused score call: `src/violawake_sdk/ensemble.py:188`
### 9.5 Shape Assumption Caveat
The current ensemble scorer assumes each registered model can consume a flattened embedding shaped like `(1, 96)` after reshape.
That makes it naturally aligned with single-embedding heads.
Contributors should note that temporal heads normally expect rank-3 inputs such as `(1, seq_len, 96)`. If temporal models are to participate in ensemble scoring directly, the ensemble input contract would need to be generalized.
This is an implementation caveat, not a user-facing API guarantee.
Refs: reshape assumption: `src/violawake_sdk/ensemble.py:170`; detector ensemble branch: `src/violawake_sdk/wake_detector.py:633`
