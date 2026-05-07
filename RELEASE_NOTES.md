# Release Notes

Update this file before each release. These notes are used as the GitHub Release body in `.github/workflows/release.yml`.

## v0.2.5 — Quality Gate Hardening + Self-Reporting Version

### Highlights

- **Quality gate silence subgrade now actually runs.** Pre-v0.2.5, when the OpenWakeWord backbone correctly rejected zero-energy audio, the silence test counted zero windows and `silence_max_score` defaulted to 0.0 — which silently passed the silence threshold for Grade A and Grade B. Models that overfit on thin TTS positive sets could ship without ever being checked against a low-energy input. v0.2.5 falls back to near-silence (1e-4 RMS gaussian noise) so the OWW backbone still produces embeddings and the silence subgrade actually exercises the model. If even near-silence produces zero embeddings, the gate forces Grade F.
- **`__version__` is now derived from package metadata.** v0.2.4 shipped to PyPI as `0.2.4` but `violawake_sdk.__version__` still reported `"0.2.2"` because the constant was hardcoded. Now it reads from `importlib.metadata.version("violawake")` and cannot drift.
- **Training job queue tolerates transient stalls.** A single slow progress write (e.g. during a backend restart warmup) used to kill the whole training job with `error_reason=timeout`. Per-event timeout bumped from 10s to 60s, and timeouts are now caught — the stalled event is dropped, the job keeps running.

### Breaking Changes

- None.

## v0.2.4 — ONNX Export Compatibility With torch 2.10+

### Highlights

- **Restore TemporalCNN ONNX export on torch 2.10+.** torch 2.10 changed `torch.onnx.export` to default to the dynamo-based exporter, which lacks an ONNX dispatcher for `aten.adaptive_max_pool2d` — the lowering used by `nn.AdaptiveMaxPool1d(1)` inside `TemporalCNN`. Training jobs failed at the export step with `DispatchError: No ONNX function found for aten.adaptive_max_pool2d`. v0.2.4 pins `export_temporal_onnx` to the legacy exporter, matching how the production reference model was originally exported.
- **Console backend dependency:** add `onnxscript` so torch's exporter machinery has its required dependency satisfied.
- **Live e2e test fixes:** use the literal wake-word string instead of a per-run unique identifier, and extend the training-status polling deadline to match real CPU training duration.

### Breaking Changes

- None.

## v0.2.2 — Quality Gate & Training Parity

### Highlights

- **Silence quality gate fix**: zero-energy audio correctly rejected by OWW backbone now scores 0.0 instead of 1.0 (was causing false Grade F on perfectly good models)
- **Training pipeline consistency**: patience=15 everywhere (CLI, SDK, Console) — was 10 in some paths
- **Console training service**: added `augment_source_files` parameter and repo-root corpus search path to match CLI pipeline
- **Standalone `train_full_pipeline.py`**: same fixes as Console for full parity

### Breaking Changes

- None.

---

## v0.2.1 — Kokoro TTS Fallback & Registry Cleanup

### Highlights

- **Kokoro TTS fallback** when Edge TTS is unavailable
- **`temporal_convgru` reserve model** added to registry
- **Registry integrity checking** via `check_registry_integrity()`
- `r3_10x_s42` MLP model marked DEPRECATED (fails live mic test, max score 0.50)
- Removed `viola_mlp_oww` and `viola_cnn_v4` from registry (never uploaded to GitHub Releases)

### Breaking Changes

- None.

---

## v0.2.0 — Temporal CNN & 8-Phase Training Pipeline

### Highlights

- **TemporalCNN production model** (`temporal_cnn`): 9-frame sliding window over OWW embeddings, d'=8.577, EER 0.8%, AUC 0.9993 — replaces MLP as default
- **8-phase training pipeline**: user positives, TTS (20 voices x 3 phrases), audiomentations augmentation, confusable negatives R1+R2, speech negatives, universal corpus, TemporalCNN training
- **Post-training quality gate** with A/B/C/F grading (Grade F blocks ONNX export)
- FocalLoss with AdamW + CosineAnnealingLR + EMA
- Group-aware stratified train/val split preventing augmentation data leakage

### Breaking Changes

- Default production model changed from `r3_10x_s42` MLP to `temporal_cnn`
- Model alias `"viola"` now resolves to `temporal_cnn`

---

## v0.1.0 — Initial Release

### Highlights

- **Wake word detection** with Temporal CNN on OpenWakeWord embeddings — EER 5.49% on benchmark v2 (700 adversarial negatives, 180 TTS positives)
- **4-gate decision policy** (RMS floor, score threshold, cooldown, playback suppression) plus optional 3-of-3 multi-window confirmation (87% FAPH reduction)
- **Full voice pipeline**: Wake -> VAD -> STT (faster-whisper) -> TTS (Kokoro-82M) in one `VoicePipeline` class
- **Training CLI** (`violawake-train`) with data augmentation (gain, time stretch, pitch shift, noise mixing, time shift), FocalLoss, EMA, and SWA weight averaging
- **Evaluation CLI** (`violawake-eval`) with EER, FAR/FRR, ROC curves, and FAPH measurement

### Breaking Changes

- None (initial release).

### Bug Fixes

- SDK inference path rewritten to use correct OWW 2-model pipeline
- Critical normalization fix: mel model expects int16-range float32, output needs mel/10+2 transform
- float32 audio input no longer silently rejected by Gate 1 RMS check

### Models

- `temporal_cnn.onnx` — Production default (~100 KB, EER 5.49%)
- `temporal_convgru.onnx` — Reserve model (~81 KB)
- `kokoro-v1.0.onnx` + `voices-v1.0.bin` — Kokoro-82M TTS (~354 MB total, Apache 2.0, hosted upstream)

### Security

- SHA-256 model integrity verification on all downloads
- HTTPS-only model download enforcement
- Placeholder hash models blocked from auto-download
- Temp file permissions set to 0o600
