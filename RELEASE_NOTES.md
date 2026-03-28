# Release Notes

Update this file before each release. These notes are used as the GitHub Release body in `.github/workflows/release.yml`.

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
