# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- VAD `process_frame()` and `is_speech()` now accept `np.ndarray` (float32, float64, int16) in addition to bytes
- `_coerce_to_bytes()` shared helper for input normalization across VAD backends
- API docs generation setup with pdoc (`docs` optional dependency)

## [0.2.2] - 2026-04-05

### Fixed
- Silence quality gate bug: zero-energy audio correctly rejected by OWW backbone now scores 0.0 instead of 1.0 (was causing false Grade F on perfectly good models)
- Training pipeline consistency: patience=15 everywhere (CLI, SDK, Console — was 10 in some paths)
- Console training service: added `augment_source_files` parameter and repo-root corpus search path to match CLI pipeline
- Standalone `train_full_pipeline.py`: same fixes as Console for full parity

## [0.2.1] - 2026-03-30

### Added
- Kokoro TTS fallback when Edge TTS is unavailable
- `temporal_convgru` reserve model in registry
- Registry integrity checking (`check_registry_integrity()`)

### Changed
- `r3_10x_s42` MLP model marked DEPRECATED in registry (fails live mic test, max score 0.50)
- Removed `viola_mlp_oww` and `viola_cnn_v4` from registry (never uploaded to GitHub Releases)

## [0.2.0] - 2026-03-28

### Added
- **TemporalCNN production model** (`temporal_cnn`): 9-frame sliding window over OWW embeddings, d'=8.577, EER 0.8%, AUC 0.9993 — replaces MLP as default
- 8-phase training pipeline: user positives → TTS (20 voices x 3 phrases) → audiomentations augmentation → confusable negatives R1 (30 words) → R2 (16 words) → speech negatives (104 phrases) → universal corpus (LibriSpeech, MUSAN) → TemporalCNN training
- Post-training quality gate with A/B/C/F grading (Grade F blocks ONNX export)
- FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05) with AdamW + CosineAnnealingLR + EMA
- Group-aware stratified train/val split preventing augmentation data leakage
- Auto-evaluation on held-out 20% test set
- `docs/PROVEN_TRAINING_RECIPE.md` — canonical pipeline documentation

### Changed
- Default production model: `temporal_cnn` (was `r3_10x_s42` MLP)
- Model alias `"viola"` now resolves to `temporal_cnn`

## [0.1.0] - 2026-03-27

### Added
- Wake word detection with 4-gate decision policy (silence guard, threshold, cooldown, playback suppression) plus optional multi-window confirmation
- 3-of-3 multi-window confirmation reducing FAPH by 87%
- Production default model: temporal_cnn (EER 5.49%, best live recall + lowest FP)
- VAD engine with 3-backend fallback (WebRTC -> Silero -> RMS)
- STT integration via faster-whisper (5 model sizes)
- TTS integration via Kokoro-82M with sentence-chunked streaming
- VoicePipeline orchestrating Wake->VAD->STT->TTS
- Training CLI (violawake-train) with augmentation pipeline (gain, time stretch, pitch shift, additive noise, time shift)
- Evaluation CLI (violawake-eval) with EER, ROC AUC, FAPH, FRR@FAR operating points
- `violawake-generate` CLI for headless TTS sample generation
- `violawake-expand-corpus` CLI for LibriSpeech/MUSAN corpus download
- `[generate]` optional extra for sample generation without the full training stack
- Quality gate with A/B/C/F grading post-training
- Auto-evaluation with a held-out 20% test set
- Safe tarball extraction with zip-slip protection
- Model registry with SHA-256 verification and auto-download
- Confusable word generation for adversarial testing

### Fixed
- SDK inference path rewritten to use correct OWW 2-model pipeline
- Critical normalization fix: mel model expects int16-range float32, output needs mel/10+2 transform
- Without normalization fix, FAPH was 65x worse (783/h vs 12/h)

### Security
- SHA-256 model integrity verification
- HTTPS-only model downloads
