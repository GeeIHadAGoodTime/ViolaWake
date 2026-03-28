# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- VAD `process_frame()` and `is_speech()` now accept `np.ndarray` (float32, float64, int16) in addition to bytes
- `_coerce_to_bytes()` shared helper for input normalization across VAD backends
- API docs generation setup with pdoc (`docs` optional dependency)

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
