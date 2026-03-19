"""Compatibility package that re-exports the ViolaWake SDK."""

from __future__ import annotations

from violawake_sdk import (
    AudioCaptureError,
    ModelLoadError,
    ModelNotFoundError,
    STTEngine,
    TTSEngine,
    VADEngine,
    ViolaWakeError,
    VoicePipeline,
    WakeDetector,
    WakewordDetector,
    __version__,
)

__all__ = [
    "WakeDetector",
    "WakewordDetector",
    "VADEngine",
    "TTSEngine",
    "STTEngine",
    "VoicePipeline",
    "ViolaWakeError",
    "ModelNotFoundError",
    "AudioCaptureError",
    "ModelLoadError",
    "__version__",
]
