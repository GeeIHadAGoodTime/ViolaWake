"""
ViolaWake SDK — Open-source wake word detection + voice pipeline.

Public API surface:
    WakeDetector   — detect a wake word in an audio stream
    VADEngine      — voice activity detection
    TTSEngine      — on-device text-to-speech (Kokoro-82M)
    STTEngine      — speech-to-text (faster-whisper)
    VoicePipeline  — bundled Wake→VAD→STT→TTS orchestration

Quick start::

    from violawake_sdk import WakeDetector

    detector = WakeDetector(threshold=0.80)
    for chunk in detector.stream_mic():
        if detector.process(chunk):
            print("Wake word detected!")
            break

See README.md or https://violawake.readthedocs.io for full documentation.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "ViolaWake Contributors"
__license__ = "Apache-2.0"

from violawake_sdk._exceptions import (
    AudioCaptureError,
    ModelLoadError,
    ModelNotFoundError,
    ViolaWakeError,
)
from violawake_sdk.pipeline import VoicePipeline
from violawake_sdk.vad import VADEngine
from violawake_sdk.wake_detector import WakeDetector, WakewordDetector

# Conditional imports for optional extras
try:
    from violawake_sdk.tts import TTSEngine
except ImportError:
    TTSEngine = None  # type: ignore[assignment,misc]

try:
    from violawake_sdk.stt import STTEngine
except ImportError:
    STTEngine = None  # type: ignore[assignment,misc]

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
