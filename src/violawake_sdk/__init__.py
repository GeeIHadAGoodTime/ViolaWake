"""
ViolaWake SDK — Open-source wake word detection + voice pipeline.

Public API surface:
    WakeDetector      — detect a wake word in an audio stream
    AsyncWakeDetector — async wrapper for asyncio-based applications
    DetectorConfig    — advanced configuration (ensemble, adaptive, speaker, power)
    VADEngine         — voice activity detection (WebRTC, Silero, RMS)
    TTSEngine         — on-device text-to-speech (Kokoro-82M)
    STTEngine         — speech-to-text (faster-whisper)
    VoicePipeline     — bundled Wake→VAD→STT→TTS orchestration
    NoiseProfiler     — noise-adaptive threshold adjustment
    PowerManager      — battery-aware power management
    FusionStrategy    — multi-model ensemble scoring
    list_models()     — discover available wake word models
    list_voices()     — discover available TTS voices

Quick start::

    from violawake_sdk import WakeDetector

    with WakeDetector(threshold=0.80) as detector:
        for chunk in detector.stream_mic():
            if detector.detect(chunk):
                print("Wake word detected!")
                break

See README.md or https://github.com/GeeIHadAGoodTime/ViolaWake for full documentation.
"""

from __future__ import annotations

__version__ = "0.2.1"
__author__ = "ViolaWake Contributors"
__license__ = "Apache-2.0"

from violawake_sdk._exceptions import (
    AudioCaptureError,
    ModelLoadError,
    ModelNotFoundError,
    PipelineError,
    VADBackendError,
    ViolaWakeError,
)
from violawake_sdk.async_detector import AsyncWakeDetector
from violawake_sdk.confidence import ConfidenceLevel, ConfidenceResult
from violawake_sdk.ensemble import FusionStrategy
from violawake_sdk.noise_profiler import NoiseProfiler
from violawake_sdk.pipeline import VoicePipeline
from violawake_sdk.power_manager import PowerManager
from violawake_sdk.vad import VADEngine
from violawake_sdk.wake_detector import (
    DetectorConfig,
    WakeDecisionPolicy,
    WakeDetector,
    WakewordDetector,  # noqa: F401 — backward compat
    validate_audio_chunk,
)

# Conditional imports for optional extras
try:
    from violawake_sdk.tts import TTSEngine
except ImportError:
    TTSEngine = None  # type: ignore[assignment,misc]

try:
    from violawake_sdk.stt import STTEngine
except ImportError:
    STTEngine = None  # type: ignore[assignment,misc]


def list_models() -> list[dict[str, str]]:
    """Return available wake word models with their descriptions.

    Each entry is a dict with keys: ``name``, ``description``, ``version``.

    Example::

        >>> from violawake_sdk import list_models
        >>> for m in list_models():
        ...     print(f"{m['name']:20s} {m['description']}")
    """
    from violawake_sdk.models import MODEL_REGISTRY

    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for name, spec in MODEL_REGISTRY.items():
        # Deduplicate aliases (e.g. "viola" -> "temporal_cnn")
        if spec.name in seen:
            continue
        # Hide deprecated, package-managed, and non-wake-word models
        if "DEPRECATED" in spec.description:
            continue
        if spec.name in ("oww_backbone", "kokoro_v1_0", "kokoro_voices_v1_0"):
            continue
        seen.add(spec.name)
        result.append(
            {
                "name": name,
                "description": spec.description,
                "version": spec.version,
            }
        )
    return result


def list_voices() -> list[str]:
    """Return available TTS voice names for use with ``TTSEngine``.

    Requires the ``[tts]`` extra to be installed for actual synthesis,
    but this function always works for discovery.

    Example::

        >>> from violawake_sdk import list_voices
        >>> list_voices()
        ['af_heart', 'af_bella', 'af_sarah', ...]
    """
    from violawake_sdk.tts import AVAILABLE_VOICES

    return list(AVAILABLE_VOICES)


__all__ = [
    # Core detection
    "DetectorConfig",
    "WakeDetector",
    "AsyncWakeDetector",
    "WakeDecisionPolicy",
    "validate_audio_chunk",
    # Confidence & scoring
    "ConfidenceResult",
    "ConfidenceLevel",
    "FusionStrategy",
    # Advanced features
    "NoiseProfiler",
    "PowerManager",
    # Pipeline components
    "VADEngine",
    "TTSEngine",
    "STTEngine",
    "VoicePipeline",
    # Exceptions
    "ViolaWakeError",
    "ModelNotFoundError",
    "AudioCaptureError",
    "ModelLoadError",
    "PipelineError",
    "VADBackendError",
    # Discovery
    "list_models",
    "list_voices",
    "__version__",
]
