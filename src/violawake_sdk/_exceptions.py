"""ViolaWake SDK exception hierarchy."""

from __future__ import annotations


class ViolaWakeError(Exception):
    """Base exception for all ViolaWake SDK errors."""


class ModelNotFoundError(ViolaWakeError):
    """Raised when a model file is not found in the cache or at the given path.

    Resolution: run ``violawake-download --model <model_name>`` to download.
    """


class ModelLoadError(ViolaWakeError):
    """Raised when a model file exists but cannot be loaded by ONNX Runtime.

    Possible causes: corrupted file, ONNX opset version mismatch.
    """


class AudioCaptureError(ViolaWakeError):
    """Raised when microphone capture fails to initialize or read frames.

    Common causes: no audio input device, device already in use,
    PortAudio not installed.
    """


class VADBackendError(ViolaWakeError):
    """Raised when the requested VAD backend is unavailable.

    Falls back to RMS heuristic if webrtcvad/silero not installed.
    """


class PipelineError(ViolaWakeError):
    """Raised when the VoicePipeline encounters an unrecoverable error."""
