"""Voice Activity Detection (VAD) module.

Supports three backends in priority order:
  1. WebRTC VAD (webrtcvad library) — preferred, high accuracy
  2. Silero VAD (PyTorch, onnxruntime) — good accuracy, larger dependency
  3. RMS heuristic — fallback, no additional dependencies required

The ``VADEngine`` class auto-selects the best available backend unless
``backend`` is explicitly specified.

Usage::

    vad = VADEngine(backend="webrtc")
    prob = vad.process_frame(audio_20ms_bytes)  # float 0.0–1.0
    is_speech = prob > 0.5
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

# Frame configuration
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20ms at 16kHz


class VADBackend(str, Enum):
    """Available VAD backends."""

    WEBRTC = "webrtc"
    SILERO = "silero"
    RMS = "rms"
    AUTO = "auto"


class _VADBackendProtocol(Protocol):
    """Protocol that all VAD backends must implement."""

    def process_frame(self, audio_bytes: bytes) -> float:
        """Process a 20ms audio frame. Returns speech probability 0.0–1.0."""
        ...

    def reset(self) -> None:
        """Reset any internal state."""
        ...


class _WebRTCVADBackend:
    """WebRTC VAD backend using the webrtcvad library."""

    def __init__(self, aggressiveness: int = 2) -> None:
        """Args:
            aggressiveness: 0–3. 0 is least aggressive (more false accepts),
                            3 is most aggressive (more false rejects). Default 2.
        """
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(aggressiveness)
        except ImportError as e:
            raise ImportError(
                "webrtcvad is not installed. Install it with: pip install 'violawake[vad]'"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to initialize WebRTC VAD: {e}") from e

    def process_frame(self, audio_bytes: bytes) -> float:
        """Returns 1.0 if speech detected, 0.0 if not (WebRTC is binary)."""
        try:
            is_speech = self._vad.is_speech(audio_bytes, sample_rate=SAMPLE_RATE)
            return 1.0 if is_speech else 0.0
        except Exception as e:
            logger.warning("WebRTC VAD error: %s", e)
            return 0.0

    def reset(self) -> None:
        """WebRTC VAD is stateless — no-op."""


class _RMSHeuristicBackend:
    """Simple RMS-based VAD heuristic.

    Returns a probability based on the signal energy relative to noise floor.
    Not as accurate as WebRTC/Silero but has zero dependencies.
    """

    def __init__(
        self,
        speech_threshold: float = 200.0,
        silence_threshold: float = 50.0,
    ) -> None:
        self._speech_threshold = speech_threshold
        self._silence_threshold = silence_threshold

    def process_frame(self, audio_bytes: bytes) -> float:
        """Returns speech probability based on RMS energy."""
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(pcm ** 2)))

        if rms >= self._speech_threshold:
            return 1.0
        elif rms <= self._silence_threshold:
            return 0.0
        else:
            # Linear interpolation in the ambiguous zone
            return (rms - self._silence_threshold) / (self._speech_threshold - self._silence_threshold)

    def reset(self) -> None:
        """Stateless — no-op."""


def _create_backend(
    backend: VADBackend, **kwargs: object
) -> tuple[VADBackend, _VADBackendProtocol]:
    """Create the specified VAD backend.

    For AUTO, tries WebRTC → RMS (Silero requires heavier deps, skip for now).
    """
    if backend == VADBackend.WEBRTC:
        return backend, _WebRTCVADBackend(**kwargs)  # type: ignore[arg-type]
    elif backend == VADBackend.RMS:
        return backend, _RMSHeuristicBackend(**kwargs)  # type: ignore[arg-type]
    elif backend == VADBackend.AUTO:
        try:
            b = _WebRTCVADBackend()
            logger.info("VAD backend: WebRTC")
            return VADBackend.WEBRTC, b
        except (ImportError, RuntimeError):
            logger.info("WebRTC VAD unavailable, falling back to RMS heuristic")
            return VADBackend.RMS, _RMSHeuristicBackend()
    elif backend == VADBackend.SILERO:
        raise NotImplementedError(
            "Silero VAD backend not yet implemented. Use 'webrtc' or 'rms'."
        )
    else:
        raise ValueError(f"Unknown VAD backend: {backend}")


class VADEngine:
    """Voice Activity Detection engine.

    Auto-selects the best available backend unless explicitly specified.

    Example::

        vad = VADEngine(backend="webrtc")  # or "silero", "rms", "auto"
        prob = vad.process_frame(audio_20ms_bytes)
        is_speech = prob > 0.5
    """

    def __init__(
        self,
        backend: str | VADBackend = VADBackend.AUTO,
        **backend_kwargs: object,
    ) -> None:
        """Initialize the VAD engine.

        Args:
            backend: One of "auto", "webrtc", "silero", "rms".
                     "auto" selects the best available backend.
            **backend_kwargs: Backend-specific arguments.
                For "webrtc": aggressiveness (0–3, default 2)
                For "rms": speech_threshold, silence_threshold
        """
        if isinstance(backend, str):
            backend = VADBackend(backend)

        self._backend_name, self._backend = _create_backend(backend, **backend_kwargs)

    @property
    def backend_name(self) -> str:
        """Name of the active backend."""
        return self._backend_name.value

    def process_frame(self, audio_bytes: bytes) -> float:
        """Process a 20ms audio frame.

        Args:
            audio_bytes: 320 samples of 16kHz mono 16-bit PCM.

        Returns:
            Speech probability in [0.0, 1.0].
            1.0 = definitely speech, 0.0 = definitely silence.
        """
        return self._backend.process_frame(audio_bytes)

    def is_speech(self, audio_bytes: bytes, threshold: float = 0.5) -> bool:
        """Convenience method: returns True if speech probability exceeds threshold."""
        return self.process_frame(audio_bytes) >= threshold

    def reset(self) -> None:
        """Reset internal state (useful between utterances)."""
        self._backend.reset()
