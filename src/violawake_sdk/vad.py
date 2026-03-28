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
    prob = vad.process_frame(audio_20ms_ndarray)  # also accepts numpy arrays
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


def _coerce_to_bytes(audio: bytes | np.ndarray) -> bytes:
    """Convert audio input to int16 PCM bytes.

    Accepts:
      - bytes/bytearray: returned as-is
      - np.ndarray float32/float64: assumed normalized to [-1.0, 1.0],
        scaled by 32768 and clipped to int16 range.  **Do NOT pass
        float arrays in int16 range** (e.g., values like 5000.0) —
        they will be multiplied by 32768 and produce clipped garbage.
        Use int16 dtype for int16-range data.
      - np.ndarray int16: converted to bytes directly

    Raises:
        TypeError: If input is not bytes or ndarray.
        ValueError: If ndarray has an unsupported dtype.
    """
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if isinstance(audio, np.ndarray):
        if audio.dtype in (np.float32, np.float64):
            return (audio * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
        if audio.dtype == np.int16:
            return audio.tobytes()
        raise ValueError(f"ndarray dtype must be float32, float64, or int16, got {audio.dtype}")
    raise TypeError(f"audio must be bytes or np.ndarray, got {type(audio).__name__}")


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

    # Valid frame sizes for WebRTC VAD: 10ms, 20ms, 30ms at 16kHz (int16 = 2 bytes/sample)
    _VALID_FRAME_BYTES = frozenset(
        {
            160 * 2,  # 10ms at 16kHz
            320 * 2,  # 20ms at 16kHz
            480 * 2,  # 30ms at 16kHz
        }
    )

    def process_frame(self, audio_bytes: bytes) -> float:
        """Returns 1.0 if speech detected, 0.0 if not (WebRTC is binary).

        Raises:
            TypeError: If audio_bytes is not bytes/bytearray.
            ValueError: If frame size doesn't match 10/20/30ms at 16kHz.
        """
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError(
                f"audio_bytes must be bytes (int16 PCM), got {type(audio_bytes).__name__}"
            )
        if len(audio_bytes) not in self._VALID_FRAME_BYTES:
            n_bytes = len(audio_bytes)
            raise ValueError(
                f"WebRTC VAD requires 10/20/30ms frames at 16kHz "
                f"(320/640/960 bytes). Got {n_bytes} bytes "
                f"({n_bytes / 2 / SAMPLE_RATE * 1000:.1f}ms)."
            )
        try:
            is_speech = self._vad.is_speech(audio_bytes, sample_rate=SAMPLE_RATE)
            return 1.0 if is_speech else 0.0
        except Exception as e:
            logger.warning("WebRTC VAD error: %s", e)
            return 0.0

    def reset(self) -> None:
        """WebRTC VAD is stateless — no-op."""


class _SileroVADBackend:
    """Silero VAD backend using torch.hub.

    Requires PyTorch (torch). Loaded lazily to avoid hard dependency.
    Install with: pip install torch

    Note: torch is NOT a hard dependency of violawake. It is only required
    when explicitly selecting the "silero" backend or when "auto" falls
    through WebRTC to Silero.
    """

    def __init__(self) -> None:
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for Silero VAD. Install with: pip install torch"
            ) from e

        try:
            # SECURITY: trust_repo=True disables the interactive safety prompt
            # AND bypasses integrity checks on the downloaded repository code.
            # This is accepted because:
            #   1. Silero VAD (snakers4/silero-vad) is a well-known, widely-used
            #      open-source project with 5k+ GitHub stars.
            #   2. The alternative — bundling the model — would increase SDK
            #      package size by ~4MB and complicate version updates.
            #   3. For production deployments that require stricter supply-chain
            #      security, pin to a known commit hash or vendor the model.
            logger.warning(
                "Loading Silero VAD from torch.hub with trust_repo=True — "
                "the remote repo is not integrity-verified. Pin to a known "
                "commit hash in production deployments."
            )
            model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        except (RuntimeError, OSError) as e:
            raise RuntimeError(
                f"Failed to load Silero VAD model from torch.hub. "
                f"This may indicate a network issue, corrupted cache, or "
                f"incompatible PyTorch version. Try clearing the hub cache "
                f"with: rm -rf ~/.cache/torch/hub/snakers4_silero-vad_master — "
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {e}") from e

        self._model = model
        self._torch = torch
        self._sample_rate = SAMPLE_RATE

    def process_frame(self, audio_bytes: bytes) -> float:
        """Returns speech probability from Silero VAD model."""
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError(
                f"audio_bytes must be bytes (int16 PCM), got {type(audio_bytes).__name__}"
            )
        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                f"audio_bytes length must be even (int16 = 2 bytes/sample), got {len(audio_bytes)}"
            )
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Silero VAD requires specific window sizes (512, 768, 1024, 1536 at
        # 16kHz).  The wake detector sends 20ms frames (320 samples) which
        # Silero rejects.  Zero-pad to 512 so short frames work transparently.
        n_samples = len(pcm)
        if n_samples < 512:
            padded = np.zeros(512, dtype=np.float32)
            padded[:n_samples] = pcm
            pcm = padded

        tensor = self._torch.from_numpy(pcm)
        try:
            prob = self._model(tensor, self._sample_rate).item()
        except Exception as e:
            logger.warning("Silero VAD error (samples=%d): %s", n_samples, e)
            return 0.0
        return float(prob)

    def reset(self) -> None:
        """Reset Silero model internal states."""
        self._model.reset_states()


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
        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError(
                f"audio_bytes must be bytes (int16 PCM), got {type(audio_bytes).__name__}"
            )
        if len(audio_bytes) % 2 != 0:
            raise ValueError(
                f"audio_bytes length must be even (int16 = 2 bytes/sample), got {len(audio_bytes)}"
            )
        pcm = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(pcm**2)))

        if rms >= self._speech_threshold:
            return 1.0
        elif rms <= self._silence_threshold:
            return 0.0
        else:
            # Linear interpolation in the ambiguous zone
            return (rms - self._silence_threshold) / (
                self._speech_threshold - self._silence_threshold
            )

    def reset(self) -> None:
        """Stateless — no-op."""


def _create_backend(
    backend: VADBackend, **kwargs: object
) -> tuple[VADBackend, _VADBackendProtocol]:
    """Create the specified VAD backend.

    For AUTO, tries WebRTC → Silero → RMS (best available).
    """
    if backend == VADBackend.WEBRTC:
        return backend, _WebRTCVADBackend(**kwargs)  # type: ignore[arg-type]
    elif backend == VADBackend.SILERO:
        return backend, _SileroVADBackend()
    elif backend == VADBackend.RMS:
        return backend, _RMSHeuristicBackend(**kwargs)  # type: ignore[arg-type]
    elif backend == VADBackend.AUTO:
        # Fallback chain: WebRTC → Silero → RMS
        try:
            b = _WebRTCVADBackend()
            logger.info("VAD backend: WebRTC")
            return VADBackend.WEBRTC, b
        except Exception as e:
            logger.debug("WebRTC VAD unavailable (%s: %s), trying Silero", type(e).__name__, e)
        try:
            b = _SileroVADBackend()
            logger.info("VAD backend: Silero")
            return VADBackend.SILERO, b
        except Exception as e:
            logger.debug(
                "Silero VAD unavailable (%s: %s), falling back to RMS heuristic",
                type(e).__name__,
                e,
            )
        logger.info("VAD backend: RMS heuristic")
        return VADBackend.RMS, _RMSHeuristicBackend()
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

    def process_frame(self, audio: bytes | np.ndarray) -> float:
        """Process a 20ms audio frame.

        Args:
            audio: 320 samples of 16kHz mono audio. Accepted formats:
                - bytes/bytearray: int16 PCM (640 bytes for 20ms)
                - np.ndarray float32/float64: assumed normalized to [-1.0, 1.0],
                  scaled by 32768 to int16. Use int16 dtype for int16-range data.
                - np.ndarray int16: converted to bytes directly

        Returns:
            Speech probability in [0.0, 1.0].
            1.0 = definitely speech, 0.0 = definitely silence.
        """
        audio_bytes = _coerce_to_bytes(audio)
        return self._backend.process_frame(audio_bytes)

    def is_speech(self, audio: bytes | np.ndarray, threshold: float = 0.5) -> bool:
        """Convenience method: returns True if speech probability exceeds threshold."""
        return self.process_frame(audio) >= threshold

    def reset(self) -> None:
        """Reset internal state (useful between utterances)."""
        self._backend.reset()

    def close(self) -> None:
        """Release backend resources."""
        self._backend = None  # type: ignore[assignment]

    def __enter__(self) -> VADEngine:
        """Enter sync context manager. Returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit sync context manager. Releases backend resources."""
        self.close()
