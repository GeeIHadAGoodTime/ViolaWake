"""Wake word detection using ViolaWake MLP + OpenWakeWord backbone."""

from __future__ import annotations

import logging
import collections
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from violawake_sdk._constants import DEFAULT_THRESHOLD as _CONST_DEFAULT_THRESHOLD
from violawake_sdk._exceptions import AudioCaptureError, ModelLoadError, ModelNotFoundError
from violawake_sdk.backends import get_backend
from violawake_sdk.confidence import ConfidenceResult, ScoreTracker
from violawake_sdk.ensemble import EnsembleScorer, FusionStrategy
from violawake_sdk.models import MODEL_REGISTRY, get_model_path
from violawake_sdk.noise_profiler import NoiseProfiler
from violawake_sdk.oww_backbone import EMBEDDING_DIM, OpenWakeWordBackbone
from violawake_sdk.power_manager import PowerManager

if TYPE_CHECKING:
    from violawake_sdk.audio_source import AudioSource
    from violawake_sdk.backends.base import BackendSession, InferenceBackend
    from violawake_sdk.speaker import SpeakerVerifyResult

logger = logging.getLogger(__name__)

# Frame configuration (matches production Viola)
SAMPLE_RATE = 16_000          # 16 kHz mono
FRAME_MS = 20                 # 20ms frames
FRAME_SAMPLES = 320           # 16000 * 0.020
DEFAULT_THRESHOLD = _CONST_DEFAULT_THRESHOLD  # Canonical value from _constants.py (ADR-002)
DEFAULT_COOLDOWN_S = 2.0      # Minimum seconds between detections
WAKE_WORD_ALIASES = {
    "viola": "temporal_cnn",
}

# OWW backbone constants
MEL_BINS = 32                 # Mel spectrogram frequency bins
MEL_FRAMES_PER_EMBEDDING = 76 # Mel frames per embedding window
MEL_STRIDE = 8                # Mel frame stride between embeddings
# Audio buffer limits
_MAX_AUDIO_SAMPLES = SAMPLE_RATE * 2   # 2s ring buffer (32000 samples)
_MAX_CHUNK_SAMPLES = SAMPLE_RATE * 10  # Maximum accepted chunk: 10s
_MAX_PROCESS_FRAME_SAMPLES = FRAME_SAMPLES * 10  # Largest non-pathological frame for process()

# Temporal model constants
_TEMPORAL_SEQ_LEN_DEFAULT = 9  # Default sequence length for temporal models


# ---------------------------------------------------------------------------
# DetectorConfig: progressive disclosure for advanced features
# ---------------------------------------------------------------------------

# Sentinel to distinguish "not provided" from explicit None/False
_UNSET: Any = object()


@dataclass
class DetectorConfig:
    """Advanced configuration for WakeDetector.

    Basic usage needs no config -- just use ``WakeDetector(threshold=0.80)``.
    Use ``DetectorConfig`` to opt-in to advanced features without cluttering
    the constructor.

    Example::

        # Simple (80% of users):
        det = WakeDetector(model="temporal_cnn", threshold=0.80)

        # Advanced (multi-model ensemble + adaptive threshold):
        det = WakeDetector(
            model="temporal_cnn",
            config=DetectorConfig(
                adaptive_threshold=True,
                confirm_count=3,
            ),
        )

    Attributes:
        models: Additional model paths for multi-model ensemble (K3).
        fusion_strategy: Score fusion strategy for ensemble (K3).
        fusion_weights: Per-model weights for weighted_average fusion (K3).
        adaptive_threshold: Enable dynamic threshold based on noise (K4).
        noise_profiler: Custom NoiseProfiler instance (K4).
        speaker_verify_fn: Post-detection speaker verification callback (K5).
        power_manager: Power management controller for duty cycling (K7).
        confirm_count: Consecutive above-threshold scores required (K2).
        score_history_size: Number of recent scores to retain (K2).
    """

    # K3: Multi-model ensemble
    models: list[str] | None = None
    fusion_strategy: FusionStrategy | str = FusionStrategy.AVERAGE
    fusion_weights: list[float] | None = None

    # K4: Adaptive threshold
    adaptive_threshold: bool = False
    noise_profiler: NoiseProfiler | None = None

    # K5: Speaker verification
    speaker_verify_fn: Callable[..., bool] | None = None

    # K7: Power management
    power_manager: PowerManager | None = None

    # K2: Confidence tracking
    confirm_count: int = 1
    score_history_size: int = 50

    def build(self, model: str = "temporal_cnn", **kwargs) -> WakeDetector:
        """Build a WakeDetector from this config.

        Convenience method that passes ``self`` as the ``config=`` argument.

        Args:
            model: Model name or path (default: ``"temporal_cnn"``).
            **kwargs: Additional WakeDetector constructor arguments
                (threshold, cooldown_s, etc.).

        Returns:
            Configured WakeDetector instance.
        """
        return WakeDetector(model=model, config=self, **kwargs)


# Names of advanced kwargs that overlap with DetectorConfig fields.
# Used to detect conflicts when both config= and individual kwargs are given.
_CONFIG_FIELD_NAMES = frozenset({
    "models", "fusion_strategy", "fusion_weights",
    "adaptive_threshold", "noise_profiler",
    "speaker_verify_fn",
    "power_manager",
    "confirm_count", "score_history_size",
})


# ---------------------------------------------------------------------------
# G1: Input validation utilities
# ---------------------------------------------------------------------------

def validate_audio_chunk(data: bytes | np.ndarray) -> np.ndarray:
    """Validate and normalize an audio chunk for use with WakeDetector.

    Accepts bytes (int16 PCM) or numpy arrays (int16, float32, float64).
    Returns a float32 numpy array suitable for processing.

    Args:
        data: Audio chunk as bytes (int16 little-endian PCM) or numpy array.

    Returns:
        Validated float32 numpy array.

    Raises:
        TypeError: If data is not bytes or ndarray.
        ValueError: If data is empty, has invalid dtype, contains only
            non-finite values, or exceeds the maximum chunk size.
    """
    if isinstance(data, bytes):
        if len(data) == 0:
            raise ValueError("Audio chunk is empty (0 bytes)")
        if len(data) % 2 != 0:
            raise ValueError(
                f"Audio bytes length must be even (int16 = 2 bytes/sample), got {len(data)}"
            )
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    elif isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError("Audio chunk is empty (0 samples)")
        if data.ndim != 1:
            raise ValueError(
                f"Audio chunk must be 1-D, got {data.ndim}-D array with shape {data.shape}"
            )
        _ALLOWED_DTYPES = (np.int16, np.float32, np.float64)
        if data.dtype not in _ALLOWED_DTYPES:
            raise ValueError(
                f"Audio chunk dtype must be one of {[str(d) for d in _ALLOWED_DTYPES]}, "
                f"got {data.dtype}"
            )
        if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
            data = np.where(np.isfinite(data), data, 0.0)
            logger.warning("Audio chunk contained non-finite values (NaN/inf); replaced with 0")
        pcm = data.astype(np.float32)
    else:
        raise TypeError(
            f"Audio chunk must be bytes or numpy ndarray, got {type(data).__name__}"
        )

    if len(pcm) > _MAX_CHUNK_SAMPLES:
        raise ValueError(
            f"Audio chunk too large: {len(pcm)} samples "
            f"(max {_MAX_CHUNK_SAMPLES} = {_MAX_CHUNK_SAMPLES // SAMPLE_RATE}s at {SAMPLE_RATE}Hz)"
        )

    return pcm


class WakeDecisionPolicy:
    """4-gate core decision pipeline (RMS floor, threshold, cooldown, playback suppression).

    Extended by WakeDetector with optional confirmation (K2), adaptive
    threshold (K4), and speaker verification (K5).

    Gate 1: Zero-input guard -- skip if RMS < 1.0 (silence / DC offset artifact)
    Gate 2: Score threshold -- skip if model score < threshold
    Gate 3: Cooldown -- ignore events within cooldown_s of last detection
    Gate 4: Listening gate -- suppress during active playback (optional)
    """

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        rms_floor: float = 1.0,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold!r}")

        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.rms_floor = rms_floor
        self._last_detection: float = 0.0

    def evaluate(
        self,
        score: float,
        rms: float = 100.0,
        is_playing: bool = False,
    ) -> bool:
        """Evaluate whether a wake word event should be triggered."""
        if rms < self.rms_floor:
            logger.debug("Gate 1 reject: RMS %.1f below floor %.1f", rms, self.rms_floor)
            return False
        if score < self.threshold:
            return False
        now = time.monotonic()
        if now - self._last_detection < self.cooldown_s:
            logger.debug(
                "Gate 3 reject: cooldown active (%.1fs remaining)",
                self.cooldown_s - (now - self._last_detection),
            )
            return False
        if is_playing:
            logger.debug("Gate 4 reject: playback active")
            return False
        self._last_detection = now
        logger.info("Wake word detected! score=%.3f", score)
        return True

    def reset_cooldown(self) -> None:
        """Reset the cooldown window (useful for testing)."""
        self._last_detection = 0.0


class WakeDetector:
    """Wake word detector using ViolaWake MLP on OpenWakeWord embeddings.

    Supports pluggable inference backends (ONNX Runtime, TFLite) via the
    ``backend`` parameter.  The default ``"auto"`` mode tries ONNX Runtime
    first, then falls back to TFLite, so users on edge devices can run
    without installing ``onnxruntime``.

    Also supports optional competitive features (all opt-in, backward compatible):

    - **K2 Confidence API**: ``get_confidence()`` and ``last_scores`` property.
    - **K3 Multi-model ensemble**: ``models`` parameter with fusion strategies.
    - **K4 Adaptive threshold**: ``adaptive_threshold`` parameter with noise profiling.
    - **K5 Speaker verification**: ``speaker_verify_fn`` callback for post-detection.
    - **K6 Audio source abstraction**: ``from_source()`` class method factory.
    - **K7 Power management**: ``power_manager`` parameter for duty cycling.

    **Threshold tuning guide:**

    - 0.70 = sensitive (more detections, more false positives)
    - 0.80 = balanced (default, recommended starting point)
    - 0.85 = conservative (fewer false positives, may miss some)
    - 0.90+ = very conservative (for noisy environments)

    Start at 0.80 and adjust based on your false accept rate.

    Args:
        model: Model name from the registry, or a path to a model file.
        threshold: Detection confidence threshold in [0.0, 1.0].
        cooldown_s: Minimum seconds between consecutive detections.
        providers: ONNX Runtime execution providers (ignored for TFLite).
        backend: Inference backend selector (``"onnx"``, ``"tflite"``, ``"auto"``).
        config: A ``DetectorConfig`` instance bundling all advanced options.
            Mutually exclusive with the individual advanced kwargs below.
        models: Additional model paths for ensemble scoring (K3).
        fusion_strategy: Score fusion strategy for ensemble (K3).
        fusion_weights: Per-model weights for weighted_average fusion (K3).
        adaptive_threshold: Enable dynamic threshold based on noise (K4).
        noise_profiler: Custom NoiseProfiler instance (K4).
        speaker_verify_fn: Post-detection speaker verification callback (K5).
        power_manager: Power management controller for duty cycling (K7).
        confirm_count: Consecutive above-threshold scores required for detection (K2).
        score_history_size: Number of recent scores to retain (K2).
    """

    _VALID_BACKENDS = ("onnx", "tflite", "auto")

    def __init__(
        self,
        model: str = "temporal_cnn",
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        providers: list[str] | None = None,
        backend: str = "auto",
        *,
        config: DetectorConfig | None = None,
        # K3: Multi-model ensemble (individual kwargs, backwards compat)
        models: list[str] | None = _UNSET,
        fusion_strategy: FusionStrategy | str = _UNSET,
        fusion_weights: list[float] | None = _UNSET,
        # K4: Adaptive threshold
        adaptive_threshold: bool = _UNSET,
        noise_profiler: NoiseProfiler | None = _UNSET,
        # K5: Speaker verification
        speaker_verify_fn: Callable[[np.ndarray], bool] | None = _UNSET,
        # K7: Power management
        power_manager: PowerManager | None = _UNSET,
        # K2: Confidence tracking
        confirm_count: int = _UNSET,
        score_history_size: int = _UNSET,
    ) -> None:
        # --- Resolve config vs individual kwargs -------------------------
        # Detect if any advanced kwarg was explicitly passed (not _UNSET)
        _locals = {
            "models": models,
            "fusion_strategy": fusion_strategy,
            "fusion_weights": fusion_weights,
            "adaptive_threshold": adaptive_threshold,
            "noise_profiler": noise_profiler,
            "speaker_verify_fn": speaker_verify_fn,
            "power_manager": power_manager,
            "confirm_count": confirm_count,
            "score_history_size": score_history_size,
        }
        explicit_kwargs = {
            name for name, val in _locals.items()
            if val is not _UNSET
        }
        if config is not None and explicit_kwargs:
            raise ValueError(
                f"Cannot specify both config= and individual advanced kwargs. "
                f"Conflicting kwargs: {sorted(explicit_kwargs)}. "
                f"Either use config=DetectorConfig(...) or pass kwargs directly, not both."
            )

        if config is not None:
            # Unpack from DetectorConfig
            models = config.models
            fusion_strategy = config.fusion_strategy
            fusion_weights = config.fusion_weights
            adaptive_threshold = config.adaptive_threshold
            noise_profiler = config.noise_profiler
            speaker_verify_fn = config.speaker_verify_fn
            power_manager = config.power_manager
            confirm_count = config.confirm_count
            score_history_size = config.score_history_size
        else:
            # Apply defaults for any _UNSET values (backwards compat path)
            if models is _UNSET:
                models = None
            if fusion_strategy is _UNSET:
                fusion_strategy = FusionStrategy.AVERAGE
            if fusion_weights is _UNSET:
                fusion_weights = None
            if adaptive_threshold is _UNSET:
                adaptive_threshold = False
            if noise_profiler is _UNSET:
                noise_profiler = None
            if speaker_verify_fn is _UNSET:
                speaker_verify_fn = None
            if power_manager is _UNSET:
                power_manager = None
            if confirm_count is _UNSET:
                confirm_count = 1
            if score_history_size is _UNSET:
                score_history_size = 50

        # G1: Input validation for public constructor parameters
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be a number, got {type(threshold).__name__}")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold!r}")
        if not isinstance(cooldown_s, (int, float)):
            raise TypeError(f"cooldown_s must be a number, got {type(cooldown_s).__name__}")
        if cooldown_s < 0:
            raise ValueError(f"cooldown_s must be >= 0, got {cooldown_s!r}")
        if backend not in self._VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {self._VALID_BACKENDS}, got {backend!r}"
            )
        if confirm_count < 1:
            raise ValueError(f"confirm_count must be >= 1, got {confirm_count}")

        self.threshold = threshold
        self._lock = threading.Lock()
        self._backbone_lock = threading.Lock()
        self._policy = WakeDecisionPolicy(threshold=threshold, cooldown_s=cooldown_s)
        self._providers = providers or ["CPUExecutionProvider"]
        self._backend: InferenceBackend = get_backend(backend, providers=self._providers)

        # K2: Confidence tracking
        self._score_tracker = ScoreTracker(
            threshold=threshold, history_size=score_history_size,
        )
        self._confirm_required = confirm_count
        self._confirm_counter = 0

        # K3: Ensemble support
        self._ensemble: EnsembleScorer | None = None
        if models and len(models) > 0:
            if isinstance(fusion_strategy, str):
                fusion_strategy = FusionStrategy(fusion_strategy)
            self._ensemble = EnsembleScorer(
                strategy=fusion_strategy,
                weights=fusion_weights,
            )

        # K4: Noise profiler / adaptive threshold
        self._adaptive_threshold = adaptive_threshold
        if noise_profiler is not None:
            self._noise_profiler: NoiseProfiler | None = noise_profiler
        elif adaptive_threshold:
            self._noise_profiler = NoiseProfiler(base_threshold=threshold)
        else:
            self._noise_profiler = None

        # K5: Speaker verification
        self._speaker_verify_fn = speaker_verify_fn

        # K7: Power manager
        self._power_manager = power_manager

        # Warn on deprecated models
        if model in MODEL_REGISTRY and "DEPRECATED" in MODEL_REGISTRY[model].description:
            import warnings
            warnings.warn(
                f"Model '{model}' is deprecated: {MODEL_REGISTRY[model].description}. "
                f"Use model='temporal_cnn' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Load models
        self._oww_backbone = self._create_oww_backbone()
        self._mlp_session = self._load_session(model)
        self._mlp_input_name = self._mlp_session.get_inputs()[0].name
        self._last_score = 0.0

        # Detect temporal vs MLP model from input shape
        mlp_input_shape = self._mlp_session.get_inputs()[0].shape
        if len(mlp_input_shape) == 3:
            # Temporal model: input is (batch, seq_len, embedding_dim)
            self._is_temporal = True
            self._temporal_seq_len = (
                mlp_input_shape[1]
                if isinstance(mlp_input_shape[1], int)
                else _TEMPORAL_SEQ_LEN_DEFAULT
            )
            self._embedding_buffer: collections.deque[np.ndarray] = collections.deque(
                maxlen=self._temporal_seq_len,
            )
            logger.info(
                "Temporal model detected: seq_len=%d", self._temporal_seq_len,
            )
        else:
            self._is_temporal = False
            self._temporal_seq_len = 0

        # K3: Load additional ensemble models
        if models and self._ensemble is not None:
            # Add primary model to ensemble
            self._ensemble.add_session(self._mlp_session, self._mlp_input_name)
            for extra_model in models:
                extra_session = self._load_session(extra_model)
                extra_input_name = extra_session.get_inputs()[0].name
                self._ensemble.add_session(extra_session, extra_input_name)

        logger.info(
            "WakeDetector initialized: model=%s, threshold=%.2f, backend=%s",
            model, threshold, self._backend.name,
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> WakeDetector:
        """Enter sync context manager. Returns self."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit sync context manager. Releases sessions and resets state."""
        self.close()

    def close(self) -> None:
        """Release inference sessions and reset internal state.

        After calling close(), the detector should not be used for inference.
        This is called automatically when using WakeDetector as a context
        manager.
        """
        self.reset()
        # Release inference session references so the underlying runtime
        # (ONNX / TFLite) can free memory immediately rather than waiting
        # for garbage collection.
        self._mlp_session = None  # type: ignore[assignment]
        if self._ensemble is not None:
            self._ensemble.clear()
        self._oww_backbone = None  # type: ignore[assignment]

    def _create_oww_backbone(self) -> OpenWakeWordBackbone:
        """Create the shared OpenWakeWord backbone."""
        return OpenWakeWordBackbone(self._backend)

    def _load_session(self, model: str) -> BackendSession:
        """Load a model file via the configured backend.

        Resolves *model* to a file path (direct path, .onnx/.tflite suffix,
        or registry lookup), then delegates to ``self._backend.load()``.

        For TFLite backends, if only a ``.onnx`` file exists in the cache
        the method looks for a sibling ``.tflite`` file with the same stem.
        """
        model_path = self._resolve_model_path(model)

        # When using the TFLite backend, prefer a .tflite sibling if the
        # resolved path is an .onnx file.
        if self._backend.name == "tflite" and model_path.suffix == ".onnx":
            tflite_sibling = model_path.with_suffix(".tflite")
            if tflite_sibling.exists():
                model_path = tflite_sibling
                logger.debug("TFLite backend: using .tflite sibling %s", model_path)
            else:
                logger.warning(
                    "TFLite backend selected but only .onnx file found at %s. "
                    "Convert with: python -c "
                    "\"from violawake_sdk.backends.tflite_backend import "
                    "convert_onnx_to_tflite; convert_onnx_to_tflite('%s')\"",
                    model_path, model_path,
                )

        try:
            session = self._backend.load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_path}: {e}") from e
        logger.debug("Loaded model via %s backend: %s", self._backend.name, model_path)
        return session

    @staticmethod
    def _resolve_model_path(model: str) -> Path:
        """Resolve a model name or path string to a concrete file path.

        Resolution order:
        1. If *model* is an existing file path, use it directly.
        2. If *model* ends with ``.onnx`` or ``.tflite``, treat as a path
           (raise if not found).
        3. Otherwise, look up *model* in the model registry / cache.
        """
        if Path(model).is_file():
            return Path(model)

        if model.endswith((".onnx", ".tflite")):
            path = Path(model)
            if not path.exists():
                raise ModelNotFoundError(
                    f"Model file not found: {model}. "
                    f"If this is a named model, omit the file extension."
                )
            return path

        try:
            return get_model_path(model)
        except FileNotFoundError as e:
            raise ModelNotFoundError(
                f"Model '{model}' not found in cache and auto-download failed or is disabled. "
                f"Run: violawake-download --model {model}"
            ) from e

    def _get_embedding(self, audio_frame: bytes | np.ndarray) -> np.ndarray:
        """Extract the OWW embedding from an audio frame.

        Returns the raw embedding vector before MLP scoring.
        Used internally for speaker verification (K5).
        """
        with self._backbone_lock:
            embedding = self._oww_backbone.last_embedding
            if embedding is None:
                _, embedding = self._oww_backbone.push_audio(audio_frame)
        if embedding is None:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)
        return embedding

    @staticmethod
    def _needs_int16_normalization(audio_frame: bytes | np.ndarray) -> bool:
        """Check whether audio_frame requires int16-to-float normalization."""
        return isinstance(audio_frame, bytes) or (
            isinstance(audio_frame, np.ndarray) and audio_frame.dtype == np.int16
        )

    @staticmethod
    def _prepare_model_audio(audio_frame: bytes | np.ndarray) -> np.ndarray:
        """Validate an audio frame and normalize it for model inference."""
        pcm = validate_audio_chunk(audio_frame)
        if WakeDetector._needs_int16_normalization(audio_frame):
            return pcm / 32768.0
        return pcm

    def process(self, audio_frame: bytes | np.ndarray) -> float:
        """Process a 20ms audio frame and return the wake word detection score.

        If ensemble mode is active (K3), returns the fused score.
        The score is recorded for confidence tracking (K2) and reported
        to the power manager (K7) if configured.

        Thread-safe: protects internal state mutation with a lock.

        Raises:
            TypeError: If audio_frame is not bytes or ndarray.
            ValueError: If audio_frame is empty, malformed, or drastically
                larger than the supported streaming frame size.
        """
        return self._process_core(self._prepare_model_audio(audio_frame), audio_frame)

    def _process_core(self, pcm: np.ndarray, raw_audio_frame: bytes | np.ndarray) -> float:
        """Internal scoring engine operating on pre-validated, normalized PCM.

        Args:
            pcm: Float32 array, already validated and normalized by _prepare_model_audio.
            raw_audio_frame: Original audio frame passed through to OWW backbone.
        """
        if pcm.shape[0] != FRAME_SAMPLES:
            # Reject pathologically large or empty frames first
            if pcm.shape[0] == 0 or pcm.shape[0] > _MAX_PROCESS_FRAME_SAMPLES:
                raise ValueError(
                    "Audio frame length is too far from the expected streaming size: "
                    f"expected {FRAME_SAMPLES} samples, got {pcm.shape[0]} "
                    f"(maximum non-pathological size is {_MAX_PROCESS_FRAME_SAMPLES})"
                )
            # Non-multiples of 320 indicate wrong sample rate — return 0.0
            if pcm.shape[0] % FRAME_SAMPLES != 0:
                logger.warning(
                    "Audio frame has %d samples (not a multiple of %d). "
                    "Expected 16kHz, 20ms frames. Returning score 0.0.",
                    pcm.shape[0], FRAME_SAMPLES,
                )
                return 0.0
        with self._backbone_lock:
            produced_embedding, embedding = self._oww_backbone.push_audio(raw_audio_frame)
            if embedding is None:
                score = self._last_score
            elif self._ensemble is not None and self._ensemble.model_count > 0:
                score = self._ensemble.score(embedding.flatten()) if produced_embedding else self._last_score
            elif self._is_temporal:
                if produced_embedding:
                    self._embedding_buffer.append(embedding.flatten())
                    if len(self._embedding_buffer) >= self._temporal_seq_len:
                        temporal_input = np.stack(list(self._embedding_buffer))
                        temporal_input = temporal_input.reshape(
                            1, self._temporal_seq_len, EMBEDDING_DIM,
                        ).astype(np.float32)
                        score = float(
                            self._mlp_session.run(None, {self._mlp_input_name: temporal_input})[0].flatten()[0]
                        )
                    else:
                        score = 0.0
                else:
                    score = self._last_score
            else:
                if produced_embedding:
                    mlp_input = embedding.reshape(1, EMBEDDING_DIM).astype(np.float32)
                    score_output = self._mlp_session.run(None, {self._mlp_input_name: mlp_input})[0]
                    score = float(np.asarray(score_output).reshape(-1)[0])
                else:
                    score = self._last_score

        with self._lock:
            self._last_score = score
            # K2: Record score for confidence tracking
            self._score_tracker.record(score)

        # K7: Report score to power manager for activity detection
        if self._power_manager is not None:
            self._power_manager.report_score(score)

        return score

    def detect(self, audio_frame: bytes | np.ndarray, is_playing: bool = False) -> bool:
        """Process a frame and apply the full decision policy.

        Integrates adaptive threshold (K4), multi-window confirmation (K2),
        speaker verification (K5), and power management (K7) when configured.

        Thread-safe: protects internal state mutation with a lock.

        Raises:
            TypeError: If audio_frame is not bytes or ndarray.
            ValueError: If audio_frame is empty or has invalid format.
        """
        # G1: Input validation (single pass — process_core skips re-validation)
        pcm = validate_audio_chunk(audio_frame)

        # Compute RMS on int16-scale PCM for the rms_floor comparison.
        # rms_floor=1.0 is calibrated for int16 scale (speech ≈ 500–5000,
        # silence ≈ 0–5).  Float32 input in [-1, 1] is scaled up so the
        # same rms_floor works regardless of input format.
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        if not self._needs_int16_normalization(audio_frame):
            # Float32/float64 input: RMS is in [0, ~0.7] — scale to int16 range
            rms *= 32768.0

        # Normalize for model inference
        if self._needs_int16_normalization(audio_frame):
            model_pcm = pcm / 32768.0
        else:
            model_pcm = pcm

        # K7: Power management -- skip frame if power manager says so
        if self._power_manager is not None:
            if not self._power_manager.should_process(pcm):
                return False

        # K4: Update noise profiler and get adaptive threshold
        if self._noise_profiler is not None and self._adaptive_threshold:
            adapted = self._noise_profiler.update(pcm)
            self._policy.threshold = adapted

        score = self._process_core(model_pcm, audio_frame)

        # K5: Pre-fetch embedding outside _lock to avoid ABBA deadlock
        # (_process_core acquires _backbone_lock -> _lock; we must not
        # acquire _backbone_lock while holding _lock).
        #
        # Trade-off: Under concurrent access, _get_embedding reads the
        # backbone's last_embedding which may have been overwritten by
        # another thread's _process_core call since our score was computed.
        # This means the embedding used for speaker verification may not
        # correspond to the score just returned by _process_core.  This is
        # accepted for performance — taking _backbone_lock across both
        # _process_core and _get_embedding would serialize all detection,
        # and the mismatch is benign (embeddings from adjacent frames are
        # nearly identical in practice).
        speaker_embedding: np.ndarray | None = None
        if self._speaker_verify_fn is not None:
            speaker_embedding = self._get_embedding(audio_frame)

        with self._lock:
            # K2: Multi-window confirmation
            if score >= self._policy.threshold:
                self._confirm_counter += 1
            else:
                self._confirm_counter = 0

            effective_detected = self._confirm_counter >= self._confirm_required

            if effective_detected:
                detected = self._policy.evaluate(score=score, rms=rms, is_playing=is_playing)
            else:
                detected = False

            if detected:
                # K5: Speaker verification post-detection
                if self._speaker_verify_fn is not None and speaker_embedding is not None:
                    if not self._speaker_verify_fn(speaker_embedding.flatten()):
                        logger.debug("Speaker verification rejected detection")
                        return False

                self._confirm_counter = 0
                return True

        return False

    def reset_cooldown(self) -> None:
        """Reset the cooldown window without clearing confirmation state or buffers."""
        with self._lock:
            self._policy.reset_cooldown()

    def reset(self) -> None:
        """Reset cooldown, confirmation state, score history, and temporal buffers.

        Lock ordering: _backbone_lock then _lock, matching _process_core
        to prevent ABBA deadlock.
        """
        with self._backbone_lock:
            with self._lock:
                self._policy.reset_cooldown()
                self._confirm_counter = 0
                self._last_score = 0.0
                self._score_tracker.reset()
                self._oww_backbone.reset()
                if self._is_temporal:
                    self._embedding_buffer.clear()

    # ------------------------------------------------------------------
    # K2: Confidence API
    # ------------------------------------------------------------------

    def get_confidence(self) -> ConfidenceResult:
        """Return a confidence assessment of the current detection state.

        Includes the raw MLP score, multi-window confirmation count,
        and a classified confidence level (LOW/MEDIUM/HIGH/CERTAIN).
        """
        return self._score_tracker.classify(
            confirm_count=self._confirm_counter,
            confirm_required=self._confirm_required,
        )

    @property
    def last_scores(self) -> tuple[float, ...]:
        """Return the recent score history (most recent last)."""
        return self._score_tracker.last_scores

    # ------------------------------------------------------------------
    # K5: Speaker verification helpers
    # ------------------------------------------------------------------

    def enroll_speaker(self, speaker_id: str, audio_frames: list[bytes | np.ndarray]) -> int:
        """Enroll a speaker by extracting embeddings from audio frames.

        Requires a ``SpeakerVerificationHook`` as the ``speaker_verify_fn``.

        Args:
            speaker_id: Unique identifier for the speaker.
            audio_frames: Audio frames to extract embeddings from.

        Returns:
            Total enrollment count for this speaker.

        Raises:
            RuntimeError: If no SpeakerVerificationHook is configured.
        """
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = self._speaker_verify_fn
        if not isinstance(hook, SpeakerVerificationHook):
            raise RuntimeError(
                "enroll_speaker requires a SpeakerVerificationHook as speaker_verify_fn"
            )

        embeddings = []
        for frame in audio_frames:
            emb = self._get_embedding(frame)
            embeddings.append(emb.flatten())

        return hook.enroll_speaker(speaker_id, embeddings)

    def verify_speaker(self, audio_frame: bytes | np.ndarray) -> SpeakerVerifyResult:
        """Verify the speaker in an audio frame against enrolled profiles.

        Args:
            audio_frame: Audio frame to verify.

        Returns:
            SpeakerVerifyResult with match details.

        Raises:
            RuntimeError: If no SpeakerVerificationHook is configured.
        """
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = self._speaker_verify_fn
        if not isinstance(hook, SpeakerVerificationHook):
            raise RuntimeError(
                "verify_speaker requires a SpeakerVerificationHook as speaker_verify_fn"
            )

        embedding = self._get_embedding(audio_frame)
        return hook.verify_speaker(embedding.flatten())

    # ------------------------------------------------------------------
    # K6: Audio source factory
    # ------------------------------------------------------------------

    @classmethod
    def from_source(
        cls,
        source: AudioSource,
        model: str = "temporal_cnn",
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        **kwargs,
    ) -> _SourceDetector:
        """Create a WakeDetector bound to an AudioSource.

        The returned object wraps a WakeDetector and provides a ``run()``
        method that reads frames from the source and runs detection.

        Args:
            source: Any object implementing the AudioSource protocol.
            model: Model name or path.
            threshold: Detection threshold.
            cooldown_s: Cooldown between detections.
            **kwargs: Additional WakeDetector keyword arguments.

        Returns:
            A _SourceDetector wrapping both the source and detector.
        """
        detector = cls(
            model=model, threshold=threshold, cooldown_s=cooldown_s, **kwargs,
        )
        return _SourceDetector(detector=detector, source=source)

    # ------------------------------------------------------------------
    # Original methods
    # ------------------------------------------------------------------

    def stream_mic(self, device_index: int | None = None) -> Generator[bytes, None, None]:
        """Generator that yields 20ms audio frames from the default microphone."""
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio is required for microphone features. "
                "Install with: pip install violawake[audio]"
            ) from None
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True,
                frames_per_buffer=FRAME_SAMPLES, input_device_index=device_index,
            )
        except Exception as e:
            pa.terminate()
            raise AudioCaptureError(
                f"Failed to open microphone: {e}. "
                f"Check that a microphone is connected and not in use by another application."
            ) from e
        logger.info("Microphone capture started (16kHz, mono, 20ms frames)")
        _MAX_CONSECUTIVE_ERRORS = 10
        consecutive_errors = 0
        try:
            while True:
                try:
                    yield stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                    consecutive_errors = 0
                except Exception as e:
                    consecutive_errors += 1
                    logger.warning("Mic read error (%d/%d): %s", consecutive_errors, _MAX_CONSECUTIVE_ERRORS, e)
                    if consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                        raise AudioCaptureError(
                            f"Microphone read failed {_MAX_CONSECUTIVE_ERRORS} consecutive times. "
                            f"Last error: {e}"
                        ) from e
                    continue
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            logger.info("Microphone capture stopped")


class _SourceDetector:
    """Wraps a WakeDetector and AudioSource for convenient streaming detection.

    Returned by ``WakeDetector.from_source()``.
    """

    def __init__(self, detector: WakeDetector, source: AudioSource) -> None:
        self.detector = detector
        self.source = source

    def run(
        self,
        on_detect: Callable[[], None] | None = None,
        max_frames: int | None = None,
        is_playing: bool = False,
    ) -> int:
        """Run detection loop reading from the audio source.

        Args:
            on_detect: Callback invoked on each wake word detection.
            max_frames: Stop after this many frames (None = until source exhausted).
            is_playing: Whether playback is active (passed to detect()).

        Returns:
            Total number of detections.
        """
        self.source.start()
        detections = 0
        frames_read = 0
        try:
            while True:
                if max_frames is not None and frames_read >= max_frames:
                    break
                frame = self.source.read_frame()
                if frame is None:
                    break
                frames_read += 1
                if self.detector.detect(frame, is_playing=is_playing):
                    detections += 1
                    if on_detect is not None:
                        on_detect()
        finally:
            self.source.stop()
        return detections

    def get_confidence(self) -> ConfidenceResult:
        """Proxy to the underlying detector's confidence API."""
        return self.detector.get_confidence()

    @property
    def last_scores(self) -> tuple[float, ...]:
        """Proxy to the underlying detector's score history."""
        return self.detector.last_scores


class WakewordDetector:
    """Deprecated compatibility wrapper — use ``WakeDetector`` instead.

    .. deprecated:: 0.1.0
        Use :class:`WakeDetector` directly. ``WakewordDetector`` will be
        removed in v1.0.
    """

    def __init__(
        self, wake_word: str = "viola", threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S, providers: list[str] | None = None,
        backend: str = "auto",
    ) -> None:
        import warnings
        warnings.warn(
            "WakewordDetector is deprecated. Use WakeDetector instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.wake_word = wake_word
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.providers = providers
        self.backend = backend
        self._detector: WakeDetector | None = None
        self._init_lock = threading.Lock()
        self._model_name = self._resolve_model_name(wake_word)

    @staticmethod
    def _resolve_model_name(wake_word: str) -> str:
        if wake_word in WAKE_WORD_ALIASES:
            return WAKE_WORD_ALIASES[wake_word]
        if wake_word in MODEL_REGISTRY:
            return wake_word
        available = ", ".join(sorted({*WAKE_WORD_ALIASES, *MODEL_REGISTRY}))
        raise KeyError(f"Unknown wakeword '{wake_word}'. Available: {available}")

    def _get_detector(self) -> WakeDetector:
        # Double-checked locking: fast path avoids lock acquisition
        if self._detector is not None:
            return self._detector
        with self._init_lock:
            if self._detector is None:
                self._detector = WakeDetector(
                    model=self._model_name, threshold=self.threshold,
                    cooldown_s=self.cooldown_s, providers=self.providers,
                    backend=self.backend,
                )
        return self._detector

    def process_audio(self, audio_frame: bytes | np.ndarray, is_playing: bool = False) -> bool:
        return self._get_detector().detect(audio_frame, is_playing=is_playing)

    def process(self, audio_frame: bytes | np.ndarray) -> float:
        return self._get_detector().process(audio_frame)

    def detect(self, audio_frame: bytes | np.ndarray, is_playing: bool = False) -> bool:
        return self._get_detector().detect(audio_frame, is_playing=is_playing)

    def stream_mic(self, device_index: int | None = None) -> Generator[bytes, None, None]:
        yield from self._get_detector().stream_mic(device_index=device_index)

    def reset(self) -> None:
        self._get_detector().reset()

    def reset_cooldown(self) -> None:
        self._get_detector().reset_cooldown()

    def get_confidence(self) -> ConfidenceResult:
        return self._get_detector().get_confidence()

    @property
    def last_scores(self) -> tuple[float, ...]:
        return self._get_detector().last_scores
