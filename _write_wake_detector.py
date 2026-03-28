"""Helper script to write wake_detector.py atomically."""
import pathlib

content = '''\
"""Wake word detection using ViolaWake MLP + OpenWakeWord backbone."""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from violawake_sdk._exceptions import AudioCaptureError, ModelLoadError, ModelNotFoundError
from violawake_sdk.backends import get_backend
from violawake_sdk.backends.base import BackendSession, InferenceBackend
from violawake_sdk.models import MODEL_REGISTRY, get_model_path

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)

# Frame configuration (matches production Viola)
SAMPLE_RATE = 16_000          # 16 kHz mono
FRAME_MS = 20                 # 20ms frames
FRAME_SAMPLES = 320           # 16000 * 0.020
DEFAULT_THRESHOLD = 0.80      # Raised from 0.50 after false-positive flood (see ADR-002)
DEFAULT_COOLDOWN_S = 2.0      # Minimum seconds between detections
WAKE_WORD_ALIASES = {
    "viola": "viola_mlp_oww",
}


class WakeDecisionPolicy:
    """4-gate decision pipeline (from production Viola)."""

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

    def evaluate(self, score: float, rms: float = 100.0, is_playing: bool = False) -> bool:
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

    Example::

        detector = WakeDetector(threshold=0.80)
        for chunk in detector.stream_mic():
            if detector.process(chunk):
                print("Wake word detected!")
                break
    """

    def __init__(
        self,
        model: str = "viola_mlp_oww",
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        providers: list[str] | None = None,
        backend: str = "auto",
    ) -> None:
        """Initialize the wake word detector.

        Args:
            model: Model name or path to .onnx/.tflite file.
            threshold: Detection threshold in [0.0, 1.0].
            cooldown_s: Minimum seconds between consecutive detections.
            providers: ONNX Runtime execution providers. Ignored for tflite.
            backend: Inference backend: "auto" (default), "onnx", or "tflite".
        """
        self.threshold = threshold
        self._policy = WakeDecisionPolicy(threshold=threshold, cooldown_s=cooldown_s)
        self._providers = providers or ["CPUExecutionProvider"]
        self._backend_name = backend
        self._backend: InferenceBackend = get_backend(backend, providers=self._providers)
        self._oww_session = self._load_session("oww_backbone")
        self._mlp_session = self._load_session(model)
        self._oww_input_name = self._oww_session.get_inputs()[0].name
        self._mlp_input_name = self._mlp_session.get_inputs()[0].name
        logger.info(
            "WakeDetector initialized: model=%s, threshold=%.2f, backend=%s",
            model, threshold, self._backend.name,
        )

    def _load_session(self, model: str) -> BackendSession:
        """Load a model via the configured backend, downloading if necessary."""
        if Path(model).is_file():
            model_path = Path(model)
        elif model.endswith((".onnx", ".tflite")):
            model_path = Path(model)
            if not model_path.exists():
                raise ModelNotFoundError(
                    f"Model file not found: {model}. "
                    f"If this is a named model, omit the extension."
                )
        else:
            try:
                base_path = get_model_path(model)
            except FileNotFoundError as e:
                raise ModelNotFoundError(
                    f"Model \\'{model}\\' not found in cache. "
                    f"Run: violawake-download --model {model}"
                ) from e
            if self._backend.name == "tflite":
                tflite_path = base_path.with_suffix(".tflite")
                model_path = tflite_path if tflite_path.exists() else base_path
            else:
                model_path = base_path
        try:
            session = self._backend.load(model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model {model_path}: {e}") from e
        logger.debug("Loaded model via %s backend: %s", self._backend.name, model_path)
        return session

    def process(self, audio_frame: bytes | np.ndarray) -> float:
        """Process a 20ms audio frame and return the wake word detection score."""
        if isinstance(audio_frame, bytes):
            pcm = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            pcm = np.asarray(audio_frame)
            if pcm.dtype == np.int16:
                pcm = pcm.astype(np.float32) / 32768.0
            else:
                pcm = pcm.astype(np.float32)
        if pcm.shape[0] != FRAME_SAMPLES:
            logger.warning(
                "Expected %d samples, got %d -- resampling not supported",
                FRAME_SAMPLES, pcm.shape[0],
            )
        oww_input = pcm.reshape(1, -1)
        embedding = self._oww_session.run(None, {self._oww_input_name: oww_input})[0]
        score = float(self._mlp_session.run(None, {self._mlp_input_name: embedding})[0][0])
        return score

    def detect(self, audio_frame: bytes | np.ndarray, is_playing: bool = False) -> bool:
        """Process a frame and apply the full 4-gate decision policy."""
        if isinstance(audio_frame, bytes):
            pcm = np.frombuffer(audio_frame, dtype=np.int16).astype(np.float32)
        else:
            pcm = np.asarray(audio_frame, dtype=np.float32)
        rms = float(np.sqrt(np.mean(pcm ** 2)))
        score = self.process(audio_frame)
        return self._policy.evaluate(score=score, rms=rms, is_playing=is_playing)

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
                format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True, frames_per_buffer=FRAME_SAMPLES,
                input_device_index=device_index,
            )
        except Exception as e:
            pa.terminate()
            raise AudioCaptureError(
                f"Failed to open microphone: {e}. "
                f"Check that a microphone is connected."
            ) from e
        logger.info("Microphone capture started (16kHz, mono, 20ms frames)")
        try:
            while True:
                try:
                    yield stream.read(FRAME_SAMPLES, exception_on_overflow=False)
                except Exception as e:
                    logger.warning("Mic read error: %s", e)
                    continue
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            logger.info("Microphone capture stopped")


class WakewordDetector:
    """Compatibility wrapper that lazy-loads WakeDetector on first use."""

    def __init__(
        self,
        wake_word: str = "viola",
        threshold: float = DEFAULT_THRESHOLD,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        providers: list[str] | None = None,
        backend: str = "auto",
    ) -> None:
        self.wake_word = wake_word
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.providers = providers
        self.backend = backend
        self._detector: WakeDetector | None = None
        self._model_name = self._resolve_model_name(wake_word)

    @staticmethod
    def _resolve_model_name(wake_word: str) -> str:
        if wake_word in WAKE_WORD_ALIASES:
            return WAKE_WORD_ALIASES[wake_word]
        if wake_word in MODEL_REGISTRY:
            return wake_word
        available = ", ".join(sorted({*WAKE_WORD_ALIASES, *MODEL_REGISTRY}))
        raise KeyError(f"Unknown wakeword \\'{wake_word}\\'. Available: {available}")

    def _get_detector(self) -> WakeDetector:
        if self._detector is None:
            self._detector = WakeDetector(
                model=self._model_name, threshold=self.threshold,
                cooldown_s=self.cooldown_s, providers=self.providers,
                backend=self.backend,
            )
        return self._detector

    def process_audio(self, audio_frame: bytes | np.ndarray, is_playing: bool = False) -> bool:
        """Process a frame and return the compatibility boolean detection result."""
        return self._get_detector().detect(audio_frame, is_playing=is_playing)

    def process(self, audio_frame: bytes | np.ndarray) -> float:
        """Return the raw wake score from the underlying detector."""
        return self._get_detector().process(audio_frame)

    def detect(self, audio_frame: bytes | np.ndarray, is_playing: bool = False) -> bool:
        """Expose the native detection API for callers migrating forward."""
        return self._get_detector().detect(audio_frame, is_playing=is_playing)

    def stream_mic(self, device_index: int | None = None) -> Generator[bytes, None, None]:
        """Delegate microphone streaming to the underlying detector."""
        yield from self._get_detector().stream_mic(device_index=device_index)
'''

target = pathlib.Path(r'J:\CLAUDE\PROJECTS\Wakeword\src\violawake_sdk\wake_detector.py')
target.write_text(content, encoding='utf-8')
print(f'Written {len(content)} bytes to {target}')
