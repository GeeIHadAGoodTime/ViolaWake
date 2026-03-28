"""Unit tests for DetectorConfig and its integration with WakeDetector.

Verifies:
- Default DetectorConfig produces same behavior as default kwargs
- Config object parameters are forwarded correctly
- Individual kwargs still work (backwards compat)
- Cannot specify both config and individual advanced kwargs (ValueError)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.ensemble import FusionStrategy
from violawake_sdk.noise_profiler import NoiseProfiler
from violawake_sdk.power_manager import PowerManager
from violawake_sdk.wake_detector import (
    FRAME_SAMPLES,
    DetectorConfig,
    WakeDetector,
)


def _loud_float32_frame() -> np.ndarray:
    """Float32 frame with RMS > 1.0 so it passes the rms_floor gate."""
    rng = np.random.default_rng(99)
    return rng.uniform(-5.0, 5.0, FRAME_SAMPLES).astype(np.float32)


# ---------------------------------------------------------------------------
# Helper to build a WakeDetector with fully mocked backend
# ---------------------------------------------------------------------------

def _make_backend_session(output_value: np.ndarray) -> MagicMock:
    """Return a mock BackendSession that always returns *output_value*."""
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    inp.shape = [1, 96]
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [output_value]
    return sess


def _make_fake_backbone() -> MagicMock:
    backbone = MagicMock()
    backbone.push_audio.return_value = (True, np.ones(96, dtype=np.float32) * 0.5)
    backbone.last_embedding = np.ones(96, dtype=np.float32) * 0.5
    return backbone


def _build_detector(
    mlp_score: float = 0.95,
    threshold: float = 0.80,
    cooldown_s: float = 0.0,
    **kwargs,
) -> WakeDetector:
    """Build a WakeDetector with mocked backend sessions."""
    mlp_sess = _make_backend_session(np.array([[mlp_score]], dtype=np.float32))

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    mock_backend.load.return_value = mlp_sess

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=_make_fake_backbone()),
    ):
        det = WakeDetector(
            threshold=threshold,
            cooldown_s=cooldown_s,
            **kwargs,
        )

    return det


# ---------------------------------------------------------------------------
# DetectorConfig dataclass tests
# ---------------------------------------------------------------------------

class TestDetectorConfigDefaults:
    """DetectorConfig defaults match the original WakeDetector defaults."""

    def test_default_values(self) -> None:
        cfg = DetectorConfig()
        assert cfg.models is None
        assert cfg.fusion_strategy == FusionStrategy.AVERAGE
        assert cfg.fusion_weights is None
        assert cfg.adaptive_threshold is False
        assert cfg.noise_profiler is None
        assert cfg.speaker_verify_fn is None
        assert cfg.power_manager is None
        assert cfg.confirm_count == 1
        assert cfg.score_history_size == 50

    def test_default_config_same_as_no_config(self, loud_noise_frame: bytes) -> None:
        """A default DetectorConfig produces the same detection behavior as no config."""
        det_no_config = _build_detector(mlp_score=0.95, cooldown_s=0.0)
        det_with_config = _build_detector(
            mlp_score=0.95, cooldown_s=0.0, config=DetectorConfig(),
        )

        score_a = det_no_config.process(loud_noise_frame)
        score_b = det_with_config.process(loud_noise_frame)
        assert abs(score_a - score_b) < 0.01

        result_a = det_no_config.detect(loud_noise_frame)
        result_b = det_with_config.detect(loud_noise_frame)
        assert result_a == result_b


# ---------------------------------------------------------------------------
# Config forwarding tests
# ---------------------------------------------------------------------------

class TestConfigForwarding:
    """Config object parameters are forwarded to WakeDetector internals."""

    def test_confirm_count_forwarded(self) -> None:
        """confirm_count=3 via config requires 3 consecutive above-threshold frames."""
        cfg = DetectorConfig(confirm_count=3)
        det = _build_detector(mlp_score=0.95, cooldown_s=0.0, config=cfg)
        frame = _loud_float32_frame()

        # First two frames should not trigger (need 3 consecutive)
        assert det.detect(frame) is False
        assert det.detect(frame) is False
        # Third frame should trigger
        assert det.detect(frame) is True

    def test_score_history_size_forwarded(self) -> None:
        """score_history_size via config is applied to the score tracker."""
        cfg = DetectorConfig(score_history_size=10)
        det = _build_detector(config=cfg)
        assert det._score_tracker._history.maxlen == 10

    def test_adaptive_threshold_forwarded(self) -> None:
        """adaptive_threshold=True via config creates a noise profiler."""
        cfg = DetectorConfig(adaptive_threshold=True)
        det = _build_detector(config=cfg)
        assert det._adaptive_threshold is True
        assert det._noise_profiler is not None

    def test_custom_noise_profiler_forwarded(self) -> None:
        """A custom noise_profiler via config is used directly."""
        profiler = NoiseProfiler(base_threshold=0.90)
        cfg = DetectorConfig(noise_profiler=profiler)
        det = _build_detector(config=cfg)
        assert det._noise_profiler is profiler

    def test_speaker_verify_fn_forwarded(self) -> None:
        """speaker_verify_fn via config is stored."""
        fn = MagicMock(return_value=True)
        cfg = DetectorConfig(speaker_verify_fn=fn)
        det = _build_detector(config=cfg)
        assert det._speaker_verify_fn is fn

    def test_power_manager_forwarded(self) -> None:
        """power_manager via config is stored."""
        pm = MagicMock(spec=PowerManager)
        cfg = DetectorConfig(power_manager=pm)
        det = _build_detector(config=cfg)
        assert det._power_manager is pm


# ---------------------------------------------------------------------------
# Backwards compatibility: individual kwargs still work
# ---------------------------------------------------------------------------

class TestBackwardsCompat:
    """All existing constructor kwargs still work without config=."""

    def test_confirm_count_kwarg(self) -> None:
        det = _build_detector(mlp_score=0.95, cooldown_s=0.0, confirm_count=2)
        frame = _loud_float32_frame()
        assert det.detect(frame) is False
        assert det.detect(frame) is True

    def test_adaptive_threshold_kwarg(self) -> None:
        det = _build_detector(adaptive_threshold=True)
        assert det._adaptive_threshold is True
        assert det._noise_profiler is not None

    def test_score_history_size_kwarg(self) -> None:
        det = _build_detector(score_history_size=5)
        assert det._score_tracker._history.maxlen == 5

    def test_speaker_verify_fn_kwarg(self) -> None:
        fn = MagicMock(return_value=True)
        det = _build_detector(speaker_verify_fn=fn)
        assert det._speaker_verify_fn is fn

    def test_no_kwargs_uses_defaults(self) -> None:
        """No advanced kwargs => default values (same as DetectorConfig defaults)."""
        det = _build_detector()
        assert det._confirm_required == 1
        assert det._adaptive_threshold is False
        assert det._noise_profiler is None
        assert det._speaker_verify_fn is None
        assert det._power_manager is None
        assert det._ensemble is None


# ---------------------------------------------------------------------------
# Conflict detection: config + individual kwargs => ValueError
# ---------------------------------------------------------------------------

class TestConfigKwargConflict:
    """Specifying both config= and individual advanced kwargs raises ValueError."""

    def test_config_plus_confirm_count(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both config="):
            _build_detector(config=DetectorConfig(), confirm_count=3)

    def test_config_plus_adaptive_threshold(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both config="):
            _build_detector(config=DetectorConfig(), adaptive_threshold=True)

    def test_config_plus_models(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both config="):
            _build_detector(config=DetectorConfig(), models=["extra_model"])

    def test_config_plus_multiple_kwargs(self) -> None:
        with pytest.raises(ValueError, match="Cannot specify both config="):
            _build_detector(
                config=DetectorConfig(),
                confirm_count=2,
                score_history_size=10,
            )

    def test_error_message_lists_conflicts(self) -> None:
        with pytest.raises(ValueError, match="confirm_count") as exc_info:
            _build_detector(config=DetectorConfig(), confirm_count=5)
        assert "confirm_count" in str(exc_info.value)

    def test_basic_kwargs_with_config_ok(self) -> None:
        """Basic params (model, threshold, cooldown_s) can be used alongside config."""
        det = _build_detector(
            threshold=0.75,
            cooldown_s=1.0,
            config=DetectorConfig(confirm_count=2),
        )
        assert det.threshold == 0.75
        assert det._confirm_required == 2


# ---------------------------------------------------------------------------
# Export test
# ---------------------------------------------------------------------------

class TestExport:
    """DetectorConfig is importable from the top-level package."""

    def test_importable_from_package(self) -> None:
        from violawake_sdk import DetectorConfig as DC
        assert DC is DetectorConfig

    def test_in_all(self) -> None:
        import violawake_sdk
        assert "DetectorConfig" in violawake_sdk.__all__
