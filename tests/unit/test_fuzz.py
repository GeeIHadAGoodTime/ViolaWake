"""H10: Fuzz testing for WakeDetector and validate_audio_chunk.

Tests:
- Random bytes as audio input
- Random dtypes (int8, int16, int32, float16, float32, float64)
- Random lengths (0 to 1M samples)
- NaN, inf, -inf injection
- Ensure no crashes, only clean errors
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.wake_detector import (
    FRAME_SAMPLES,
    WakeDetector,
    validate_audio_chunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_backend_session(score: float = 0.50):
    """Create a mock BackendSession (returned by backend.load())."""
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [np.array([[score]], dtype=np.float32)]
    return sess


def _create_detector(score: float = 0.50) -> WakeDetector:
    """Build a WakeDetector with fully mocked backend."""
    oww_sess = _mock_backend_session(score=0.10)  # OWW returns embeddings
    oww_sess.run.return_value = [np.ones((1, 96), dtype=np.float32) * 0.1]
    mlp_sess = _mock_backend_session(score=score)

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    sessions = [oww_sess, mlp_sess]
    mock_backend.load.side_effect = sessions

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
            push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.1)),
        )),
    ):
        return WakeDetector(
            model="viola_mlp_oww",
            threshold=0.80,
            cooldown_s=0.0,
        )


# ---------------------------------------------------------------------------
# Fuzz: validate_audio_chunk
# ---------------------------------------------------------------------------

class TestFuzzValidateAudioChunk:
    """Fuzz the validate_audio_chunk() utility function."""

    @pytest.mark.parametrize("seed", range(50))
    def test_random_bytes(self, seed: int) -> None:
        """Random bytes should either validate or raise ValueError/TypeError cleanly."""
        rng = np.random.default_rng(seed)
        length = rng.integers(0, 10_000)
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()

        try:
            result = validate_audio_chunk(data)
            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32
        except (ValueError, TypeError):
            pass  # Expected for invalid inputs

    def test_empty_bytes(self) -> None:
        """Empty bytes should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            validate_audio_chunk(b"")

    def test_odd_length_bytes(self) -> None:
        """Odd-length bytes should raise ValueError (not valid int16)."""
        with pytest.raises(ValueError, match="even"):
            validate_audio_chunk(b"\x00\x01\x02")

    def test_single_sample_bytes(self) -> None:
        """Two bytes (one int16 sample) should be valid."""
        result = validate_audio_chunk(b"\x00\x10")
        assert len(result) == 1

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float16, np.uint8, np.uint16])
    def test_invalid_dtypes(self, dtype: np.dtype) -> None:
        """Non-allowed dtypes should raise ValueError."""
        data = np.zeros(100, dtype=dtype)
        with pytest.raises(ValueError, match="dtype"):
            validate_audio_chunk(data)

    @pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
    def test_valid_dtypes(self, dtype: np.dtype) -> None:
        """Allowed dtypes should pass validation."""
        data = np.ones(100, dtype=dtype)
        result = validate_audio_chunk(data)
        assert result.dtype == np.float32

    def test_empty_ndarray(self) -> None:
        """Empty ndarray should raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            validate_audio_chunk(np.array([], dtype=np.float32))

    def test_2d_ndarray(self) -> None:
        """2-D ndarray should raise ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            validate_audio_chunk(np.zeros((10, 2), dtype=np.float32))

    def test_3d_ndarray(self) -> None:
        """3-D ndarray should raise ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            validate_audio_chunk(np.zeros((2, 5, 3), dtype=np.float32))

    def test_non_bytes_non_ndarray(self) -> None:
        """Non-bytes, non-ndarray inputs should raise TypeError."""
        with pytest.raises(TypeError, match="bytes or numpy"):
            validate_audio_chunk([1.0, 2.0, 3.0])  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="bytes or numpy"):
            validate_audio_chunk("hello")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="bytes or numpy"):
            validate_audio_chunk(42)  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="bytes or numpy"):
            validate_audio_chunk(None)  # type: ignore[arg-type]


class TestFuzzNaNInf:
    """Fuzz with NaN, inf, -inf values."""

    def test_all_nan(self) -> None:
        """All-NaN array should not crash validate_audio_chunk."""
        data = np.full(100, np.nan, dtype=np.float32)
        result = validate_audio_chunk(data)
        # NaNs should be replaced with 0
        assert np.all(np.isfinite(result))

    def test_all_inf(self) -> None:
        """All-inf array should not crash."""
        data = np.full(100, np.inf, dtype=np.float32)
        result = validate_audio_chunk(data)
        assert np.all(np.isfinite(result))

    def test_all_neg_inf(self) -> None:
        """All -inf array should not crash."""
        data = np.full(100, -np.inf, dtype=np.float32)
        result = validate_audio_chunk(data)
        assert np.all(np.isfinite(result))

    def test_mixed_nan_inf(self) -> None:
        """Mixed NaN/inf/normal values should be cleaned."""
        data = np.array([1.0, np.nan, -np.inf, 0.5, np.inf, -1.0], dtype=np.float32)
        result = validate_audio_chunk(data)
        assert np.all(np.isfinite(result))
        # Non-NaN/inf values should be preserved
        assert result[0] == 1.0
        assert result[3] == 0.5
        assert result[5] == -1.0

    def test_nan_in_float64(self) -> None:
        """NaN in float64 array should be handled."""
        data = np.array([1.0, np.nan, 2.0], dtype=np.float64)
        result = validate_audio_chunk(data)
        assert np.all(np.isfinite(result))

    def test_sparse_nan(self) -> None:
        """Array with sparse NaNs should clean only the NaN positions."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000).astype(np.float32)
        # Inject NaN at random positions
        nan_idx = rng.choice(1000, 50, replace=False)
        data[nan_idx] = np.nan
        result = validate_audio_chunk(data)
        assert np.all(np.isfinite(result))
        # Check non-NaN positions are preserved
        non_nan_mask = np.ones(1000, dtype=bool)
        non_nan_mask[nan_idx] = False
        np.testing.assert_array_equal(result[non_nan_mask], data[non_nan_mask])


class TestFuzzRandomLengths:
    """Fuzz with random array lengths."""

    @pytest.mark.parametrize("length", [1, 2, 3, 10, 100, 320, 640, 16000, 32000])
    def test_various_lengths_float32(self, length: int) -> None:
        """Various float32 array lengths should pass validation."""
        data = np.ones(length, dtype=np.float32)
        result = validate_audio_chunk(data)
        assert len(result) == length

    @pytest.mark.parametrize("length", [1, 2, 3, 10, 100, 320, 640, 16000, 32000])
    def test_various_lengths_int16(self, length: int) -> None:
        """Various int16 array lengths should pass validation."""
        data = np.ones(length, dtype=np.int16)
        result = validate_audio_chunk(data)
        assert len(result) == length

    def test_max_chunk_size_boundary(self) -> None:
        """Chunk at exactly the max size should pass."""
        from violawake_sdk.wake_detector import _MAX_CHUNK_SAMPLES
        data = np.ones(_MAX_CHUNK_SAMPLES, dtype=np.float32)
        result = validate_audio_chunk(data)
        assert len(result) == _MAX_CHUNK_SAMPLES

    def test_over_max_chunk_size(self) -> None:
        """Chunk exceeding max size should raise ValueError."""
        from violawake_sdk.wake_detector import _MAX_CHUNK_SAMPLES
        data = np.ones(_MAX_CHUNK_SAMPLES + 1, dtype=np.float32)
        with pytest.raises(ValueError, match="too large"):
            validate_audio_chunk(data)


# ---------------------------------------------------------------------------
# Fuzz: WakeDetector.detect() with adversarial inputs
# ---------------------------------------------------------------------------

class TestFuzzDetector:
    """Fuzz the full WakeDetector.detect() method with adversarial inputs."""

    @pytest.mark.parametrize("seed", range(30))
    def test_random_bytes_detect(self, seed: int) -> None:
        """Random even-length bytes should not crash detect()."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(seed)
        # Max 3200 samples (10x FRAME_SAMPLES) — process() rejects larger
        length = rng.integers(1, FRAME_SAMPLES * 10) * 2  # Ensure even length
        data = rng.integers(0, 256, length, dtype=np.uint8).tobytes()

        # Should not raise (random bytes are valid int16 PCM within frame limit)
        result = detector.detect(data)
        assert isinstance(result, bool)

    def test_nan_injection_detect(self) -> None:
        """NaN-filled float32 array should not crash detect()."""
        detector = _create_detector(score=0.50)
        data = np.full(FRAME_SAMPLES, np.nan, dtype=np.float32)
        # Should handle gracefully -- NaNs are cleaned by validate_audio_chunk
        result = detector.detect(data)
        assert isinstance(result, bool)

    def test_inf_injection_detect(self) -> None:
        """Inf-filled float32 array should not crash detect()."""
        detector = _create_detector(score=0.50)
        data = np.full(FRAME_SAMPLES, np.inf, dtype=np.float32)
        result = detector.detect(data)
        assert isinstance(result, bool)

    def test_neg_inf_injection_detect(self) -> None:
        """-Inf-filled float32 array should not crash detect()."""
        detector = _create_detector(score=0.50)
        data = np.full(FRAME_SAMPLES, -np.inf, dtype=np.float32)
        result = detector.detect(data)
        assert isinstance(result, bool)

    def test_max_int16_values(self) -> None:
        """Extreme int16 values (32767, -32768) should not crash."""
        detector = _create_detector(score=0.50)
        data = np.full(FRAME_SAMPLES, 32767, dtype=np.int16).tobytes()
        result = detector.detect(data)
        assert isinstance(result, bool)

        data = np.full(FRAME_SAMPLES, -32768, dtype=np.int16).tobytes()
        result = detector.detect(data)
        assert isinstance(result, bool)

    def test_alternating_extreme_values(self) -> None:
        """Rapidly alternating max/min values should not crash."""
        detector = _create_detector(score=0.50)
        data = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        data[::2] = 32767
        data[1::2] = -32768
        result = detector.detect(data.tobytes())
        assert isinstance(result, bool)

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.float16])
    def test_invalid_dtype_detect(self, dtype: np.dtype) -> None:
        """Invalid dtypes should raise ValueError, not crash."""
        detector = _create_detector(score=0.50)
        data = np.zeros(FRAME_SAMPLES, dtype=dtype)
        with pytest.raises(ValueError, match="dtype"):
            detector.detect(data)

    def test_string_input_raises_type_error(self) -> None:
        """String input should raise TypeError."""
        detector = _create_detector(score=0.50)
        with pytest.raises(TypeError, match="bytes or numpy"):
            detector.detect("not audio")  # type: ignore[arg-type]

    def test_list_input_raises_type_error(self) -> None:
        """List input should raise TypeError."""
        detector = _create_detector(score=0.50)
        with pytest.raises(TypeError, match="bytes or numpy"):
            detector.detect([1, 2, 3])  # type: ignore[arg-type]

    def test_none_input_raises_type_error(self) -> None:
        """None input should raise TypeError."""
        detector = _create_detector(score=0.50)
        with pytest.raises(TypeError, match="bytes or numpy"):
            detector.detect(None)  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", range(10))
    def test_random_float32_detect(self, seed: int) -> None:
        """Random float32 arrays should not crash detect()."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(seed)
        length = rng.integers(1, FRAME_SAMPLES * 10)
        data = rng.standard_normal(length).astype(np.float32)
        result = detector.detect(data)
        assert isinstance(result, bool)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_float32_with_nan_inf(self, seed: int) -> None:
        """Random float32 with injected NaN/inf should not crash."""
        detector = _create_detector(score=0.50)
        rng = np.random.default_rng(seed)
        length = rng.integers(10, FRAME_SAMPLES * 10)
        data = rng.standard_normal(length).astype(np.float32)
        # Inject NaN/inf at random positions
        n_bad = rng.integers(1, min(length, 50))
        bad_idx = rng.choice(length, n_bad, replace=False)
        bad_values = rng.choice([np.nan, np.inf, -np.inf], n_bad)
        data[bad_idx] = bad_values

        result = detector.detect(data)
        assert isinstance(result, bool)


class TestFuzzConstructor:
    """Fuzz WakeDetector constructor parameters."""

    @pytest.mark.parametrize("threshold", [-0.1, -1.0, 1.1, 2.0, 100.0, float("nan")])
    def test_invalid_threshold(self, threshold: float) -> None:
        """Invalid thresholds should raise ValueError."""
        with pytest.raises((ValueError, TypeError)):
            _create_detector_with_threshold(threshold)

    @pytest.mark.parametrize("threshold", [0.0, 0.5, 0.8, 1.0])
    def test_valid_threshold(self, threshold: float) -> None:
        """Valid thresholds should not raise."""
        oww_sess = _mock_backend_session()
        oww_sess.run.return_value = [np.ones((1, 96), dtype=np.float32) * 0.1]
        mlp_sess = _mock_backend_session()

        mock_backend = MagicMock()
        mock_backend.name = "onnx"
        mock_backend.load.side_effect = [oww_sess, mlp_sess]

        fake_path = Path("/fake/model.onnx")

        with (
            patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
            patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
                push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.1)),
            )),
        ):
            detector = WakeDetector(
                model="viola_mlp_oww", threshold=threshold,
                cooldown_s=0.0,
            )
            assert detector.threshold == threshold

    @pytest.mark.parametrize("cooldown", [-0.1, -1.0, -100.0])
    def test_invalid_cooldown(self, cooldown: float) -> None:
        """Negative cooldown should raise ValueError."""
        oww_sess = _mock_backend_session()
        oww_sess.run.return_value = [np.ones((1, 96), dtype=np.float32) * 0.1]
        mlp_sess = _mock_backend_session()

        mock_backend = MagicMock()
        mock_backend.name = "onnx"
        mock_backend.load.side_effect = [oww_sess, mlp_sess]

        fake_path = Path("/fake/model.onnx")

        with (
            patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
            patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
                push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.1)),
            )),
        ):
            with pytest.raises(ValueError, match="cooldown"):
                WakeDetector(
                    model="viola_mlp_oww", threshold=0.80,
                    cooldown_s=cooldown,
                )


def _mock_backend_session(score: float = 0.50):
    """Create a mock BackendSession."""
    sess = MagicMock()
    inp = MagicMock()
    inp.name = "input"
    sess.get_inputs.return_value = [inp]
    sess.run.return_value = [np.array([[score]], dtype=np.float32)]
    return sess


def _create_detector_with_threshold(threshold: float) -> WakeDetector:
    """Helper that may raise on invalid threshold."""
    oww_sess = _mock_backend_session()
    oww_sess.run.return_value = [np.ones((1, 96), dtype=np.float32) * 0.1]
    mlp_sess = _mock_backend_session()

    mock_backend = MagicMock()
    mock_backend.name = "onnx"
    mock_backend.load.side_effect = [oww_sess, mlp_sess]

    fake_path = Path("/fake/model.onnx")

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock(
            push_audio=MagicMock(return_value=(True, np.ones((1, 96), dtype=np.float32) * 0.1)),
        )),
    ):
        return WakeDetector(
            model="viola_mlp_oww", threshold=threshold,
            cooldown_s=0.0,
        )
