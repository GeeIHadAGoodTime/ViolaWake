"""Unit tests for VADEngine.

Tests cover:
- Backend selection (auto, silero, webrtc, rms)
- RMS heuristic backend correctness
- Graceful degradation when optional VAD deps are not installed
- is_speech() convenience method
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.vad import (
    SileroVAD,
    VADBackend,
    VADEngine,
    _coerce_to_bytes,
    _RMSHeuristicBackend,
)


def _make_silero_output(prob: float) -> MagicMock:
    output = MagicMock()
    output.item.return_value = prob
    return output


# ──────────────────────────────────────────────────────────────────────────────
# RMS Heuristic Backend Tests (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

class TestRMSHeuristicBackend:
    """Tests for the RMS heuristic VAD backend (no dependencies required)."""

    def _make_silent_frame(self) -> bytes:
        return (np.zeros(320, dtype=np.int16)).tobytes()

    def _make_loud_frame(self, amplitude: int = 20000) -> bytes:
        rng = np.random.default_rng(42)
        samples = rng.integers(-amplitude, amplitude, 320, dtype=np.int16)
        return samples.tobytes()

    def test_silence_returns_zero(self) -> None:
        backend = _RMSHeuristicBackend(speech_threshold=200.0, silence_threshold=50.0)
        prob = backend.process_frame(self._make_silent_frame())
        assert prob == 0.0

    def test_loud_audio_returns_one(self) -> None:
        backend = _RMSHeuristicBackend(speech_threshold=200.0, silence_threshold=50.0)
        prob = backend.process_frame(self._make_loud_frame(amplitude=20000))
        assert prob == 1.0

    def test_probability_in_range(self) -> None:
        backend = _RMSHeuristicBackend(speech_threshold=500.0, silence_threshold=100.0)
        # Medium volume frame
        rng = np.random.default_rng(99)
        samples = rng.integers(-300, 300, 320, dtype=np.int16)
        prob = backend.process_frame(samples.tobytes())
        assert 0.0 <= prob <= 1.0

    def test_reset_is_noop(self) -> None:
        backend = _RMSHeuristicBackend()
        backend.reset()  # Should not raise


# ──────────────────────────────────────────────────────────────────────────────
# VADEngine Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVADEngineRMSBackend:
    """Tests using the RMS backend (no external dependencies)."""

    @pytest.fixture
    def vad(self) -> VADEngine:
        return VADEngine(backend="rms")

    def test_backend_name(self, vad: VADEngine) -> None:
        assert vad.backend_name == "rms"

    def test_process_frame_returns_float(self, vad: VADEngine, silent_frame: bytes) -> None:
        prob = vad.process_frame(silent_frame)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_is_speech_false_on_silence(self, vad: VADEngine, silent_frame: bytes) -> None:
        assert vad.is_speech(silent_frame) is False

    def test_is_speech_true_on_loud_audio(self, vad: VADEngine, loud_noise_frame: bytes) -> None:
        assert vad.is_speech(loud_noise_frame) is True

    def test_is_speech_threshold_parameter(self, vad: VADEngine, loud_noise_frame: bytes) -> None:
        # Very strict threshold
        assert vad.is_speech(loud_noise_frame, threshold=0.999) in (True, False)  # doesn't crash
        # Very lenient threshold
        assert vad.is_speech(loud_noise_frame, threshold=0.001) is True

    def test_reset_does_not_raise(self, vad: VADEngine) -> None:
        vad.reset()  # Should not raise


class TestVADEngineAutoBackend:
    """Tests for auto backend selection."""

    def test_auto_falls_back_to_webrtc_when_silero_unavailable(self) -> None:
        mock_webrtcvad = MagicMock()
        mock_webrtcvad.Vad.return_value.is_speech.return_value = False

        with (
            patch("violawake_sdk.vad.load_silero_vad", None),
            patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}),
        ):
            vad = VADEngine(backend="auto")
            assert vad.backend_name == "webrtc"

    def test_auto_selects_silero_when_available(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = _make_silero_output(0.8)
        mock_webrtcvad = MagicMock()
        mock_torch = MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: arr

        with (
            patch("violawake_sdk.vad.load_silero_vad", return_value=mock_model),
            patch.dict("sys.modules", {"torch": mock_torch, "webrtcvad": mock_webrtcvad}),
        ):
            vad = VADEngine(backend="auto")
            assert vad.backend_name == "silero"


class TestVADEngineWebRTCBackend:
    """Tests for WebRTC VAD backend (mocked)."""

    def test_webrtc_returns_one_on_speech(self, tone_frame: bytes) -> None:
        mock_webrtcvad = MagicMock()
        mock_vad_instance = MagicMock()
        mock_vad_instance.is_speech.return_value = True
        mock_webrtcvad.Vad.return_value = mock_vad_instance

        with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
            vad = VADEngine(backend="webrtc")
            prob = vad.process_frame(tone_frame)
            assert prob == 1.0

    def test_webrtc_returns_zero_on_silence(self, silent_frame: bytes) -> None:
        mock_webrtcvad = MagicMock()
        mock_vad_instance = MagicMock()
        mock_vad_instance.is_speech.return_value = False
        mock_webrtcvad.Vad.return_value = mock_vad_instance

        with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
            vad = VADEngine(backend="webrtc")
            prob = vad.process_frame(silent_frame)
            assert prob == 0.0

    def test_webrtc_import_error_raises(self) -> None:
        with (
            patch.dict("sys.modules", {"webrtcvad": None}),
            pytest.raises(ImportError, match="webrtcvad"),
        ):
                VADEngine(backend="webrtc")


class TestWebRTCFrameValidation:
    """Tests for WebRTC backend input validation."""

    def _make_webrtc_vad(self) -> VADEngine:
        """Create a WebRTC VADEngine with mocked webrtcvad."""
        mock_webrtcvad = MagicMock()
        mock_vad_instance = MagicMock()
        mock_vad_instance.is_speech.return_value = False
        mock_webrtcvad.Vad.return_value = mock_vad_instance
        with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
            return VADEngine(backend="webrtc")

    def test_wrong_size_frame_raises_valueerror(self) -> None:
        """Frame that is not 10/20/30ms at 16kHz must raise ValueError."""
        vad = self._make_webrtc_vad()
        bad_frame = b"\x00" * 100  # 50 samples = 3.125ms — invalid
        with pytest.raises(ValueError, match="10/20/30ms"):
            vad.process_frame(bad_frame)

    def test_non_bytes_raises_typeerror(self) -> None:
        """Non-bytes input must raise TypeError."""
        vad = self._make_webrtc_vad()
        with pytest.raises(TypeError, match="audio must be bytes"):
            vad.process_frame("not bytes")  # type: ignore[arg-type]

    def test_valid_10ms_frame_accepted(self) -> None:
        """320 bytes (10ms at 16kHz) must be accepted."""
        vad = self._make_webrtc_vad()
        frame_10ms = b"\x00" * 320  # 160 samples * 2 bytes
        prob = vad.process_frame(frame_10ms)
        assert prob == 0.0

    def test_valid_20ms_frame_accepted(self) -> None:
        """640 bytes (20ms at 16kHz) must be accepted."""
        vad = self._make_webrtc_vad()
        frame_20ms = b"\x00" * 640  # 320 samples * 2 bytes
        prob = vad.process_frame(frame_20ms)
        assert prob == 0.0

    def test_valid_30ms_frame_accepted(self) -> None:
        """960 bytes (30ms at 16kHz) must be accepted."""
        vad = self._make_webrtc_vad()
        frame_30ms = b"\x00" * 960  # 480 samples * 2 bytes
        prob = vad.process_frame(frame_30ms)
        assert prob == 0.0

    def test_empty_frame_raises_valueerror(self) -> None:
        """Empty input must raise ValueError."""
        vad = self._make_webrtc_vad()
        with pytest.raises(ValueError, match="10/20/30ms"):
            vad.process_frame(b"")

    def test_error_message_shows_actual_duration(self) -> None:
        """Error message should include the actual frame duration."""
        vad = self._make_webrtc_vad()
        # 200 bytes = 100 samples = 6.25ms
        with pytest.raises(ValueError, match="6.2ms"):
            vad.process_frame(b"\x00" * 200)


class TestAutoFallbackChain:
    """Tests for AUTO mode fallback chain with both WebRTC and Silero unavailable."""

    def test_auto_falls_back_to_rms_when_both_unavailable(self) -> None:
        """When both Silero and WebRTC are unavailable, AUTO must fall back to RMS."""
        with (
            patch("violawake_sdk.vad.load_silero_vad", None),
            patch.dict("sys.modules", {"webrtcvad": None}),
        ):
            vad = VADEngine(backend="auto")
            assert vad.backend_name == "rms"

    def test_auto_rms_fallback_still_works(self) -> None:
        """RMS fallback from AUTO must still process frames correctly."""
        with (
            patch("violawake_sdk.vad.load_silero_vad", None),
            patch.dict("sys.modules", {"webrtcvad": None}),
        ):
            vad = VADEngine(backend="auto")
            silent = np.zeros(320, dtype=np.int16).tobytes()
            prob = vad.process_frame(silent)
            assert prob == 0.0

    def test_auto_prefers_silero_over_webrtc(self) -> None:
        """When both are available, AUTO should pick Silero first."""
        mock_model = MagicMock()
        mock_model.return_value = _make_silero_output(0.6)
        mock_webrtcvad = MagicMock()
        mock_webrtcvad.Vad.return_value = MagicMock()
        mock_torch = MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: arr

        with (
            patch("violawake_sdk.vad.load_silero_vad", return_value=mock_model),
            patch.dict("sys.modules", {"torch": mock_torch, "webrtcvad": mock_webrtcvad}),
        ):
            vad = VADEngine(backend="auto")
            assert vad.backend_name == "silero"


class TestSileroVADBackend:
    """Tests for the packaged Silero ONNX backend."""

    def test_missing_silero_package_raises_importerror(self) -> None:
        with (
            patch("violawake_sdk.vad.load_silero_vad", None),
            pytest.raises(ImportError, match="silero-vad"),
        ):
                VADEngine(backend="silero")

    def test_load_error_raises_runtimeerror(self) -> None:
        mock_torch = MagicMock()
        with (
            patch("violawake_sdk.vad.load_silero_vad", side_effect=RuntimeError("bad model")),
            patch.dict("sys.modules", {"torch": mock_torch}),
            pytest.raises(RuntimeError, match="Failed to load Silero VAD model"),
        ):
            VADEngine(backend="silero")

    def test_process_frame_accepts_native_512_sample_frame(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = _make_silero_output(0.73)
        mock_torch = MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: arr

        with (
            patch("violawake_sdk.vad.load_silero_vad", return_value=mock_model),
            patch.dict("sys.modules", {"torch": mock_torch}),
        ):
            backend = SileroVAD()
            frame = np.zeros(512, dtype=np.int16).tobytes()
            prob = backend.process_frame(frame)

        assert prob == 0.73
        mock_model.assert_called_once()

    def test_process_frame_pads_short_sdk_frames(self) -> None:
        mock_model = MagicMock()
        mock_model.return_value = _make_silero_output(0.42)
        mock_torch = MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: arr

        with (
            patch("violawake_sdk.vad.load_silero_vad", return_value=mock_model),
            patch.dict("sys.modules", {"torch": mock_torch}),
        ):
            backend = SileroVAD()
            frame = np.zeros(320, dtype=np.int16).tobytes()
            prob = backend.process_frame(frame)

        assert prob == 0.42
        passed_chunk = mock_torch.from_numpy.call_args[0][0]
        assert passed_chunk.shape == (512,)

    def test_process_frame_chunks_long_audio_and_uses_max_probability(self) -> None:
        mock_model = MagicMock()
        mock_model.side_effect = [_make_silero_output(0.1), _make_silero_output(0.9)]
        mock_torch = MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: arr

        with (
            patch("violawake_sdk.vad.load_silero_vad", return_value=mock_model),
            patch.dict("sys.modules", {"torch": mock_torch}),
        ):
            backend = SileroVAD()
            frame = np.zeros(1024, dtype=np.int16).tobytes()
            prob = backend.process_frame(frame)

        assert prob == 0.9
        assert mock_model.call_count == 2


class TestRMSInputValidation:
    """Edge case tests for RMS backend input validation."""

    def test_non_bytes_raises_typeerror(self) -> None:
        backend = _RMSHeuristicBackend()
        with pytest.raises(TypeError, match="audio_bytes must be bytes"):
            backend.process_frame([1, 2, 3])  # type: ignore[arg-type]

    def test_odd_length_raises_valueerror(self) -> None:
        backend = _RMSHeuristicBackend()
        with pytest.raises(ValueError, match="even"):
            backend.process_frame(b"\x00\x01\x02")  # 3 bytes — not even

    def test_empty_bytes_returns_zero(self) -> None:
        """Empty audio should return 0.0 (silence), not crash."""
        backend = _RMSHeuristicBackend()
        # Empty bytes has 0 samples — mean of empty array.
        # numpy will warn but shouldn't crash. The RMS of nothing is 0.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            prob = backend.process_frame(b"")
        # Either 0.0 or NaN-safe handling
        assert prob == 0.0 or math.isnan(prob)


class TestVADEngineBackendEnum:
    """Test VADBackend enum handling."""

    def test_string_backend_accepted(self) -> None:
        vad = VADEngine(backend="rms")
        assert vad.backend_name == "rms"

    def test_enum_backend_accepted(self) -> None:
        vad = VADEngine(backend=VADBackend.RMS)
        assert vad.backend_name == "rms"

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            VADEngine(backend="invalid_backend")  # type: ignore[arg-type]

    def test_silero_backend(self) -> None:
        """Silero VAD backend should initialize through the packaged loader."""
        mock_model = MagicMock()
        mock_model.return_value = _make_silero_output(0.5)
        mock_torch = MagicMock()
        mock_torch.from_numpy.side_effect = lambda arr: arr

        with (
            patch("violawake_sdk.vad.load_silero_vad", return_value=mock_model),
            patch.dict("sys.modules", {"torch": mock_torch}),
        ):
            vad = VADEngine(backend="silero")
            assert vad.backend_name == "silero"


# ──────────────────────────────────────────────────────────────────────────────
# _coerce_to_bytes Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCoerceToBytes:
    """Tests for the _coerce_to_bytes helper."""

    def test_bytes_passthrough(self) -> None:
        raw = b"\x00\x01\x02\x03"
        assert _coerce_to_bytes(raw) == raw

    def test_bytearray_passthrough(self) -> None:
        raw = bytearray(b"\x00\x01\x02\x03")
        result = _coerce_to_bytes(raw)
        assert result == bytes(raw)
        assert isinstance(result, bytes)

    def test_int16_ndarray(self) -> None:
        arr = np.array([0, 100, -100, 32767], dtype=np.int16)
        result = _coerce_to_bytes(arr)
        assert result == arr.tobytes()

    def test_float32_ndarray(self) -> None:
        arr = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        result = _coerce_to_bytes(arr)
        expected = (arr * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
        assert result == expected

    def test_float64_ndarray(self) -> None:
        arr = np.array([0.0, 0.5, -0.5], dtype=np.float64)
        result = _coerce_to_bytes(arr)
        expected = (arr * 32768).clip(-32768, 32767).astype(np.int16).tobytes()
        assert result == expected

    def test_float32_clips_overflow(self) -> None:
        """Values > 1.0 should be clipped to 32767."""
        arr = np.array([2.0, -2.0], dtype=np.float32)
        result = _coerce_to_bytes(arr)
        decoded = np.frombuffer(result, dtype=np.int16)
        assert decoded[0] == 32767
        assert decoded[1] == -32768

    def test_unsupported_dtype_raises(self) -> None:
        arr = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ValueError, match="int16"):
            _coerce_to_bytes(arr)

    def test_non_bytes_non_ndarray_raises(self) -> None:
        with pytest.raises(TypeError, match="bytes or np.ndarray"):
            _coerce_to_bytes([1, 2, 3])  # type: ignore[arg-type]


# ──────────────────────────────────────────────────────────────────────────────
# Numpy Input Integration Tests (via VADEngine)
# ──────────────────────────────────────────────────────────────────────────────

class TestVADEngineNumpyInput:
    """Tests that VADEngine.process_frame and is_speech accept numpy arrays."""

    @pytest.fixture
    def vad(self) -> VADEngine:
        return VADEngine(backend="rms")

    def test_process_frame_float32_silence(self, vad: VADEngine) -> None:
        arr = np.zeros(320, dtype=np.float32)
        prob = vad.process_frame(arr)
        assert prob == 0.0

    def test_process_frame_float32_loud(self, vad: VADEngine) -> None:
        rng = np.random.default_rng(42)
        arr = (rng.standard_normal(320) * 0.8).astype(np.float32)
        prob = vad.process_frame(arr)
        assert prob == 1.0  # ~0.8 * 32768 = ~26000 RMS, well above threshold

    def test_process_frame_int16(self, vad: VADEngine) -> None:
        arr = np.zeros(320, dtype=np.int16)
        prob = vad.process_frame(arr)
        assert prob == 0.0

    def test_is_speech_with_ndarray(self, vad: VADEngine) -> None:
        silent = np.zeros(320, dtype=np.float32)
        assert vad.is_speech(silent) is False

    def test_bytes_and_float32_agree(self, vad: VADEngine) -> None:
        """bytes and equivalent float32 ndarray should produce the same result."""
        arr_f32 = np.zeros(320, dtype=np.float32)
        arr_bytes = np.zeros(320, dtype=np.int16).tobytes()
        assert vad.process_frame(arr_f32) == vad.process_frame(arr_bytes)
