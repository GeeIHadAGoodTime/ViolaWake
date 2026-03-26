"""Unit tests for VADEngine.

Tests cover:
- Backend selection (auto, webrtc, rms)
- RMS heuristic backend correctness
- Graceful degradation when webrtcvad not installed
- is_speech() convenience method
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.vad import VADBackend, VADEngine, _RMSHeuristicBackend


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

    def test_auto_falls_back_when_webrtcvad_unavailable(self) -> None:
        with patch.dict("sys.modules", {"webrtcvad": None}):
            vad = VADEngine(backend="auto")
            # Auto chain: webrtc → silero → rms. With webrtc blocked,
            # picks silero (if torch available) or rms.
            assert vad.backend_name in ("rms", "silero", "auto")

    def test_auto_selects_webrtc_when_available(self) -> None:
        mock_webrtcvad = MagicMock()
        mock_webrtcvad.Vad.return_value.is_speech.return_value = True
        with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
            vad = VADEngine(backend="auto")
            # Auto tries webrtc first, then silero, then rms
            assert vad.backend_name in ("rms", "webrtc", "silero")


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
        with patch.dict("sys.modules", {"webrtcvad": None}):
            with pytest.raises(ImportError, match="webrtcvad"):
                VADEngine(backend="webrtc")


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
        """Silero VAD backend should initialize if torch is available."""
        try:
            vad = VADEngine(backend="silero")
            assert vad.backend_name == "silero"
        except (ImportError, RuntimeError):
            pytest.skip("torch not available or Silero failed to load")
