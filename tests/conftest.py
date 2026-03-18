"""Shared pytest fixtures for ViolaWake SDK tests.

Design principles:
- No hardware required for unit tests (mock everything)
- No model files required for unit tests (mock ONNX Runtime)
- Synthetic audio only (numpy-generated, deterministic)
- No network access during tests
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Audio Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320  # 20ms at 16kHz


@pytest.fixture
def silent_frame() -> bytes:
    """20ms of silence at 16kHz mono 16-bit PCM."""
    return (np.zeros(FRAME_SAMPLES, dtype=np.int16)).tobytes()


@pytest.fixture
def noise_frame() -> bytes:
    """20ms of white noise at 16kHz mono 16-bit PCM (deterministic seed)."""
    rng = np.random.default_rng(42)
    samples = rng.integers(-1000, 1000, FRAME_SAMPLES, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def loud_noise_frame() -> bytes:
    """20ms of loud white noise (RMS > 1000) — should pass zero-input guard."""
    rng = np.random.default_rng(123)
    samples = rng.integers(-20000, 20000, FRAME_SAMPLES, dtype=np.int16)
    return samples.tobytes()


@pytest.fixture
def tone_frame() -> bytes:
    """20ms of 440Hz sine wave at 16kHz mono 16-bit PCM."""
    t = np.linspace(0, 0.02, FRAME_SAMPLES, endpoint=False)
    signal = (np.sin(2 * math.pi * 440 * t) * 10000).astype(np.int16)
    return signal.tobytes()


@pytest.fixture
def audio_3s() -> np.ndarray:
    """3 seconds of white noise as float32 array at 16kHz."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(3 * SAMPLE_RATE).astype(np.float32) * 0.1


@pytest.fixture
def audio_silence_3s() -> np.ndarray:
    """3 seconds of silence as float32 array at 16kHz."""
    return np.zeros(3 * SAMPLE_RATE, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Mock Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ort_session_low():
    """Mock ONNX Runtime session that always returns a low score (0.05)."""
    mock = MagicMock()
    mock.get_inputs.return_value = [MagicMock(name="input")]
    mock.run.return_value = [np.array([[0.05]], dtype=np.float32)]
    return mock


@pytest.fixture
def mock_ort_session_high():
    """Mock ONNX Runtime session that always returns a high score (0.95)."""
    mock = MagicMock()
    mock.get_inputs.return_value = [MagicMock(name="input")]
    mock.run.return_value = [np.array([[0.95]], dtype=np.float32)]
    return mock


@pytest.fixture
def mock_ort_session_threshold():
    """Mock ONNX Runtime session that returns exactly the default threshold (0.80)."""
    mock = MagicMock()
    mock.get_inputs.return_value = [MagicMock(name="input")]
    mock.run.return_value = [np.array([[0.80]], dtype=np.float32)]
    return mock


@pytest.fixture
def mock_pyaudio():
    """Mock pyaudio.PyAudio for tests that don't need real mic input."""
    with patch("pyaudio.PyAudio") as mock_pa:
        mock_stream = MagicMock()
        mock_stream.read.return_value = b"\x00" * 640  # 320 int16 samples
        mock_pa.return_value.open.return_value = mock_stream
        yield mock_pa


# ──────────────────────────────────────────────────────────────────────────────
# Model Cache Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Temporary directory as model cache root. Overrides VIOLAWAKE_MODEL_DIR."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def fake_onnx_model(tmp_path: Path) -> Path:
    """Create a fake (empty) .onnx file for testing path resolution."""
    model_file = tmp_path / "fake_model.onnx"
    model_file.write_bytes(b"fake_onnx_content")
    return model_file
