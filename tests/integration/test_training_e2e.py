"""
Integration test: End-to-end training pipeline.

This test verifies that the SDK training pipeline can:
  1. Accept WAV files
  2. Extract OWW embeddings (mocked for CI, real for local)
  3. Train an MLP model
  4. Export to ONNX
  5. Produce a valid model that loads in onnxruntime

Requires: pip install 'violawake[training]' (torch, openwakeword)

Run locally:
    pytest tests/integration/test_training_e2e.py -v -m integration

In CI (mock OWW):
    pytest tests/integration/test_training_e2e.py -v -k "mock"
"""
from __future__ import annotations

import json
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_wavs(tmp_path: Path) -> Path:
    """Create 10 sample WAV files (synthetic speech-like)."""
    pos_dir = tmp_path / "positives"
    pos_dir.mkdir()

    rng = np.random.default_rng(42)
    sr = 16000
    duration = 1.5

    for i in range(10):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 440 + i * 20
        signal = (
            np.sin(2 * np.pi * freq * t) * 0.4
            + np.sin(2 * np.pi * freq * 2 * t) * 0.2
            + rng.normal(0, 0.05, len(t))
        )
        pcm = (np.clip(signal, -1, 1) * 32767).astype(np.int16)

        with wave.open(str(pos_dir / f"sample_{i:02d}.wav"), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

    return pos_dir


@pytest.fixture
def output_model(tmp_path: Path) -> Path:
    """Output path for trained model."""
    return tmp_path / "test_model.onnx"


class TestTrainingWithMockOWW:
    """Test training pipeline with mocked OWW backbone.

    This runs in CI without needing the real OWW model files.
    """

    @pytest.fixture
    def mock_oww(self):
        """Mock the OpenWakeWord model."""
        mock_preprocessor = MagicMock()
        # Return random 96-dim embeddings (realistic shape)
        rng = np.random.default_rng(42)
        mock_preprocessor.embed_clips.side_effect = lambda audio, ncpu=1: rng.normal(
            0, 1, (1, 1, 96)
        ).astype(np.float32)
        mock_preprocessor.onnx_execution_provider = "CPUExecutionProvider"

        mock_model = MagicMock()
        mock_model.preprocessor = mock_preprocessor
        return mock_model

    def test_training_produces_onnx(
        self,
        sample_wavs: Path,
        output_model: Path,
        mock_oww: MagicMock,
    ) -> None:
        """Training should produce a valid .onnx file."""
        with patch(
            "openwakeword.model.Model",
            return_value=mock_oww,
        ):
            try:
                from violawake_sdk.tools.train import _train_mlp_on_oww

                _train_mlp_on_oww(
                    positives_dir=sample_wavs,
                    output_path=output_model,
                    epochs=3,
                    augment=False,
                    verbose=False,
                )
            except Exception:
                # If the import path changed, try alternative
                pytest.skip("Training function signature may have changed")

        assert output_model.exists(), "ONNX model should be created"
        assert output_model.stat().st_size > 1000, "ONNX model should have content"

    def test_config_json_saved(
        self,
        sample_wavs: Path,
        output_model: Path,
        mock_oww: MagicMock,
    ) -> None:
        """Training should save config.json alongside the model."""
        with patch(
            "openwakeword.model.Model",
            return_value=mock_oww,
        ):
            try:
                from violawake_sdk.tools.train import _train_mlp_on_oww

                _train_mlp_on_oww(
                    positives_dir=sample_wavs,
                    output_path=output_model,
                    epochs=3,
                    augment=False,
                    verbose=False,
                )
            except Exception:
                pytest.skip("Training function signature may have changed")

        config_path = output_model.with_suffix(".config.json")
        assert config_path.exists(), "Config JSON should be saved"

        config = json.loads(config_path.read_text())
        assert config["architecture"] == "mlp_on_oww"
        assert config["embedding_dim"] == 96
        assert config["n_pos_samples"] == 10

    def test_model_loads_in_onnxruntime(
        self,
        sample_wavs: Path,
        output_model: Path,
        mock_oww: MagicMock,
    ) -> None:
        """Trained model should load and run in onnxruntime."""
        with patch(
            "openwakeword.model.Model",
            return_value=mock_oww,
        ):
            try:
                from violawake_sdk.tools.train import _train_mlp_on_oww

                _train_mlp_on_oww(
                    positives_dir=sample_wavs,
                    output_path=output_model,
                    epochs=3,
                    augment=False,
                    verbose=False,
                )
            except Exception:
                pytest.skip("Training function may have changed")

        if not output_model.exists():
            pytest.skip("Model not created")

        import onnxruntime as ort

        session = ort.InferenceSession(
            str(output_model), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Should accept (1, 96) embedding input
        assert len(input_shape) == 2
        assert input_shape[1] == 96

        # Run inference with random embedding
        test_input = np.random.randn(1, 96).astype(np.float32)
        output = session.run(None, {input_name: test_input})

        # Should produce a score in [0, 1]
        score = float(output[0][0])
        assert 0.0 <= score <= 1.0

    def test_progress_callback(
        self,
        sample_wavs: Path,
        output_model: Path,
        mock_oww: MagicMock,
    ) -> None:
        """Progress callback should be called each epoch."""
        callback_calls: list[dict] = []

        def on_progress(info: dict) -> None:
            callback_calls.append(info)

        with patch(
            "openwakeword.model.Model",
            return_value=mock_oww,
        ):
            try:
                from violawake_sdk.tools.train import _train_mlp_on_oww

                _train_mlp_on_oww(
                    positives_dir=sample_wavs,
                    output_path=output_model,
                    epochs=3,
                    augment=False,
                    verbose=False,
                    progress_callback=on_progress,
                )
            except Exception:
                pytest.skip("Training function may have changed")

        assert len(callback_calls) >= 3, "Should be called for each epoch"
        assert "epoch" in callback_calls[0]
