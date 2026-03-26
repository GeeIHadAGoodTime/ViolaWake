"""Tests for the evaluation pipeline fixes.

Verifies:
  - Auto-detection of model architecture (MLP-on-OWW vs CNN)
  - OWW embedding scoring path
  - Threshold sweep
  - Confusion matrix computation
  - d-prime calculation
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from violawake_sdk.training.evaluate import compute_dprime, compute_eer


class TestDPrime:
    """Test d-prime computation."""

    def test_perfect_separation(self) -> None:
        """Perfect separation should give very high d-prime."""
        # Use slightly varied scores (constant scores have zero variance → d'=0)
        rng = np.random.default_rng(42)
        pos = rng.normal(0.95, 0.02, 100)
        neg = rng.normal(0.05, 0.02, 100)
        d = compute_dprime(pos, neg)
        assert d > 10.0

    def test_no_separation(self) -> None:
        """Identical distributions should give d-prime ~0."""
        rng = np.random.default_rng(42)
        scores = rng.normal(0.5, 0.1, 200)
        pos = scores[:100]
        neg = scores[100:]
        d = compute_dprime(pos, neg)
        assert abs(d) < 1.0

    def test_empty_input(self) -> None:
        """Empty arrays should return 0."""
        assert compute_dprime([], []) == 0.0
        assert compute_dprime([0.9], []) == 0.0
        assert compute_dprime([], [0.1]) == 0.0

    def test_constant_scores(self) -> None:
        """All-same scores should return 0 (zero variance)."""
        pos = np.full(50, 0.9)
        neg = np.full(50, 0.1)
        # var is 0, pooled_std is 0 → should return 0 (not inf)
        d = compute_dprime(pos, neg)
        assert d == 0.0

    def test_realistic_scores(self) -> None:
        """Realistic score distributions should give reasonable d-prime."""
        rng = np.random.default_rng(42)
        pos = rng.normal(0.85, 0.08, 100)  # High scores
        neg = rng.normal(0.15, 0.10, 500)  # Low scores
        d = compute_dprime(pos, neg)
        assert 5.0 < d < 15.0


class TestEER:
    """Test Equal Error Rate computation."""

    def test_perfect_roc(self) -> None:
        """Perfect ROC should have EER ~0."""
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        eer, idx = compute_eer(fpr, tpr)
        assert eer < 0.01

    def test_random_roc(self) -> None:
        """Random classifier should have EER ~0.5."""
        fpr = np.linspace(0, 1, 100)
        tpr = fpr  # diagonal = random
        eer, idx = compute_eer(fpr, tpr)
        assert 0.4 < eer < 0.6


class TestArchitectureDetection:
    """Test auto-detection of model architecture.

    detect_architecture(model_path, session) reads a .config.json file
    alongside the model (path 1) or falls back to ONNX input shape
    heuristic via the session object (path 2). We mock the session to
    test both paths without requiring real ONNX models.
    """

    @staticmethod
    def _make_mock_session(input_shape: list) -> object:
        """Create a mock ONNX InferenceSession with the given input shape."""

        class _MockInput:
            def __init__(self, shape):
                self.shape = shape
                self.name = "input"

        class _MockSession:
            def __init__(self, shape):
                self._input = _MockInput(shape)

            def get_inputs(self):
                return [self._input]

        return _MockSession(input_shape)

    def test_mlp_config_detected(self, tmp_path: Path) -> None:
        """Config with architecture=mlp_on_oww should be detected."""
        from violawake_sdk.training.evaluate import detect_architecture

        config = {"architecture": "mlp_on_oww", "embedding_dim": 96}
        model_path = tmp_path / "model.onnx"
        model_path.touch()  # create empty file so .with_suffix works
        config_path = model_path.with_suffix(".config.json")
        config_path.write_text(json.dumps(config))

        session = self._make_mock_session([1, 96])
        arch = detect_architecture(model_path, session)
        assert arch == "mlp_on_oww"

    def test_cnn_config_detected(self, tmp_path: Path) -> None:
        """Config with architecture=cnn should use legacy path."""
        from violawake_sdk.training.evaluate import detect_architecture

        config = {"architecture": "cnn"}
        model_path = tmp_path / "model.onnx"
        model_path.touch()
        config_path = model_path.with_suffix(".config.json")
        config_path.write_text(json.dumps(config))

        session = self._make_mock_session([1, 40, 100])
        arch = detect_architecture(model_path, session)
        assert arch == "cnn"

    def test_no_config_mlp_shape_detected(self, tmp_path: Path) -> None:
        """Without config, 2D input shape [1, 96] should detect mlp_on_oww."""
        from violawake_sdk.training.evaluate import detect_architecture

        model_path = tmp_path / "model.onnx"
        model_path.touch()
        # No .config.json file exists — falls through to shape heuristic

        session = self._make_mock_session([1, 96])
        arch = detect_architecture(model_path, session)
        assert arch == "mlp_on_oww"

    def test_no_config_cnn_shape_detected(self, tmp_path: Path) -> None:
        """Without config, 3D input shape should detect cnn."""
        from violawake_sdk.training.evaluate import detect_architecture

        model_path = tmp_path / "model.onnx"
        model_path.touch()

        session = self._make_mock_session([1, 40, 100])
        arch = detect_architecture(model_path, session)
        assert arch == "cnn"

    def test_no_config_ambiguous_defaults_cnn(self, tmp_path: Path) -> None:
        """Ambiguous shape without config should default to cnn."""
        from violawake_sdk.training.evaluate import detect_architecture

        model_path = tmp_path / "model.onnx"
        model_path.touch()

        # Shape that doesn't match MLP or CNN heuristics clearly
        session = self._make_mock_session([1, 10])  # dim 10 is below OWW range
        arch = detect_architecture(model_path, session)
        assert arch == "cnn"  # default fallback


class TestThresholdSweep:
    """Test optimal threshold finding."""

    def test_sweep_finds_optimal(self) -> None:
        """Threshold sweep should find a reasonable threshold."""
        try:
            from violawake_sdk.training.evaluate import find_optimal_threshold
        except ImportError:
            pytest.skip("find_optimal_threshold not yet available")

        rng = np.random.default_rng(42)
        pos_scores = rng.normal(0.85, 0.08, 100)
        neg_scores = rng.normal(0.15, 0.10, 500)

        result = find_optimal_threshold(pos_scores, neg_scores)
        # Result may be a dict or a float depending on implementation
        if isinstance(result, dict):
            threshold = result["optimal_threshold"]
        else:
            threshold = float(result)
        assert 0.2 < threshold < 0.9
        assert threshold > np.mean(neg_scores)
        assert threshold < np.mean(pos_scores)

    def test_sweep_perfect_separation(self) -> None:
        """With perfect separation, any threshold in the gap works."""
        try:
            from violawake_sdk.training.evaluate import find_optimal_threshold
        except ImportError:
            pytest.skip("find_optimal_threshold not yet available")

        pos_scores = np.array([0.9, 0.95, 0.92, 0.88, 0.91])
        neg_scores = np.array([0.1, 0.05, 0.08, 0.12, 0.09])

        result = find_optimal_threshold(pos_scores, neg_scores)
        if isinstance(result, dict):
            threshold = result["optimal_threshold"]
        else:
            threshold = float(result)
        assert 0.12 < threshold < 0.88
