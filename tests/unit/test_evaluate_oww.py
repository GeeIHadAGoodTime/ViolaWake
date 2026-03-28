"""Tests for the evaluation pipeline fixes.

Verifies:
  - Auto-detection of model architecture (MLP-on-OWW vs CNN)
  - OWW embedding scoring path
  - Threshold sweep
  - Confusion matrix computation
  - d-prime calculation
  - CSV score dumping with strict validation
  - evaluate_onnx_model with mocked ONNX sessions
  - _build_oww_scorer and _build_cnn_scorer with mocked backends
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn required for evaluation tests")

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


class TestConfusionMatrix:
    """Test compute_confusion_matrix."""

    def test_perfect_classification(self) -> None:
        """All positives above threshold, all negatives below."""
        from violawake_sdk.training.evaluate import compute_confusion_matrix

        pos = np.array([0.9, 0.95, 0.85])
        neg = np.array([0.1, 0.05, 0.15])
        cm = compute_confusion_matrix(pos, neg, threshold=0.5)
        assert cm["tp"] == 3
        assert cm["fn"] == 0
        assert cm["fp"] == 0
        assert cm["tn"] == 3
        assert cm["precision"] == 1.0
        assert cm["recall"] == 1.0
        assert cm["f1"] == 1.0

    def test_all_misclassified(self) -> None:
        """All positives below threshold, all negatives above."""
        from violawake_sdk.training.evaluate import compute_confusion_matrix

        pos = np.array([0.1, 0.2])
        neg = np.array([0.8, 0.9])
        cm = compute_confusion_matrix(pos, neg, threshold=0.5)
        assert cm["tp"] == 0
        assert cm["fn"] == 2
        assert cm["fp"] == 2
        assert cm["tn"] == 0
        assert cm["precision"] == 0.0
        assert cm["recall"] == 0.0
        assert cm["f1"] == 0.0

    def test_mixed(self) -> None:
        """Some correct, some incorrect."""
        from violawake_sdk.training.evaluate import compute_confusion_matrix

        pos = np.array([0.8, 0.3])  # 1 TP, 1 FN
        neg = np.array([0.6, 0.2])  # 1 FP, 1 TN
        cm = compute_confusion_matrix(pos, neg, threshold=0.5)
        assert cm["tp"] == 1
        assert cm["fn"] == 1
        assert cm["fp"] == 1
        assert cm["tn"] == 1


class TestDumpScoresCSV:
    """Test _dump_scores_csv CSV output and validation."""

    def test_csv_written_correctly(self, tmp_path: Path) -> None:
        """CSV should contain correct headers and rows."""
        from violawake_sdk.training.evaluate import _dump_scores_csv

        pos_files = [Path("a.wav"), Path("b.wav")]
        pos_scores = [0.9, 0.8]
        neg_files = [Path("c.wav"), Path("d.wav")]
        neg_scores = [0.1, 0.2]
        csv_path = tmp_path / "scores.csv"

        _dump_scores_csv(pos_files, pos_scores, neg_files, neg_scores, 0.5, csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["file", "label", "score", "threshold_pass"]
        assert len(rows) == 5  # header + 2 pos + 2 neg
        assert rows[1][1] == "positive"
        assert rows[3][1] == "negative"

    def test_csv_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Should create parent directories."""
        from violawake_sdk.training.evaluate import _dump_scores_csv

        csv_path = tmp_path / "sub" / "dir" / "scores.csv"
        _dump_scores_csv([Path("a.wav")], [0.9], [Path("b.wav")], [0.1], 0.5, csv_path)
        assert csv_path.exists()

    def test_csv_mismatch_raises(self, tmp_path: Path) -> None:
        """Mismatched files/scores lengths should raise ValueError."""
        from violawake_sdk.training.evaluate import _dump_scores_csv

        csv_path = tmp_path / "scores.csv"
        with pytest.raises(ValueError, match="Positive files/scores mismatch"):
            _dump_scores_csv([Path("a.wav")], [0.9, 0.8], [], [], 0.5, csv_path)

        with pytest.raises(ValueError, match="Negative files/scores mismatch"):
            _dump_scores_csv([], [], [Path("a.wav")], [0.1, 0.2], 0.5, csv_path)


class TestBuildOWWScorer:
    """Test _build_oww_scorer with mocked OWW and ONNX backends."""

    def test_missing_openwakeword_raises(self) -> None:
        """Should raise ImportError if openwakeword is not installed."""
        from violawake_sdk.training.evaluate import _build_oww_scorer

        mock_session = mock.MagicMock()
        with mock.patch.dict("sys.modules", {"openwakeword": None, "openwakeword.model": None}):
            with pytest.raises(ImportError, match="openwakeword required"):
                _build_oww_scorer(mock_session, "input")

    def test_returns_callable(self) -> None:
        """With mocked OWW, should return a callable scorer."""
        from violawake_sdk.training.evaluate import _build_oww_scorer

        mock_session = mock.MagicMock()
        mock_session.run.return_value = [np.array([[0.85]])]

        # Mock the OWW model and preprocessor
        mock_preprocessor = mock.MagicMock()
        mock_preprocessor.embed_clips.return_value = np.random.randn(1, 3, 96).astype(np.float32)

        mock_oww_model = mock.MagicMock()
        mock_oww_model.preprocessor = mock_preprocessor

        mock_oww_module = mock.MagicMock()
        mock_oww_module.Model.return_value = mock_oww_model

        with mock.patch.dict("sys.modules", {"openwakeword": mock.MagicMock(), "openwakeword.model": mock_oww_module}):
            scorer = _build_oww_scorer(mock_session, "input")

        assert callable(scorer)


class TestBuildCNNScorer:
    """Test _build_cnn_scorer with mocked ONNX backend.

    _build_cnn_scorer imports from violawake_sdk.audio which requires torchaudio.
    We mock the audio module to avoid the broken torchaudio import.
    """

    @staticmethod
    def _mock_audio_imports():
        """Create mock audio module to avoid torchaudio dependency."""
        mock_audio = mock.MagicMock()
        mock_audio.load_audio = mock.MagicMock(return_value=np.zeros(16000, dtype=np.float32))
        mock_audio.center_crop = mock.MagicMock(return_value=np.zeros(16000, dtype=np.float32))
        mock_audio.compute_features = mock.MagicMock(return_value=np.zeros((40, 100), dtype=np.float32))

        mock_constants = mock.MagicMock()
        mock_constants.CLIP_SAMPLES = 16000
        return mock_audio, mock_constants

    def test_returns_callable(self) -> None:
        """Should return a callable scorer function."""
        mock_audio, mock_constants = self._mock_audio_imports()

        with mock.patch.dict("sys.modules", {"violawake_sdk.audio": mock_audio, "violawake_sdk._constants": mock_constants}):
            from violawake_sdk.training.evaluate import _build_cnn_scorer

            mock_session = mock.MagicMock()
            mock_session.run.return_value = [np.array([[0.75]])]
            scorer = _build_cnn_scorer(mock_session, "input")

        assert callable(scorer)

    def test_scorer_returns_float_on_valid_audio(self, tmp_path: Path) -> None:
        """Scorer should return a float when scoring a file."""
        mock_audio, mock_constants = self._mock_audio_imports()

        with mock.patch.dict("sys.modules", {"violawake_sdk.audio": mock_audio, "violawake_sdk._constants": mock_constants}):
            from violawake_sdk.training.evaluate import _build_cnn_scorer

            mock_session = mock.MagicMock()
            mock_session.run.return_value = [np.array([[0.42]])]
            scorer = _build_cnn_scorer(mock_session, "input")

        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        result = scorer(wav_file)
        assert isinstance(result, float)
        assert abs(result - 0.42) < 0.01

    def test_scorer_returns_none_on_load_failure(self, tmp_path: Path) -> None:
        """Scorer should return None if load_audio returns None."""
        mock_audio, mock_constants = self._mock_audio_imports()
        mock_audio.load_audio.return_value = None

        with mock.patch.dict("sys.modules", {"violawake_sdk.audio": mock_audio, "violawake_sdk._constants": mock_constants}):
            from violawake_sdk.training.evaluate import _build_cnn_scorer

            mock_session = mock.MagicMock()
            scorer = _build_cnn_scorer(mock_session, "input")

        wav_file = tmp_path / "test.wav"
        wav_file.touch()
        result = scorer(wav_file)
        assert result is None


class TestEvaluateOnnxModel:
    """Test evaluate_onnx_model with fully mocked ONNX sessions.

    This avoids needing real ONNX models or audio files by mocking
    build_model_scorer to return a deterministic scoring function.
    """

    def _make_test_dir(self, tmp_path: Path, n_pos: int = 5, n_neg: int = 10) -> Path:
        """Create a test directory with dummy wav files."""
        test_dir = tmp_path / "test_data"
        pos_dir = test_dir / "positives"
        neg_dir = test_dir / "negatives"
        pos_dir.mkdir(parents=True)
        neg_dir.mkdir(parents=True)

        for i in range(n_pos):
            (pos_dir / f"pos_{i}.wav").touch()
        for i in range(n_neg):
            (neg_dir / f"neg_{i}.wav").touch()

        return test_dir

    def test_missing_positives_dir(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError if positives/ is missing."""
        from violawake_sdk.training.evaluate import evaluate_onnx_model

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "negatives").mkdir()

        with pytest.raises(FileNotFoundError, match="positives"):
            evaluate_onnx_model("fake.onnx", test_dir)

    def test_missing_negatives_dir(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError if negatives/ is missing."""
        from violawake_sdk.training.evaluate import evaluate_onnx_model

        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "positives").mkdir()

        with pytest.raises(FileNotFoundError, match="negatives"):
            evaluate_onnx_model("fake.onnx", test_dir)

    def test_full_evaluation_mocked(self, tmp_path: Path) -> None:
        """Full evaluation with mocked scorer should return all expected keys."""
        from violawake_sdk.training.evaluate import evaluate_onnx_model

        test_dir = self._make_test_dir(tmp_path, n_pos=10, n_neg=20)

        # Create a deterministic scorer: positives get high scores, negatives get low
        rng = np.random.default_rng(42)
        pos_scores_iter = iter(rng.uniform(0.8, 1.0, 10).tolist())
        neg_scores_iter = iter(rng.uniform(0.0, 0.2, 20).tolist())
        call_count = {"pos": 0, "neg": 0}

        def mock_scorer(wav_path: Path) -> float:
            if "positives" in str(wav_path):
                call_count["pos"] += 1
                return next(pos_scores_iter)
            else:
                call_count["neg"] += 1
                return next(neg_scores_iter)

        with mock.patch(
            "violawake_sdk.training.evaluate.build_model_scorer",
            return_value=("mlp_on_oww", mock_scorer),
        ):
            results = evaluate_onnx_model("fake.onnx", test_dir, threshold=0.5)

        # Verify all expected keys
        expected_keys = {
            "d_prime", "far_per_hour", "frr", "roc_auc",
            "tp_scores", "fp_scores", "tp_mean", "fp_mean",
            "n_positives", "n_negatives", "threshold_used",
            "architecture",
            "optimal_threshold", "optimal_far", "optimal_frr", "eer_approx",
            "confusion_matrix",
        }
        assert expected_keys.issubset(set(results.keys()))

        # Verify counts
        assert results["n_positives"] == 10
        assert results["n_negatives"] == 20
        assert results["architecture"] == "mlp_on_oww"
        assert results["threshold_used"] == 0.5

        # With well-separated scores, d_prime should be high
        assert results["d_prime"] > 5.0
        assert results["roc_auc"] > 0.9

        # Confusion matrix should be a dict with expected keys
        cm = results["confusion_matrix"]
        assert "tp" in cm and "fp" in cm and "tn" in cm and "fn" in cm

    def test_evaluation_with_csv_dump(self, tmp_path: Path) -> None:
        """evaluate_onnx_model should write CSV when dump_scores_csv is provided."""
        from violawake_sdk.training.evaluate import evaluate_onnx_model

        test_dir = self._make_test_dir(tmp_path, n_pos=3, n_neg=5)
        csv_path = tmp_path / "dump.csv"

        rng = np.random.default_rng(99)
        pos_iter = iter(rng.uniform(0.7, 1.0, 3).tolist())
        neg_iter = iter(rng.uniform(0.0, 0.3, 5).tolist())

        def mock_scorer(wav_path: Path) -> float:
            if "positives" in str(wav_path):
                return next(pos_iter)
            return next(neg_iter)

        with mock.patch(
            "violawake_sdk.training.evaluate.build_model_scorer",
            return_value=("cnn", mock_scorer),
        ):
            evaluate_onnx_model("fake.onnx", test_dir, dump_scores_csv=csv_path)

        assert csv_path.exists()
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        # header + 3 pos + 5 neg = 9 rows
        assert len(rows) == 9

    def test_no_valid_positives_raises(self, tmp_path: Path) -> None:
        """Should raise RuntimeError if scorer returns None for all positives."""
        from violawake_sdk.training.evaluate import evaluate_onnx_model

        test_dir = self._make_test_dir(tmp_path, n_pos=3, n_neg=3)

        def mock_scorer(wav_path: Path) -> float | None:
            if "positives" in str(wav_path):
                return None  # All positives fail
            return 0.1

        with mock.patch(
            "violawake_sdk.training.evaluate.build_model_scorer",
            return_value=("mlp_on_oww", mock_scorer),
        ):
            with pytest.raises(RuntimeError, match="No valid positive"):
                evaluate_onnx_model("fake.onnx", test_dir)
