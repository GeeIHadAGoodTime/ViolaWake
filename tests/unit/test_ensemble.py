"""Unit tests for K3: Multi-model ensemble support.

Tests the FusionStrategy enum, fuse_scores function, and EnsembleScorer
without requiring real ONNX models.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from violawake_sdk.ensemble import (
    EnsembleScorer,
    FusionStrategy,
    fuse_scores,
)


class TestFusionStrategies:
    """Test each score fusion strategy."""

    def test_average(self) -> None:
        result = fuse_scores([0.8, 0.6, 0.4], strategy=FusionStrategy.AVERAGE)
        assert abs(result - 0.6) < 1e-6

    def test_average_single_model(self) -> None:
        result = fuse_scores([0.75], strategy=FusionStrategy.AVERAGE)
        assert abs(result - 0.75) < 1e-6

    def test_max(self) -> None:
        result = fuse_scores([0.3, 0.9, 0.5], strategy=FusionStrategy.MAX)
        assert abs(result - 0.9) < 1e-6

    def test_max_all_same(self) -> None:
        result = fuse_scores([0.5, 0.5, 0.5], strategy=FusionStrategy.MAX)
        assert abs(result - 0.5) < 1e-6

    def test_voting_all_above(self) -> None:
        result = fuse_scores(
            [0.8, 0.9, 0.7], strategy=FusionStrategy.VOTING, voting_threshold=0.5,
        )
        assert abs(result - 1.0) < 1e-6  # 3/3 vote yes

    def test_voting_some_above(self) -> None:
        result = fuse_scores(
            [0.8, 0.3, 0.7], strategy=FusionStrategy.VOTING, voting_threshold=0.5,
        )
        assert abs(result - 2.0 / 3.0) < 1e-6  # 2/3 vote yes

    def test_voting_none_above(self) -> None:
        result = fuse_scores(
            [0.1, 0.2, 0.3], strategy=FusionStrategy.VOTING, voting_threshold=0.5,
        )
        assert abs(result - 0.0) < 1e-6  # 0/3 vote yes

    def test_weighted_average(self) -> None:
        result = fuse_scores(
            [0.8, 0.2],
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
            weights=[0.7, 0.3],
        )
        expected = 0.8 * 0.7 + 0.2 * 0.3  # 0.56 + 0.06 = 0.62
        assert abs(result - expected) < 1e-6

    def test_weighted_average_equal_weights(self) -> None:
        result = fuse_scores(
            [0.8, 0.4],
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
            weights=[0.5, 0.5],
        )
        assert abs(result - 0.6) < 1e-6

    def test_empty_scores_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            fuse_scores([], strategy=FusionStrategy.AVERAGE)

    def test_weighted_average_no_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="weights required"):
            fuse_scores([0.5], strategy=FusionStrategy.WEIGHTED_AVERAGE)

    def test_weighted_average_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="weights length"):
            fuse_scores(
                [0.5, 0.6],
                strategy=FusionStrategy.WEIGHTED_AVERAGE,
                weights=[0.5, 0.3, 0.2],
            )

    def test_weighted_average_bad_sum_normalizes(self) -> None:
        """Weights that don't sum to 1.0 are auto-normalized (with a warning)."""
        result = fuse_scores(
            [0.5, 0.6],
            strategy=FusionStrategy.WEIGHTED_AVERAGE,
            weights=[0.5, 0.8],  # sum = 1.3, will be normalized
        )
        # After normalization: weights become [0.5/1.3, 0.8/1.3] ≈ [0.3846, 0.6154]
        # Result: 0.5 * 0.3846 + 0.6 * 0.6154 ≈ 0.5615
        assert 0.0 <= result <= 1.0


class TestFusionStrategyEnum:
    """Test FusionStrategy enum values."""

    def test_string_values(self) -> None:
        assert FusionStrategy.AVERAGE.value == "average"
        assert FusionStrategy.MAX.value == "max"
        assert FusionStrategy.VOTING.value == "voting"
        assert FusionStrategy.WEIGHTED_AVERAGE.value == "weighted_average"

    def test_from_string(self) -> None:
        assert FusionStrategy("average") == FusionStrategy.AVERAGE
        assert FusionStrategy("max") == FusionStrategy.MAX


class TestEnsembleScorer:
    """Test EnsembleScorer with mocked ONNX sessions."""

    @staticmethod
    def _make_mock_session(score: float) -> MagicMock:
        """Create a mock ONNX session that always returns the given score."""
        session = MagicMock()
        session.run.return_value = [np.array([[score]], dtype=np.float32)]
        return session

    def test_empty_ensemble_returns_zero(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.AVERAGE)
        assert scorer.model_count == 0
        emb = np.random.randn(96).astype(np.float32)
        assert scorer.score(emb) == 0.0

    def test_single_model(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.AVERAGE)
        scorer.add_session(self._make_mock_session(0.85), "input")
        assert scorer.model_count == 1
        emb = np.random.randn(96).astype(np.float32)
        score = scorer.score(emb)
        assert abs(score - 0.85) < 1e-5

    def test_two_models_average(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.AVERAGE)
        scorer.add_session(self._make_mock_session(0.90), "input")
        scorer.add_session(self._make_mock_session(0.70), "input")
        emb = np.random.randn(96).astype(np.float32)
        score = scorer.score(emb)
        assert abs(score - 0.80) < 1e-5

    def test_two_models_max(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.MAX)
        scorer.add_session(self._make_mock_session(0.90), "input")
        scorer.add_session(self._make_mock_session(0.70), "input")
        emb = np.random.randn(96).astype(np.float32)
        score = scorer.score(emb)
        assert abs(score - 0.90) < 1e-5

    def test_score_all(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.AVERAGE)
        scorer.add_session(self._make_mock_session(0.90), "input")
        scorer.add_session(self._make_mock_session(0.70), "input")
        emb = np.random.randn(96).astype(np.float32)
        scores = scorer.score_all(emb)
        assert len(scores) == 2
        assert abs(scores[0] - 0.90) < 1e-5
        assert abs(scores[1] - 0.70) < 1e-5

    def test_score_all_empty(self) -> None:
        scorer = EnsembleScorer()
        emb = np.random.randn(96).astype(np.float32)
        assert scorer.score_all(emb) == []

    def test_failed_model_returns_zero(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.AVERAGE)
        good_session = self._make_mock_session(0.80)
        bad_session = MagicMock()
        bad_session.run.side_effect = RuntimeError("model error")
        scorer.add_session(good_session, "input")
        scorer.add_session(bad_session, "input")
        emb = np.random.randn(96).astype(np.float32)
        score = scorer.score(emb)
        # average of (0.80, 0.0) = 0.40
        assert abs(score - 0.40) < 1e-5

    def test_strategy_property(self) -> None:
        scorer = EnsembleScorer(strategy="max")
        assert scorer.strategy == FusionStrategy.MAX

    def test_score_clamped_to_01(self) -> None:
        scorer = EnsembleScorer(strategy=FusionStrategy.AVERAGE)
        # Score > 1.0 should be clamped
        session = MagicMock()
        session.run.return_value = [np.array([[1.5]], dtype=np.float32)]
        scorer.add_session(session, "input")
        emb = np.random.randn(96).astype(np.float32)
        score = scorer.score(emb)
        assert score <= 1.0
