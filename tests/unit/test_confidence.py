"""Unit tests for K2: Score Verifier / Confidence API.

Tests the ScoreTracker, ConfidenceLevel classification, and ConfidenceResult
without requiring model files or hardware.
"""

from __future__ import annotations

import pytest

from violawake_sdk.confidence import (
    ConfidenceLevel,
    ConfidenceResult,
    ScoreTracker,
)


class TestScoreTracker:
    """Test ScoreTracker recording and history."""

    def test_empty_tracker_returns_zero(self) -> None:
        tracker = ScoreTracker(threshold=0.80)
        assert tracker.latest_score == 0.0
        assert tracker.last_scores == ()

    def test_record_single_score(self) -> None:
        tracker = ScoreTracker(threshold=0.80)
        tracker.record(0.75)
        assert tracker.latest_score == 0.75
        assert tracker.last_scores == (0.75,)

    def test_record_multiple_scores(self) -> None:
        tracker = ScoreTracker(threshold=0.80)
        for s in [0.1, 0.5, 0.9]:
            tracker.record(s)
        assert tracker.latest_score == 0.9
        assert tracker.last_scores == (0.1, 0.5, 0.9)

    def test_history_size_limit(self) -> None:
        tracker = ScoreTracker(threshold=0.80, history_size=5)
        for i in range(10):
            tracker.record(float(i) / 10.0)
        scores = tracker.last_scores
        assert len(scores) == 5
        # Should retain the last 5: 0.5, 0.6, 0.7, 0.8, 0.9
        assert scores == (0.5, 0.6, 0.7, 0.8, 0.9)

    def test_reset_clears_history(self) -> None:
        tracker = ScoreTracker(threshold=0.80)
        tracker.record(0.5)
        tracker.record(0.6)
        tracker.reset()
        assert tracker.latest_score == 0.0
        assert tracker.last_scores == ()


class TestConfidenceClassification:
    """Test the classify() method with various score/confirm combinations."""

    def test_no_scores_recorded_returns_low_confidence(self) -> None:
        tracker = ScoreTracker(threshold=0.0)
        result = tracker.classify(confirm_count=1, confirm_required=1)
        assert result.confidence == ConfidenceLevel.LOW
        assert result.raw_score == 0.0
        assert result.score_history == ()

    def test_low_confidence(self) -> None:
        tracker = ScoreTracker(threshold=0.80, medium_ratio=0.75, high_ratio=0.90)
        tracker.record(0.10)
        result = tracker.classify(confirm_count=0, confirm_required=3)
        assert result.confidence == ConfidenceLevel.LOW
        assert result.raw_score == 0.10
        assert result.confirm_count == 0
        assert result.confirm_required == 3

    def test_medium_confidence(self) -> None:
        # medium_boundary = 0.80 * 0.75 = 0.60
        tracker = ScoreTracker(threshold=0.80, medium_ratio=0.75, high_ratio=0.90)
        tracker.record(0.65)
        result = tracker.classify(confirm_count=1, confirm_required=3)
        assert result.confidence == ConfidenceLevel.MEDIUM

    def test_high_confidence(self) -> None:
        # high_boundary = 0.80 * 0.90 = 0.72
        tracker = ScoreTracker(threshold=0.80, medium_ratio=0.75, high_ratio=0.90)
        tracker.record(0.75)
        result = tracker.classify(confirm_count=2, confirm_required=3)
        assert result.confidence == ConfidenceLevel.HIGH

    def test_certain_confidence(self) -> None:
        tracker = ScoreTracker(threshold=0.80, medium_ratio=0.75, high_ratio=0.90)
        tracker.record(0.90)
        result = tracker.classify(confirm_count=3, confirm_required=3)
        assert result.confidence == ConfidenceLevel.CERTAIN

    def test_high_score_but_insufficient_confirms_is_high_not_certain(self) -> None:
        tracker = ScoreTracker(threshold=0.80)
        tracker.record(0.95)
        result = tracker.classify(confirm_count=1, confirm_required=3)
        # Score >= threshold but confirm_count < confirm_required -> HIGH (not CERTAIN)
        assert result.confidence == ConfidenceLevel.HIGH

    def test_score_history_in_result(self) -> None:
        tracker = ScoreTracker(threshold=0.80, history_size=3)
        for s in [0.1, 0.5, 0.9]:
            tracker.record(s)
        result = tracker.classify(confirm_count=0, confirm_required=1)
        assert result.score_history == (0.1, 0.5, 0.9)


class TestConfidenceResult:
    """Test ConfidenceResult dataclass properties."""

    def test_frozen(self) -> None:
        result = ConfidenceResult(
            raw_score=0.85,
            confirm_count=2,
            confirm_required=3,
            confidence=ConfidenceLevel.HIGH,
            score_history=(0.80, 0.85),
        )
        with pytest.raises(AttributeError):
            result.raw_score = 0.99  # type: ignore[misc]

    def test_confidence_level_values(self) -> None:
        assert ConfidenceLevel.LOW == "LOW"
        assert ConfidenceLevel.MEDIUM == "MEDIUM"
        assert ConfidenceLevel.HIGH == "HIGH"
        assert ConfidenceLevel.CERTAIN == "CERTAIN"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_threshold(self) -> None:
        tracker = ScoreTracker(threshold=0.0)
        tracker.record(0.0)
        result = tracker.classify(confirm_count=1, confirm_required=1)
        # score >= threshold (0.0 >= 0.0) and confirms met -> CERTAIN
        assert result.confidence == ConfidenceLevel.CERTAIN

    def test_one_threshold(self) -> None:
        tracker = ScoreTracker(threshold=1.0)
        tracker.record(0.99)
        result = tracker.classify(confirm_count=1, confirm_required=1)
        # 0.99 < 1.0 -> not CERTAIN. high_boundary = 1.0 * 0.90 = 0.90 -> HIGH
        assert result.confidence == ConfidenceLevel.HIGH

    def test_history_size_one(self) -> None:
        tracker = ScoreTracker(threshold=0.80, history_size=1)
        tracker.record(0.5)
        tracker.record(0.9)
        assert tracker.last_scores == (0.9,)
        assert tracker.latest_score == 0.9
