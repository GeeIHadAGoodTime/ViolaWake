"""Stable: K2 score verifier / confidence API.

Provides confidence classification and score history tracking for wake word
detection scores. Used by ``WakeDetector.get_confidence()`` and
``WakeDetector.last_scores``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence classification for wake word detection."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CERTAIN = "CERTAIN"


@dataclass(frozen=True)
class ConfidenceResult:
    """Result from get_confidence() with full detection context.

    Attributes:
        raw_score: The most recent MLP/CNN output score in [0.0, 1.0].
        confirm_count: Number of consecutive above-threshold scores in the
            current multi-window confirmation sequence.
        confirm_required: Total consecutive scores required for detection.
        confidence: Classified confidence level.
        score_history: Recent score history (most recent last).
    """

    raw_score: float
    confirm_count: int
    confirm_required: int
    confidence: ConfidenceLevel
    score_history: tuple[float, ...]


class ScoreTracker:
    """Tracks detection scores and provides confidence classification.

    Maintains a fixed-size deque of recent scores and classifies the current
    detection state based on configurable thresholds.

    Args:
        threshold: The detection threshold (same as WakeDecisionPolicy).
        history_size: Maximum number of scores to retain. Default 50.
        medium_ratio: Fraction of threshold to qualify as MEDIUM. Default 0.75.
        high_ratio: Fraction of threshold to qualify as HIGH. Default 0.90.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        history_size: int = 50,
        medium_ratio: float = 0.75,
        high_ratio: float = 0.90,
    ) -> None:
        self._threshold = threshold
        self._history: deque[float] = deque(maxlen=history_size)
        self._medium_boundary = threshold * medium_ratio
        self._high_boundary = threshold * high_ratio

    def record(self, score: float) -> None:
        """Record a new score into history."""
        self._history.append(score)

    @property
    def last_scores(self) -> tuple[float, ...]:
        """Return the recent score history as a tuple (most recent last)."""
        return tuple(self._history)

    @property
    def latest_score(self) -> float:
        """Return the most recent score, or 0.0 if no scores recorded."""
        if not self._history:
            return 0.0
        return self._history[-1]

    def classify(self, confirm_count: int, confirm_required: int) -> ConfidenceResult:
        """Classify the current detection confidence.

        The classification logic:
        - LOW: no scores recorded yet
        - CERTAIN: score >= threshold AND confirm_count >= confirm_required
        - HIGH: score >= high_boundary (90% of threshold by default)
        - MEDIUM: score >= medium_boundary (75% of threshold by default)
        - LOW: everything else

        Args:
            confirm_count: Current consecutive above-threshold count.
            confirm_required: Required consecutive count for detection.

        Returns:
            ConfidenceResult with full context.
        """
        if not self._history:
            return ConfidenceResult(
                raw_score=0.0,
                confirm_count=confirm_count,
                confirm_required=confirm_required,
                confidence=ConfidenceLevel.LOW,
                score_history=(),
            )

        raw = self.latest_score

        if raw >= self._threshold and confirm_count >= confirm_required:
            level = ConfidenceLevel.CERTAIN
        elif raw >= self._high_boundary:
            level = ConfidenceLevel.HIGH
        elif raw >= self._medium_boundary:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return ConfidenceResult(
            raw_score=raw,
            confirm_count=confirm_count,
            confirm_required=confirm_required,
            confidence=level,
            score_history=self.last_scores,
        )

    def reset(self) -> None:
        """Clear all recorded scores."""
        self._history.clear()
