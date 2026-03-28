"""Experimental: K3 multi-model ensemble support.

Provides score fusion strategies for combining predictions from multiple
MLP/CNN models. Used by ``WakeDetector`` when initialized with multiple
models.

Note: Ensemble scoring is an advanced feature. For most use cases, a single
model with multi-window confirmation (``confirm_count=3``) provides
sufficient accuracy.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """Score fusion strategy for multi-model ensemble."""

    AVERAGE = "average"
    MAX = "max"
    VOTING = "voting"
    WEIGHTED_AVERAGE = "weighted_average"


def fuse_scores(
    scores: list[float],
    strategy: FusionStrategy = FusionStrategy.AVERAGE,
    weights: list[float] | None = None,
    voting_threshold: float = 0.5,
) -> float:
    """Combine multiple model scores into a single fused score.

    Args:
        scores: List of individual model scores in [0.0, 1.0].
        strategy: How to combine the scores.
        weights: Per-model weights for WEIGHTED_AVERAGE. Must sum to 1.0.
            Ignored for other strategies.
        voting_threshold: Per-model threshold for VOTING strategy. A model
            "votes yes" if its score >= this threshold. The fused score is
            the fraction of models that voted yes.

    Returns:
        Fused score in [0.0, 1.0].

    Raises:
        ValueError: If scores is empty, or weights mismatch, or weights don't sum to ~1.
    """
    if not scores:
        raise ValueError("scores list must not be empty")

    if strategy == FusionStrategy.AVERAGE:
        return float(np.mean(scores))

    elif strategy == FusionStrategy.MAX:
        return float(max(scores))

    elif strategy == FusionStrategy.VOTING:
        votes = sum(1 for s in scores if s >= voting_threshold)
        return votes / len(scores)

    elif strategy == FusionStrategy.WEIGHTED_AVERAGE:
        if weights is None:
            raise ValueError("weights required for WEIGHTED_AVERAGE strategy")
        if len(weights) != len(scores):
            raise ValueError(
                f"weights length ({len(weights)}) must match scores length ({len(scores)})"
            )
        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 0.001:
            # Normalize weights to sum to exactly 1.0 and warn
            logger.warning(
                "Ensemble weights sum to %.6f (not 1.0); normalizing",
                weight_sum,
            )
            weights = [w / weight_sum for w in weights]
        return float(np.dot(scores, weights))

    else:
        raise ValueError(f"Unknown fusion strategy: {strategy}")


class EnsembleScorer:
    """Manages multiple model sessions and fuses their scores.

    This class holds references to ONNX sessions and provides a single
    ``score()`` method that runs all models and returns the fused result.

    Args:
        strategy: Score fusion strategy. Default AVERAGE.
        weights: Per-model weights for WEIGHTED_AVERAGE.
        voting_threshold: Per-model vote threshold for VOTING.
    """

    def __init__(
        self,
        strategy: FusionStrategy | str = FusionStrategy.AVERAGE,
        weights: list[float] | None = None,
        voting_threshold: float = 0.5,
    ) -> None:
        if isinstance(strategy, str):
            strategy = FusionStrategy(strategy)
        self._strategy = strategy
        self._weights = weights
        self._voting_threshold = voting_threshold
        self._sessions: list[object] = []
        self._input_names: list[str] = []

    @property
    def model_count(self) -> int:
        """Number of models in the ensemble."""
        return len(self._sessions)

    @property
    def strategy(self) -> FusionStrategy:
        """Active fusion strategy."""
        return self._strategy

    def clear(self) -> None:
        """Remove all registered sessions, releasing references."""
        self._sessions.clear()
        self._input_names.clear()

    def add_session(self, session: object, input_name: str) -> None:
        """Register an ONNX inference session.

        Args:
            session: An onnxruntime.InferenceSession.
            input_name: The name of the model's input tensor.
        """
        self._sessions.append(session)
        self._input_names.append(input_name)

        if (
            self._strategy == FusionStrategy.WEIGHTED_AVERAGE
            and self._weights is not None
            and len(self._weights) != len(self._sessions)
        ):
            logger.warning(
                "Ensemble has %d models but %d weights — "
                "will fail at score() time unless corrected",
                len(self._sessions),
                len(self._weights),
            )

    def _compute_scores(self, embedding: np.ndarray) -> list[float]:
        """Run all models on the embedding and return per-model scores.

        Validates the embedding shape, runs inference on each registered
        session, and clamps scores to [0.0, 1.0].  Failed models contribute
        0.0 with a logged warning.

        Args:
            embedding: 1-D or 2-D float32 embedding vector.

        Returns:
            List of per-model scores (one per registered session).
        """
        if embedding.ndim == 0:
            raise ValueError("Embedding must be at least 1-D, got scalar")
        if embedding.ndim > 2:
            raise ValueError(
                f"Embedding must be 1-D or 2-D, got {embedding.ndim}-D with shape {embedding.shape}"
            )

        emb_2d = embedding.reshape(1, -1)
        scores: list[float] = []

        for idx, (session, input_name) in enumerate(
            zip(self._sessions, self._input_names, strict=False)
        ):
            try:
                result = session.run(None, {input_name: emb_2d})  # type: ignore[union-attr]
                score = float(np.asarray(result[0]).flatten()[0])
                score = max(0.0, min(1.0, score))
                scores.append(score)
            except Exception as e:
                logger.warning(
                    "Ensemble model %d/%d inference failed: %s",
                    idx,
                    len(self._sessions),
                    e,
                )
                scores.append(0.0)

        return scores

    def score(self, embedding: np.ndarray) -> float:
        """Run all models on the embedding and return the fused score.

        Args:
            embedding: 1-D float32 embedding vector (e.g. 96-dim from OWW).

        Returns:
            Fused score in [0.0, 1.0].
        """
        if not self._sessions:
            return 0.0

        individual_scores = self._compute_scores(embedding)

        return fuse_scores(
            individual_scores,
            strategy=self._strategy,
            weights=self._weights,
            voting_threshold=self._voting_threshold,
        )

    def score_all(self, embedding: np.ndarray) -> list[float]:
        """Run all models and return individual scores (no fusion).

        Args:
            embedding: 1-D float32 embedding vector.

        Returns:
            List of per-model scores.
        """
        if not self._sessions:
            return []

        return self._compute_scores(embedding)
