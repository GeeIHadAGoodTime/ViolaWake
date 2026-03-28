"""Experimental — not evaluated on speaker benchmarks: K5 speaker verification.

EXPERIMENTAL: Speaker verification using OWW wake word embeddings. These
embeddings are optimized for wake word discrimination, not speaker
recognition. Verification accuracy has not been evaluated on standard speaker
benchmarks. Use for convenience gating only — not as a security boundary.

Provides a lightweight speaker verification system that uses cosine similarity
on embeddings. Supports speaker enrollment and verification as callbacks
that integrate with ``WakeDetector`` post-detection.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeakerVerifyResult:
    """Result from speaker verification.

    Attributes:
        is_verified: True if the speaker matched an enrolled profile.
        speaker_id: ID of the best-matching speaker, or None.
        similarity: Cosine similarity score of the best match.
        threshold: The threshold used for verification.
    """

    is_verified: bool
    speaker_id: str | None
    similarity: float
    threshold: float


class CosineScorer:
    """Computes cosine similarity between embedding vectors.

    This is a stateless utility — it does not store speaker profiles.
    """

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First embedding vector (1-D).
            b: Second embedding vector (1-D).

        Returns:
            Cosine similarity in [-1.0, 1.0]. Higher = more similar.
        """
        a = a.flatten().astype(np.float64)
        b = b.flatten().astype(np.float64)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    @staticmethod
    def distance(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance (1 - similarity) between two vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine distance in [0.0, 2.0]. Lower = more similar.
        """
        return 1.0 - CosineScorer.similarity(a, b)


class SpeakerProfile:
    """Enrolled speaker profile with averaged embedding.

    Stores multiple enrollment embeddings and computes a centroid for
    efficient verification. The centroid is updated incrementally as
    new embeddings are enrolled.

    Args:
        speaker_id: Unique identifier for this speaker.
    """

    def __init__(self, speaker_id: str) -> None:
        self.speaker_id = speaker_id
        self._embeddings: list[np.ndarray] = []
        self._centroid: np.ndarray | None = None

    @property
    def n_enrollments(self) -> int:
        """Number of enrollment embeddings."""
        return len(self._embeddings)

    @property
    def embeddings(self) -> list[np.ndarray]:
        """List of enrollment embedding vectors (read-only copy)."""
        return list(self._embeddings)

    @property
    def centroid(self) -> np.ndarray | None:
        """Averaged embedding centroid, or None if no enrollments."""
        return self._centroid

    def add_embedding(self, embedding: np.ndarray) -> None:
        """Add an enrollment embedding and update the centroid.

        Args:
            embedding: 1-D float32 embedding vector.
        """
        emb = embedding.flatten().astype(np.float32)
        self._embeddings.append(emb)

        if self._centroid is None:
            self._centroid = emb.copy()
        else:
            # Incremental centroid update: new_centroid = old * (n-1)/n + new/n
            n = len(self._embeddings)
            self._centroid = self._centroid * ((n - 1) / n) + emb / n

    def verify(self, embedding: np.ndarray) -> float:
        """Compare an embedding against the enrolled centroid.

        Args:
            embedding: 1-D float32 embedding to verify.

        Returns:
            Cosine similarity score. Returns 0.0 if no enrollments.
        """
        if self._centroid is None:
            return 0.0
        return CosineScorer.similarity(embedding, self._centroid)


class SpeakerVerificationHook:
    """Post-detection speaker verification with enrollment and persistence.

    Manages multiple speaker profiles and verifies incoming embeddings
    against all enrolled speakers. Can be used as a callback in WakeDetector.

    Args:
        threshold: Minimum cosine similarity to consider a match. Default 0.65.
    """

    # Maximum embeddings per speaker loaded from disk to prevent DoS
    # from tampered JSON files claiming millions of embeddings.
    _MAX_EMBEDDINGS_PER_SPEAKER = 1000

    def __init__(self, threshold: float = 0.65) -> None:
        self._threshold = threshold
        self._profiles: dict[str, SpeakerProfile] = {}
        self._scorer = CosineScorer()
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float:
        """Verification threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def enrolled_speakers(self) -> list[str]:
        """List of enrolled speaker IDs."""
        return list(self._profiles.keys())

    def enroll_speaker(
        self,
        speaker_id: str,
        embeddings: list[np.ndarray] | np.ndarray,
    ) -> int:
        """Enroll a speaker with one or more embedding samples.

        If the speaker is already enrolled, the new embeddings are added
        to the existing profile (incrementally updates the centroid).

        Thread-safe: protects ``_profiles`` mutations with a lock.

        Args:
            speaker_id: Unique speaker identifier.
            embeddings: Single embedding or list of embeddings.

        Returns:
            Total number of enrollments for this speaker after adding.
        """
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = [embeddings]
        elif isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            embeddings = [embeddings[i] for i in range(embeddings.shape[0])]

        with self._lock:
            if speaker_id not in self._profiles:
                self._profiles[speaker_id] = SpeakerProfile(speaker_id)

            profile = self._profiles[speaker_id]

            for emb in embeddings:
                profile.add_embedding(emb)

            logger.info(
                "Enrolled speaker '%s': %d total embeddings",
                speaker_id,
                profile.n_enrollments,
            )
            return profile.n_enrollments

    def remove_speaker(self, speaker_id: str) -> bool:
        """Remove an enrolled speaker.

        Thread-safe: protects ``_profiles`` mutations with a lock.

        Args:
            speaker_id: Speaker to remove.

        Returns:
            True if the speaker was found and removed.
        """
        with self._lock:
            if speaker_id in self._profiles:
                del self._profiles[speaker_id]
                logger.info("Removed speaker '%s'", speaker_id)
                return True
            return False

    def verify_speaker(self, embedding: np.ndarray) -> SpeakerVerifyResult:
        """Verify an embedding against all enrolled speakers.

        Finds the best-matching enrolled speaker and checks if the similarity
        exceeds the threshold.

        Thread-safe: takes a snapshot of profiles under lock, then scores
        outside the lock to avoid holding it during computation.

        Args:
            embedding: 1-D float32 embedding from the wake word detection.

        Returns:
            SpeakerVerifyResult with match details.
        """
        with self._lock:
            if not self._profiles:
                return SpeakerVerifyResult(
                    is_verified=False,
                    speaker_id=None,
                    similarity=0.0,
                    threshold=self._threshold,
                )
            # Snapshot profiles under lock for iteration safety
            profiles_snapshot = list(self._profiles.items())
            threshold = self._threshold

        best_id: str | None = None
        best_sim = -1.0

        for speaker_id, profile in profiles_snapshot:
            sim = profile.verify(embedding)
            if sim > best_sim:
                best_sim = sim
                best_id = speaker_id

        is_verified = best_sim >= threshold

        if is_verified:
            logger.debug(
                "Speaker verified: id=%s, similarity=%.3f (threshold=%.3f)",
                best_id,
                best_sim,
                threshold,
            )
        else:
            logger.debug(
                "Speaker rejected: best_id=%s, similarity=%.3f (threshold=%.3f)",
                best_id,
                best_sim,
                threshold,
            )

        return SpeakerVerifyResult(
            is_verified=is_verified,
            speaker_id=best_id if is_verified else None,
            similarity=best_sim,
            threshold=threshold,
        )

    def __call__(self, embedding: np.ndarray) -> bool:
        """Callable interface for use as speaker_verify_fn callback.

        Args:
            embedding: 1-D float32 embedding.

        Returns:
            True if any enrolled speaker matches.
        """
        return self.verify_speaker(embedding).is_verified

    def save(self, path: str | Path) -> None:
        """Persist all enrolled profiles to disk.

        Uses JSON for metadata and numpy .npz for embeddings (safe
        serialization -- no pickle/arbitrary code execution risk).

        Args:
            path: File path for the saved profiles (metadata JSON).
                  Embeddings are saved as a sibling .npz file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata: dict = {
            "threshold": self._threshold,
            "profiles": {},
        }
        embedding_arrays: dict[str, np.ndarray] = {}

        for sid, profile in self._profiles.items():
            metadata["profiles"][sid] = {
                "speaker_id": sid,
                "n_embeddings": profile.n_enrollments,
            }
            for i, emb in enumerate(profile.embeddings):
                embedding_arrays[f"{sid}__emb_{i}"] = emb

        # Save metadata as JSON
        json_path = path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Save embeddings as .npz (safe numpy format)
        npz_path = path.with_suffix(".npz")
        if embedding_arrays:
            np.savez(npz_path, **embedding_arrays)
        elif npz_path.exists():
            npz_path.unlink()

        logger.info("Speaker profiles saved to %s (%d speakers)", json_path, len(self._profiles))

    def load(self, path: str | Path) -> None:
        """Load enrolled profiles from disk.

        Reads JSON metadata and numpy .npz embeddings (safe
        deserialization -- no pickle).

        Args:
            path: File path to load from (will look for .json and .npz siblings).

        Raises:
            FileNotFoundError: If the metadata file does not exist.
        """
        path = Path(path)
        json_path = path.with_suffix(".json")

        # Support loading from the exact path given or the .json sibling
        if not json_path.exists():
            if not path.exists():
                raise FileNotFoundError(f"Speaker profiles not found: {path}")
            # Try reading the given path as JSON directly
            json_path = path

        with open(json_path, encoding="utf-8") as f:
            metadata = json.load(f)

        npz_path = path.with_suffix(".npz")
        embeddings_data: dict[str, np.ndarray] = {}
        if npz_path.exists():
            with np.load(npz_path) as npz:
                for key in npz.files:
                    embeddings_data[key] = npz[key]

        self._threshold = metadata.get("threshold", self._threshold)
        self._profiles.clear()

        for sid, pdata in metadata["profiles"].items():
            profile = SpeakerProfile(sid)
            n_emb = pdata.get("n_embeddings", 0)
            if n_emb > self._MAX_EMBEDDINGS_PER_SPEAKER:
                logger.warning(
                    "Speaker '%s' claims %d embeddings, capping to %d to prevent DoS",
                    sid,
                    n_emb,
                    self._MAX_EMBEDDINGS_PER_SPEAKER,
                )
                n_emb = self._MAX_EMBEDDINGS_PER_SPEAKER
            for i in range(n_emb):
                key = f"{sid}__emb_{i}"
                if key in embeddings_data:
                    profile.add_embedding(embeddings_data[key].astype(np.float32))
            self._profiles[sid] = profile

        logger.info("Speaker profiles loaded from %s (%d speakers)", json_path, len(self._profiles))
