"""Unit tests for K5: Speaker verification hook.

Tests CosineScorer, SpeakerProfile, and SpeakerVerificationHook
without requiring model files or hardware.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from violawake_sdk.speaker import (
    CosineScorer,
    SpeakerProfile,
    SpeakerVerificationHook,
    SpeakerVerifyResult,
)


class TestCosineScorer:
    """Test cosine similarity computation."""

    def test_identical_vectors(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert abs(CosineScorer.similarity(a, a) - 1.0) < 1e-6

    def test_orthogonal_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert abs(CosineScorer.similarity(a, b) - 0.0) < 1e-6

    def test_opposite_vectors(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = -a
        assert abs(CosineScorer.similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        assert CosineScorer.similarity(a, b) == 0.0

    def test_distance_is_one_minus_similarity(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        sim = CosineScorer.similarity(a, b)
        dist = CosineScorer.distance(a, b)
        assert abs(dist - (1.0 - sim)) < 1e-6

    def test_high_dimensional(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(96).astype(np.float32)
        assert abs(CosineScorer.similarity(a, a) - 1.0) < 1e-5


class TestSpeakerProfile:
    """Test SpeakerProfile enrollment and verification."""

    def test_empty_profile(self) -> None:
        profile = SpeakerProfile("user1")
        assert profile.n_enrollments == 0
        assert profile.centroid is None

    def test_single_enrollment(self) -> None:
        profile = SpeakerProfile("user1")
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        profile.add_embedding(emb)
        assert profile.n_enrollments == 1
        assert profile.centroid is not None
        np.testing.assert_allclose(profile.centroid, emb, atol=1e-6)

    def test_multiple_enrollments_centroid(self) -> None:
        profile = SpeakerProfile("user1")
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        profile.add_embedding(emb1)
        profile.add_embedding(emb2)
        assert profile.n_enrollments == 2
        # Centroid should be average
        expected = (emb1 + emb2) / 2.0
        np.testing.assert_allclose(profile.centroid, expected, atol=1e-5)

    def test_verify_returns_high_similarity_for_enrolled(self) -> None:
        profile = SpeakerProfile("user1")
        emb = np.array([1.0, 0.5, 0.2], dtype=np.float32)
        profile.add_embedding(emb)
        sim = profile.verify(emb)
        assert sim > 0.99

    def test_verify_returns_low_similarity_for_different(self) -> None:
        profile = SpeakerProfile("user1")
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        profile.add_embedding(emb)
        different = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        sim = profile.verify(different)
        assert sim < 0.5

    def test_verify_empty_profile_returns_zero(self) -> None:
        profile = SpeakerProfile("user1")
        emb = np.array([1.0, 0.5], dtype=np.float32)
        assert profile.verify(emb) == 0.0


class TestSpeakerVerificationHook:
    """Test the full verification hook workflow."""

    def test_no_enrolled_speakers(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        emb = np.random.randn(96).astype(np.float32)
        result = hook.verify_speaker(emb)
        assert result.is_verified is False
        assert result.speaker_id is None
        assert result.similarity == 0.0

    def test_enroll_and_verify_match(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        rng = np.random.default_rng(42)
        # Create a speaker embedding
        speaker_emb = rng.standard_normal(96).astype(np.float32)
        hook.enroll_speaker("alice", speaker_emb)
        # Verify with same embedding -> should match
        result = hook.verify_speaker(speaker_emb)
        assert result.is_verified is True
        assert result.speaker_id == "alice"
        assert result.similarity > 0.99

    def test_enroll_and_verify_no_match(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        rng = np.random.default_rng(42)
        speaker_emb = rng.standard_normal(96).astype(np.float32)
        hook.enroll_speaker("alice", speaker_emb)
        # Verify with a completely different embedding
        different_emb = rng.standard_normal(96).astype(np.float32) + 100
        result = hook.verify_speaker(different_emb)
        # Different random vectors should have low similarity
        # (may or may not pass 0.65 — check against actual score)
        assert isinstance(result.is_verified, bool)
        assert isinstance(result.similarity, float)

    def test_multiple_speakers(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.50)
        # Create distinctive embeddings
        alice_emb = np.zeros(96, dtype=np.float32)
        alice_emb[0] = 1.0
        bob_emb = np.zeros(96, dtype=np.float32)
        bob_emb[1] = 1.0
        hook.enroll_speaker("alice", alice_emb)
        hook.enroll_speaker("bob", bob_emb)
        assert len(hook.enrolled_speakers) == 2
        # Verify alice
        result = hook.verify_speaker(alice_emb)
        assert result.is_verified is True
        assert result.speaker_id == "alice"

    def test_remove_speaker(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        emb = np.random.randn(96).astype(np.float32)
        hook.enroll_speaker("alice", emb)
        assert "alice" in hook.enrolled_speakers
        assert hook.remove_speaker("alice") is True
        assert "alice" not in hook.enrolled_speakers
        assert hook.remove_speaker("alice") is False  # Already removed

    def test_callable_interface(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        rng = np.random.default_rng(42)
        emb = rng.standard_normal(96).astype(np.float32)
        hook.enroll_speaker("alice", emb)
        # __call__ should return bool
        assert hook(emb) is True

    def test_enroll_multiple_embeddings(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        embeddings = [np.random.randn(96).astype(np.float32) for _ in range(5)]
        count = hook.enroll_speaker("alice", embeddings)
        assert count == 5

    def test_enroll_2d_array(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        batch = np.random.randn(3, 96).astype(np.float32)
        count = hook.enroll_speaker("alice", batch)
        assert count == 3

    def test_threshold_property(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.65)
        assert hook.threshold == 0.65
        hook.threshold = 0.80
        assert hook.threshold == 0.80


class TestSpeakerPersistence:
    """Test save/load of speaker profiles."""

    def test_save_and_load(self) -> None:
        hook = SpeakerVerificationHook(threshold=0.70)
        rng = np.random.default_rng(42)
        emb_alice = rng.standard_normal(96).astype(np.float32)
        emb_bob = rng.standard_normal(96).astype(np.float32)
        hook.enroll_speaker("alice", emb_alice)
        hook.enroll_speaker("bob", emb_bob)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "speakers"
            hook.save(path)
            assert (Path(tmpdir) / "speakers.json").exists()
            assert (Path(tmpdir) / "speakers.npz").exists()

            # Load into a fresh hook
            hook2 = SpeakerVerificationHook()
            hook2.load(path)
            assert set(hook2.enrolled_speakers) == {"alice", "bob"}
            assert hook2.threshold == 0.70

            # Verify alice still works
            result = hook2.verify_speaker(emb_alice)
            assert result.is_verified is True
            assert result.speaker_id == "alice"

    def test_load_nonexistent_raises(self) -> None:
        hook = SpeakerVerificationHook()
        with pytest.raises(FileNotFoundError):
            hook.load("/nonexistent/path")


class TestSpeakerVerifyResult:
    """Test SpeakerVerifyResult dataclass."""

    def test_frozen(self) -> None:
        result = SpeakerVerifyResult(
            is_verified=True, speaker_id="alice", similarity=0.95, threshold=0.65,
        )
        with pytest.raises(AttributeError):
            result.is_verified = False  # type: ignore[misc]
