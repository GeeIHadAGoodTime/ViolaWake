"""Unit tests for WakeDecisionPolicy (4-gate detection pipeline).

These tests verify the decision logic without requiring model files or hardware.
All test cases use synthetic scores and do not touch ONNX Runtime.
"""

from __future__ import annotations

import time

import pytest

from violawake_sdk.wake_detector import WakeDecisionPolicy


# ──────────────────────────────────────────────────────────────────────────────
# Table-driven gate tests
# ──────────────────────────────────────────────────────────────────────────────

class TestDecisionPolicyGates:
    """Test each gate independently and in combination."""

    @pytest.mark.parametrize("score,rms,is_playing,in_cooldown,expected,description", [
        # Normal detections (all gates pass)
        (0.90, 500.0, False, False, True,  "normal detection above threshold"),
        (0.80, 500.0, False, False, True,  "detection at exactly threshold"),
        (1.00, 500.0, False, False, True,  "maximum score"),

        # Gate 1: Zero-input guard
        (0.95, 0.0,   False, False, False, "RMS=0 (complete silence/DC artifact)"),
        (0.95, 0.5,   False, False, False, "RMS below floor (0.5 < 1.0)"),
        (0.95, 1.0,   False, False, True,  "RMS exactly at floor — should pass"),

        # Gate 2: Score threshold
        (0.79, 500.0, False, False, False, "just below threshold"),
        (0.50, 500.0, False, False, False, "far below threshold"),
        (0.00, 500.0, False, False, False, "minimum score"),

        # Gate 3: Cooldown (tested via in_cooldown parameter)
        # Note: in real usage, cooldown is time-based. See test_cooldown_timing below.

        # Gate 4: Playback gate
        (0.95, 500.0, True,  False, False, "playback active — suppress detection"),
        (0.95, 500.0, False, False, True,  "playback NOT active — allow detection"),
    ])
    def test_gate_combinations(
        self,
        score: float,
        rms: float,
        is_playing: bool,
        in_cooldown: bool,
        expected: bool,
        description: str,
    ) -> None:
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=2.0, rms_floor=1.0)

        # Pre-set cooldown if needed
        if in_cooldown:
            policy._last_detection = time.monotonic()  # just now = in cooldown

        result = policy.evaluate(score=score, rms=rms, is_playing=is_playing)
        assert result == expected, f"FAILED: {description} (score={score}, rms={rms}, playing={is_playing})"

    def test_threshold_boundary_below(self) -> None:
        """Score just below threshold should not detect."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        assert policy.evaluate(score=0.799, rms=500.0) is False

    def test_threshold_boundary_above(self) -> None:
        """Score just above threshold should detect."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        assert policy.evaluate(score=0.801, rms=500.0) is True

    def test_custom_threshold(self) -> None:
        """Custom threshold should override default 0.80."""
        policy = WakeDecisionPolicy(threshold=0.70, cooldown_s=0.0)
        assert policy.evaluate(score=0.75, rms=500.0) is True

        policy_strict = WakeDecisionPolicy(threshold=0.90, cooldown_s=0.0)
        assert policy_strict.evaluate(score=0.85, rms=500.0) is False


class TestDecisionPolicyCooldown:
    """Test cooldown timing behavior."""

    def test_no_repeat_within_cooldown(self) -> None:
        """Second detection within cooldown window should be suppressed."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=2.0)
        # First detection
        assert policy.evaluate(score=0.95, rms=500.0) is True
        # Immediate second detection — in cooldown
        assert policy.evaluate(score=0.95, rms=500.0) is False

    def test_detection_after_cooldown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Detection should succeed after cooldown window expires."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=1.0)

        # First detection at t=0
        assert policy.evaluate(score=0.95, rms=500.0) is True

        # Simulate time passing beyond cooldown (monkeypatch time.monotonic)
        fake_now = time.monotonic() + 2.0
        monkeypatch.setattr(
            "violawake_sdk.wake_detector.time.monotonic",
            lambda: fake_now,
        )

        # Should detect again
        assert policy.evaluate(score=0.95, rms=500.0) is True

    def test_reset_cooldown(self) -> None:
        """reset_cooldown() should immediately allow the next detection."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=60.0)
        assert policy.evaluate(score=0.95, rms=500.0) is True
        # In cooldown — blocked
        assert policy.evaluate(score=0.95, rms=500.0) is False
        # Reset
        policy.reset_cooldown()
        # Should detect again
        assert policy.evaluate(score=0.95, rms=500.0) is True

    def test_zero_cooldown(self) -> None:
        """Zero cooldown should allow back-to-back detections."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        assert policy.evaluate(score=0.95, rms=500.0) is True
        assert policy.evaluate(score=0.95, rms=500.0) is True


class TestDecisionPolicyRMSFloor:
    """Test zero-input guard (Gate 1)."""

    def test_default_rms_floor(self) -> None:
        """Default RMS floor is 1.0."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        assert policy.rms_floor == 1.0

    def test_custom_rms_floor(self) -> None:
        """Custom RMS floor should be respected."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0, rms_floor=50.0)
        # Below floor
        assert policy.evaluate(score=0.95, rms=49.9) is False
        # At floor
        assert policy.evaluate(score=0.95, rms=50.0) is True

    def test_rms_default_parameter(self) -> None:
        """Default RMS parameter (100.0) should pass the floor."""
        policy = WakeDecisionPolicy(threshold=0.80, cooldown_s=0.0)
        # Using default rms=100.0 (non-silent)
        assert policy.evaluate(score=0.95) is True
