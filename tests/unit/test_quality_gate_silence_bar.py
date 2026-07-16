"""Ratchet: the training quality gate's silence subgrade must derive its bar from
the deployment threshold, never a hardcoded constant.

Guards the #1465 fix (CL-20260714-4c23 / #1184): the silence C->F cliff was a
hardcoded 0.50, disconnected from the 0.80 deployment threshold, which blocked ~75%
of real models whose near-silence score sat below the threshold (unable to fire at
deployment) for run-to-run training variance rather than real false-fire risk.

These tests are written to RED on the pre-fix shape (silence cliff hardcoded at
0.50, or any bar that does not move with the deployment threshold) and GREEN on the
derived-from-threshold implementation.
"""

from __future__ import annotations

import pytest

from violawake_sdk._constants import DEFAULT_THRESHOLD
from violawake_sdk.tools.train import _grade_quality

CLEAN = dict(speech_fp_rate=0.0, confusable_fp_rate=0.0)  # noqa: C408

# The 15 grade-F failing silence_max_scores observed on the box (2026-07-12..14,
# jobs 71-90), from the gate's own eval logs. Post-fix, only those at/above the
# 0.80 deployment threshold should still fail; the rest are real models that cannot
# fire at deployment and should pass (grade C).
OBSERVED_FAIL_SILENCE = [
    0.53, 0.54, 0.55, 0.56, 0.63, 0.63, 0.66, 0.68,
    0.76, 0.78, 0.81, 0.81, 0.90, 0.92, 0.95,
]


def test_silence_cliff_is_the_deployment_threshold_not_hardcoded_050() -> None:
    """The C->F silence cliff is the deployment threshold (0.80), not 0.50.

    RED on the old hardcoded-0.50 grader (which returned F for silence 0.60);
    GREEN on the derived bar.
    """
    t = DEFAULT_THRESHOLD  # 0.80
    # A near-silence score below the deployment threshold cannot false-fire at
    # deployment, so it must NOT force grade F. The pre-fix grader returned "F"
    # here (0.60 >= 0.50); the fixed grader returns "C".
    assert _grade_quality(silence_max_score=0.60, deployment_threshold=t, **CLEAN) == "C"
    assert _grade_quality(silence_max_score=0.79, deployment_threshold=t, **CLEAN) == "C"
    # At/above the deployment threshold it WOULD false-fire => grade F.
    assert _grade_quality(silence_max_score=t, deployment_threshold=t, **CLEAN) == "F"
    assert _grade_quality(silence_max_score=0.81, deployment_threshold=t, **CLEAN) == "F"

    # Founder projection on the real observed failures: only silence >= 0.80 stays F.
    graded = [_grade_quality(silence_max_score=s, deployment_threshold=t, **CLEAN)
              for s in OBSERVED_FAIL_SILENCE]
    still_f = [s for s, g in zip(OBSERVED_FAIL_SILENCE, graded) if g == "F"]
    assert still_f == [0.81, 0.81, 0.90, 0.92, 0.95]
    assert all(g == "C" for s, g in zip(OBSERVED_FAIL_SILENCE, graded) if s < t)


def test_silence_bar_tracks_the_deployment_threshold() -> None:
    """Every silence bar (A/B/C-cliff) moves with the deployment threshold.

    This reds on ANY hardcoded silence bar (the old 0.50, or a naive replacement
    that hardcodes 0.80): the outcome must flip when the threshold changes.
    """
    # C->F cliff tracks the threshold in BOTH directions.
    assert _grade_quality(silence_max_score=0.70, deployment_threshold=0.60, **CLEAN) == "F"
    assert _grade_quality(silence_max_score=0.55, deployment_threshold=0.60, **CLEAN) == "C"
    # 0.55 fails at threshold 0.50 but passes at 0.80 -> proves the bar is derived,
    # not any fixed constant.
    assert _grade_quality(silence_max_score=0.55, deployment_threshold=0.50, **CLEAN) == "F"
    assert _grade_quality(silence_max_score=0.55, deployment_threshold=0.80, **CLEAN) == "C"

    # A/B safety-margin tiers are derived too (0.25 / 0.375 of the threshold).
    # At threshold 0.80: A<0.20, B<0.30.
    assert _grade_quality(silence_max_score=0.19, deployment_threshold=0.80, **CLEAN) == "A"
    assert _grade_quality(silence_max_score=0.25, deployment_threshold=0.80, **CLEAN) == "B"
    # At threshold 0.40 the same absolute scores drop a tier (A<0.10, B<0.15, C<0.40).
    assert _grade_quality(silence_max_score=0.19, deployment_threshold=0.40, **CLEAN) == "C"
    assert _grade_quality(silence_max_score=0.08, deployment_threshold=0.40, **CLEAN) == "A"


@pytest.mark.parametrize("bad_axis", ["speech", "confusable"])
def test_speech_and_confusable_axes_still_gate(bad_axis: str) -> None:
    """The non-silence axes are unchanged: a clean-silence model that false-fires
    on speech/confusables still fails, so relaxing the silence bar did not open a
    hole on the real false-fire axes."""
    kw = dict(silence_max_score=0.0, deployment_threshold=DEFAULT_THRESHOLD,  # noqa: C408
              speech_fp_rate=0.0, confusable_fp_rate=0.0)
    kw[f"{bad_axis}_fp_rate"] = 0.5  # 50% false-positive rate on that axis
    assert _grade_quality(**kw) == "F"
