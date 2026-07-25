"""Ratchet: the training quality gate's silence subgrade must be a false-fire RATE
measured on real no-wake audio through the runtime path -- never a single max draw
against a score bar.

Supersedes the #1465 shape (silence cliff derived from the deployment threshold).
That fix was correct as far as it went -- it removed a hardcoded 0.50 bar -- but the
subgrade underneath it was still invalid, and production kept failing real users:
12 of the 21 training jobs run after the #1465 fix deployed still failed, every one
of them on this subgrade alone, with speech FP and confusable FP both 0.0%
(wakeword-backend-1 job_queue.db + container eval logs, read 2026-07-24, #2611).

Root cause the tests below lock down (reproduced on the box against six real
deployed models, src/violawake_sdk/tools/train.py):

  1. n=1. The probe was ONE fixed-seed (42) clip, center-cropped from 10s to 1.5s
     by _prepare_audio_for_oww, yielding silence_window_count == 1 -- the
     advertised "max over windows" was a single forward pass. Across alternative
     probe draws, models that had PASSED scored F on 10-37% of them.
  2. Unphysical input. The probe was white noise at float RMS 1e-4 => int16 RMS
     3.29. Real recorded room tone measures int16 RMS 224-3782, and the runtime's
     own RMS floor comment puts speech at 500-5000. The gate was scoring a regime
     ~100-1000x quieter than any microphone produces, where model output is
     arbitrary.
  3. Wrong path. It scored in batch mode while the runtime streams; measured
     divergence up to 0.368 on the same audio (#1487).

These tests RED on that shape (a grader keyed on a single silence max score) and
GREEN on the rate-based grader.
"""

from __future__ import annotations

import inspect

import pytest

from violawake_sdk._constants import DEFAULT_THRESHOLD
from violawake_sdk.tools.train import _grade_quality

CLEAN = dict(speech_fp_rate=0.0, confusable_fp_rate=0.0)  # noqa: C408


def test_silence_subgrade_is_a_rate_not_a_single_max_score() -> None:
    """The silence axis is graded on a false-fire RATE.

    REDs on the pre-fix grader, whose silence parameter was a max score compared
    against the deployment threshold: under that shape a rate of 0.0 (a perfectly
    clean model) and a rate of 0.9 (a model that fires on nine of ten room-tone
    windows) both sit below 0.80 and both graded "A".
    """
    sig = inspect.signature(_grade_quality)
    assert "silence_fp_rate" in sig.parameters, (
        "the silence subgrade must be expressed as a false-fire rate; a "
        "silence_max_score parameter is the pre-fix single-draw shape"
    )
    assert "silence_max_score" not in sig.parameters

    t = DEFAULT_THRESHOLD
    # A model that fires on 90% of real room-tone windows is broken and must fail.
    # The pre-fix grader read 0.90 as a score below... nothing, and graded it "A".
    assert _grade_quality(silence_fp_rate=0.90, deployment_threshold=t, **CLEAN) == "F"
    assert _grade_quality(silence_fp_rate=0.10, deployment_threshold=t, **CLEAN) == "F"
    # A clean model passes at the top tier.
    assert _grade_quality(silence_fp_rate=0.0, deployment_threshold=t, **CLEAN) == "A"


def test_silence_rate_tiers_match_the_speech_subgrade() -> None:
    """Silence uses the same rate tiers as speech, so all three axes answer one
    question: how often would this model fire on no-wake audio?"""
    t = DEFAULT_THRESHOLD
    for rate, expected in [(0.019, "A"), (0.02, "B"), (0.049, "B"), (0.05, "C"),
                           (0.099, "C"), (0.10, "F")]:
        assert _grade_quality(silence_fp_rate=rate, deployment_threshold=t, **CLEAN) == expected, (
            f"silence rate {rate} should grade {expected}"
        )
        # The speech axis, held at the same rate with silence clean, agrees.
        assert _grade_quality(
            speech_fp_rate=rate, confusable_fp_rate=0.0, silence_fp_rate=0.0,
            deployment_threshold=t,
        ) == expected


def test_unmeasurable_silence_does_not_fail_the_model() -> None:
    """When no room tone could be extracted, the model is graded on the axes we DID
    measure, not failed for our own missing measurement.

    The pre-fix code set silence_max_score = 1.0 and forced grade F in this case
    ("conservative: force Grade F"), which charged the user for our gap. Genuinely
    quiet input is independently rejected at runtime by the RMS floor
    (wake_detector.py Gate 1), and the speech/confusable axes still gate below.
    """
    t = DEFAULT_THRESHOLD
    assert _grade_quality(silence_fp_rate=None, deployment_threshold=t, **CLEAN) == "A"
    # ...but an unmeasurable silence axis does NOT excuse the other axes.
    assert _grade_quality(
        speech_fp_rate=0.5, confusable_fp_rate=0.0, silence_fp_rate=None,
        deployment_threshold=t,
    ) == "F"


def test_no_silence_score_bar_is_hardcoded() -> None:
    """Carried forward from the #1465 ratchet: no silence bar may be a hardcoded
    score constant (the old 0.50, or a naive replacement hardcoding 0.80).

    The rate tiers are thresholds on a RATE, and every rate is computed AT the
    deployment threshold by the caller, so changing the deployment threshold
    changes which windows count as firing -- the bar still tracks the threshold it
    protects, without any score constant living in the grader.
    """
    src = inspect.getsource(_grade_quality)
    for banned in ("0.50", "0.5 *", "0.80 *", "0.375 *", "0.25 *"):
        assert banned not in src, f"hardcoded silence score bar {banned!r} in _grade_quality"


@pytest.mark.parametrize("bad_axis", ["speech", "confusable"])
def test_speech_and_confusable_axes_still_gate(bad_axis: str) -> None:
    """The non-silence axes are unchanged: a clean-silence model that false-fires on
    speech/confusables still fails, so reworking the silence subgrade did not open a
    hole on the other false-fire axes."""
    kw = dict(silence_fp_rate=0.0, deployment_threshold=DEFAULT_THRESHOLD,  # noqa: C408
              speech_fp_rate=0.0, confusable_fp_rate=0.0)
    kw[f"{bad_axis}_fp_rate"] = 0.5  # 50% false-positive rate on that axis
    assert _grade_quality(**kw) == "F"
