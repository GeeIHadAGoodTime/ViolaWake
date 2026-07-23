"""Ratchet: the training quality gate's silence subgrade must be measured from
several independent near-silence probes (aggregated by MEDIAN), never a single
fixed realization's max window score.

Guards the #1775 residual fix (post-#1465, CL-20260714-4c23 / #1184 line of
investigation): #1465 fixed the C->F cliff's BAR (a hardcoded 0.50 disconnected
from the 0.80 deployment threshold). It did not fix the STATISTIC measured
against that bar -- pre-this-fix, the silence subgrade came from exactly ONE
fixed 10s near-silence clip (numpy seed=42), scored as the single worst
(max) sliding window over that one clip. That makes the subgrade a one-shot
test: a model with an idiosyncratic peak response to that ONE specific noise
realization was graded identically to a model that is systematically prone to
firing on quiet backgrounds, even though only the latter is a real deployment
risk. The historical grade-F reads (CL-20260714-4c23, n=20) already showed the
same wake word passing on one retrain and failing on the next from exactly
this kind of run-to-run variance landing on either side of a single-sample
test -- consistent with a one-shot oracle, not a robust one.

The fix scores _NEAR_SILENCE_PROBE_COUNT independent realizations and
aggregates with the MEDIAN (see _aggregate_silence_probe_scores), not the max
of one clip. Median is deliberately not a relaxation: a model that is
genuinely risky keeps failing because most/all independent probes trigger it
(true-positive catch rate for systematic risk is unchanged); only a spike
confined to a minority of probes -- the one-shot-idiosyncrasy failure mode --
stops being sufficient on its own to force grade F.

These tests RED on the pre-fix shape (a single-probe max, i.e.
``_aggregate_silence_probe_scores`` behaving like ``max()`` on one score, or
``_NEAR_SILENCE_PROBE_COUNT`` collapsed back to 1) and GREEN on the
median-of-independent-probes implementation.
"""

from __future__ import annotations

from violawake_sdk._constants import DEFAULT_THRESHOLD
from violawake_sdk.tools.train import (
    _NEAR_SILENCE_PROBE_COUNT,
    _NEAR_SILENCE_PROBE_SEED_BASE,
    _aggregate_silence_probe_scores,
    _grade_quality,
)


def test_probe_count_is_more_than_one() -> None:
    """The whole point of the fix is >1 independent probe, aggregated robustly.

    RED on the pre-fix shape (_NEAR_SILENCE_PROBE_COUNT == 1, a single clip).
    """
    assert _NEAR_SILENCE_PROBE_COUNT >= 3
    # Odd count keeps the median well-defined as an actual observed score,
    # not an average of two middle values.
    assert _NEAR_SILENCE_PROBE_COUNT % 2 == 1


def test_seed_42_is_still_the_first_probe() -> None:
    """Historical single-probe measurements (seed=42) stay comparable.

    All CL-20260714-4c23 / OBSERVED_FAIL_SILENCE historical scores were taken
    from the seed=42 clip; the fix must not discard that continuity.
    """
    assert _NEAR_SILENCE_PROBE_SEED_BASE == 42


def test_single_idiosyncratic_spike_does_not_force_the_aggregate_above_threshold() -> None:
    """A model that spikes on exactly ONE of several independent probes must
    NOT have that spike alone reflected as the aggregate silence score.

    RED on the pre-fix behavior: the pre-fix subgrade WAS that single probe's
    max score (there was only one probe), so an idiosyncratic 0.95 spike on
    that one realization directly became the reported (and graded) score.
    GREEN on the fix: with several probes, one outlier spike is outvoted by
    the rest and the median stays well below the deployment threshold.
    """
    t = DEFAULT_THRESHOLD  # 0.80
    # One clip spikes above threshold; every other independent probe is calm.
    per_clip_scores = [0.95, 0.05, 0.06, 0.04, 0.05]
    aggregated = _aggregate_silence_probe_scores(per_clip_scores)

    # The pre-fix statistic (max of the single available score) would have
    # been 0.95 -- at/above threshold, forcing grade F. The fixed statistic
    # must NOT equal the lone spike, and must sit below the deployment
    # threshold so this model is not wrongly failed on silence grounds.
    assert aggregated != max(per_clip_scores)
    assert aggregated < t
    assert _grade_quality(
        speech_fp_rate=0.0, confusable_fp_rate=0.0, silence_max_score=aggregated, deployment_threshold=t
    ) != "F"


def test_systematically_risky_model_still_fails() -> None:
    """A model that spikes across MOST/ALL independent probes must still be
    caught -- the fix must not blunt true-positive detection of real risk.

    GREEN both pre- and post-fix in spirit (systematic risk was always
    caught), but this proves the new aggregation preserves that catch rate:
    the median of a consistently-high score set is itself high.
    """
    t = DEFAULT_THRESHOLD  # 0.80
    per_clip_scores = [0.90, 0.88, 0.93, 0.85, 0.91]
    aggregated = _aggregate_silence_probe_scores(per_clip_scores)

    assert aggregated >= t
    assert _grade_quality(
        speech_fp_rate=0.0, confusable_fp_rate=0.0, silence_max_score=aggregated, deployment_threshold=t
    ) == "F"


def test_aggregate_is_the_median_not_the_max_or_mean() -> None:
    """Pins the exact statistic so a future edit can't silently swap it back
    to max() (the pre-fix, one-shot-vulnerable behavior) without this test
    turning RED.
    """
    per_clip_scores = [0.10, 0.20, 0.30, 0.40, 0.90]
    aggregated = _aggregate_silence_probe_scores(per_clip_scores)

    assert aggregated == 0.30  # median of the 5 values
    assert aggregated != max(per_clip_scores)
    assert aggregated != sum(per_clip_scores) / len(per_clip_scores)  # not the mean either


def test_empty_probe_list_keeps_the_conservative_safety_floor() -> None:
    """No silence-class input scored at all -> still force the 1.0 safety
    floor (unchanged from the pre-fix "no silence-class input could be
    scored" fallback in _run_quality_gate).
    """
    assert _aggregate_silence_probe_scores([]) == 1.0
    assert (
        _grade_quality(
            speech_fp_rate=0.0,
            confusable_fp_rate=0.0,
            silence_max_score=_aggregate_silence_probe_scores([]),
            deployment_threshold=DEFAULT_THRESHOLD,
        )
        == "F"
    )
