"""The silence subgrade must not report a number its measurement cannot support.

Two independent review findings on the room-tone rework (GeeIHadAGoodTime/Viola#1487,
comment 5121584192) name the same disease from opposite ends: the gate printing a
confident-looking silence figure that no measurement stands behind.

  1. **A missing measurement graded as a clean one.** ``_extract_room_tone`` returns
     None whenever a recording holds no extractable quiet segment -- a tightly
     trimmed clip, or a room noisy enough that nothing sits below the spoken peak.
     The first draft mapped that to a 0.0 rate, so a model nobody had scored on the
     silence axis graded "A". The retired synthetic probe was unphysical but it was
     *always there*; room tone from user recordings is physical but sometimes
     absent, which turned a universal safety net into a silent one.

  2. **A rate over samples that are not independent.** The streaming extractor
     slides one 80ms embedding at a time (``OWW_CHUNK_SAMPLES`` = 1280 samples) over
     a ``seq_len``-embedding window, so at the production ``seq_len`` of 9 adjacent
     windows share 8 of 9 frames. Dividing firings by that count claims a precision
     the sample has not got, and lets one sustained false-fire burst enter the
     numerator dozens of times.

Both are fixed here, and both tests RED on the pre-fix shapes: a grader that reads
None as clean, and a rate taken over the raw self-overlapping window count.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from violawake_sdk.tools.train import (
    _ROOM_TONE_MIN_SAMPLES,
    _RUNTIME_RMS_FLOOR,
    _SILENCE_MIN_INDEPENDENT_WINDOWS,
    _SYNTHETIC_ROOM_TONE_RMS_I16,
    _decimate_to_independent_windows,
    _extract_room_tone,
    _grade_quality,
    _int16_rms,
    _run_quality_gate,
    _silence_subgrade,
    _synthetic_room_tone,
)

SR = 16000
CLEAN = dict(speech_fp_rate=0.0, confusable_fp_rate=0.0)  # noqa: C408


# ---------------------------------------------------------------------------
# Finding 1 -- a missing measurement must fail closed, and must be rare
# ---------------------------------------------------------------------------


def test_an_unscored_silence_axis_cannot_grade_as_clean() -> None:
    """The grader refuses to clear a model on an axis it has no number for.

    REDs on the pre-fix grader body, which opened with
    ``silence = 0.0 if silence_fp_rate is None else silence_fp_rate``.
    """
    assert _grade_quality(silence_fp_rate=None, deployment_threshold=0.80, **CLEAN) == "F"
    src = inspect.getsource(_grade_quality)
    assert "0.0 if silence_fp_rate is None" not in src, (
        "an unmeasured silence axis must not be coerced to a clean 0.0 rate"
    )


def test_the_quality_gate_falls_back_to_a_probe_rather_than_failing_the_user() -> None:
    """Fail-closed is only fair if the gate first tries hard not to reach it.

    How somebody trimmed their recordings is not a property of their model, so it
    must decide the grade in neither direction: not "A" via a missing measurement,
    and not "F" via a punished one. ``_run_quality_gate`` therefore measures the
    same axis on a synthetic probe at a real room-tone level when the user's own
    audio cannot power it, and only the failure of THAT reaches the fail-closed
    branch.
    """
    src = inspect.getsource(_run_quality_gate)
    assert "_synthetic_room_tone()" in src, (
        "the gate must fall back to a real-level probe before failing the axis closed"
    )
    # The fallback is triggered by an under-powered sample, not only an empty one.
    assert "_SILENCE_MIN_INDEPENDENT_WINDOWS" in src
    # Short per-clip yields are pooled, so the fallback is a last resort rather than
    # the common path for anyone who records tightly.
    assert "pooled_samples" in src


def test_the_fallback_probe_sits_at_a_real_room_tone_level() -> None:
    """The fallback must not smuggle back the regime that caused all of this.

    The retired probe sat at int16 RMS 3.288, which a measured amplitude sweep
    (Viola#2611 comment 5121856038) later showed to be the MAXIMUM of the streaming
    false-fire curve for the released temporal_cnn -- the gate evaluated every model
    at close to its worst point. Real recorded room tone measures int16 RMS
    224-3782. A fallback probe is only legitimate inside that band.
    """
    probe = _synthetic_room_tone()
    rms = _int16_rms(probe)
    assert 224.0 <= rms <= 3782.0, (
        f"fallback probe at int16 RMS {rms:.1f} is outside the measured real "
        "room-tone band 224-3782"
    )
    assert rms > 100.0 * _RUNTIME_RMS_FLOOR, "probe must be audio the runtime scores"
    assert abs(rms - _SYNTHETIC_ROOM_TONE_RMS_I16) < 1.0, "probe level must be exact, not a draw"
    # Long enough to power the rate at the production seq_len of 9: >= 12 independent
    # windows needs >= 12 * 9 embeddings = 108 * 80ms = 8.64s.
    assert len(probe) >= SR * 9, "fallback probe too short to support an independent-window rate"


def test_short_room_tone_yields_are_pooled_across_recordings() -> None:
    """A 0.9s yield is useless alone and useful five times over.

    ``_extract_room_tone`` takes the floor as a parameter precisely so the caller can
    apply it to the pooled total. REDs on a hardcoded per-clip floor.
    """
    rng = np.random.default_rng(11)
    quiet = rng.standard_normal(SR).astype(np.float32) * (300.0 / 32767.0)
    loud = rng.standard_normal(SR // 2).astype(np.float32) * (3000.0 / 32767.0)
    tight_clip = np.concatenate([quiet, loud])  # 1.5s: ~0.9s of usable room tone

    assert _extract_room_tone(tight_clip) is None, "per-clip floor still rejects it alone"
    partial = _extract_room_tone(tight_clip, min_samples=4800)
    assert partial is not None, "the caller must be able to recover a short yield"
    assert len(partial) < _ROOM_TONE_MIN_SAMPLES
    # Five such clips clear the pooled floor that one cannot.
    assert 5 * len(partial) >= _ROOM_TONE_MIN_SAMPLES


# ---------------------------------------------------------------------------
# Finding 2 -- the rate must be taken over independent samples
# ---------------------------------------------------------------------------


def test_overlapping_windows_are_decimated_to_independent_samples() -> None:
    """Adjacent stream windows share seq_len-1 frames, so only every seq_len-th
    window is an independent look. REDs on any code that rates the raw count."""
    seq_len = 9
    scores = np.arange(27, dtype=np.float32)  # one clip, 27 overlapping windows
    kept = _decimate_to_independent_windows(scores, [0] * 27, seq_len)
    assert kept.tolist() == [0.0, 9.0, 18.0], "must keep exactly the zero-overlap subset"
    assert len(kept) == 3


def test_decimation_is_per_clip_so_clips_do_not_absorb_each_other() -> None:
    """Two clips of 9 windows are two independent looks, not one.

    A global stride over the concatenated stream would keep window 0 of clip A and
    then window 0 of clip B only by luck of alignment; decimating per source clip
    keeps the first window of every clip.
    """
    seq_len = 9
    scores = np.array([1.0] * 9 + [2.0] * 9, dtype=np.float32)
    sources = [0] * 9 + [1] * 9
    kept = _decimate_to_independent_windows(scores, sources, seq_len)
    assert sorted(kept.tolist()) == [1.0, 2.0], "each clip contributes its own first look"


def test_the_denominator_becomes_the_real_sample_size_without_moving_the_estimate() -> None:
    """What decimation fixes is the CLAIMED PRECISION, not the point estimate.

    A single 0.8s burst inside ~10s of room tone spans ~10 of the 80ms-hop windows.
    Rated raw that reads "10 firings in 117 samples"; there are not 117 samples,
    there are 13 independent looks at the audio, and the same event is ~1 of those.
    Both forms describe the event similarly -- which is the point: decimation shrinks
    numerator and denominator together, so it is NOT a silent re-calibration of the
    A/B/C rate bars. Measured over 4000 randomised burst layouts, mean raw rate
    0.1171 vs mean independent rate 0.1138 (bias 0.3 percentage points, r = 0.94).

    So this test pins both halves: the estimate stays put, and the reported sample
    size stops being ~9x the truth.
    """
    seq_len = 9
    n = 117
    scores = np.zeros(n, dtype=np.float32)
    scores[40:50] = 0.95  # one sustained burst above a 0.80 threshold
    sources = [0] * n

    raw_rate = float((scores >= 0.80).mean())
    kept = _decimate_to_independent_windows(scores, sources, seq_len)
    independent_rate = float((kept >= 0.80).mean())

    assert len(kept) == 13, "117 overlapping windows are 13 independent looks at seq_len 9"
    assert raw_rate == 10 / 117
    # One event contributes about one independent firing, not ten.
    assert (kept >= 0.80).sum() == 1
    # The estimate is preserved to within the granularity one independent look buys,
    # so the grade bars keep meaning what they meant.
    assert abs(independent_rate - raw_rate) <= 3.0 / len(kept)


def test_decimation_does_not_re_calibrate_the_grade_bars() -> None:
    """Swept across randomised burst layouts, the two forms agree in aggregate.

    This is the guard against a future 'simplification' that de-correlates by
    collapsing bursts to events or by switching to a per-cooldown-slot statistic:
    either would be defensible in isolation but would move every model's silence
    number relative to the shared 2%/5%/10% tiers, which is a grading-policy change
    and not a statistics fix.
    """
    seq_len = 9
    rng = np.random.default_rng(0)
    raw_rates: list[float] = []
    independent_rates: list[float] = []
    for _ in range(400):
        n = int(rng.integers(60, 400))
        scores = np.zeros(n, dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            start = int(rng.integers(0, max(1, n - 15)))
            scores[start : start + int(rng.integers(3, 20))] = 0.95
        kept = _decimate_to_independent_windows(scores, [0] * n, seq_len)
        raw_rates.append(float((scores >= 0.80).mean()))
        independent_rates.append(float((kept >= 0.80).mean()))

    bias = float(np.mean(independent_rates)) - float(np.mean(raw_rates))
    assert abs(bias) < 0.02, (
        f"decimation shifted the mean rate by {bias:+.4f} -- that is a re-calibration"
    )


def test_the_gate_still_decimates_to_measure_its_own_statistical_power() -> None:
    """Wiring: the independent subset must still be computed, and still gate power.

    Decimation did not stop mattering when the rate moved off it (see
    ``test_a_crossing_max_can_never_coexist_with_a_zero_rate``). It is what makes
    ``silence_window_count`` an honest count of independent looks, which is the
    only thing that can answer "was this measurement powered at all". Deleting it
    along with the old rate population would silently let a 3-look sample grade a
    customer's model.
    """
    src = inspect.getsource(_run_quality_gate)
    assert "_decimate_to_independent_windows(raw, source_indices, seq_len)" in src
    assert '"silence_window_count": silence_window_count' in src
    # ...and the stream denominator is recorded beside it, so a 0.0% that means
    # "no crossing anywhere" is distinguishable from one that only means
    # "the denominator was too coarse to show a crossing".
    assert '"silence_stream_window_count": silence_stream_window_count' in src


def test_an_underpowered_sample_is_not_reported_as_a_rate() -> None:
    """A rate over three looks cannot resolve a 2%/5%/10% bar.

    The gate requires a minimum number of INDEPENDENT windows before it will call
    the axis measured, so a coarse sample becomes either a fallback-probe
    measurement or a fail-closed verdict -- never a falsely precise percentage.
    Asserted behaviourally on ``_silence_subgrade``: a stream long enough to look
    convincing does not rescue a sample with too few independent looks in it.
    """
    assert _SILENCE_MIN_INDEPENDENT_WINDOWS >= 12, (
        "fewer than ~12 independent looks cannot resolve the 10% grade-F bar"
    )
    underpowered = np.full(_SILENCE_MIN_INDEPENDENT_WINDOWS - 1, 0.05, dtype=np.float32)
    long_stream = np.full(400, 0.05, dtype=np.float32)

    rate, _max_score, independent_count, stream_count = _silence_subgrade(
        underpowered, long_stream, 0.80
    )

    assert rate is None, "an under-powered sample must fail closed, not report a rate"
    assert independent_count == _SILENCE_MIN_INDEPENDENT_WINDOWS - 1
    assert stream_count == 400


def test_decimation_cannot_preserve_a_max_only_a_rate() -> None:
    """The mechanism behind the reported-max defect, on the real decimator.

    Decimation is a sampling step. It shrinks numerator and denominator together,
    so it preserves a RATE in expectation -- which is exactly why the rate is
    taken over it. A MAX has no such property: dropping 8 of every 9 windows can
    only ever discard the worst one, so a max read off the decimated subset is
    biased low by construction and never high.
    """
    seq_len = 9
    n = 367  # the real window count from a 30s synthetic room-tone probe
    raw = np.full(n, 0.05, dtype=np.float32)
    # Put the worst window at an index decimation drops (kept indices are 0 % 9).
    worst_index = 4
    assert worst_index % seq_len != 0
    raw[worst_index] = 0.79  # just under the 0.80 deployment threshold

    kept = _decimate_to_independent_windows(raw, [0] * n, seq_len)

    assert float(raw.max()) == pytest.approx(0.79, abs=1e-6)
    assert float(kept.max()) == pytest.approx(0.05, abs=1e-6), (
        "the decimated subset dropped the model's worst no-wake window"
    )
    assert float(kept.max()) < float(raw.max())


def test_the_reported_silence_max_is_the_worst_streamed_window() -> None:
    """The operator-facing max must come from the full stream, not the subset.

    ``_run_quality_gate`` prints this number directly beside
    ``threshold=0.80``, so it reads as "the model's worst score on no-wake
    audio versus the bar it must stay under". Measured on the released
    temporal_cnn against this gate's own synthetic room-tone probe (#2611,
    2026-07-30): raw streaming max 0.29-0.50 vs decimated max 0.09-0.16 over
    the same audio -- a 2-4x understatement of the model's real worst case.

    REDs on the pre-fix shape ``silence_max_score = float(silence_window_scores.max())``,
    which read the max off the decimated subset that
    ``test_decimation_cannot_preserve_a_max_only_a_rate`` proves is biased low.
    """
    gate_src = inspect.getsource(_run_quality_gate)
    subgrade_src = inspect.getsource(_silence_subgrade)

    assert "silence_max_score = float(silence_window_scores.max())" not in gate_src, (
        "the reported max is being read off the decimated subset again"
    )
    # The streaming scorer hands back the full stream alongside the independent
    # subset, and the max comes off that stream.
    assert "silence_window_scores, silence_stream_scores = _score_windows_streaming(" in gate_src
    # _score_windows_streaming is nested inside _run_quality_gate, so its body is
    # part of gate_src: it must hand back the full stream, not a pre-reduced max.
    assert "return independent, raw" in gate_src
    assert "float(stream.max())" in subgrade_src

    # Behavioural: the worst window survives to the reported max even when
    # decimation drops it.
    n = 367
    stream = np.full(n, 0.05, dtype=np.float32)
    stream[4] = 0.79  # index 4 is not a multiple of seq_len 9, so decimation drops it
    independent = _decimate_to_independent_windows(stream, [0] * n, 9)

    _rate, max_score, _ind, _cnt = _silence_subgrade(independent, stream, 0.80)

    assert max_score == pytest.approx(0.79, abs=1e-6), (
        "the reported max lost the model's worst no-wake window"
    )


def test_a_crossing_max_can_never_coexist_with_a_zero_rate() -> None:
    """The invariant that closes the resolution-floor hole (#1487, #2611).

    A rate cannot express a value smaller than one over its own denominator. Taken
    over ~151 independent windows the smallest non-zero rate is 0.66%, so every
    model whose true no-wake false-fire rate sits below that was reported as
    exactly 0.0% -- while the same function printed the honest full-stream max
    right beside it. Production, 2026-08-05: jobs 156 and 157 were graded A at
    "Silence FP rate: 0.0%" with ``max=0.92`` and ``max=0.90`` against
    ``threshold=0.80``. The gate was certifying as clean two models it had just
    watched cross the firing threshold on the customer's own room tone, and
    ``confirm_count`` defaults to 1 in ``wake_detector.py``, so a single crossing
    is a real false wake.

    REDs on the pre-fix shape, where the rate came from the decimated subset: the
    crossing below sits at an index decimation drops, so the old rate is 0.0 while
    the max is 0.92.
    """
    seq_len = 9
    n = 1359  # ~151 independent windows, the real count from a production job
    stream = np.full(n, 0.05, dtype=np.float32)
    crossing_index = 4  # dropped by decimation (kept indices are 0 % 9)
    assert crossing_index % seq_len != 0
    stream[crossing_index] = 0.92
    independent = _decimate_to_independent_windows(stream, [0] * n, seq_len)

    assert float((independent >= 0.80).mean()) == 0.0, (
        "this fixture must reproduce the old 0.0%-rate shape to be a valid RED"
    )

    rate, max_score, independent_count, stream_count = _silence_subgrade(independent, stream, 0.80)

    assert rate is not None
    assert max_score >= 0.80
    assert rate > 0.0, (
        "a model that crosses the deployment threshold on no-wake audio was rated 0.0%"
    )
    assert stream_count > independent_count, "the stream must be the wider population"


def test_the_rate_and_the_max_always_come_from_the_same_numbers() -> None:
    """Property: over random streams, max >= threshold implies rate > 0.

    The invariant above, checked against arbitrary score shapes rather than one
    hand-built fixture, so a future change that re-splits the two populations
    cannot pass by dodging the specific case.
    """
    rng = np.random.default_rng(20260805)
    threshold = 0.80
    saw_a_crossing = False
    for _ in range(300):
        n = int(rng.integers(200, 1500))
        stream = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
        independent = _decimate_to_independent_windows(stream, [0] * n, 9)
        rate, max_score, _ind, _cnt = _silence_subgrade(independent, stream, threshold)
        assert rate is not None
        if max_score >= threshold:
            saw_a_crossing = True
            assert rate > 0.0, f"max {max_score:.3f} crossed {threshold} yet the rate was 0.0"
        else:
            assert rate == 0.0
    assert saw_a_crossing, "the fixture never produced a crossing -- the property went untested"
