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
    assert abs(bias) < 0.02, f"decimation shifted the mean rate by {bias:+.4f} -- that is a re-calibration"


def test_the_gate_rates_the_decimated_scores_not_the_raw_stream() -> None:
    """Wiring: the scorer feeding the rate must be the decimated one.

    REDs on the pre-fix wiring, which returned ``model(X_qc)...flatten()`` straight
    from the streaming extractor and took ``.mean()`` over that.
    """
    src = inspect.getsource(_run_quality_gate)
    assert "_decimate_to_independent_windows(raw, source_indices, seq_len)" in src
    # The rate's denominator is the independent count, and it is what gets reported.
    assert '"silence_window_count": silence_window_count' in src


def test_an_underpowered_sample_is_not_reported_as_a_rate() -> None:
    """A rate over three looks cannot resolve a 2%/5%/10% bar.

    The gate requires a minimum number of independent windows before it will call
    the axis measured, so a coarse sample becomes either a fallback-probe
    measurement or a fail-closed verdict -- never a falsely precise percentage.
    """
    assert _SILENCE_MIN_INDEPENDENT_WINDOWS >= 12, (
        "fewer than ~12 independent looks cannot resolve the 10% grade-F bar"
    )
    src = inspect.getsource(_run_quality_gate)
    assert "len(silence_window_scores) >= _SILENCE_MIN_INDEPENDENT_WINDOWS" in src
