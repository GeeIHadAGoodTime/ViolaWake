"""A dead single TTS voice must not grade the customer's model F (#2611 C-303).

GeeIHadAGoodTime/Viola#2611 comment 5121856038: the quality gate's speech and
confusable probes were synthesized with exactly one edge-tts voice
(``EDGE_TTS_VOICES[0]``, ``en-US-GuyNeural``) and NO fallback. Microsoft has
already retired seven voices from this pool server-side, on this exact system
(CL-20260717-b117) -- requesting a retired voice completes the WebSocket
handshake but never sends audio, so every attempt fails deterministically. If
the one voice the gate happens to use dies the same way, ``speech_files``
(and/or ``confusable_files``) comes back empty, and the pre-fix
``_fp_rate([]) == 1.0`` fails EVERY tier in ``_grade_quality`` -- the customer
sees "Speech FP rate: 100.0%" and a grade-F verdict, which looks like a
measurement of their model but is actually a measurement of nothing.

The repo already has exactly the fix this needed: ``_KokoroFallback``
(#1768), used by ``_generate_tts_positives``/``_generate_confusable_negatives``
so one dead edge-tts voice only ever loses ITS OWN sample. The gate was the
one caller not using it.

The fix, tested below in two layers (this repo's convention for
``_run_quality_gate``, see test_quality_gate_silence_bar.py /
test_silence_subgrade_measurement_integrity.py):

  1. Two small extracted, directly-testable units:
     ``_synthesize_gate_probe`` (per-sample edge-tts-then-Kokoro fallback) and
     ``_require_gate_probes_measurable`` (raises ``QualityGateUnavailableError``
     -- an outage, never a customer verdict -- only when an axis is
     COMPLETELY unmeasurable even after the fallback).
  2. ``inspect.getsource`` wiring assertions on ``_run_quality_gate`` itself,
     proving those units are actually used in place of the old bare
     ``_edge_tts_synthesize`` calls -- this repo's established way of proving
     the heavy integration function (torch model + embeddings, expensive to
     drive end-to-end in a unit test) is wired correctly.

Every test in this file either fails to collect (the pre-fix module has none
of these symbols) or asserts something false of the pre-fix source/behavior --
both are RED on the pre-fix shape (bb1a0e9, current master) and GREEN on the
fix. Verified directly against that commit before landing (see PR description).
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from violawake_sdk.tools.train import (
    ModelQualityGateError,
    QualityGateUnavailableError,
    TrainingError,
    _KokoroFallback,
    _require_gate_probes_measurable,
    _run_quality_gate,
    _synthesize_gate_probe,
)

# ---------------------------------------------------------------------------
# QualityGateUnavailableError itself: a sibling of ModelQualityGateError, not
# a subclass -- the console backend's matchers key on class NAME across the
# MRO (console/tests/test_quality_gate_not_a_breaker_fault.py), so inheriting
# from ModelQualityGateError would make an infra outage silently look like an
# expected grade-F verdict: downgraded to a non-paging warning AND exempted
# from the user's circuit breaker. Neither must happen.
# ---------------------------------------------------------------------------


def test_outage_error_is_a_training_error_but_not_a_quality_gate_error() -> None:
    assert issubclass(QualityGateUnavailableError, TrainingError)
    assert not issubclass(QualityGateUnavailableError, ModelQualityGateError)
    mro_names = {c.__name__ for c in QualityGateUnavailableError.__mro__}
    assert "ModelQualityGateError" not in mro_names, (
        "QualityGateUnavailableError must not carry ModelQualityGateError in its "
        "MRO -- the console backend's classify_exception and "
        "_is_expected_training_outcome match by class name across the MRO, and "
        "an infra outage must not be classified as an expected grade-F verdict"
    )


# ---------------------------------------------------------------------------
# _synthesize_gate_probe: the per-sample fallback unit
# ---------------------------------------------------------------------------


def test_synthesize_gate_probe_falls_back_to_kokoro_when_edge_tts_fails(
    tmp_path: Path,
) -> None:
    kokoro = _KokoroFallback()
    kokoro_calls: list[tuple[str, int]] = []

    def _fake_kokoro_synthesize(self, text, output_path, *, rotate_index):
        kokoro_calls.append((text, rotate_index))
        output_path.write_bytes(b"kokoro-wav")
        return True

    with (
        patch("violawake_sdk.tools.train._edge_tts_synthesize", return_value=False),
        patch.object(_KokoroFallback, "ready", lambda self: True),
        patch.object(_KokoroFallback, "synthesize", _fake_kokoro_synthesize),
    ):
        out_path = tmp_path / "probe.wav"
        ok = _synthesize_gate_probe(
            "what time is it", "en-US-GuyNeural", out_path, kokoro, rotate_index=7
        )

    assert ok is True
    assert out_path.exists()
    assert kokoro_calls == [("what time is it", 7)], (
        "the failed voice's text and rotate_index must reach Kokoro unchanged"
    )


def test_synthesize_gate_probe_does_not_touch_kokoro_when_edge_tts_succeeds(
    tmp_path: Path,
) -> None:
    """The common case (edge-tts healthy) must be unchanged: no Kokoro probe,
    no Kokoro synthesis call -- the single-voice-is-fast-and-deterministic
    design is preserved for the case that isn't an outage."""
    kokoro = _KokoroFallback()
    ready_calls: list[None] = []

    def _fake_edge_tts(text, voice, output_path, *, check_cancelled=None):
        output_path.write_bytes(b"edge-wav")
        return True

    with (
        patch("violawake_sdk.tools.train._edge_tts_synthesize", side_effect=_fake_edge_tts),
        patch.object(_KokoroFallback, "ready", lambda self: ready_calls.append(None) or True),
    ):
        out_path = tmp_path / "probe.wav"
        ok = _synthesize_gate_probe(
            "play some music", "en-US-GuyNeural", out_path, kokoro, rotate_index=0
        )

    assert ok is True
    assert out_path.read_bytes() == b"edge-wav"
    assert ready_calls == [], "kokoro.ready() must not even be probed when edge-tts already worked"


def test_synthesize_gate_probe_reports_failure_when_both_providers_are_down(
    tmp_path: Path,
) -> None:
    """A genuine total outage (both TTS providers down) must fail honestly --
    no file, no swallowed exception -- so the caller can detect it."""
    kokoro = _KokoroFallback()
    with (
        patch("violawake_sdk.tools.train._edge_tts_synthesize", return_value=False),
        patch.object(_KokoroFallback, "ready", lambda self: False),
    ):
        out_path = tmp_path / "probe.wav"
        ok = _synthesize_gate_probe(
            "good morning", "en-US-GuyNeural", out_path, kokoro, rotate_index=0
        )

    assert ok is False
    assert not out_path.exists()


# ---------------------------------------------------------------------------
# _require_gate_probes_measurable: the honest-outage guard
# ---------------------------------------------------------------------------


def _wav(tmp_path: Path, name: str) -> Path:
    p = tmp_path / name
    p.write_bytes(b"wav")
    return p


def test_require_gate_probes_measurable_passes_when_both_axes_have_material(
    tmp_path: Path,
) -> None:
    speech = [_wav(tmp_path, "s.wav")]
    confusable = [_wav(tmp_path, "c.wav")]
    _require_gate_probes_measurable(speech, confusable, "en-US-GuyNeural")  # must not raise


def test_require_gate_probes_measurable_raises_outage_not_a_quality_verdict_when_speech_is_empty(
    tmp_path: Path,
) -> None:
    """This is the exact pre-fix failure shape: the single voice died, so
    speech_files is []. Pre-fix, that number silently became a 1.0 FP rate
    and a grade-F ModelQualityGateError -- indistinguishable from a real bad
    model. Post-fix it must be a distinctly-typed, honestly-worded outage."""
    confusable = [_wav(tmp_path, "c.wav")]

    with pytest.raises(QualityGateUnavailableError) as exc_info:
        _require_gate_probes_measurable([], confusable, "en-US-GuyNeural")

    message = str(exc_info.value)
    assert "speech" in message
    assert "infrastructure" in message.lower() or "outage" in message.lower()
    # It must NOT be shaped like the customer-facing grade-F message (no
    # "quality check" / "wasn't saved" / detection-threshold language) --
    # this is a different failure and must read like one.
    assert "quality check" not in message
    assert "detection" not in message


def test_require_gate_probes_measurable_raises_outage_when_confusables_is_empty(
    tmp_path: Path,
) -> None:
    speech = [_wav(tmp_path, "s.wav")]
    with pytest.raises(QualityGateUnavailableError, match="confusable"):
        _require_gate_probes_measurable(speech, [], "en-US-GuyNeural")


def test_require_gate_probes_measurable_names_both_axes_when_both_are_empty(
    tmp_path: Path,
) -> None:
    with pytest.raises(QualityGateUnavailableError) as exc_info:
        _require_gate_probes_measurable([], [], "en-US-GuyNeural")
    message = str(exc_info.value)
    assert "speech" in message
    assert "confusable" in message


def test_require_gate_probes_measurable_does_not_care_how_many_files_just_that_there_are_some(
    tmp_path: Path,
) -> None:
    """Partial recovery (most phrases died, one Kokoro sample survived) must
    still be graded genuinely -- this guard only catches TOTAL failure. It is
    not a minimum-sample-size/statistical-power gate (unlike the silence
    axis's _SILENCE_MIN_INDEPENDENT_WINDOWS, which is a separate, deliberate,
    already-shipped design and out of scope for this fix)."""
    one_file = [_wav(tmp_path, "only_survivor.wav")]
    _require_gate_probes_measurable(one_file, one_file, "en-US-GuyNeural")  # must not raise


# ---------------------------------------------------------------------------
# Wiring: _run_quality_gate must actually use both units, for BOTH axes
# ---------------------------------------------------------------------------


def test_run_quality_gate_uses_the_fallback_for_both_speech_and_confusable_axes() -> None:
    """REDs on the pre-fix source, which called `_edge_tts_synthesize(phrase,
    voice, out_path)` / `_edge_tts_synthesize(word, voice, out_path)` directly
    with no fallback for either axis."""
    src = inspect.getsource(_run_quality_gate)
    assert src.count("_synthesize_gate_probe(") == 2, (
        "both the speech loop and the confusable loop must route through the "
        "Kokoro-fallback helper, not call _edge_tts_synthesize directly"
    )
    assert "kokoro = _KokoroFallback()" in src


def test_run_quality_gate_guards_before_scoring_not_after() -> None:
    """REDs on the pre-fix source, which had no guard at all and let an empty
    probe list flow straight into embedding extraction / _fp_rate([]) == 1.0.
    The guard call must appear in the source, and it must appear before the
    silence-probe section starts (so an unmeasurable speech/confusable axis
    never reaches scoring)."""
    src = inspect.getsource(_run_quality_gate)
    assert "_require_gate_probes_measurable(speech_files, confusable_files, voice)" in src

    guard_pos = src.index("_require_gate_probes_measurable(speech_files")
    silence_section_pos = src.index("Silence subgrade probes")
    assert guard_pos < silence_section_pos, (
        "the outage guard must run before the silence-probe section, so an "
        "unmeasurable speech/confusable axis is caught before any scoring work"
    )


def test_the_bare_pre_fix_call_shape_is_gone() -> None:
    """Locks out a regression back to the exact pre-fix line shape (direct,
    fallback-less _edge_tts_synthesize calls feeding speech_files/
    confusable_files with no other guard in between)."""
    src = inspect.getsource(_run_quality_gate)
    assert "_edge_tts_synthesize(phrase, voice, out_path)" not in src
    assert "_edge_tts_synthesize(word, voice, out_path)" not in src


# ---------------------------------------------------------------------------
# Control: a model that genuinely false-fires on speech must still fail --
# proves the fix does not weaken the gate, only stops it lying about outages.
# ---------------------------------------------------------------------------


def test_fp_rate_logic_itself_is_untouched_and_still_fails_closed_on_empty() -> None:
    """_fp_rate's own fail-closed behavior on an empty array (rate 1.0) is
    NOT removed by this fix -- it stays as defense in depth for any other
    caller. What changed is that _run_quality_gate's speech/confusable axes
    can no longer REACH it empty without the fallback+guard running first.
    This is asserted straight from the gate's own docstring contract instead
    of duplicating _grade_quality's existing coverage (already proven in
    test_quality_gate_silence_bar.py::test_speech_and_confusable_axes_still_gate,
    which stays green, unmodified, after this fix)."""
    src = inspect.getsource(_run_quality_gate)
    assert "return 1.0" in src, (
        "the underlying _fp_rate fail-closed-on-empty behavior must remain as "
        "a safety net; this fix adds a guard in front of it, it does not "
        "remove the net"
    )


def test_outage_guard_only_fires_on_total_failure_not_on_a_bad_model() -> None:
    """A model that would genuinely score every real (non-empty) probe above
    threshold is untouched by this guard -- it only inspects whether files
    exist, never their content/scores. Re-affirms the two extracted units
    compose correctly: successful synthesis (even 100% via Kokoro) always
    clears _require_gate_probes_measurable and reaches real scoring."""
    kokoro = _KokoroFallback()

    def _edge_tts_always_fails(text, voice, output_path, *, check_cancelled=None):
        return False

    def _kokoro_always_succeeds(self, text, output_path, *, rotate_index):
        output_path.write_bytes(f"kokoro-{rotate_index}".encode())
        return True

    speech_files: list[Path] = []
    with (
        patch(
            "violawake_sdk.tools.train._edge_tts_synthesize",
            side_effect=_edge_tts_always_fails,
        ),
        patch.object(_KokoroFallback, "ready", lambda self: True),
        patch.object(_KokoroFallback, "synthesize", _kokoro_always_succeeds),
        tempfile.TemporaryDirectory() as td,
    ):
        for i, phrase in enumerate(["what time is it", "play some music"]):
            out_path = Path(td) / f"s{i}.wav"
            ok = _synthesize_gate_probe(phrase, "en-US-GuyNeural", out_path, kokoro, rotate_index=i)
            assert ok
            speech_files.append(out_path)

    # Every sample came from Kokoro (edge-tts always failed), yet the axis is
    # fully measurable -- the model would still be scored on real audio, not
    # given a free pass.
    _require_gate_probes_measurable(speech_files, speech_files, "en-US-GuyNeural")
