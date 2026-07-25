"""Ratchet (#1775): a TTS outage must never be reported as a model-quality verdict.

The training quality gate synthesizes its OWN speech and confusable negatives
with edge-tts. If that synthesis fails -- a voice retired server-side by
Microsoft (CL-20260717-b117), a network failure, throttling -- the negative sets
come back empty and `_fp_rate` returns 1.0 computed over ZERO scored samples.
Pre-fix that silently became grade F, and the customer was told their model
"would trigger on the wrong sound" about a model the gate never scored.

Proven before the fix by executing the real `_run_quality_gate`: a model that
outputs 0.0 for every window -- incapable of false-firing on any input at any
threshold -- graded **A** with TTS healthy and **F** with TTS dead, reporting
"Speech FP rate: 100.0% (0 phrases)".

These tests RED on the pre-fix shape (a grade is returned when the gate has no
negative material) and GREEN on the fix (QualityGateUnavailableError instead).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required to drive the quality gate")
import numpy as np

from violawake_sdk.tools import train as T

# Resolved defensively, NOT imported at module scope: on the pre-fix code these
# symbols do not exist, and a hard import would turn this ratchet into a
# collection error. A collection error only proves a name is missing; the point
# of this file is to prove the pre-fix BEHAVIOR was wrong -- that the gate
# returned grade F for a model incapable of false-firing. Resolving lazily lets
# the behavioral test actually RUN against the old code and fail on its verdict.
QualityGateUnavailableError = getattr(T, "QualityGateUnavailableError", None)
_require_quality_gate_coverage = getattr(T, "_require_quality_gate_coverage", None)
_MIN_QUALITY_GATE_COVERAGE = getattr(T, "_MIN_QUALITY_GATE_COVERAGE", None)


class _NeverFiresModel(torch.nn.Module):
    """Objectively excellent: 0.0 on every window, so it cannot false-fire."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), dtype=torch.float32)

    def eval(self):  # noqa: D102
        return self

    def to(self, *a, **k):  # noqa: D102
        return self


def _fake_embeddings(audio_files, tag, verbose=True, seq_len=9):
    """One embedding window per file, so the gate can score whatever exists."""
    embs = [np.zeros((seq_len, 96), dtype=np.float32) for _ in audio_files]
    return embs, list(range(len(audio_files))), ["" for _ in audio_files]


def _need_fix_symbols() -> None:
    """Clear red on pre-fix code instead of a confusing TypeError/NameError."""
    assert QualityGateUnavailableError is not None, (
        "pre-fix: the gate cannot distinguish 'my own test material is missing' "
        "from 'this model is unfit'"
    )
    assert _require_quality_gate_coverage is not None, (
        "pre-fix: the gate has no coverage floor, so an empty negative set "
        "silently becomes a grade-F verdict about the user's model"
    )


def _run_gate(monkeypatch, *, tts_works: bool):
    monkeypatch.setattr(T, "_extract_temporal_embeddings", _fake_embeddings)
    # Kokoro is the in-process fallback; a total TTS outage means it is down too.
    monkeypatch.setattr(T._KokoroFallback, "ready", lambda self: False)

    def _synth(text, voice, output_path, *, check_cancelled=None):
        if not tts_works:
            return False
        T._save_wav(np.zeros(16000, dtype=np.float32), Path(output_path))
        return True

    monkeypatch.setattr(T, "_edge_tts_synthesize", _synth)
    return T._run_quality_gate(
        _NeverFiresModel(),
        "cpu",
        seq_len=9,
        embedding_dim=96,
        wake_word="citadel",
        deployment_threshold=0.80,
        positive_files=None,
        verbose=False,
    )


def test_healthy_tts_still_grades_a_never_firing_model_well(monkeypatch) -> None:
    """Control: with TTS healthy the gate works and does not fail the model."""
    grade, metrics = _run_gate(monkeypatch, tts_works=True)
    assert grade != "F"
    assert metrics["speech_sample_count"] > 0
    assert metrics["confusable_sample_count"] > 0


def test_tts_outage_raises_instead_of_grading_the_model_f(monkeypatch) -> None:
    """The load-bearing ratchet.

    RED pre-fix: the gate returned grade "F" for a model that cannot false-fire.
    GREEN post-fix: it refuses to grade at all and raises an infrastructure
    error, so the failure is attributed to us, not to the customer.

    Written so it RUNS on the pre-fix code (see the lazy symbol resolution
    above) and fails on the verdict itself, not on a missing import.
    """
    graded = None
    raised: BaseException | None = None
    try:
        graded, _metrics = _run_gate(monkeypatch, tts_works=False)
    except Exception as exc:  # noqa: BLE001 - the type is asserted below
        raised = exc

    assert graded is None, (
        f"the quality gate returned grade {graded!r} for a model that outputs 0.0 on "
        "every input and therefore cannot false-fire at any threshold. The negative "
        "sets were empty because TTS was down, so this grade describes our outage, "
        "not the user's model."
    )
    assert QualityGateUnavailableError is not None, (
        "the gate has no distinct error for 'could not build my own test material', "
        "so a TTS outage can only be reported as a model-quality verdict"
    )
    assert isinstance(raised, QualityGateUnavailableError), (
        f"expected QualityGateUnavailableError, got {type(raised).__name__}: {raised}"
    )

    message = str(raised)
    # The user must not be told to re-record for our outage.
    assert "not with your recordings" in message
    assert "try training again" in message


def test_unscorable_negatives_also_raise_even_though_the_wavs_exist(monkeypatch) -> None:
    """The SECOND route to an empty score array, which a files-only check misses.

    TTS can succeed -- the WAVs are on disk -- and the gate still ends up with
    nothing to score, because `_extract_temporal_embeddings` swallows per-file
    embedding failures (a broken/missing OWW backbone, an onnxruntime fault) and
    returns whatever survived, which can be nothing. `_fp_rate` then returns 1.0
    over ZERO samples and the model is failed for it.

    Reds on any implementation that enforces coverage on synthesized FILES
    instead of on SCORED SAMPLES.
    """
    monkeypatch.setattr(T._KokoroFallback, "ready", lambda self: False)

    def _synth(text, voice, output_path, *, check_cancelled=None):
        T._save_wav(np.zeros(16000, dtype=np.float32), Path(output_path))
        return True  # synthesis is HEALTHY; every wav is written

    monkeypatch.setattr(T, "_edge_tts_synthesize", _synth)
    # ...but nothing can be embedded.
    monkeypatch.setattr(
        T, "_extract_temporal_embeddings", lambda files, tag, verbose=True, seq_len=9: ([], [], [])
    )

    graded = None
    raised: BaseException | None = None
    try:
        graded, _m = T._run_quality_gate(
            _NeverFiresModel(),
            "cpu",
            seq_len=9,
            embedding_dim=96,
            wake_word="citadel",
            deployment_threshold=0.80,
            positive_files=None,
            verbose=False,
        )
    except Exception as exc:  # noqa: BLE001 - type asserted below
        raised = exc

    assert graded is None, (
        f"the gate returned grade {graded!r} with ZERO scored negatives. The wav "
        "files existed, so a coverage check counting synthesized files would have "
        "passed while the gate still judged the model on nothing."
    )
    assert QualityGateUnavailableError is not None
    assert isinstance(raised, QualityGateUnavailableError), (
        f"expected QualityGateUnavailableError, got {type(raised).__name__}: {raised}"
    )


def test_the_outage_error_is_not_a_model_quality_verdict() -> None:
    """It must not be a ModelQualityGateError: that class is classified EXPECTED
    and deliberately does not page ops. A TTS outage is our bug and must page."""
    _need_fix_symbols()
    assert not issubclass(QualityGateUnavailableError, T.ModelQualityGateError)
    assert issubclass(QualityGateUnavailableError, T.TrainingError)


def test_coverage_floor_rejects_an_empty_negative_set() -> None:
    """Zero scored material is the exact shape that made _fp_rate return 1.0."""
    _need_fix_symbols()
    with pytest.raises(QualityGateUnavailableError):
        _require_quality_gate_coverage(0, 50, "speech phrases")


def test_coverage_floor_rejects_a_severely_undersized_negative_set() -> None:
    """Partial synthesis is graded on too few samples to be meaningful: with 3
    of 50 phrases a single false positive is a 33% rate, far above the 10% bar."""
    _need_fix_symbols()
    with pytest.raises(QualityGateUnavailableError):
        _require_quality_gate_coverage(3, 50, "speech phrases")


def test_coverage_floor_accepts_a_full_or_adequately_covered_set() -> None:
    """The floor must not fire on healthy or mildly-degraded synthesis, or it
    would become a new spurious failure mode of its own."""
    _need_fix_symbols()
    _require_quality_gate_coverage(50, 50, "speech phrases")
    _require_quality_gate_coverage(25, 50, "speech phrases")
    _require_quality_gate_coverage(20, 20, "confusable words")
    # Nothing requested => nothing to enforce.
    _require_quality_gate_coverage(0, 0, "confusable words")


def test_coverage_floor_is_a_real_fraction_not_just_a_zero_check() -> None:
    """Reds on a naive 'only reject exactly zero' implementation."""
    _need_fix_symbols()
    assert 0.0 < _MIN_QUALITY_GATE_COVERAGE <= 1.0
    below = max(0, int(50 * _MIN_QUALITY_GATE_COVERAGE) - 1)
    with pytest.raises(QualityGateUnavailableError):
        _require_quality_gate_coverage(below, 50, "speech phrases")


def test_quality_gate_uses_the_kokoro_fallback_when_edge_tts_fails(monkeypatch) -> None:
    """One retired edge-tts voice must not empty the gate's negative sets: the
    gate must fall back per-sample the way the positives generator does (#1768).
    Reds on the pre-fix gate, which called edge-tts with no fallback at all."""
    monkeypatch.setattr(T, "_extract_temporal_embeddings", _fake_embeddings)
    monkeypatch.setattr(T, "_edge_tts_synthesize", lambda *a, **k: False)

    used: list[str] = []

    def _kokoro_synth(self, text, output_path, *, rotate_index):
        used.append(text)
        T._save_wav(np.zeros(16000, dtype=np.float32), Path(output_path))
        return True

    monkeypatch.setattr(T._KokoroFallback, "ready", lambda self: True)
    monkeypatch.setattr(T._KokoroFallback, "synthesize", _kokoro_synth)

    grade, metrics = T._run_quality_gate(
        _NeverFiresModel(),
        "cpu",
        seq_len=9,
        embedding_dim=96,
        wake_word="citadel",
        deployment_threshold=0.80,
        positive_files=None,
        verbose=False,
    )

    assert used, "Kokoro fallback was never used when edge-tts failed"
    assert metrics["speech_sample_count"] > 0
    assert metrics["confusable_sample_count"] > 0
    assert grade != "F"
