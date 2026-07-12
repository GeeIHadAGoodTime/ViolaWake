"""Regression test: a model that fails the deployment quality gate (grade F) is
classified as an EXPECTED training outcome, NOT an unexpected "bug".

Incident (GlitchTip violawake issue 28, 2026-07-12): the SDK's grade-F quality
gate raised a bare ``RuntimeError``. The Console backend's generic job handler
caught it and ``log_exception`` -> ``classify_exception`` mapped an unknown
RuntimeError to ``(UNEXPECTED_ERROR, "bug", logging.ERROR)``. Logging at ERROR
makes the default Sentry LoggingIntegration (event_level=ERROR) capture it as a
production error event -- so every user whose recordings were too weak to train a
deployable model paged ops. Grade F is an EXPECTED outcome, not a code bug.

The fix: the SDK raises a typed ``ModelQualityGateError`` and ``classify_exception``
maps it to ``(EXPECTED_ERROR, "model_quality", logging.WARNING)`` -- WARNING is
below the Sentry event_level, so it becomes a breadcrumb, not a captured event.

The negative assertion pins the OLD bug shape: a bare RuntimeError with the same
message is still ``bug``/ERROR, proving the fix is the typed exception and not an
accidental message-substring match.
"""

from __future__ import annotations

import logging

from app.monitoring import EXPECTED_ERROR, UNEXPECTED_ERROR, classify_exception

from violawake_sdk.tools.train import ModelQualityGateError

GRADE_F_MESSAGE = (
    "Model failed the quality gate with grade F; ONNX export was blocked. "
    "See /tmp/x.config.json for quality metrics."
)


def test_quality_gate_error_is_expected_and_below_sentry_level() -> None:
    classification = classify_exception(ModelQualityGateError(GRADE_F_MESSAGE))
    assert classification.kind == EXPECTED_ERROR
    assert classification.reason == "model_quality"
    # Below ERROR -> the default Sentry LoggingIntegration will not capture it.
    assert classification.log_level < logging.ERROR


def test_bare_runtimeerror_grade_f_message_is_still_the_old_bug_shape() -> None:
    # The OLD (pre-fix) shape: a bare RuntimeError, even with the identical
    # message, must NOT be treated as expected -- classification keys off the
    # TYPE, never the message text.
    classification = classify_exception(RuntimeError(GRADE_F_MESSAGE))
    assert classification.kind == UNEXPECTED_ERROR
    assert classification.log_level == logging.ERROR


def test_quality_gate_error_subclass_is_also_expected() -> None:
    # MRO-name matching keeps subclasses classified as expected too.
    class StricterQualityGateError(ModelQualityGateError):
        pass

    classification = classify_exception(StricterQualityGateError(GRADE_F_MESSAGE))
    assert classification.kind == EXPECTED_ERROR
    assert classification.log_level < logging.ERROR
