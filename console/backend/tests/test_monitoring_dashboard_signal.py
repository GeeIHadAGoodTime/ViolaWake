"""Regression test for #1482: grade-F block-rate loses ALL GlitchTip visibility
once PR#5 lands.

PR#5 (glitchfix/quality-gate-noise) reclassified ``ModelQualityGateError``
(grade-F training block) from ERROR to WARNING in ``classify_exception`` so it
stops paging ops -- correct, since a grade-F block is an EXPECTED outcome, not
a bug. But the Sentry/GlitchTip ``LoggingIntegration`` only auto-captures a
dashboard-visible *event* from a stdlib ``logging`` call at/above its default
``event_level=ERROR`` (``app/middleware.py`` calls ``sentry_sdk.init()`` with
no explicit ``event_level`` override). A WARNING-level ``logger.log()`` call
becomes a breadcrumb only, attached to nothing -- so after PR#5, GlitchTip
issue 28's event count (the at-a-glance signal used to notice the ~75%
false-block-rate pattern behind #1465) simply stops growing, forever, with no
error and no test failure to notice it.

The fix: ``classify_exception`` flags ``dashboard_signal=True`` for
``model_quality``, and ``log_exception`` (app/monitoring.py) explicitly calls
``sentry_sdk.capture_message`` for any such classification. An explicit
capture call is a direct Sentry API, not filtered by ``LoggingIntegration``'s
``event_level`` -- it always creates an event when Sentry is initialized,
regardless of the passed ``level=``. This restores a durable, at-a-glance
GlitchTip signal for the grade-F block rate without re-enabling paging (the
explicit capture keeps ``level="warning"``, matching the classification).

The negative assertions pin the OLD (silent) shape: with no
``dashboard_signal`` wiring, ``log_exception`` would classify, log, and return
-- and never touch ``sentry_sdk.capture_message`` at all for a WARNING-level
outcome. That silence is exactly what #1482 reports: no crash, no red test,
just a dashboard that quietly stops moving.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
import sentry_sdk
from app.monitoring import log_exception

from violawake_sdk.tools.train import ModelQualityGateError

GRADE_F_MESSAGE = (
    "Model failed the quality gate with grade F; ONNX export was blocked. "
    "See /tmp/x.config.json for quality metrics."
)


class _FakeScope:
    """Records set_tag/set_extra/fingerprint calls made inside push_scope()."""

    def __init__(self) -> None:
        self.tags: dict[str, Any] = {}
        self.extras: dict[str, Any] = {}
        self.fingerprint: list[str] | None = None

    def set_tag(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def set_extra(self, key: str, value: Any) -> None:
        self.extras[key] = value


@pytest.fixture()
def fake_sentry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub sentry_sdk as a real, initialized SDK without a network client.

    Patches only the entry points ``app.monitoring._emit_dashboard_signal``
    calls (``is_initialized``, ``push_scope``, ``capture_message``), on the
    real installed ``sentry_sdk`` module -- the same module the production
    code imports lazily inside the function.
    """
    scope = _FakeScope()
    capture_message = MagicMock(return_value="fake-event-id")

    @contextmanager
    def fake_push_scope():
        yield scope

    monkeypatch.setattr(sentry_sdk, "is_initialized", lambda: True)
    monkeypatch.setattr(sentry_sdk, "push_scope", fake_push_scope)
    monkeypatch.setattr(sentry_sdk, "capture_message", capture_message)

    return {"scope": scope, "capture_message": capture_message}


@pytest.fixture()
def uninitialized_sentry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub sentry_sdk as installed but NOT initialized (no DSN configured)."""
    capture_message = MagicMock()
    monkeypatch.setattr(sentry_sdk, "is_initialized", lambda: False)
    monkeypatch.setattr(sentry_sdk, "capture_message", capture_message)
    return {"capture_message": capture_message}


def test_grade_f_block_emits_a_sentry_capture_despite_warning_log_level(
    fake_sentry: dict[str, Any],
) -> None:
    logger = logging.getLogger("test.violawake.dashboard_signal")

    classification = log_exception(
        logger,
        ModelQualityGateError(GRADE_F_MESSAGE),
        message="Training job failed",
        source="training",
        extra={"job_id": 87, "wake_word": "jarvis"},
    )

    assert classification.log_level == logging.WARNING  # PR#5's non-paging intent stays intact
    # ... but the dashboard signal must have fired anyway (this is the fix):
    fake_sentry["capture_message"].assert_called_once()
    (message,), kwargs = fake_sentry["capture_message"].call_args
    assert "model_quality" in message
    assert kwargs["level"] == "warning"

    scope = fake_sentry["scope"]
    assert scope.tags["error_reason"] == "model_quality"
    assert scope.tags["error_type"] == "ModelQualityGateError"
    assert scope.fingerprint == ["dashboard-signal", "model_quality", "ModelQualityGateError"]
    assert scope.extras["job_id"] == 87


def test_grade_f_block_is_silent_when_sentry_is_not_initialized(
    uninitialized_sentry: dict[str, Any],
) -> None:
    # No DSN configured -> nothing to capture into, and no exception raised.
    logger = logging.getLogger("test.violawake.dashboard_signal")

    log_exception(
        logger,
        ModelQualityGateError(GRADE_F_MESSAGE),
        message="Training job failed",
        source="training",
    )

    uninitialized_sentry["capture_message"].assert_not_called()


def test_an_unexpected_bug_does_not_get_a_duplicate_dashboard_capture(
    fake_sentry: dict[str, Any],
) -> None:
    # A real bug is ERROR level -> already auto-captured by the Sentry
    # LoggingIntegration's default event_level. The explicit dashboard-signal
    # path must not ALSO fire for it (that would double the event).
    logger = logging.getLogger("test.violawake.dashboard_signal")

    log_exception(
        logger,
        RuntimeError("boom"),
        message="Unhandled request exception",
        source="request",
    )

    fake_sentry["capture_message"].assert_not_called()
