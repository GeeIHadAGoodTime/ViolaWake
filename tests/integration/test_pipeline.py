"""Integration tests for VoicePipeline command flow."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.pipeline import _STATE_IDLE, _STATE_RESPONDING, VoicePipeline

pytestmark = pytest.mark.integration


def _make_pipeline(*, enable_tts: bool) -> VoicePipeline:
    with (
        patch("violawake_sdk.pipeline.WakeDetector"),
        patch("violawake_sdk.pipeline.VADEngine"),
    ):
        return VoicePipeline(enable_tts=enable_tts)


def test_command_handler_receives_transcribed_text() -> None:
    pipeline = _make_pipeline(enable_tts=False)
    received: list[str] = []
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "play jazz"
    mock_stt.prewarm = MagicMock()

    @pipeline.on_command
    def handle(text: str) -> None:
        received.append(text)
        return None

    with patch.object(pipeline, "_get_stt", return_value=mock_stt):
        pipeline._transcribe_and_respond(b"\x00" * 640)

    assert received == ["play jazz"]


def test_tts_response_spoken_on_handler_return() -> None:
    pipeline = _make_pipeline(enable_tts=True)
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "play jazz"
    mock_stt.prewarm = MagicMock()
    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = np.zeros(100, dtype=np.float32)

    @pipeline.on_command
    def handle(_text: str) -> str:
        return "Playing jazz"

    with (
        patch.object(pipeline, "_get_stt", return_value=mock_stt),
        patch.object(pipeline, "_get_tts", return_value=mock_tts),
    ):
        pipeline._transcribe_and_respond(b"\x00" * 640)

    mock_tts.synthesize.assert_called_once_with("Playing jazz")
    mock_tts.play.assert_called_once()


def test_state_machine_returns_to_idle_after_command() -> None:
    pipeline = _make_pipeline(enable_tts=True)
    pipeline._state = _STATE_IDLE
    states_seen: list[str] = []
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = "test command"
    mock_stt.prewarm = MagicMock()

    @pipeline.on_command
    def handle(_text: str) -> None:
        states_seen.append(pipeline._state)
        return None

    with patch.object(pipeline, "_get_stt", return_value=mock_stt):
        pipeline._transcribe_and_respond(b"\x00" * 640)

    assert states_seen == [_STATE_RESPONDING]
    assert pipeline._state == _STATE_IDLE
