"""Additional unit tests for violawake_sdk.pipeline public behavior."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from violawake_sdk.pipeline import VoicePipeline, _STATE_IDLE, _STATE_TRANSCRIBING


def _build_pipeline(*, streaming_stt: bool = False, enable_tts: bool = False) -> tuple[VoicePipeline, MagicMock]:
    with (
        patch("violawake_sdk.pipeline.WakeDetector") as wake_cls,
        patch("violawake_sdk.pipeline.VADEngine") as vad_cls,
    ):
        wake = MagicMock()
        wake.detect.return_value = False
        wake.stream_mic.return_value = iter([])
        wake_cls.return_value = wake

        vad = MagicMock()
        vad.is_speech.return_value = False
        vad_cls.return_value = vad

        pipeline = VoicePipeline(streaming_stt=streaming_stt, enable_tts=enable_tts)

    return pipeline, wake


def test_on_command_returns_registered_handler() -> None:
    pipeline, _wake = _build_pipeline()

    def handle(text: str) -> str | None:
        return text

    registered = pipeline.on_command(handle)

    assert registered is handle
    assert pipeline._command_handlers == [handle]


def test_close_stops_pipeline_and_releases_engines() -> None:
    pipeline, wake = _build_pipeline()
    pipeline._stt = object()
    pipeline._tts = object()

    with patch.object(pipeline, "stop") as stop_mock:
        pipeline.close()

    stop_mock.assert_called_once()
    wake.close.assert_called_once()
    assert pipeline._stt is None
    assert pipeline._tts is None


def test_context_manager_calls_close_on_exit() -> None:
    pipeline, _wake = _build_pipeline()

    with patch.object(pipeline, "close") as close_mock:
        with pipeline as entered:
            assert entered is pipeline

    close_mock.assert_called_once()


def test_transcribe_and_respond_streaming_concatenates_segments() -> None:
    pipeline, _wake = _build_pipeline(streaming_stt=True)
    pipeline._stt = MagicMock()
    pipeline._stt.transcribe_streaming.return_value = iter(
        [
            SimpleNamespace(text="turn", start=0.0, end=0.2),
            SimpleNamespace(text=" on lights ", start=0.2, end=0.6),
        ]
    )

    with patch.object(pipeline, "_dispatch_command") as dispatch_mock:
        pipeline._transcribe_and_respond(b"\x01\x00\x02\x00")

    dispatch_mock.assert_called_once_with("turn  on lights")
    assert pipeline._state == _STATE_IDLE


def test_start_worker_skips_new_spawn_when_worker_is_alive() -> None:
    pipeline, _wake = _build_pipeline()
    existing_worker = MagicMock()
    existing_worker.is_alive.return_value = True
    pipeline._worker_thread = existing_worker
    pipeline._state = _STATE_TRANSCRIBING

    with patch("violawake_sdk.pipeline.threading.Thread") as thread_cls:
        pipeline._start_worker(b"\x00\x00")

    thread_cls.assert_not_called()
    assert pipeline._state == _STATE_IDLE
