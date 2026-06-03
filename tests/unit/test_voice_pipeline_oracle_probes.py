"""Oracle probes for VoicePipeline failure modes from the lane ledger."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk._exceptions import PipelineError
from violawake_sdk.pipeline import (
    FRAME_SAMPLES,
    MAX_COMMAND_DURATION_S,
    SAMPLE_RATE,
    VoicePipeline,
    _STATE_TRANSCRIBING,
)


def _build_pipeline(*, enable_tts: bool = False) -> VoicePipeline:
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

        return VoicePipeline(enable_tts=enable_tts)


def test_noop_stt_empty_text_is_a_pipeline_error() -> None:
    """Probe: a no-op STT engine returning empty text must not pass silently."""
    pipeline = _build_pipeline()
    pipeline._state = _STATE_TRANSCRIBING
    stt = MagicMock()
    stt.transcribe.return_value = ""

    with patch.object(pipeline, "_get_stt", return_value=stt):
        with pytest.raises(PipelineError, match="empty transcription"):
            pipeline._transcribe_and_respond(b"\x01\x00" * FRAME_SAMPLES)


def test_stt_import_failure_is_a_pipeline_error() -> None:
    """Probe: STT dependency/prewarm failures must keep their real cause visible."""
    pipeline = _build_pipeline()
    pipeline._state = _STATE_TRANSCRIBING

    with patch.object(
        pipeline,
        "_get_stt",
        side_effect=ImportError("faster-whisper is installed but failed to import: DLL"),
    ):
        with pytest.raises(PipelineError, match="STT unavailable"):
            pipeline._transcribe_and_respond(b"\x01\x00" * FRAME_SAMPLES)


def test_tts_wrong_voice_is_a_pipeline_error_from_command_path() -> None:
    """Probe: a misconfigured TTS voice must raise instead of disappearing in logs."""
    pipeline = _build_pipeline(enable_tts=True)

    @pipeline.on_command
    def handler(_text: str) -> str:
        return "spoken response"

    with patch.object(pipeline, "_get_tts", side_effect=ValueError("Unknown voice 'bad'")):
        with pytest.raises(PipelineError, match="TTS playback failed"):
            pipeline._dispatch_command("hello")


def test_vad_always_on_stops_recording_at_max_duration() -> None:
    """Probe: an always-on VAD cannot keep the pipeline listening forever."""
    pipeline = _build_pipeline()
    max_frames = int(MAX_COMMAND_DURATION_S / (FRAME_SAMPLES / SAMPLE_RATE))
    frames = [b"\x00" * (FRAME_SAMPLES * 2)] * (max_frames + 2)
    pipeline._wake_detector.detect.side_effect = [True] + [False] * (len(frames) - 1)
    pipeline._wake_detector.stream_mic.return_value = iter(frames)
    pipeline._vad.is_speech.return_value = True

    with patch.object(pipeline, "_start_worker") as start_worker:
        pipeline._run_loop()

    start_worker.assert_called_once()
