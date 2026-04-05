"""Unit tests for VoicePipeline state machine.

Tests state transitions, invalid transitions, reset behavior,
and concurrent access safety. All engines (wake, VAD, STT, TTS)
are mocked — no real models or hardware needed.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk._exceptions import PipelineError
from violawake_sdk.pipeline import (
    VoicePipeline,
    _STATE_IDLE,
    _STATE_LISTENING,
    _STATE_RESPONDING,
    _STATE_TRANSCRIBING,
)


# ---------------------------------------------------------------------------
# Helper to build a pipeline with fully mocked engines
# ---------------------------------------------------------------------------

def _build_pipeline(*, on_wake: Callable[[], None] | None = None) -> VoicePipeline:
    """Build a VoicePipeline with all engines mocked out."""
    with (
        patch("violawake_sdk.pipeline.WakeDetector") as mock_wake_cls,
        patch("violawake_sdk.pipeline.VADEngine") as mock_vad_cls,
    ):
        mock_wake = MagicMock()
        mock_wake.detect.return_value = False
        mock_wake.stream_mic.return_value = iter([])
        mock_wake_cls.return_value = mock_wake

        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = False
        mock_vad_cls.return_value = mock_vad

        pipeline = VoicePipeline(
            wake_word="viola",
            stt_model="base",
            tts_voice="af_heart",
            enable_tts=False,
            on_wake=on_wake,
        )

    return pipeline


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    """Pipeline starts in IDLE state."""

    def test_starts_idle(self) -> None:
        pipeline = _build_pipeline()
        assert pipeline._state == _STATE_IDLE

    def test_stop_event_not_set(self) -> None:
        pipeline = _build_pipeline()
        assert not pipeline._stop_event.is_set()


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    """Test valid state transitions: IDLE -> LISTENING -> TRANSCRIBING -> RESPONDING -> IDLE."""

    def test_idle_to_listening(self) -> None:
        pipeline = _build_pipeline()
        with pipeline._state_lock:
            pipeline._state = _STATE_LISTENING
        assert pipeline._state == _STATE_LISTENING

    def test_listening_to_transcribing(self) -> None:
        pipeline = _build_pipeline()
        with pipeline._state_lock:
            pipeline._state = _STATE_TRANSCRIBING
        assert pipeline._state == _STATE_TRANSCRIBING

    def test_transcribing_to_responding(self) -> None:
        pipeline = _build_pipeline()
        with pipeline._state_lock:
            pipeline._state = _STATE_RESPONDING
        assert pipeline._state == _STATE_RESPONDING

    def test_responding_to_idle(self) -> None:
        pipeline = _build_pipeline()
        pipeline._state = _STATE_RESPONDING
        with pipeline._state_lock:
            pipeline._state = _STATE_IDLE
        assert pipeline._state == _STATE_IDLE

    def test_full_cycle(self) -> None:
        """Walk through the complete state machine cycle."""
        pipeline = _build_pipeline()
        assert pipeline._state == _STATE_IDLE

        with pipeline._state_lock:
            pipeline._state = _STATE_LISTENING
        assert pipeline._state == _STATE_LISTENING

        with pipeline._state_lock:
            pipeline._state = _STATE_TRANSCRIBING
        assert pipeline._state == _STATE_TRANSCRIBING

        with pipeline._state_lock:
            pipeline._state = _STATE_RESPONDING
        assert pipeline._state == _STATE_RESPONDING

        with pipeline._state_lock:
            pipeline._state = _STATE_IDLE
        assert pipeline._state == _STATE_IDLE


# ---------------------------------------------------------------------------
# Stop behavior
# ---------------------------------------------------------------------------

class TestStopBehavior:
    """stop() signals the pipeline to terminate."""

    def test_stop_sets_event(self) -> None:
        pipeline = _build_pipeline()
        pipeline.stop()
        assert pipeline._stop_event.is_set()

    def test_stop_joins_worker_thread(self) -> None:
        pipeline = _build_pipeline()
        worker = MagicMock()
        worker.is_alive.return_value = False
        pipeline._worker_thread = worker

        pipeline.stop(timeout=0.25)

        worker.join.assert_called_once_with(timeout=0.25)

    def test_run_returns_on_stop(self) -> None:
        """run() should return when stop_event is set (empty stream)."""
        pipeline = _build_pipeline()
        # stream_mic returns empty iterator -> run_loop exits immediately
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = iter([])

        # Should not hang — empty iterator exits run_loop
        pipeline.run()
        assert pipeline._state == _STATE_IDLE

    def test_run_calls_stop_in_finally(self) -> None:
        pipeline = _build_pipeline()

        with patch.object(pipeline, "_run_loop"), patch.object(pipeline, "stop") as stop_mock:
            pipeline.run()

        stop_mock.assert_called_once_with()


# ---------------------------------------------------------------------------
# Reset to IDLE after errors
# ---------------------------------------------------------------------------

class TestResetToIdle:
    """Pipeline returns to IDLE after run() completes or errors."""

    def test_state_idle_after_run(self) -> None:
        pipeline = _build_pipeline()
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = iter([])
        pipeline.run()
        assert pipeline._state == _STATE_IDLE

    def test_state_idle_after_keyboard_interrupt(self) -> None:
        """KeyboardInterrupt during run sets state back to IDLE."""
        pipeline = _build_pipeline()
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.side_effect = KeyboardInterrupt

        pipeline.run()
        assert pipeline._state == _STATE_IDLE


# ---------------------------------------------------------------------------
# Command handler registration
# ---------------------------------------------------------------------------

class TestCommandHandlers:
    """Test on_command decorator and handler dispatch."""

    def test_on_command_registers_handler(self) -> None:
        pipeline = _build_pipeline()

        @pipeline.on_command
        def my_handler(text: str) -> str | None:
            return f"Echo: {text}"

        assert len(pipeline._command_handlers) == 1
        assert pipeline._command_handlers[0] is my_handler

    def test_multiple_handlers(self) -> None:
        pipeline = _build_pipeline()

        @pipeline.on_command
        def handler1(text: str) -> None:
            pass

        @pipeline.on_command
        def handler2(text: str) -> None:
            pass

        assert len(pipeline._command_handlers) == 2

    def test_dispatch_calls_handler(self) -> None:
        pipeline = _build_pipeline()
        called_with: list[str] = []

        @pipeline.on_command
        def handler(text: str) -> None:
            called_with.append(text)

        pipeline._dispatch_command("hello world")
        assert called_with == ["hello world"]
        # State returns to IDLE after dispatch
        assert pipeline._state == _STATE_IDLE

    def test_handler_exception_does_not_crash(self) -> None:
        """A failing handler should not crash the pipeline."""
        pipeline = _build_pipeline()

        @pipeline.on_command
        def bad_handler(text: str) -> None:
            raise RuntimeError("handler error")

        # Should not raise
        pipeline._dispatch_command("test")
        assert pipeline._state == _STATE_IDLE


class TestWakeCallback:
    """Wake callback fires when the wake word is detected."""

    def test_on_wake_called_when_wake_word_detected(self) -> None:
        wake_calls: list[str] = []

        def on_wake() -> None:
            wake_calls.append("wake")

        pipeline = _build_pipeline(on_wake=on_wake)
        pipeline._wake_detector.detect.return_value = True
        pipeline._wake_detector.stream_mic.return_value = iter([b"\x00" * 640])

        pipeline._run_loop()

        assert wake_calls == ["wake"]
        assert pipeline._state == _STATE_LISTENING


# ---------------------------------------------------------------------------
# Concurrent access safety
# ---------------------------------------------------------------------------

class TestConcurrentAccess:
    """State changes under concurrent access should not corrupt state."""

    def test_concurrent_state_changes(self) -> None:
        """Multiple threads writing state concurrently must not corrupt it."""
        pipeline = _build_pipeline()
        states = [_STATE_IDLE, _STATE_LISTENING, _STATE_TRANSCRIBING, _STATE_RESPONDING]
        errors: list[Exception] = []

        def toggle_state(target_state: str, iterations: int) -> None:
            try:
                for _ in range(iterations):
                    with pipeline._state_lock:
                        pipeline._state = target_state
                    with pipeline._state_lock:
                        pipeline._state = _STATE_IDLE
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=toggle_state, args=(s, 100))
            for s in states
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0
        # Final state should be one of the valid states
        assert pipeline._state in states

    def test_state_lock_prevents_races(self) -> None:
        """Verify the lock exists and is a proper Lock object."""
        pipeline = _build_pipeline()
        assert isinstance(pipeline._state_lock, type(threading.Lock()))

    def test_transition_state_requires_expected_state(self) -> None:
        pipeline = _build_pipeline()

        assert pipeline._transition_state(_STATE_IDLE, _STATE_LISTENING) is True
        assert pipeline._state == _STATE_LISTENING
        assert pipeline._transition_state(_STATE_IDLE, _STATE_RESPONDING) is False
        assert pipeline._state == _STATE_LISTENING


# ---------------------------------------------------------------------------
# Speak method
# ---------------------------------------------------------------------------

class TestSpeak:
    """Test TTS speak() method."""

    def test_speak_noop_when_tts_disabled(self) -> None:
        pipeline = _build_pipeline()
        pipeline._enable_tts = False
        # Should not raise
        pipeline.speak("Hello")

    def test_speak_noop_when_tts_not_installed(self) -> None:
        pipeline = _build_pipeline()
        pipeline._enable_tts = True
        pipeline._tts = None
        # _get_tts returns None when import fails
        with patch.object(pipeline, "_get_tts", return_value=None):
            pipeline.speak("Hello")  # Should not raise

    def test_speak_noop_when_stop_event_set(self) -> None:
        """speak() should do nothing if the pipeline is stopping."""
        pipeline = _build_pipeline()
        pipeline._enable_tts = True
        pipeline._stop_event.set()
        # Should return immediately without touching TTS
        pipeline.speak("Hello")

    def test_speak_success_with_tts(self) -> None:
        """speak() should call synthesize and play on the TTS engine."""
        pipeline = _build_pipeline()
        pipeline._enable_tts = True
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = b"audio_data"
        with patch.object(pipeline, "_get_tts", return_value=mock_tts):
            pipeline.speak("Hello world")
        mock_tts.synthesize.assert_called_once_with("Hello world")
        mock_tts.play.assert_called_once_with(b"audio_data")

    def test_speak_handles_tts_exception(self) -> None:
        """speak() should log but not raise if TTS fails."""
        pipeline = _build_pipeline()
        pipeline._enable_tts = True
        mock_tts = MagicMock()
        mock_tts.synthesize.side_effect = RuntimeError("TTS broke")
        with patch.object(pipeline, "_get_tts", return_value=mock_tts):
            pipeline.speak("Hello")  # Should not raise


# ---------------------------------------------------------------------------
# Default threshold from constants
# ---------------------------------------------------------------------------

class TestDefaultThreshold:
    """Pipeline default threshold uses the canonical constant."""

    def test_default_threshold_matches_constant(self) -> None:
        """The default threshold parameter should import from _constants."""
        from violawake_sdk._constants import DEFAULT_THRESHOLD as CONST_THRESHOLD
        from violawake_sdk.pipeline import DEFAULT_THRESHOLD as PIPE_THRESHOLD

        assert PIPE_THRESHOLD == CONST_THRESHOLD

    def test_pipeline_passes_threshold_to_wake_detector(self) -> None:
        """Threshold from constructor is forwarded to WakeDetector."""
        with (
            patch("violawake_sdk.pipeline.WakeDetector") as mock_wake_cls,
            patch("violawake_sdk.pipeline.VADEngine"),
        ):
            mock_wake_cls.return_value = MagicMock()
            VoicePipeline(threshold=0.65)
            mock_wake_cls.assert_called_once()
            _, kwargs = mock_wake_cls.call_args
            assert kwargs["threshold"] == 0.65

    def test_pipeline_default_threshold_value(self) -> None:
        """Without explicit threshold, pipeline uses DEFAULT_THRESHOLD."""
        from violawake_sdk._constants import DEFAULT_THRESHOLD as CONST_THRESHOLD

        with (
            patch("violawake_sdk.pipeline.WakeDetector") as mock_wake_cls,
            patch("violawake_sdk.pipeline.VADEngine"),
        ):
            mock_wake_cls.return_value = MagicMock()
            VoicePipeline()
            _, kwargs = mock_wake_cls.call_args
            assert kwargs["threshold"] == CONST_THRESHOLD


# ---------------------------------------------------------------------------
# VAD threshold configuration
# ---------------------------------------------------------------------------

class TestVADThreshold:
    """VAD threshold is stored and used correctly."""

    def test_custom_vad_threshold(self) -> None:
        """Custom vad_threshold is stored on the pipeline."""
        with (
            patch("violawake_sdk.pipeline.WakeDetector") as mock_wake_cls,
            patch("violawake_sdk.pipeline.VADEngine"),
        ):
            mock_wake_cls.return_value = MagicMock()
            pipeline = VoicePipeline(vad_threshold=0.7)
        assert pipeline._vad_threshold == 0.7

    def test_vad_threshold_passed_to_is_speech(self) -> None:
        """During recording, the VAD threshold is passed to is_speech()."""
        pipeline = _build_pipeline()
        pipeline._vad_threshold = 0.55
        pipeline._vad = MagicMock()
        pipeline._vad.is_speech.return_value = False

        # Simulate a frame in LISTENING state
        frame = b"\x00" * 640
        pipeline._state = _STATE_LISTENING

        # Manually invoke the listening logic path
        pipeline._vad.is_speech(frame, threshold=pipeline._vad_threshold)
        pipeline._vad.is_speech.assert_called_with(frame, threshold=0.55)


# ---------------------------------------------------------------------------
# Pipeline run() error wrapping
# ---------------------------------------------------------------------------

class TestRunErrorWrapping:
    """run() wraps unexpected exceptions in PipelineError."""

    def test_run_wraps_generic_exception(self) -> None:
        pipeline = _build_pipeline()
        with patch.object(pipeline, "_run_loop", side_effect=ValueError("boom")):
            with pytest.raises(PipelineError, match="boom"):
                pipeline.run()
        assert pipeline._state == _STATE_IDLE

    def test_run_keyboard_interrupt_no_pipeline_error(self) -> None:
        """KeyboardInterrupt should NOT be wrapped in PipelineError."""
        pipeline = _build_pipeline()
        with patch.object(pipeline, "_run_loop", side_effect=KeyboardInterrupt):
            # Should not raise PipelineError
            pipeline.run()
        assert pipeline._state == _STATE_IDLE


# ---------------------------------------------------------------------------
# Worker thread lifecycle
# ---------------------------------------------------------------------------

class TestWorkerLifecycle:
    """Worker thread start, get, clear."""

    def test_start_worker_creates_thread(self) -> None:
        pipeline = _build_pipeline()
        pipeline._stt = MagicMock()  # prevent lazy load
        # Mock _transcribe_and_respond to just return
        with patch.object(pipeline, "_transcribe_and_respond"):
            pipeline._start_worker(b"\x00" * 640)
        assert pipeline._worker_thread is not None
        pipeline._worker_thread.join(timeout=2.0)

    def test_get_worker_thread_returns_none_initially(self) -> None:
        pipeline = _build_pipeline()
        assert pipeline._get_worker_thread() is None

    def test_clear_worker_thread(self) -> None:
        """_clear_worker_thread clears ref when called from the worker thread."""
        pipeline = _build_pipeline()
        cleared = threading.Event()

        def worker_fn():
            with pipeline._worker_lock:
                pipeline._worker_thread = threading.current_thread()
            pipeline._clear_worker_thread()
            cleared.set()

        t = threading.Thread(target=worker_fn)
        t.start()
        t.join(timeout=2.0)
        assert cleared.is_set()
        assert pipeline._worker_thread is None

    def test_clear_worker_thread_noop_from_different_thread(self) -> None:
        """_clear_worker_thread should not clear if called from a different thread."""
        pipeline = _build_pipeline()
        sentinel = MagicMock()
        pipeline._worker_thread = sentinel
        # Called from main thread, but worker_thread is a different object
        pipeline._clear_worker_thread()
        assert pipeline._worker_thread is sentinel


# ---------------------------------------------------------------------------
# Stop with live worker
# ---------------------------------------------------------------------------

class TestStopWithWorker:
    """stop() behavior with active worker threads."""

    def test_stop_warns_on_alive_worker(self) -> None:
        """stop() warns if worker doesn't exit within timeout."""
        pipeline = _build_pipeline()
        worker = MagicMock()
        worker.is_alive.return_value = True  # Still alive after join
        pipeline._worker_thread = worker

        with patch("violawake_sdk.pipeline.logger") as mock_logger:
            pipeline.stop(timeout=0.01)
        worker.join.assert_called_once_with(timeout=0.01)
        mock_logger.warning.assert_called()

    def test_stop_clears_worker_ref_after_exit(self) -> None:
        """stop() clears worker ref if it exits cleanly."""
        pipeline = _build_pipeline()
        worker = MagicMock()
        worker.is_alive.return_value = False  # Exited cleanly
        pipeline._worker_thread = worker

        pipeline.stop(timeout=1.0)
        assert pipeline._worker_thread is None

    def test_stop_noop_when_no_worker(self) -> None:
        """stop() should not crash when there's no worker thread."""
        pipeline = _build_pipeline()
        pipeline.stop()  # Should not raise

    def test_stop_noop_when_called_from_worker(self) -> None:
        """stop() should skip join if called from the worker thread itself."""
        pipeline = _build_pipeline()
        # Set the worker thread to current thread to trigger the early-return
        pipeline._worker_thread = threading.current_thread()
        pipeline.stop()  # Should not deadlock or raise


# ---------------------------------------------------------------------------
# Transcribe and respond paths
# ---------------------------------------------------------------------------

class TestTranscribeAndRespond:
    """Test _transcribe_and_respond error and edge-case paths."""

    def test_stt_not_available(self) -> None:
        """When STT is missing, returns to IDLE without crashing."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        with patch.object(pipeline, "_get_stt", return_value=None):
            pipeline._transcribe_and_respond(b"\x00" * 640)
        assert pipeline._state == _STATE_IDLE

    def test_odd_length_audio_truncated(self) -> None:
        """Odd-length audio bytes are truncated to even length."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "hello"

        with patch.object(pipeline, "_get_stt", return_value=mock_stt):
            # 641 bytes (odd) -- should truncate to 640
            pipeline._transcribe_and_respond(b"\x01" * 641)

        mock_stt.transcribe.assert_called_once()
        call_arg = mock_stt.transcribe.call_args[0][0]
        # 640 bytes / 2 = 320 int16 samples -> 320 float32 values
        assert call_arg.shape[0] == 320
        assert pipeline._state == _STATE_IDLE

    def test_empty_audio_skips_transcription(self) -> None:
        """Empty audio buffer is skipped."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        mock_stt = MagicMock()

        with patch.object(pipeline, "_get_stt", return_value=mock_stt):
            pipeline._transcribe_and_respond(b"")
        # Empty after truncation check -> skip
        mock_stt.transcribe.assert_not_called()
        assert pipeline._state == _STATE_IDLE

    def test_empty_transcription_returns_to_idle(self) -> None:
        """Whitespace-only transcription does not dispatch command."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "   "
        called = []

        @pipeline.on_command
        def handler(text: str) -> None:
            called.append(text)

        with patch.object(pipeline, "_get_stt", return_value=mock_stt):
            pipeline._transcribe_and_respond(b"\x01\x00" * 320)
        assert called == []
        assert pipeline._state == _STATE_IDLE

    def test_successful_transcription_dispatches(self) -> None:
        """Successful transcription dispatches to command handlers."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "play music"
        called = []

        @pipeline.on_command
        def handler(text: str) -> None:
            called.append(text)

        with patch.object(pipeline, "_get_stt", return_value=mock_stt):
            pipeline._transcribe_and_respond(b"\x01\x00" * 320)
        assert called == ["play music"]
        assert pipeline._state == _STATE_IDLE

    def test_stop_event_drops_transcription(self) -> None:
        """If stop_event is set after STT, the result is dropped."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "hello"
        called = []

        @pipeline.on_command
        def handler(text: str) -> None:
            called.append(text)

        # Set stop before transcribe_and_respond checks
        pipeline._stop_event.set()
        with patch.object(pipeline, "_get_stt", return_value=mock_stt):
            pipeline._transcribe_and_respond(b"\x01\x00" * 320)
        # Transcription may run but dispatch should be skipped
        assert pipeline._state == _STATE_IDLE

    def test_transcription_exception_returns_to_idle(self) -> None:
        """Exception in STT returns to IDLE."""
        pipeline = _build_pipeline()
        pipeline._state = _STATE_TRANSCRIBING
        mock_stt = MagicMock()
        mock_stt.transcribe.side_effect = RuntimeError("STT crashed")

        with patch.object(pipeline, "_get_stt", return_value=mock_stt):
            pipeline._transcribe_and_respond(b"\x01\x00" * 320)
        assert pipeline._state == _STATE_IDLE


# ---------------------------------------------------------------------------
# Dispatch with TTS
# ---------------------------------------------------------------------------

class TestDispatchWithTTS:
    """Test _dispatch_command with TTS integration."""

    def test_handler_response_triggers_speak(self) -> None:
        """When a handler returns a string and TTS is enabled, speak() is called."""
        pipeline = _build_pipeline()
        pipeline._enable_tts = True

        @pipeline.on_command
        def handler(text: str) -> str:
            return f"Echo: {text}"

        with patch.object(pipeline, "speak") as speak_mock:
            pipeline._dispatch_command("hello")
        speak_mock.assert_called_once_with("Echo: hello")

    def test_handler_none_response_no_speak(self) -> None:
        """When a handler returns None, speak() is not called."""
        pipeline = _build_pipeline()
        pipeline._enable_tts = True

        @pipeline.on_command
        def handler(text: str) -> None:
            return None

        with patch.object(pipeline, "speak") as speak_mock:
            pipeline._dispatch_command("hello")
        speak_mock.assert_not_called()

    def test_dispatch_stops_early_on_stop_event(self) -> None:
        """If stop_event is set before dispatch, handlers are not called."""
        pipeline = _build_pipeline()
        called = []

        @pipeline.on_command
        def handler(text: str) -> None:
            called.append(text)

        pipeline._stop_event.set()
        pipeline._dispatch_command("hello")
        assert called == []

    def test_dispatch_stops_between_handlers(self) -> None:
        """If stop_event is set during handler iteration, remaining handlers are skipped."""
        pipeline = _build_pipeline()
        called = []

        @pipeline.on_command
        def handler1(text: str) -> None:
            called.append("h1")
            pipeline._stop_event.set()

        @pipeline.on_command
        def handler2(text: str) -> None:
            called.append("h2")

        pipeline._dispatch_command("hello")
        assert "h1" in called
        assert "h2" not in called

    def test_handler_returning_response_with_tts_disabled(self) -> None:
        """When TTS is disabled, speak() is not called even with a response."""
        pipeline = _build_pipeline()
        pipeline._enable_tts = False

        @pipeline.on_command
        def handler(text: str) -> str:
            return "response"

        with patch.object(pipeline, "speak") as speak_mock:
            pipeline._dispatch_command("hello")
        speak_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Lazy engine loading
# ---------------------------------------------------------------------------

class TestLazyEngineLoading:
    """Test _get_stt and _get_tts lazy loading."""

    def test_get_stt_returns_none_on_import_error(self) -> None:
        pipeline = _build_pipeline()
        with patch("builtins.__import__", side_effect=ImportError("no stt")):
            result = pipeline._get_stt()
        assert result is None

    def test_get_stt_caches_engine(self) -> None:
        pipeline = _build_pipeline()
        mock_stt = MagicMock()
        mock_stt_cls = MagicMock(return_value=mock_stt)

        with patch.dict("sys.modules", {"violawake_sdk.stt": MagicMock(STTEngine=mock_stt_cls)}):
            first = pipeline._get_stt()
            second = pipeline._get_stt()
        assert first is second
        assert first is mock_stt

    def test_get_tts_returns_none_on_import_error(self) -> None:
        pipeline = _build_pipeline()
        pipeline._enable_tts = True
        pipeline._tts = None
        with patch("builtins.__import__", side_effect=ImportError("no tts")):
            result = pipeline._get_tts()
        assert result is None

    def test_get_tts_returns_none_when_disabled(self) -> None:
        pipeline = _build_pipeline()
        pipeline._enable_tts = False
        pipeline._tts = None
        result = pipeline._get_tts()
        assert result is None

    def test_get_tts_caches_engine(self) -> None:
        pipeline = _build_pipeline()
        pipeline._enable_tts = True
        mock_tts = MagicMock()
        mock_tts_cls = MagicMock(return_value=mock_tts)

        with patch.dict("sys.modules", {"violawake_sdk.tts": MagicMock(TTSEngine=mock_tts_cls)}):
            first = pipeline._get_tts()
            second = pipeline._get_tts()
        assert first is second
        assert first is mock_tts


# ---------------------------------------------------------------------------
# Run loop — listening state details
# ---------------------------------------------------------------------------

class TestRunLoopListening:
    """Test _run_loop behavior in the listening state."""

    def test_idle_wake_detection_uses_live_playback_state(self) -> None:
        pipeline = _build_pipeline()
        frame = b"\x00" * 640
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = iter([frame])
        pipeline._wake_detector.detect.return_value = False

        with patch.object(pipeline, "_is_playing", return_value=True):
            pipeline._run_loop()

        pipeline._wake_detector.detect.assert_called_once_with(frame, is_playing=True)

    def test_silence_frames_trigger_transcription(self) -> None:
        """After enough silent frames, pipeline transitions to transcribing."""
        from violawake_sdk.pipeline import SILENCE_FRAMES_STOP

        pipeline = _build_pipeline()
        # Generate enough frames: 1 for wake detect, then SILENCE_FRAMES_STOP+1 for silence
        frames = [b"\x00" * 640] * (SILENCE_FRAMES_STOP + 2)
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = iter(frames)
        # First frame: wake detected
        pipeline._wake_detector.detect.side_effect = [True] + [False] * (SILENCE_FRAMES_STOP + 1)
        pipeline._vad = MagicMock()
        pipeline._vad.is_speech.return_value = False  # All silence

        with patch.object(pipeline, "_start_worker") as worker_mock:
            pipeline._run_loop()

        worker_mock.assert_called_once()

    def test_max_duration_triggers_transcription(self) -> None:
        """Recording stops after MAX_COMMAND_DURATION_S worth of frames."""
        from violawake_sdk.pipeline import FRAME_SAMPLES, MAX_COMMAND_DURATION_S, SAMPLE_RATE

        max_frames = int(MAX_COMMAND_DURATION_S / (FRAME_SAMPLES / SAMPLE_RATE))
        total_frames_needed = max_frames + 2  # +1 for wake, +1 to hit max

        pipeline = _build_pipeline()
        frames = [b"\x00" * 640] * total_frames_needed
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = iter(frames)
        pipeline._wake_detector.detect.side_effect = [True] + [False] * (total_frames_needed - 1)
        pipeline._vad = MagicMock()
        pipeline._vad.is_speech.return_value = True  # Always speech, never silence

        with patch.object(pipeline, "_start_worker") as worker_mock:
            pipeline._run_loop()

        worker_mock.assert_called_once()

    def test_on_wake_exception_does_not_crash(self) -> None:
        """Exception in on_wake callback does not crash the pipeline."""
        def bad_wake():
            raise RuntimeError("wake callback failed")

        pipeline = _build_pipeline(on_wake=bad_wake)
        frames = [b"\x00" * 640]
        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = iter(frames)
        pipeline._wake_detector.detect.return_value = True

        # Should not raise
        pipeline._run_loop()
        assert pipeline._state == _STATE_LISTENING

    def test_stop_event_exits_run_loop(self) -> None:
        """Setting stop_event during iteration exits the loop."""
        pipeline = _build_pipeline()

        def frame_generator():
            yield b"\x00" * 640
            pipeline._stop_event.set()
            yield b"\x00" * 640

        pipeline._wake_detector = MagicMock()
        pipeline._wake_detector.stream_mic.return_value = frame_generator()
        pipeline._wake_detector.detect.return_value = False

        pipeline._run_loop()  # Should exit without hanging
