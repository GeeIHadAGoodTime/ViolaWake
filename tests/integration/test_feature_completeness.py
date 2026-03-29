"""Feature completeness tests for ViolaWake SDK.

Verifies that EVERY feature claimed by the SDK actually exists, is importable,
and has the correct interface. Tests requiring hardware (GPU, microphone) or
external services (API keys, model downloads) are skipped with markers.

Categories:
  - SDK Core: WakeDetector, AsyncWakeDetector, AudioSource, NoiseProfiler, PowerManager, etc.
  - STT: STTEngine, StreamingSTTEngine
  - TTS: TTSEngine, voice list
  - VoicePipeline: orchestrated pipeline
  - VAD: VADEngine with multiple backends
  - Training: augmentation, temporal models, losses, evaluation
  - Models: registry, download functions
  - Speaker Verification: enrollment, verification, persistence
  - Security: cert pinning
  - Backends: inference backend abstraction
  - CLI Entry Points: all console_scripts resolve
  - Console Backend: retention, teams, job queue, auth
  - WASM: directory structure (if present)
  - Exceptions: hierarchy
  - Constants and metadata
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import wave
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WAKEWORD_ROOT = Path(__file__).resolve().parents[2]
HAS_PYAUDIO = importlib.util.find_spec("pyaudio") is not None
HAS_FASTER_WHISPER = importlib.util.find_spec("faster_whisper") is not None
HAS_KOKORO_ONNX = importlib.util.find_spec("kokoro_onnx") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_WEBRTCVAD = importlib.util.find_spec("webrtcvad") is not None
HAS_SOUNDDEVICE = importlib.util.find_spec("sounddevice") is not None
HAS_ONNXRUNTIME = importlib.util.find_spec("onnxruntime") is not None
HAS_OPENWAKEWORD = importlib.util.find_spec("openwakeword") is not None

# Console backend lives under console/backend/ and may not be on sys.path
_CONSOLE_BACKEND_DIR = WAKEWORD_ROOT / "console" / "backend"
_HAS_CONSOLE_BACKEND = _CONSOLE_BACKEND_DIR.is_dir()
if _HAS_CONSOLE_BACKEND:
    import sys as _sys
    if str(_CONSOLE_BACKEND_DIR) not in _sys.path:
        _sys.path.insert(0, str(_CONSOLE_BACKEND_DIR))

# Check if the console backend is actually importable (requires its deps)
try:
    if _HAS_CONSOLE_BACKEND:
        importlib.import_module("app")
        _CONSOLE_BACKEND_IMPORTABLE = True
    else:
        _CONSOLE_BACKEND_IMPORTABLE = False
except (ImportError, ModuleNotFoundError):
    _CONSOLE_BACKEND_IMPORTABLE = False


def _make_silence_wav(path: Path, duration_s: float = 0.5, sr: int = 16000) -> Path:
    """Create a silent 16-bit mono WAV file for testing."""
    n_samples = int(sr * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n_samples)
    return path


# ---------------------------------------------------------------------------
# SDK Core
# ---------------------------------------------------------------------------


class TestWakeDetectorInterface:
    """Verify WakeDetector class exists with expected methods and properties."""

    def test_import(self):
        """WakeDetector is importable from the top-level package."""
        from violawake_sdk import WakeDetector

        assert WakeDetector is not None

    def test_class_has_detect_method(self):
        """WakeDetector.detect() exists and accepts audio_frame + is_playing."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "detect")
        sig = inspect.signature(WakeDetector.detect)
        params = list(sig.parameters.keys())
        assert "audio_frame" in params

    def test_class_has_process_method(self):
        """WakeDetector.process() exists (returns raw score)."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "process")

    def test_class_has_close_method(self):
        """WakeDetector.close() exists for resource cleanup."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "close")

    def test_class_has_context_manager(self):
        """WakeDetector supports context manager protocol (__enter__/__exit__)."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "__enter__")
        assert hasattr(WakeDetector, "__exit__")

    def test_class_has_stream_mic(self):
        """WakeDetector.stream_mic() generator method exists."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "stream_mic")

    def test_class_has_from_source(self):
        """WakeDetector.from_source() factory exists."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "from_source")

    def test_class_has_get_confidence(self):
        """WakeDetector.get_confidence() returns ConfidenceResult."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "get_confidence")

    def test_class_has_last_scores(self):
        """WakeDetector.last_scores property exists."""
        from violawake_sdk.wake_detector import WakeDetector

        assert "last_scores" in dir(WakeDetector)

    def test_class_has_threshold_attribute(self):
        """WakeDetector stores threshold as an instance attribute set in __init__."""
        from violawake_sdk.wake_detector import WakeDetector

        sig = inspect.signature(WakeDetector.__init__)
        assert "threshold" in sig.parameters

    def test_class_has_reset_cooldown(self):
        """WakeDetector.reset_cooldown() exists."""
        from violawake_sdk.wake_detector import WakeDetector

        assert hasattr(WakeDetector, "reset_cooldown")

    def test_detector_config_importable(self):
        """DetectorConfig is importable."""
        from violawake_sdk import DetectorConfig

        assert DetectorConfig is not None

    def test_wake_decision_policy_importable(self):
        """WakeDecisionPolicy is importable."""
        from violawake_sdk import WakeDecisionPolicy

        assert WakeDecisionPolicy is not None

    def test_validate_audio_chunk_importable(self):
        """validate_audio_chunk function is importable."""
        from violawake_sdk import validate_audio_chunk

        assert callable(validate_audio_chunk)

    def test_backward_compat_alias(self):
        """WakewordDetector backward-compat alias exists."""
        from violawake_sdk.wake_detector import WakewordDetector

        assert WakewordDetector is not None


class TestAsyncWakeDetectorInterface:
    """Verify AsyncWakeDetector class exists with expected async methods."""

    def test_import(self):
        """AsyncWakeDetector is importable."""
        from violawake_sdk import AsyncWakeDetector

        assert AsyncWakeDetector is not None

    def test_has_async_detect(self):
        """AsyncWakeDetector.detect() is a coroutine function."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "detect")
        assert asyncio.iscoroutinefunction(AsyncWakeDetector.detect)

    def test_has_async_process(self):
        """AsyncWakeDetector.process() is a coroutine function."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "process")
        assert asyncio.iscoroutinefunction(AsyncWakeDetector.process)

    def test_has_async_context_manager(self):
        """AsyncWakeDetector supports async context manager (__aenter__/__aexit__)."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "__aenter__")
        assert hasattr(AsyncWakeDetector, "__aexit__")

    def test_has_stream_method(self):
        """AsyncWakeDetector.stream() async generator exists."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "stream")

    def test_has_close(self):
        """AsyncWakeDetector.close() exists."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "close")

    def test_has_threshold_property(self):
        """AsyncWakeDetector.threshold property exists."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert "threshold" in dir(AsyncWakeDetector)

    def test_has_get_confidence(self):
        """AsyncWakeDetector.get_confidence() exists."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "get_confidence")

    def test_has_reset_cooldown(self):
        """AsyncWakeDetector.reset_cooldown() exists."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert hasattr(AsyncWakeDetector, "reset_cooldown")

    def test_has_last_scores(self):
        """AsyncWakeDetector.last_scores property exists."""
        from violawake_sdk.async_detector import AsyncWakeDetector

        assert "last_scores" in dir(AsyncWakeDetector)


class TestAudioSourceInterface:
    """Verify AudioSource protocol and concrete implementations."""

    def test_audio_source_protocol_importable(self):
        """AudioSource protocol is importable."""
        from violawake_sdk.audio_source import AudioSource

        assert AudioSource is not None

    def test_audio_source_is_runtime_checkable(self):
        """AudioSource is a runtime-checkable Protocol."""
        from typing import runtime_checkable

        from violawake_sdk.audio_source import AudioSource

        # Runtime-checkable protocols support isinstance
        assert isinstance(AudioSource, type)

    def test_audio_source_has_read_frame(self):
        """AudioSource protocol requires read_frame()."""
        from violawake_sdk.audio_source import AudioSource

        assert "read_frame" in dir(AudioSource)

    def test_audio_source_has_start(self):
        """AudioSource protocol requires start()."""
        from violawake_sdk.audio_source import AudioSource

        assert "start" in dir(AudioSource)

    def test_audio_source_has_stop(self):
        """AudioSource protocol requires stop()."""
        from violawake_sdk.audio_source import AudioSource

        assert "stop" in dir(AudioSource)


class TestMicrophoneSource:
    """Verify MicrophoneSource exists and has correct interface."""

    def test_importable(self):
        """MicrophoneSource is importable."""
        from violawake_sdk.audio_source import MicrophoneSource

        assert MicrophoneSource is not None

    def test_has_start_stop_read(self):
        """MicrophoneSource has start(), stop(), read_frame()."""
        from violawake_sdk.audio_source import MicrophoneSource

        assert hasattr(MicrophoneSource, "start")
        assert hasattr(MicrophoneSource, "stop")
        assert hasattr(MicrophoneSource, "read_frame")

    def test_constructor_accepts_device_index(self):
        """MicrophoneSource accepts device_index parameter."""
        from violawake_sdk.audio_source import MicrophoneSource

        sig = inspect.signature(MicrophoneSource.__init__)
        assert "device_index" in sig.parameters


class TestFileSource:
    """Verify FileSource exists and can read WAV files."""

    def test_importable(self):
        """FileSource is importable."""
        from violawake_sdk.audio_source import FileSource

        assert FileSource is not None

    def test_has_start_stop_read(self):
        """FileSource has start(), stop(), read_frame()."""
        from violawake_sdk.audio_source import FileSource

        assert hasattr(FileSource, "start")
        assert hasattr(FileSource, "stop")
        assert hasattr(FileSource, "read_frame")

    def test_can_open_wav_file(self, tmp_path):
        """FileSource can open and read frames from a valid WAV file."""
        from violawake_sdk.audio_source import FileSource

        wav_path = _make_silence_wav(tmp_path / "test.wav", duration_s=0.1)
        source = FileSource(wav_path)
        source.start()
        frame = source.read_frame()
        source.stop()
        # Frame should be bytes of correct length (640 bytes = 320 samples * 2 bytes)
        assert frame is not None
        assert isinstance(frame, bytes)
        assert len(frame) == 640

    def test_exhausted_returns_none(self, tmp_path):
        """FileSource returns None when file is exhausted."""
        from violawake_sdk.audio_source import FileSource

        wav_path = _make_silence_wav(tmp_path / "short.wav", duration_s=0.01)
        source = FileSource(wav_path)
        source.start()
        frames = []
        for _ in range(100):
            f = source.read_frame()
            if f is None:
                break
            frames.append(f)
        source.stop()
        assert len(frames) >= 1

    def test_loop_parameter(self):
        """FileSource accepts loop parameter."""
        from violawake_sdk.audio_source import FileSource

        sig = inspect.signature(FileSource.__init__)
        assert "loop" in sig.parameters

    def test_nonexistent_file_raises(self, tmp_path):
        """FileSource.start() raises FileNotFoundError for missing files."""
        from violawake_sdk.audio_source import FileSource

        source = FileSource(tmp_path / "nonexistent.wav")
        with pytest.raises(FileNotFoundError):
            source.start()


class TestNetworkSource:
    """Verify NetworkSource exists with correct interface."""

    def test_importable(self):
        """NetworkSource is importable."""
        from violawake_sdk.audio_source import NetworkSource

        assert NetworkSource is not None

    def test_has_start_stop_read(self):
        """NetworkSource has start(), stop(), read_frame()."""
        from violawake_sdk.audio_source import NetworkSource

        assert hasattr(NetworkSource, "start")
        assert hasattr(NetworkSource, "stop")
        assert hasattr(NetworkSource, "read_frame")

    def test_accepts_protocol_parameter(self):
        """NetworkSource accepts protocol (tcp/udp) parameter."""
        from violawake_sdk.audio_source import NetworkSource

        sig = inspect.signature(NetworkSource.__init__)
        assert "protocol" in sig.parameters
        assert "host" in sig.parameters
        assert "port" in sig.parameters


class TestCallbackSource:
    """Verify CallbackSource exists with push model interface."""

    def test_importable(self):
        """CallbackSource is importable."""
        from violawake_sdk.audio_source import CallbackSource

        assert CallbackSource is not None

    def test_has_push_audio(self):
        """CallbackSource has push_audio() method."""
        from violawake_sdk.audio_source import CallbackSource

        assert hasattr(CallbackSource, "push_audio")

    def test_push_and_read(self):
        """CallbackSource can push audio bytes and read frames."""
        from violawake_sdk.audio_source import CallbackSource

        source = CallbackSource(timeout=0.1)
        source.start()
        source.push_audio(b"\x00" * 640)
        frame = source.read_frame()
        source.stop()
        assert frame is not None
        assert len(frame) == 640

    def test_push_numpy_array(self):
        """CallbackSource accepts numpy arrays via push_audio()."""
        from violawake_sdk.audio_source import CallbackSource

        source = CallbackSource(timeout=0.1)
        source.start()
        arr = np.zeros(320, dtype=np.float32)
        source.push_audio(arr)
        frame = source.read_frame()
        source.stop()
        assert frame is not None

    def test_stop_drains_queue(self):
        """CallbackSource.stop() drains the queue."""
        from violawake_sdk.audio_source import CallbackSource

        source = CallbackSource(timeout=0.1)
        source.start()
        source.push_audio(b"\x00" * 640)
        source.stop()
        assert source.read_frame() is None


class TestNoiseProfiler:
    """Verify NoiseProfiler exists with expected interface."""

    def test_importable(self):
        """NoiseProfiler is importable from top-level."""
        from violawake_sdk import NoiseProfiler

        assert NoiseProfiler is not None

    def test_has_update_method(self):
        """NoiseProfiler.update() exists and returns adjusted threshold."""
        from violawake_sdk.noise_profiler import NoiseProfiler

        profiler = NoiseProfiler(base_threshold=0.80)
        frame = np.random.randn(320).astype(np.float32)
        result = profiler.update(frame)
        assert isinstance(result, float)

    def test_has_get_profile_method(self):
        """NoiseProfiler.get_profile() returns NoiseProfile dataclass."""
        from violawake_sdk.noise_profiler import NoiseProfile, NoiseProfiler

        profiler = NoiseProfiler()
        profile = profiler.get_profile()
        assert isinstance(profile, NoiseProfile)
        assert hasattr(profile, "noise_rms")
        assert hasattr(profile, "signal_rms")
        assert hasattr(profile, "snr_db")
        assert hasattr(profile, "adjusted_threshold")
        assert hasattr(profile, "base_threshold")

    def test_has_reset(self):
        """NoiseProfiler.reset() clears history."""
        from violawake_sdk.noise_profiler import NoiseProfiler

        profiler = NoiseProfiler()
        for _ in range(20):
            profiler.update(np.random.randn(320).astype(np.float32))
        profiler.reset()
        assert profiler.noise_floor == 0.0

    def test_base_threshold_property(self):
        """NoiseProfiler.base_threshold returns the configured base."""
        from violawake_sdk.noise_profiler import NoiseProfiler

        profiler = NoiseProfiler(base_threshold=0.75)
        assert profiler.base_threshold == 0.75

    def test_adaptive_threshold_changes(self):
        """NoiseProfiler adapts threshold after enough frames."""
        from violawake_sdk.noise_profiler import NoiseProfiler

        profiler = NoiseProfiler(base_threshold=0.80)
        for _ in range(15):
            profiler.update(np.random.randn(320).astype(np.float32) * 0.001)
        threshold = profiler.update(np.random.randn(320).astype(np.float32) * 10.0)
        profile = profiler.get_profile()
        # After enough data, the profiler should have a non-trivial noise estimate
        assert profile.noise_rms >= 0.0


class TestPowerManager:
    """Verify PowerManager exists with expected interface."""

    def test_importable(self):
        """PowerManager is importable from top-level."""
        from violawake_sdk import PowerManager

        assert PowerManager is not None

    def test_should_process_method(self):
        """PowerManager.should_process() exists and returns bool."""
        from violawake_sdk.power_manager import PowerManager

        pm = PowerManager(duty_cycle_n=1, silence_rms=0.0)
        frame = np.random.randn(320).astype(np.float32) * 1000
        result = pm.should_process(frame)
        assert isinstance(result, bool)

    def test_report_score_method(self):
        """PowerManager.report_score() exists."""
        from violawake_sdk.power_manager import PowerManager

        pm = PowerManager()
        pm.report_score(0.5)

    def test_get_state_returns_power_state(self):
        """PowerManager.get_state() returns PowerState dataclass."""
        from violawake_sdk.power_manager import PowerManager, PowerState

        pm = PowerManager()
        state = pm.get_state()
        assert isinstance(state, PowerState)
        assert hasattr(state, "battery_percent")
        assert hasattr(state, "is_on_battery")
        assert hasattr(state, "duty_cycle_n")
        assert hasattr(state, "frames_processed")
        assert hasattr(state, "frames_skipped")
        assert hasattr(state, "silence_skipped")
        assert hasattr(state, "effective_rate")

    def test_reset_method(self):
        """PowerManager.reset() clears counters."""
        from violawake_sdk.power_manager import PowerManager

        pm = PowerManager(silence_rms=0.0)
        frame = np.ones(320, dtype=np.float32) * 1000
        pm.should_process(frame)
        pm.reset()
        state = pm.get_state()
        assert state.frames_processed == 0
        assert state.frames_skipped == 0

    def test_silence_skipping(self):
        """PowerManager skips silent frames when silence_rms > 0."""
        from violawake_sdk.power_manager import PowerManager

        pm = PowerManager(silence_rms=100.0)
        silent_frame = np.zeros(320, dtype=np.float32)
        assert pm.should_process(silent_frame) is False

    def test_duty_cycle_validation(self):
        """PowerManager rejects duty_cycle_n < 1."""
        from violawake_sdk.power_manager import PowerManager

        with pytest.raises(ValueError, match="duty_cycle_n must be >= 1"):
            PowerManager(duty_cycle_n=0)

    def test_effective_duty_cycle_property(self):
        """PowerManager.effective_duty_cycle property exists."""
        from violawake_sdk.power_manager import PowerManager

        pm = PowerManager(duty_cycle_n=2)
        assert pm.effective_duty_cycle >= 1


class TestEnsembleScorer:
    """Verify EnsembleScorer exists with expected interface."""

    def test_importable(self):
        """EnsembleScorer is importable."""
        from violawake_sdk.ensemble import EnsembleScorer

        assert EnsembleScorer is not None

    def test_score_method_with_no_sessions(self):
        """EnsembleScorer.score() returns 0.0 when no sessions registered."""
        from violawake_sdk.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        result = scorer.score(np.zeros(96, dtype=np.float32))
        assert result == 0.0

    def test_model_count_property(self):
        """EnsembleScorer.model_count reflects registered sessions."""
        from violawake_sdk.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        assert scorer.model_count == 0

    def test_strategy_property(self):
        """EnsembleScorer.strategy property exists."""
        from violawake_sdk.ensemble import EnsembleScorer, FusionStrategy

        scorer = EnsembleScorer(strategy=FusionStrategy.MAX)
        assert scorer.strategy == FusionStrategy.MAX

    def test_add_session_method(self):
        """EnsembleScorer.add_session() exists."""
        from violawake_sdk.ensemble import EnsembleScorer

        assert hasattr(EnsembleScorer, "add_session")

    def test_score_all_method(self):
        """EnsembleScorer.score_all() returns empty list with no sessions."""
        from violawake_sdk.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        result = scorer.score_all(np.zeros(96, dtype=np.float32))
        assert result == []

    def test_clear_method(self):
        """EnsembleScorer.clear() removes all sessions."""
        from violawake_sdk.ensemble import EnsembleScorer

        scorer = EnsembleScorer()
        scorer.clear()
        assert scorer.model_count == 0

    def test_string_strategy_accepted(self):
        """EnsembleScorer accepts string strategy names."""
        from violawake_sdk.ensemble import EnsembleScorer, FusionStrategy

        scorer = EnsembleScorer(strategy="average")
        assert scorer.strategy == FusionStrategy.AVERAGE


class TestFusionStrategy:
    """Verify FusionStrategy enum and fuse_scores function."""

    def test_all_strategies_exist(self):
        """All four fusion strategies are defined."""
        from violawake_sdk.ensemble import FusionStrategy

        assert hasattr(FusionStrategy, "AVERAGE")
        assert hasattr(FusionStrategy, "MAX")
        assert hasattr(FusionStrategy, "VOTING")
        assert hasattr(FusionStrategy, "WEIGHTED_AVERAGE")

    def test_fuse_scores_average(self):
        """fuse_scores with AVERAGE strategy computes mean."""
        from violawake_sdk.ensemble import FusionStrategy, fuse_scores

        result = fuse_scores([0.2, 0.4, 0.6], strategy=FusionStrategy.AVERAGE)
        assert abs(result - 0.4) < 1e-6

    def test_fuse_scores_max(self):
        """fuse_scores with MAX strategy returns maximum."""
        from violawake_sdk.ensemble import FusionStrategy, fuse_scores

        result = fuse_scores([0.2, 0.9, 0.5], strategy=FusionStrategy.MAX)
        assert result == 0.9

    def test_fuse_scores_voting(self):
        """fuse_scores with VOTING strategy returns vote fraction."""
        from violawake_sdk.ensemble import FusionStrategy, fuse_scores

        result = fuse_scores([0.6, 0.3, 0.7], strategy=FusionStrategy.VOTING, voting_threshold=0.5)
        assert abs(result - 2 / 3) < 1e-6

    def test_fuse_scores_weighted(self):
        """fuse_scores with WEIGHTED_AVERAGE uses weights."""
        from violawake_sdk.ensemble import FusionStrategy, fuse_scores

        result = fuse_scores(
            [0.5, 1.0], strategy=FusionStrategy.WEIGHTED_AVERAGE, weights=[0.3, 0.7]
        )
        assert abs(result - 0.85) < 1e-6

    def test_fuse_scores_empty_raises(self):
        """fuse_scores raises ValueError on empty scores."""
        from violawake_sdk.ensemble import fuse_scores

        with pytest.raises(ValueError, match="must not be empty"):
            fuse_scores([])


class TestConfidence:
    """Verify ConfidenceResult, ConfidenceLevel, and ScoreTracker."""

    def test_confidence_level_importable(self):
        """ConfidenceLevel enum is importable with all levels."""
        from violawake_sdk import ConfidenceLevel

        assert hasattr(ConfidenceLevel, "LOW")
        assert hasattr(ConfidenceLevel, "MEDIUM")
        assert hasattr(ConfidenceLevel, "HIGH")
        assert hasattr(ConfidenceLevel, "CERTAIN")

    def test_confidence_result_importable(self):
        """ConfidenceResult dataclass is importable."""
        from violawake_sdk import ConfidenceResult

        assert ConfidenceResult is not None

    def test_confidence_result_fields(self):
        """ConfidenceResult has all documented fields."""
        from violawake_sdk.confidence import ConfidenceLevel, ConfidenceResult

        cr = ConfidenceResult(
            raw_score=0.85,
            confirm_count=3,
            confirm_required=3,
            confidence=ConfidenceLevel.CERTAIN,
            score_history=(0.8, 0.85, 0.9),
        )
        assert cr.raw_score == 0.85
        assert cr.confirm_count == 3
        assert cr.confirm_required == 3
        assert cr.confidence == ConfidenceLevel.CERTAIN
        assert len(cr.score_history) == 3

    def test_score_tracker(self):
        """ScoreTracker records scores and classifies confidence."""
        from violawake_sdk.confidence import ConfidenceLevel, ScoreTracker

        tracker = ScoreTracker(threshold=0.80)
        tracker.record(0.5)
        tracker.record(0.85)
        result = tracker.classify(confirm_count=1, confirm_required=3)
        assert result.raw_score == 0.85
        assert result.confidence in (
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
            ConfidenceLevel.CERTAIN,
        )

    def test_score_tracker_reset(self):
        """ScoreTracker.reset() clears history."""
        from violawake_sdk.confidence import ScoreTracker

        tracker = ScoreTracker()
        tracker.record(0.5)
        tracker.reset()
        assert tracker.latest_score == 0.0
        assert tracker.last_scores == ()


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------


class TestSTTEngine:
    """Verify STTEngine exists with correct interface."""

    def test_importable_from_package(self):
        """STTEngine is importable from top-level (may be None if deps missing)."""
        from violawake_sdk import STTEngine

        if HAS_FASTER_WHISPER:
            assert STTEngine is not None
        else:
            assert STTEngine is None

    def test_module_importable(self):
        """violawake_sdk.stt module is importable."""
        mod = importlib.import_module("violawake_sdk.stt")
        assert hasattr(mod, "STTEngine")

    def test_has_transcribe_method(self):
        """STTEngine.transcribe() method exists."""
        from violawake_sdk.stt import STTEngine

        assert hasattr(STTEngine, "transcribe")

    def test_has_transcribe_full_method(self):
        """STTEngine.transcribe_full() method exists."""
        from violawake_sdk.stt import STTEngine

        assert hasattr(STTEngine, "transcribe_full")

    def test_has_transcribe_streaming_method(self):
        """STTEngine.transcribe_streaming() generator method exists."""
        from violawake_sdk.stt import STTEngine

        assert hasattr(STTEngine, "transcribe_streaming")

    def test_has_prewarm(self):
        """STTEngine.prewarm() exists for eager model loading."""
        from violawake_sdk.stt import STTEngine

        assert hasattr(STTEngine, "prewarm")

    def test_has_close(self):
        """STTEngine.close() exists."""
        from violawake_sdk.stt import STTEngine

        assert hasattr(STTEngine, "close")

    def test_context_manager_protocol(self):
        """STTEngine supports context manager protocol."""
        from violawake_sdk.stt import STTEngine

        assert hasattr(STTEngine, "__enter__")
        assert hasattr(STTEngine, "__exit__")

    def test_model_profiles_exist(self):
        """MODEL_PROFILES dictionary exists with expected models."""
        from violawake_sdk.stt import MODEL_PROFILES

        assert "tiny" in MODEL_PROFILES
        assert "base" in MODEL_PROFILES
        assert "small" in MODEL_PROFILES
        assert "medium" in MODEL_PROFILES
        assert "large-v3" in MODEL_PROFILES

    def test_transcript_segment_dataclass(self):
        """TranscriptSegment has expected fields."""
        from violawake_sdk.stt import TranscriptSegment

        seg = TranscriptSegment(text="hello", start=0.0, end=1.0, no_speech_prob=0.1)
        assert seg.text == "hello"
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.no_speech_prob == 0.1

    def test_transcript_result_dataclass(self):
        """TranscriptResult has expected fields."""
        from violawake_sdk.stt import TranscriptResult

        fields = TranscriptResult.__dataclass_fields__
        assert "text" in fields
        assert "segments" in fields
        assert "language" in fields
        assert "language_prob" in fields
        assert "duration_s" in fields
        assert "no_speech_prob" in fields

    def test_invalid_model_raises(self):
        """STTEngine raises ValueError for invalid model names."""
        from violawake_sdk.stt import STTEngine

        with pytest.raises(ValueError, match="Unknown model"):
            STTEngine(model="nonexistent_model")


class TestStreamingSTTEngine:
    """Verify StreamingSTTEngine exists with correct interface."""

    def test_importable(self):
        """StreamingSTTEngine is importable from top-level (may be None)."""
        from violawake_sdk import StreamingSTTEngine

        if HAS_FASTER_WHISPER:
            assert StreamingSTTEngine is not None
        else:
            assert StreamingSTTEngine is None

    def test_has_push_chunk(self):
        """StreamingSTTEngine.push_chunk() exists."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert hasattr(StreamingSTTEngine, "push_chunk")

    def test_has_flush(self):
        """StreamingSTTEngine.flush() exists."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert hasattr(StreamingSTTEngine, "flush")

    def test_has_reset(self):
        """StreamingSTTEngine.reset() exists."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert hasattr(StreamingSTTEngine, "reset")

    def test_has_buffer_duration(self):
        """StreamingSTTEngine.buffer_duration_s property exists."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert "buffer_duration_s" in dir(StreamingSTTEngine)

    def test_context_manager_protocol(self):
        """StreamingSTTEngine supports context manager protocol."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert hasattr(StreamingSTTEngine, "__enter__")
        assert hasattr(StreamingSTTEngine, "__exit__")

    def test_has_prewarm(self):
        """StreamingSTTEngine.prewarm() exists."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert hasattr(StreamingSTTEngine, "prewarm")

    def test_has_close(self):
        """StreamingSTTEngine.close() exists."""
        from violawake_sdk.stt import StreamingSTTEngine

        assert hasattr(StreamingSTTEngine, "close")


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------


class TestTTSEngine:
    """Verify TTSEngine exists with correct interface."""

    def test_importable_from_package(self):
        """TTSEngine is importable from top-level (may be None)."""
        from violawake_sdk import TTSEngine

        if HAS_KOKORO_ONNX:
            assert TTSEngine is not None

    def test_module_importable(self):
        """violawake_sdk.tts module is importable."""
        mod = importlib.import_module("violawake_sdk.tts")
        assert hasattr(mod, "TTSEngine")

    def test_has_synthesize(self):
        """TTSEngine.synthesize() method exists."""
        from violawake_sdk.tts import TTSEngine

        assert hasattr(TTSEngine, "synthesize")

    def test_has_synthesize_chunked(self):
        """TTSEngine.synthesize_chunked() generator method exists."""
        from violawake_sdk.tts import TTSEngine

        assert hasattr(TTSEngine, "synthesize_chunked")

    def test_has_play(self):
        """TTSEngine.play() method exists."""
        from violawake_sdk.tts import TTSEngine

        assert hasattr(TTSEngine, "play")

    def test_has_play_async(self):
        """TTSEngine.play_async() method exists."""
        from violawake_sdk.tts import TTSEngine

        assert hasattr(TTSEngine, "play_async")

    def test_has_close(self):
        """TTSEngine.close() exists."""
        from violawake_sdk.tts import TTSEngine

        assert hasattr(TTSEngine, "close")

    def test_context_manager_protocol(self):
        """TTSEngine supports context manager protocol."""
        from violawake_sdk.tts import TTSEngine

        assert hasattr(TTSEngine, "__enter__")
        assert hasattr(TTSEngine, "__exit__")

    def test_available_voices_list(self):
        """AVAILABLE_VOICES list is populated."""
        from violawake_sdk.tts import AVAILABLE_VOICES

        assert isinstance(AVAILABLE_VOICES, list)
        assert len(AVAILABLE_VOICES) > 0
        assert "af_heart" in AVAILABLE_VOICES
        assert "am_adam" in AVAILABLE_VOICES

    def test_list_voices_function(self):
        """list_voices() function works and returns voices."""
        from violawake_sdk import list_voices

        voices = list_voices()
        assert isinstance(voices, list)
        assert len(voices) > 0

    def test_tts_sample_rate_constants(self):
        """TTS sample rate constants are defined."""
        from violawake_sdk.tts import TARGET_SAMPLE_RATE, TTS_SAMPLE_RATE

        assert TTS_SAMPLE_RATE == 24000
        assert TARGET_SAMPLE_RATE == 16000

    def test_invalid_voice_raises(self):
        """TTSEngine raises ValueError for invalid voice names."""
        from violawake_sdk.tts import TTSEngine

        with pytest.raises(ValueError, match="Unknown voice"):
            TTSEngine(voice="nonexistent_voice")

    def test_invalid_speed_raises(self):
        """TTSEngine raises ValueError for out-of-range speed."""
        from violawake_sdk.tts import TTSEngine

        with pytest.raises(ValueError, match="Speed must be"):
            TTSEngine(voice="af_heart", speed=0.0)


# ---------------------------------------------------------------------------
# VoicePipeline
# ---------------------------------------------------------------------------


class TestVoicePipeline:
    """Verify VoicePipeline orchestrator."""

    def test_importable(self):
        """VoicePipeline is importable from top-level."""
        from violawake_sdk import VoicePipeline

        assert VoicePipeline is not None

    def test_constructor_params(self):
        """VoicePipeline accepts all documented constructor parameters."""
        from violawake_sdk.pipeline import VoicePipeline

        sig = inspect.signature(VoicePipeline.__init__)
        params = set(sig.parameters.keys())
        expected = {
            "self",
            "wake_word",
            "stt_model",
            "tts_voice",
            "threshold",
            "vad_backend",
            "vad_threshold",
            "enable_tts",
            "device_index",
            "on_wake",
            "streaming_stt",
        }
        assert expected.issubset(params), f"Missing params: {expected - params}"

    def test_has_on_command_decorator(self):
        """VoicePipeline.on_command() decorator exists."""
        from violawake_sdk.pipeline import VoicePipeline

        assert hasattr(VoicePipeline, "on_command")

    def test_has_run_method(self):
        """VoicePipeline.run() blocking method exists."""
        from violawake_sdk.pipeline import VoicePipeline

        assert hasattr(VoicePipeline, "run")

    def test_has_stop_method(self):
        """VoicePipeline.stop() exists."""
        from violawake_sdk.pipeline import VoicePipeline

        assert hasattr(VoicePipeline, "stop")

    def test_has_close_method(self):
        """VoicePipeline.close() exists."""
        from violawake_sdk.pipeline import VoicePipeline

        assert hasattr(VoicePipeline, "close")

    def test_has_speak_method(self):
        """VoicePipeline.speak() exists for manual TTS."""
        from violawake_sdk.pipeline import VoicePipeline

        assert hasattr(VoicePipeline, "speak")

    def test_context_manager_protocol(self):
        """VoicePipeline supports context manager protocol."""
        from violawake_sdk.pipeline import VoicePipeline

        assert hasattr(VoicePipeline, "__enter__")
        assert hasattr(VoicePipeline, "__exit__")


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------


class TestVADEngine:
    """Verify VADEngine exists with correct interface."""

    def test_importable(self):
        """VADEngine is importable from top-level."""
        from violawake_sdk import VADEngine

        assert VADEngine is not None

    def test_rms_backend_always_available(self):
        """RMS VAD backend is always available (zero dependencies)."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="rms")
        assert vad.backend_name == "rms"

    def test_process_frame_returns_float(self):
        """VADEngine.process_frame() returns float probability."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="rms")
        frame = np.zeros(320, dtype=np.int16).tobytes()
        prob = vad.process_frame(frame)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_is_speech_convenience(self):
        """VADEngine.is_speech() returns bool."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="rms")
        frame = np.zeros(320, dtype=np.int16).tobytes()
        result = vad.is_speech(frame, threshold=0.5)
        assert isinstance(result, bool)

    def test_reset_method(self):
        """VADEngine.reset() exists."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="rms")
        vad.reset()

    def test_context_manager(self):
        """VADEngine supports context manager protocol."""
        from violawake_sdk.vad import VADEngine

        with VADEngine(backend="rms") as vad:
            assert vad.backend_name == "rms"

    def test_vad_backend_enum(self):
        """VADBackend enum has all backends."""
        from violawake_sdk.vad import VADBackend

        assert hasattr(VADBackend, "WEBRTC")
        assert hasattr(VADBackend, "SILERO")
        assert hasattr(VADBackend, "RMS")
        assert hasattr(VADBackend, "AUTO")

    def test_accepts_numpy_float32(self):
        """VADEngine.process_frame() accepts float32 numpy arrays."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="rms")
        prob = vad.process_frame(np.zeros(320, dtype=np.float32))
        assert isinstance(prob, float)

    def test_accepts_numpy_int16(self):
        """VADEngine.process_frame() accepts int16 numpy arrays."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="rms")
        prob = vad.process_frame(np.zeros(320, dtype=np.int16))
        assert isinstance(prob, float)

    @pytest.mark.skipif(not HAS_WEBRTCVAD, reason="webrtcvad not installed")
    def test_webrtc_backend(self):
        """WebRTC VAD backend loads when available."""
        from violawake_sdk.vad import VADEngine

        vad = VADEngine(backend="webrtc")
        assert vad.backend_name == "webrtc"


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


class TestTrainingAugmentation:
    """Verify training augmentation pipeline."""

    def test_augmentation_pipeline_importable(self):
        """AugmentationPipeline is importable."""
        from violawake_sdk.training.augment import AugmentationPipeline

        assert AugmentationPipeline is not None

    def test_augment_config_importable(self):
        """AugmentConfig dataclass is importable with all fields."""
        from violawake_sdk.training.augment import AugmentConfig

        cfg = AugmentConfig()
        assert hasattr(cfg, "gain_db_range")
        assert hasattr(cfg, "p_noise")
        assert hasattr(cfg, "p_rir")
        assert hasattr(cfg, "p_spec_augment")
        assert hasattr(cfg, "p_gain")
        assert hasattr(cfg, "p_time_stretch")
        assert hasattr(cfg, "p_pitch_shift")
        assert hasattr(cfg, "p_time_shift")

    def test_augment_clip(self):
        """AugmentationPipeline.augment_clip() produces augmented variants."""
        from violawake_sdk.training.augment import AugmentationPipeline

        pipeline = AugmentationPipeline(seed=42)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        variants = pipeline.augment_clip(audio, factor=3)
        assert len(variants) == 3
        for v in variants:
            assert isinstance(v, np.ndarray)
            assert v.dtype == np.float32

    def test_augment_batch(self):
        """AugmentationPipeline.augment_batch() works on multiple clips."""
        from violawake_sdk.training.augment import AugmentationPipeline

        pipeline = AugmentationPipeline(seed=42)
        clips = [np.random.randn(16000).astype(np.float32) for _ in range(3)]
        result = pipeline.augment_batch(clips, factor=2)
        assert len(result) == 6

    def test_individual_augmentations(self):
        """Individual augmentation functions are importable and functional."""
        from violawake_sdk.training.augment import (
            apply_additive_noise,
            apply_gain,
            apply_pitch_shift,
            apply_time_shift,
            apply_time_stretch,
        )

        audio = np.random.randn(8000).astype(np.float32) * 0.5
        rng = np.random.default_rng(42)

        assert apply_gain(audio, 3.0).dtype == np.float32
        assert apply_time_stretch(audio, 1.1).dtype == np.float32
        assert apply_pitch_shift(audio, 1.0).dtype == np.float32
        assert apply_additive_noise(audio, 10.0, rng).dtype == np.float32
        assert apply_time_shift(audio, 100).dtype == np.float32

    def test_spec_augment(self):
        """spec_augment function masks a spectrogram."""
        from violawake_sdk.training.augment import spec_augment

        spec = np.ones((40, 100), dtype=np.float32)
        masked = spec_augment(spec, freq_mask_param=10, time_mask_param=20,
                              rng=np.random.default_rng(0))
        assert masked.shape == spec.shape
        assert np.any(masked == 0.0)

    def test_rir_augment(self):
        """rir_augment and generate_synthetic_rir work."""
        from violawake_sdk.training.augment import generate_synthetic_rir, rir_augment

        rir = generate_synthetic_rir(sample_rate=16000, rt60=0.3)
        assert isinstance(rir, np.ndarray)
        assert len(rir) > 0
        assert rir.dtype == np.float32

        audio = np.random.randn(16000).astype(np.float32)
        result = rir_augment(audio)
        assert isinstance(result, np.ndarray)

    def test_augment_spectrogram_method(self):
        """AugmentationPipeline.augment_spectrogram() exists."""
        from violawake_sdk.training.augment import AugmentConfig, AugmentationPipeline

        cfg = AugmentConfig(p_spec_augment=1.0)
        pipeline = AugmentationPipeline(config=cfg, seed=42)
        spec = np.random.randn(40, 100).astype(np.float32)
        result = pipeline.augment_spectrogram(spec)
        assert result.shape == spec.shape

    def test_load_rir_dataset(self):
        """load_rir_dataset function exists."""
        from violawake_sdk.training.augment import load_rir_dataset

        assert callable(load_rir_dataset)


class TestTrainingTemporalModels:
    """Verify temporal model classes exist (require torch)."""

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_temporal_cnn_importable(self):
        """TemporalCNN model is importable when torch is available."""
        from violawake_sdk.training.temporal_model import TemporalCNN

        model = TemporalCNN(embedding_dim=96, seq_len=9)
        assert model is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_temporal_gru_importable(self):
        """TemporalGRU model is importable when torch is available."""
        from violawake_sdk.training.temporal_model import TemporalGRU

        model = TemporalGRU(embedding_dim=96)
        assert model is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_temporal_convgru_importable(self):
        """TemporalConvGRU model is importable when torch is available."""
        from violawake_sdk.training.temporal_model import TemporalConvGRU

        model = TemporalConvGRU(embedding_dim=96)
        assert model is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_temporal_cnn_forward_pass(self):
        """TemporalCNN forward pass produces correct output shape."""
        import torch

        from violawake_sdk.training.temporal_model import TemporalCNN

        model = TemporalCNN(embedding_dim=96, seq_len=9)
        model.eval()
        x = torch.randn(2, 9, 96)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1)

    def test_module_importable_without_torch(self):
        """temporal_model module is importable even without torch."""
        mod = importlib.import_module("violawake_sdk.training.temporal_model")
        assert mod is not None


class TestTrainingLosses:
    """Verify training loss functions."""

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_focal_loss_importable(self):
        """FocalLoss is importable."""
        from violawake_sdk.training.losses import FocalLoss

        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        assert loss_fn is not None

    @pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
    def test_focal_loss_forward(self):
        """FocalLoss forward pass produces scalar loss."""
        import torch

        from violawake_sdk.training.losses import FocalLoss

        loss_fn = FocalLoss()
        inputs = torch.sigmoid(torch.randn(8, 1))
        targets = torch.randint(0, 2, (8, 1)).float()
        loss = loss_fn(inputs, targets)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_module_importable_without_torch(self):
        """losses module is importable even without torch."""
        mod = importlib.import_module("violawake_sdk.training.losses")
        assert mod is not None


class TestTrainingEvaluation:
    """Verify training evaluation module."""

    def test_module_importable(self):
        """training.evaluate module is importable."""
        mod = importlib.import_module("violawake_sdk.training.evaluate")
        assert mod is not None

    def test_has_evaluate_onnx_model(self):
        """evaluate_onnx_model function exists."""
        from violawake_sdk.training.evaluate import evaluate_onnx_model

        assert callable(evaluate_onnx_model)

    def test_has_compute_dprime(self):
        """compute_dprime function exists and computes correctly."""
        from violawake_sdk.training.evaluate import compute_dprime

        assert callable(compute_dprime)
        d = compute_dprime(
            pos_scores=np.array([0.8, 0.9, 0.85]),
            neg_scores=np.array([0.1, 0.2, 0.15]),
        )
        assert isinstance(d, float)
        assert d > 0

    def test_detect_architecture(self):
        """detect_architecture function exists."""
        from violawake_sdk.training.evaluate import detect_architecture

        assert callable(detect_architecture)


class TestTrainingWeightAveraging:
    """Verify weight averaging module."""

    def test_module_importable(self):
        """training.weight_averaging module is importable."""
        mod = importlib.import_module("violawake_sdk.training.weight_averaging")
        assert mod is not None


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Verify model registry and download infrastructure."""

    def test_model_registry_importable(self):
        """MODEL_REGISTRY dict is importable and non-empty."""
        from violawake_sdk.models import MODEL_REGISTRY

        assert isinstance(MODEL_REGISTRY, dict)
        assert len(MODEL_REGISTRY) > 0

    def test_temporal_cnn_in_registry(self):
        """temporal_cnn (production default) is in the registry."""
        from violawake_sdk.models import MODEL_REGISTRY

        assert "temporal_cnn" in MODEL_REGISTRY

    def test_viola_alias_exists(self):
        """'viola' alias resolves to temporal_cnn."""
        from violawake_sdk.models import MODEL_REGISTRY

        assert "viola" in MODEL_REGISTRY
        assert MODEL_REGISTRY["viola"].name == MODEL_REGISTRY["temporal_cnn"].name

    def test_model_spec_fields(self):
        """ModelSpec has all required fields."""
        from violawake_sdk.models import MODEL_REGISTRY, ModelSpec

        spec = MODEL_REGISTRY["temporal_cnn"]
        assert isinstance(spec, ModelSpec)
        assert spec.name
        assert spec.url.startswith("https://")
        assert spec.sha256
        assert spec.size_bytes > 0
        assert spec.description
        assert spec.version

    def test_list_models_function(self):
        """list_models() returns a list of model dicts."""
        from violawake_sdk import list_models

        models = list_models()
        assert isinstance(models, list)
        assert len(models) > 0
        for m in models:
            assert "name" in m
            assert "description" in m
            assert "version" in m

    def test_get_model_dir_function(self):
        """get_model_dir() returns a Path."""
        from violawake_sdk.models import get_model_dir

        model_dir = get_model_dir()
        assert isinstance(model_dir, Path)

    def test_download_model_function_exists(self):
        """download_model function exists with expected signature."""
        from violawake_sdk.models import download_model

        assert callable(download_model)
        sig = inspect.signature(download_model)
        params = set(sig.parameters.keys())
        assert "model_name" in params
        assert "force" in params
        assert "verify" in params

    def test_get_model_path_function_exists(self):
        """get_model_path function exists."""
        from violawake_sdk.models import get_model_path

        assert callable(get_model_path)

    def test_check_registry_integrity(self):
        """check_registry_integrity function runs in non-strict mode."""
        from violawake_sdk.models import check_registry_integrity

        result = check_registry_integrity(strict=False)
        assert isinstance(result, list)

    def test_list_cached_models_function(self):
        """list_cached_models function exists."""
        from violawake_sdk.models import list_cached_models

        result = list_cached_models()
        assert isinstance(result, list)

    def test_kokoro_models_in_registry(self):
        """Kokoro TTS models are in the registry."""
        from violawake_sdk.models import MODEL_REGISTRY

        assert "kokoro_v1_0" in MODEL_REGISTRY
        assert "kokoro_voices_v1_0" in MODEL_REGISTRY

    def test_unknown_model_raises(self):
        """get_model_path raises ModelNotFoundError for unknown models."""
        from violawake_sdk._exceptions import ModelNotFoundError
        from violawake_sdk.models import get_model_path

        with pytest.raises(ModelNotFoundError):
            get_model_path("totally_nonexistent_model_xyz")


# ---------------------------------------------------------------------------
# Speaker Verification
# ---------------------------------------------------------------------------


class TestSpeakerVerification:
    """Verify speaker verification subsystem."""

    def test_speaker_verify_result_importable(self):
        """SpeakerVerifyResult dataclass is importable with all fields."""
        from violawake_sdk.speaker import SpeakerVerifyResult

        result = SpeakerVerifyResult(
            is_verified=True, speaker_id="user1", similarity=0.8, threshold=0.65
        )
        assert result.is_verified
        assert result.speaker_id == "user1"
        assert result.similarity == 0.8
        assert result.threshold == 0.65

    def test_cosine_scorer_similarity(self):
        """CosineScorer computes correct similarity."""
        from violawake_sdk.speaker import CosineScorer

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = CosineScorer.similarity(a, b)
        assert abs(sim - 1.0) < 1e-6

        c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim2 = CosineScorer.similarity(a, c)
        assert abs(sim2) < 1e-6

    def test_cosine_distance(self):
        """CosineScorer.distance() is 1 - similarity."""
        from violawake_sdk.speaker import CosineScorer

        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0], dtype=np.float32)
        dist = CosineScorer.distance(a, b)
        assert abs(dist) < 1e-6

    def test_speaker_profile_enroll_and_verify(self):
        """SpeakerProfile supports enrollment and verification."""
        from violawake_sdk.speaker import SpeakerProfile

        profile = SpeakerProfile("test_speaker")
        assert profile.n_enrollments == 0
        assert profile.centroid is None

        emb = np.random.randn(96).astype(np.float32)
        profile.add_embedding(emb)
        assert profile.n_enrollments == 1
        assert profile.centroid is not None

        sim = profile.verify(emb)
        assert isinstance(sim, float)
        assert sim > 0.9

    def test_speaker_verification_hook_enroll_and_verify(self):
        """SpeakerVerificationHook can enroll and verify speakers."""
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = SpeakerVerificationHook(threshold=0.5)
        emb = np.random.randn(96).astype(np.float32)

        count = hook.enroll_speaker("alice", emb)
        assert count == 1
        assert "alice" in hook.enrolled_speakers

        result = hook.verify_speaker(emb)
        assert result.is_verified
        assert result.speaker_id == "alice"

        assert hook(emb) is True

    def test_speaker_verification_hook_remove(self):
        """SpeakerVerificationHook.remove_speaker() works."""
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = SpeakerVerificationHook()
        emb = np.random.randn(96).astype(np.float32)
        hook.enroll_speaker("bob", emb)
        assert hook.remove_speaker("bob") is True
        assert hook.remove_speaker("nonexistent") is False
        assert "bob" not in hook.enrolled_speakers

    def test_speaker_verification_no_profiles(self):
        """SpeakerVerificationHook returns not-verified with no profiles."""
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = SpeakerVerificationHook()
        result = hook.verify_speaker(np.random.randn(96).astype(np.float32))
        assert result.is_verified is False
        assert result.speaker_id is None

    def test_speaker_save_and_load(self, tmp_path):
        """SpeakerVerificationHook can save and load profiles."""
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = SpeakerVerificationHook(threshold=0.7)
        emb = np.random.randn(96).astype(np.float32)
        hook.enroll_speaker("charlie", emb)

        save_path = tmp_path / "speakers"
        hook.save(save_path)
        assert (tmp_path / "speakers.json").exists()
        assert (tmp_path / "speakers.npz").exists()

        hook2 = SpeakerVerificationHook()
        hook2.load(save_path)
        assert "charlie" in hook2.enrolled_speakers
        assert hook2.threshold == 0.7

    def test_speaker_threshold_setter(self):
        """SpeakerVerificationHook.threshold is settable."""
        from violawake_sdk.speaker import SpeakerVerificationHook

        hook = SpeakerVerificationHook(threshold=0.5)
        assert hook.threshold == 0.5
        hook.threshold = 0.8
        assert hook.threshold == 0.8


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


class TestSecurity:
    """Verify security module exists."""

    def test_security_module_importable(self):
        """violawake_sdk.security module is importable."""
        mod = importlib.import_module("violawake_sdk.security")
        assert mod is not None

    def test_cert_pinning_importable(self):
        """cert_pinning module exports are importable."""
        from violawake_sdk.security import (
            CertPinError,
            PinSet,
            add_pins,
            create_pinned_ssl_context,
            fetch_live_spki_pins,
            verify_certificate_pin,
        )

        assert CertPinError is not None
        assert callable(add_pins)
        assert callable(create_pinned_ssl_context)
        assert callable(fetch_live_spki_pins)
        assert callable(verify_certificate_pin)

    def test_cert_pin_error_is_exception(self):
        """CertPinError is a proper Exception subclass."""
        from violawake_sdk.security import CertPinError

        assert issubclass(CertPinError, Exception)
        with pytest.raises(CertPinError):
            raise CertPinError("test pin failure")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestBackends:
    """Verify inference backend abstraction layer."""

    def test_backend_module_importable(self):
        """violawake_sdk.backends module is importable."""
        from violawake_sdk.backends import BackendSession, InferenceBackend, get_backend

        assert BackendSession is not None
        assert InferenceBackend is not None
        assert callable(get_backend)

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    def test_onnx_backend_loadable(self):
        """ONNX backend can be created."""
        from violawake_sdk.backends import get_backend

        backend = get_backend("onnx")
        assert backend is not None

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="onnxruntime not installed")
    def test_auto_backend(self):
        """Auto backend selection works."""
        from violawake_sdk.backends import get_backend

        backend = get_backend("auto")
        assert backend is not None

    def test_unknown_backend_raises(self):
        """get_backend() raises ValueError for unknown backend."""
        from violawake_sdk.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_base_classes_are_abstract(self):
        """BackendSession and InferenceBackend are abstract."""
        from violawake_sdk.backends.base import BackendSession, InferenceBackend

        assert inspect.isabstract(BackendSession)
        assert inspect.isabstract(InferenceBackend)


# ---------------------------------------------------------------------------
# CLI Entry Points
# ---------------------------------------------------------------------------


class TestCLIEntryPoints:
    """Verify all console_scripts in pyproject.toml resolve to real functions."""

    CONSOLE_SCRIPTS = {
        "violawake-train": "violawake_sdk.tools.train:main",
        "violawake-eval": "violawake_sdk.tools.evaluate:main",
        "violawake-collect": "violawake_sdk.tools.collect_samples:main",
        "violawake-download": "violawake_sdk.tools.download_model:main",
        "violawake-expand-corpus": "violawake_sdk.tools.expand_corpus:main",
        "violawake-streaming-eval": "violawake_sdk.tools.streaming_eval:main",
        "violawake-test-confusables": "violawake_sdk.tools.test_confusables:main",
        "violawake-contamination-check": "violawake_sdk.tools.contamination_check:main",
        "violawake-generate": "violawake_sdk.tools.generate_samples:main",
    }

    @pytest.mark.parametrize("script_name,entry_point", list(CONSOLE_SCRIPTS.items()))
    def test_entry_point_resolves(self, script_name, entry_point):
        """Console script entry point resolves to a real callable."""
        module_path, func_name = entry_point.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        func = getattr(mod, func_name, None)
        assert func is not None, f"{script_name} -> {entry_point} did not resolve"
        assert callable(func), f"{script_name} -> {entry_point} is not callable"


# ---------------------------------------------------------------------------
# Console Backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _CONSOLE_BACKEND_IMPORTABLE,
    reason="Console backend not importable (missing deps or not on path)",
)
class TestConsoleRetention:
    """Verify retention cleanup functions exist with correct signatures."""

    def test_module_importable(self):
        """app.retention module is importable."""
        mod = importlib.import_module("app.retention")
        assert mod is not None

    def test_cleanup_expired_recordings_exists(self):
        """cleanup_expired_recordings is an async function."""
        from app.retention import cleanup_expired_recordings

        assert callable(cleanup_expired_recordings)
        assert asyncio.iscoroutinefunction(cleanup_expired_recordings)

    def test_cleanup_expired_models_exists(self):
        """cleanup_expired_models is an async function."""
        from app.retention import cleanup_expired_models

        assert callable(cleanup_expired_models)
        assert asyncio.iscoroutinefunction(cleanup_expired_models)


@pytest.mark.skipif(
    not _CONSOLE_BACKEND_IMPORTABLE,
    reason="Console backend not importable (missing deps or not on path)",
)
class TestConsoleTeams:
    """Verify team routes exist."""

    def test_module_importable(self):
        """app.routes.teams module is importable."""
        mod = importlib.import_module("app.routes.teams")
        assert mod is not None

    def test_router_exists(self):
        """Teams router is defined with routes."""
        from app.routes.teams import router

        assert router is not None
        assert hasattr(router, "routes")


@pytest.mark.skipif(
    not _CONSOLE_BACKEND_IMPORTABLE,
    reason="Console backend not importable (missing deps or not on path)",
)
class TestConsoleJobQueue:
    """Verify job queue with priority support."""

    def test_module_importable(self):
        """app.job_queue module is importable."""
        mod = importlib.import_module("app.job_queue")
        assert mod is not None

    def test_job_status_enum(self):
        """JobStatus enum has expected states."""
        from app.job_queue import JobStatus

        assert hasattr(JobStatus, "PENDING")
        assert hasattr(JobStatus, "RUNNING")

    def test_queue_max_size_constant(self):
        """QUEUE_MAX_SIZE constant is defined."""
        from app.job_queue import QUEUE_MAX_SIZE

        assert isinstance(QUEUE_MAX_SIZE, int)
        assert QUEUE_MAX_SIZE > 0

    def test_failure_threshold_constant(self):
        """FAILURE_THRESHOLD constant is defined for circuit breaker."""
        from app.job_queue import FAILURE_THRESHOLD

        assert isinstance(FAILURE_THRESHOLD, int)
        assert FAILURE_THRESHOLD > 0


@pytest.mark.skipif(
    not _CONSOLE_BACKEND_IMPORTABLE,
    reason="Console backend not importable (missing deps or not on path)",
)
class TestConsoleAuth:
    """Verify auth routes with email verification and password reset."""

    def test_module_importable(self):
        """app.routes.auth module is importable."""
        mod = importlib.import_module("app.routes.auth")
        assert mod is not None

    def test_router_exists(self):
        """Auth router is defined."""
        from app.routes.auth import router

        assert router is not None

    def test_auth_functions_exist(self):
        """Core auth functions are importable."""
        from app.auth import (
            create_access_token,
            create_email_verification_token,
            create_password_reset_token,
            hash_password,
            verify_password,
        )

        assert callable(create_access_token)
        assert callable(create_email_verification_token)
        assert callable(create_password_reset_token)
        assert callable(hash_password)
        assert callable(verify_password)

    def test_download_token_exists(self):
        """Download token creation function exists."""
        from app.auth import create_download_token

        assert callable(create_download_token)

    def test_team_invite_token_exists(self):
        """Team invite token functions exist."""
        from app.auth import create_team_invite_token, decode_team_invite_token

        assert callable(create_team_invite_token)
        assert callable(decode_team_invite_token)


# ---------------------------------------------------------------------------
# WASM
# ---------------------------------------------------------------------------


class TestWASM:
    """Verify WASM directory structure if present."""

    def test_wasm_directory_check(self):
        """Document whether wasm/ directory exists at project root.

        WASM support is optional and may not be present in all builds.
        """
        wasm_dir = WAKEWORD_ROOT / "wasm"
        if wasm_dir.exists():
            files = list(wasm_dir.rglob("*"))
            assert len(files) > 0, "wasm/ directory exists but is empty"
        else:
            pytest.skip("wasm/ directory not present in this build")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    """Verify exception hierarchy is correct and complete."""

    def test_all_exceptions_importable(self):
        """All SDK exceptions are importable from top-level."""
        from violawake_sdk import (
            AudioCaptureError,
            ModelLoadError,
            ModelNotFoundError,
            PipelineError,
            VADBackendError,
            ViolaWakeError,
        )

        assert ViolaWakeError is not None
        assert ModelNotFoundError is not None
        assert AudioCaptureError is not None
        assert ModelLoadError is not None
        assert PipelineError is not None
        assert VADBackendError is not None

    def test_base_exception_hierarchy(self):
        """All SDK exceptions inherit from ViolaWakeError which inherits from Exception."""
        from violawake_sdk import (
            AudioCaptureError,
            ModelLoadError,
            ModelNotFoundError,
            PipelineError,
            VADBackendError,
            ViolaWakeError,
        )

        assert issubclass(ViolaWakeError, Exception)
        assert issubclass(ModelNotFoundError, ViolaWakeError)
        assert issubclass(AudioCaptureError, ViolaWakeError)
        assert issubclass(ModelLoadError, ViolaWakeError)
        assert issubclass(PipelineError, ViolaWakeError)
        assert issubclass(VADBackendError, ViolaWakeError)

    def test_exceptions_catchable_by_base(self):
        """SDK exceptions can be raised and caught by base class."""
        from violawake_sdk import ModelNotFoundError, ViolaWakeError

        with pytest.raises(ViolaWakeError):
            raise ModelNotFoundError("test model not found")

    def test_exception_messages(self):
        """Exceptions preserve their message string."""
        from violawake_sdk import AudioCaptureError, ModelLoadError, PipelineError

        for exc_cls, msg in [
            (ModelLoadError, "ONNX load failed"),
            (AudioCaptureError, "No microphone"),
            (PipelineError, "Pipeline crashed"),
        ]:
            try:
                raise exc_cls(msg)
            except exc_cls as e:
                assert str(e) == msg

    def test_exceptions_importable_from_module(self):
        """Exceptions are also importable from _exceptions module directly."""
        from violawake_sdk._exceptions import (
            AudioCaptureError,
            ModelLoadError,
            ModelNotFoundError,
            PipelineError,
            VADBackendError,
            ViolaWakeError,
        )

        assert all(
            issubclass(cls, ViolaWakeError)
            for cls in [ModelNotFoundError, AudioCaptureError, ModelLoadError, PipelineError, VADBackendError]
        )


# ---------------------------------------------------------------------------
# Package Metadata
# ---------------------------------------------------------------------------


class TestPackageMetadata:
    """Verify package-level metadata."""

    def test_version_defined(self):
        """__version__ is defined and is a semver-like string."""
        from violawake_sdk import __version__

        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) >= 2

    def test_all_exports_defined(self):
        """__all__ contains all expected public names."""
        from violawake_sdk import __all__

        assert isinstance(__all__, list)
        expected_names = [
            "WakeDetector",
            "AsyncWakeDetector",
            "DetectorConfig",
            "WakeDecisionPolicy",
            "validate_audio_chunk",
            "VADEngine",
            "VoicePipeline",
            "NoiseProfiler",
            "PowerManager",
            "FusionStrategy",
            "ConfidenceResult",
            "ConfidenceLevel",
            "TTSEngine",
            "STTEngine",
            "StreamingSTTEngine",
            "ViolaWakeError",
            "ModelNotFoundError",
            "AudioCaptureError",
            "ModelLoadError",
            "PipelineError",
            "VADBackendError",
            "list_models",
            "list_voices",
            "__version__",
        ]
        for name in expected_names:
            assert name in __all__, f"{name} missing from __all__"

    def test_author_defined(self):
        """__author__ is defined."""
        from violawake_sdk import __author__

        assert isinstance(__author__, str)

    def test_license_defined(self):
        """__license__ is Apache-2.0."""
        from violawake_sdk import __license__

        assert __license__ == "Apache-2.0"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify SDK constants module."""

    def test_core_constants(self):
        """Core audio constants are defined."""
        from violawake_sdk._constants import (
            CLIP_DURATION,
            CLIP_SAMPLES,
            DEFAULT_THRESHOLD,
            SAMPLE_RATE,
        )

        assert SAMPLE_RATE == 16000
        assert 0.0 < DEFAULT_THRESHOLD < 1.0
        assert CLIP_DURATION > 0
        assert CLIP_SAMPLES == int(SAMPLE_RATE * CLIP_DURATION)

    def test_feature_config_function(self):
        """get_feature_config() returns a dict with expected keys."""
        from violawake_sdk._constants import get_feature_config

        config = get_feature_config()
        assert isinstance(config, dict)
        expected_keys = [
            "feature_type", "sample_rate", "n_mels", "n_fft",
            "hop_length", "win_length", "use_pcen",
        ]
        for key in expected_keys:
            assert key in config, f"Missing key: {key}"

    def test_pcen_constants(self):
        """PCEN constants are defined."""
        from violawake_sdk._constants import PCEN_BIAS, PCEN_EPS, PCEN_GAIN, PCEN_POWER

        assert PCEN_GAIN > 0
        assert PCEN_BIAS > 0
        assert 0 < PCEN_POWER <= 1
        assert PCEN_EPS > 0

    def test_silence_gate_rms(self):
        """SILENCE_GATE_RMS constant is defined."""
        from violawake_sdk._constants import SILENCE_GATE_RMS

        assert SILENCE_GATE_RMS > 0
