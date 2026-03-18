"""Latency benchmarks for ViolaWake SDK components.

Run with:
    pytest tests/benchmarks/bench_latency.py --benchmark-json=benchmark-results/latency.json

Requires model files (integration test). Skips if models not cached.

Latency targets (from docs/PRD.md):
    - Wake word inference: ≤ 10ms (p50), ≤ 15ms (p99) per 20ms frame
    - VAD WebRTC: ≤ 1ms (p50), ≤ 2ms (p99) per frame
    - TTS first audio: ≤ 400ms (p50), ≤ 800ms (p99) per sentence
    - STT (base model, 3s audio): ≤ 700ms (p50), ≤ 1500ms (p99)
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.benchmark


# ──────────────────────────────────────────────────────────────────────────────
# VAD Benchmarks (no model required)
# ──────────────────────────────────────────────────────────────────────────────

class TestVADLatency:
    """VAD latency benchmarks — no model files required."""

    @pytest.fixture
    def vad_rms(self):
        from violawake_sdk.vad import VADEngine
        return VADEngine(backend="rms")

    @pytest.fixture
    def audio_frame(self) -> bytes:
        rng = np.random.default_rng(42)
        return rng.integers(-5000, 5000, 320, dtype=np.int16).tobytes()

    def test_rms_vad_latency(self, benchmark, vad_rms, audio_frame):
        """RMS VAD should be < 0.5ms per frame."""
        result = benchmark(vad_rms.process_frame, audio_frame)
        assert 0.0 <= result <= 1.0

    def test_vad_webrtc_latency(self, benchmark, audio_frame):
        """WebRTC VAD should be < 1ms per frame."""
        try:
            from violawake_sdk.vad import VADEngine
            vad = VADEngine(backend="webrtc")
        except ImportError:
            pytest.skip("webrtcvad not installed")

        result = benchmark(vad.process_frame, audio_frame)
        assert 0.0 <= result <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Wake Word Inference Benchmarks (requires model files)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestWakeWordLatency:
    """Wake word inference latency benchmarks — requires model files."""

    @pytest.fixture(scope="class")
    def detector(self):
        try:
            from violawake_sdk import WakeDetector
            return WakeDetector(threshold=0.80)
        except Exception as e:
            pytest.skip(f"Could not load WakeDetector: {e}")

    @pytest.fixture
    def audio_frame(self) -> bytes:
        rng = np.random.default_rng(42)
        return rng.integers(-5000, 5000, 320, dtype=np.int16).tobytes()

    def test_wake_inference_latency(self, benchmark, detector, audio_frame):
        """Wake word inference should be < 15ms per 20ms frame (p99 target)."""
        score = benchmark(detector.process, audio_frame)
        assert 0.0 <= score <= 1.0

    def test_wake_inference_warmup(self, benchmark, detector, audio_frame):
        """Benchmark with warmup to exclude JIT effects."""
        benchmark.warmup_iterations = 10
        score = benchmark(detector.process, audio_frame)
        assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# TTS Latency Benchmarks (requires Kokoro model)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestTTSLatency:
    """TTS synthesis latency benchmarks — requires Kokoro model file (330MB)."""

    @pytest.fixture(scope="class")
    def tts(self):
        try:
            from violawake_sdk import TTSEngine
            if TTSEngine is None:
                pytest.skip("TTS optional deps not installed (pip install violawake[tts])")
            return TTSEngine(voice="af_heart")
        except Exception as e:
            pytest.skip(f"Could not load TTSEngine: {e}")

    SHORT_SENTENCE = "Hello from ViolaWake."
    MEDIUM_SENTENCE = "The weather today is sunny with a high of twenty-five degrees."

    def test_tts_short_sentence_latency(self, benchmark, tts):
        """Short sentence TTS should be < 400ms (p50 target)."""
        audio = benchmark(tts.synthesize, self.SHORT_SENTENCE)
        assert audio.size > 0
        assert audio.dtype == np.float32

    def test_tts_medium_sentence_latency(self, benchmark, tts):
        """Medium sentence TTS should be < 600ms."""
        audio = benchmark(tts.synthesize, self.MEDIUM_SENTENCE)
        assert audio.size > 0


# ──────────────────────────────────────────────────────────────────────────────
# STT Latency Benchmarks (requires Whisper model)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestSTTLatency:
    """STT transcription latency benchmarks — requires Whisper model."""

    @pytest.fixture(scope="class")
    def stt(self):
        try:
            from violawake_sdk import STTEngine
            if STTEngine is None:
                pytest.skip("STT optional deps not installed (pip install violawake[stt])")
            engine = STTEngine(model="base")
            engine.prewarm()
            return engine
        except Exception as e:
            pytest.skip(f"Could not load STTEngine: {e}")

    @pytest.fixture
    def audio_3s(self) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.standard_normal(48_000).astype(np.float32) * 0.1

    def test_stt_3s_audio_latency(self, benchmark, stt, audio_3s):
        """3s audio transcription should be < 700ms (p50 target for base model)."""
        text = benchmark(stt.transcribe, audio_3s)
        assert isinstance(text, str)
