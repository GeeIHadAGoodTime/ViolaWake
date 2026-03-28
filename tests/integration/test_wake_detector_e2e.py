"""Integration test: WakeDetector end-to-end with real OWW models.

Requires openwakeword and onnxruntime installed. Uses real mel + embedding
models from OWW, plus the r3_10x_s42 MLP from experiments/models/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

WAKEWORD = Path(__file__).resolve().parent.parent.parent
MLP_PATH = WAKEWORD / "experiments" / "models" / "r3_10x_s42.onnx"
POSITIVE_DIR = WAKEWORD / "eval_clean" / "positives"

# Skip entire module if MLP model not available
pytestmark = pytest.mark.skipif(
    not MLP_PATH.exists(),
    reason=f"MLP model not found: {MLP_PATH}",
)


@pytest.fixture
def detector():
    """Create a WakeDetector with r3_10x_s42 MLP for fast testing."""
    from violawake_sdk.wake_detector import WakeDetector
    return WakeDetector(
        model=str(MLP_PATH),
        threshold=0.80,
        cooldown_s=0.0,
    )


@pytest.fixture
def detector_strict():
    """Create a WakeDetector with higher threshold."""
    from violawake_sdk.wake_detector import WakeDetector
    return WakeDetector(
        model=str(MLP_PATH),
        threshold=0.90,
        cooldown_s=0.0,
    )


def load_audio_clip(path: Path, target_samples: int = 24000) -> tuple[np.ndarray, np.ndarray]:
    """Load audio file, return (float32, int16) arrays."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)))
    else:
        audio = audio[:target_samples]
    audio_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return audio.astype(np.float32), audio_i16


class TestWakeDetectorE2E:
    """End-to-end tests with real models."""

    def test_positive_audio_triggers(self, detector) -> None:
        """Feeding a positive 'viola' clip should produce a high score."""
        wavs = sorted(POSITIVE_DIR.rglob("*viola*.wav"))
        if not wavs:
            pytest.skip("No positive audio files found")

        audio_f32, audio_i16 = load_audio_clip(wavs[0])

        # Feed as 80ms chunks (1280 samples) -- matches OWW default
        chunk_size = 1280
        scores = []
        for i in range(len(audio_i16) // chunk_size):
            chunk = audio_i16[i * chunk_size:(i + 1) * chunk_size]
            score = detector.process(chunk)
            if score > 0:
                scores.append(score)

        assert len(scores) > 0, "No scores produced (need enough audio for mel extraction)"
        max_score = max(scores)
        assert max_score > 0.80, f"Max score {max_score:.4f} should be > 0.80 for positive audio"

    def test_silence_does_not_trigger(self, detector) -> None:
        """Silence should produce low scores."""
        silence = np.zeros(24000, dtype=np.int16)

        chunk_size = 1280
        scores = []
        for i in range(len(silence) // chunk_size):
            chunk = silence[i * chunk_size:(i + 1) * chunk_size]
            score = detector.process(chunk)
            if score > 0:
                scores.append(score)

        if scores:
            assert max(scores) < 0.50, f"Silence should not trigger (max={max(scores):.4f})"

    def test_noise_does_not_trigger(self) -> None:
        """Random noise should not trigger detection."""
        from violawake_sdk.wake_detector import WakeDetector

        det = WakeDetector(
            model=str(MLP_PATH),
            threshold=0.80,
            cooldown_s=0.0,
        )

        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(24000) * 1000).astype(np.int16)

        chunk_size = 1280
        detections = 0
        for i in range(len(noise) // chunk_size):
            chunk = noise[i * chunk_size:(i + 1) * chunk_size]
            if det.detect(chunk):
                detections += 1

        # Noise may occasionally score high but detection should be rare
        assert detections <= 2, f"Noise triggered too many detections: {detections}"

    def test_process_returns_valid_scores(self, detector) -> None:
        """process() should return valid float scores for any input."""
        wavs = sorted(POSITIVE_DIR.rglob("*viola*.wav"))
        if not wavs:
            pytest.skip("No positive audio files found")

        audio_f32, audio_i16 = load_audio_clip(wavs[0])

        chunk_size = 1280
        for i in range(len(audio_i16) // chunk_size):
            chunk = audio_i16[i * chunk_size:(i + 1) * chunk_size]
            score = detector.process(chunk)
            assert isinstance(score, float)

    def test_policy_reset_works(self, detector) -> None:
        """Policy reset should allow fresh detection."""
        rng = np.random.default_rng(42)
        noise = (rng.standard_normal(24000) * 1000).astype(np.int16)

        # Feed some audio
        detector.process(noise[:1280])
        detector.process(noise[1280:2560])

        # Reset cooldown
        detector._policy.reset_cooldown()
        # Should still work after reset
        score = detector.process(noise[2560:3840])
        assert isinstance(score, float)
