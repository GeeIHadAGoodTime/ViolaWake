"""Lane 1 wake-detection oracle tests.

These tests lock the documented wake contract at the SDK boundary:
16 kHz mono, 20 ms frames, 320-sample stride, 96-dim OWW embeddings,
default threshold 0.80, and all four decision-policy gates.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from violawake_sdk._constants import DEFAULT_THRESHOLD as CONSTANT_DEFAULT_THRESHOLD
from violawake_sdk.audio_source import FRAME_SAMPLES as SOURCE_FRAME_SAMPLES
from violawake_sdk.oww_backbone import EMBEDDING_DIM
from violawake_sdk.wake_detector import (
    DEFAULT_THRESHOLD,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    WakeDetector,
)


def _backend_session(score: float) -> MagicMock:
    session = MagicMock()
    input_meta = MagicMock()
    input_meta.name = "input"
    input_meta.shape = [1, EMBEDDING_DIM]
    session.get_inputs.return_value = [input_meta]
    session.run.return_value = [np.array([[score]], dtype=np.float32)]
    return session


def _fake_backbone(embedding_dim: int = EMBEDDING_DIM) -> MagicMock:
    embedding = np.ones(embedding_dim, dtype=np.float32) * 0.5
    backbone = MagicMock()
    backbone.push_audio.return_value = (True, embedding)
    backbone.last_embedding = embedding
    return backbone


def _detector(score: float, *, cooldown_s: float = 0.0) -> WakeDetector:
    backend = MagicMock()
    backend.name = "onnx"
    backend.load.return_value = _backend_session(score)

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=Path("/fake/model.onnx")),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=_fake_backbone()),
    ):
        return WakeDetector(cooldown_s=cooldown_s)


def test_sdk_entry_audio_contract_constants_are_locked() -> None:
    assert SAMPLE_RATE == 16_000
    assert FRAME_MS == 20
    assert FRAME_SAMPLES == 320
    assert SOURCE_FRAME_SAMPLES == 320
    assert EMBEDDING_DIM == 96


def test_default_threshold_is_080_and_rejects_050_score(loud_noise_frame: bytes) -> None:
    assert CONSTANT_DEFAULT_THRESHOLD == 0.80
    assert DEFAULT_THRESHOLD == 0.80

    detector = _detector(score=0.50)
    assert detector.threshold == 0.80
    assert detector.detect(loud_noise_frame) is False


def test_detect_exercises_all_four_policy_gates_end_to_end(
    loud_noise_frame: bytes,
    silent_frame: bytes,
) -> None:
    assert _detector(score=0.95).detect(loud_noise_frame) is True
    assert _detector(score=0.95).detect(silent_frame) is False
    assert _detector(score=0.79).detect(loud_noise_frame) is False

    cooldown_detector = _detector(score=0.95, cooldown_s=60.0)
    assert cooldown_detector.detect(loud_noise_frame) is True
    assert cooldown_detector.detect(loud_noise_frame) is False

    assert _detector(score=0.95).detect(loud_noise_frame, is_playing=True) is False
