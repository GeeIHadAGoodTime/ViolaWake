"""Additional public-API tests for violawake_sdk.wake_detector."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk._exceptions import AudioCaptureError
from violawake_sdk.wake_detector import (
    DEFAULT_THRESHOLD,
    EMBEDDING_DIM,
    FRAME_SAMPLES,
    WakeDetector,
    WakewordDetector,
)


def _make_backend_session(
    output_value: np.ndarray | None = None,
    *,
    shape: list[int] | None = None,
) -> MagicMock:
    session = MagicMock()
    model_input = MagicMock()
    model_input.name = "input"
    model_input.shape = shape or [1, EMBEDDING_DIM]
    session.get_inputs.return_value = [model_input]
    session.run.return_value = [
        output_value if output_value is not None else np.array([[0.95]], dtype=np.float32)
    ]
    return session


def _make_fake_backbone(embedding: np.ndarray | None = None) -> MagicMock:
    embedding = embedding if embedding is not None else np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.5
    backbone = MagicMock()
    backbone.push_audio.return_value = (True, embedding)
    backbone.last_embedding = embedding
    return backbone


def _build_detector(
    *,
    mlp_score: float = 0.95,
    cooldown_s: float = 0.0,
    speaker_verify_fn: object | None = None,
) -> tuple[WakeDetector, MagicMock, MagicMock]:
    session = _make_backend_session(np.array([[mlp_score]], dtype=np.float32))
    backend = MagicMock()
    backend.name = "onnx"
    backend.load.return_value = session
    backbone = _make_fake_backbone()

    with (
        patch("violawake_sdk.wake_detector.get_backend", return_value=backend),
        patch.object(WakeDetector, "_resolve_model_path", return_value=Path("C:/fake/model.onnx")),
        patch.object(WakeDetector, "_create_oww_backbone", return_value=backbone),
    ):
        detector = WakeDetector(
            threshold=DEFAULT_THRESHOLD,
            cooldown_s=cooldown_s,
            speaker_verify_fn=speaker_verify_fn,
        )

    return detector, backbone, session


class TestWakeDetectorPublicAPI:
    def test_close_releases_resources(self) -> None:
        detector, _backbone, _session = _build_detector()
        detector._ensemble = MagicMock()

        with patch.object(detector, "reset") as reset_mock:
            detector.close()

        reset_mock.assert_called_once()
        detector._ensemble.clear.assert_called_once()
        assert detector._mlp_session is None
        assert detector._oww_backbone is None

    def test_reset_cooldown_allows_detection_after_manual_reset(self, loud_noise_frame: bytes) -> None:
        detector, _backbone, _session = _build_detector(cooldown_s=60.0)

        assert detector.detect(loud_noise_frame) is True
        assert detector.detect(loud_noise_frame) is False

        detector.reset_cooldown()

        assert detector.detect(loud_noise_frame) is True

    def test_from_source_runs_detection_loop_and_stops_source(self) -> None:
        source = MagicMock()
        source.read_frame.side_effect = [b"\x00\x00" * FRAME_SAMPLES, b"\x00\x00" * FRAME_SAMPLES, None]

        with (
            patch("violawake_sdk.wake_detector.get_backend") as get_backend_mock,
            patch.object(
                WakeDetector, "_resolve_model_path", return_value=Path("C:/fake/model.onnx")
            ),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=_make_fake_backbone()),
        ):
            backend = MagicMock()
            backend.name = "onnx"
            backend.load.return_value = _make_backend_session()
            get_backend_mock.return_value = backend
            runner = WakeDetector.from_source(source)

        with patch.object(runner.detector, "detect", side_effect=[False, True]) as detect_mock:
            on_detect = MagicMock()
            detections = runner.run(on_detect=on_detect)

        assert detections == 1
        assert runner.last_scores == ()
        source.start.assert_called_once()
        source.stop.assert_called_once()
        assert detect_mock.call_count == 2
        on_detect.assert_called_once()

    def test_stream_mic_raises_helpful_import_error_when_pyaudio_missing(self) -> None:
        detector, _backbone, _session = _build_detector()
        original_import = builtins.__import__

        def _import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pyaudio":
                raise ImportError("missing pyaudio")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_import):
            generator = detector.stream_mic()
            with pytest.raises(ImportError, match="pyaudio is required"):
                next(generator)

    def test_stream_mic_wraps_open_failure(self) -> None:
        detector, _backbone, _session = _build_detector()
        pa_instance = MagicMock()
        pa_instance.open.side_effect = OSError("busy")
        fake_pyaudio = SimpleNamespace(PyAudio=MagicMock(return_value=pa_instance), paInt16=8)

        with patch.dict(sys.modules, {"pyaudio": fake_pyaudio}):
            generator = detector.stream_mic()
            with pytest.raises(AudioCaptureError, match="Failed to open microphone"):
                next(generator)

        pa_instance.terminate.assert_called_once()

    def test_stream_mic_yields_frames_and_cleans_up_stream(self) -> None:
        detector, _backbone, _session = _build_detector()
        stream = MagicMock()
        stream.read.side_effect = [b"\x01\x02" * FRAME_SAMPLES]
        pa_instance = MagicMock()
        pa_instance.open.return_value = stream
        fake_pyaudio = SimpleNamespace(PyAudio=MagicMock(return_value=pa_instance), paInt16=8)

        with patch.dict(sys.modules, {"pyaudio": fake_pyaudio}):
            generator = detector.stream_mic(device_index=7)
            frame = next(generator)
            generator.close()

        assert frame == b"\x01\x02" * FRAME_SAMPLES
        assert pa_instance.open.call_args.kwargs["input_device_index"] == 7
        stream.stop_stream.assert_called_once()
        stream.close.assert_called_once()
        pa_instance.terminate.assert_called_once()

    def test_enroll_speaker_uses_hook_embeddings(self, loud_noise_frame: bytes) -> None:
        fake_module = ModuleType("violawake_sdk.speaker")

        class SpeakerVerificationHook:
            pass

        class Hook(SpeakerVerificationHook):
            def __init__(self) -> None:
                self.calls: list[tuple[str, list[np.ndarray]]] = []

            def enroll_speaker(self, speaker_id: str, embeddings: list[np.ndarray]) -> int:
                self.calls.append((speaker_id, embeddings))
                return len(embeddings)

        fake_module.SpeakerVerificationHook = SpeakerVerificationHook
        hook = Hook()
        detector, _backbone, _session = _build_detector(speaker_verify_fn=hook)

        with patch.dict(sys.modules, {"violawake_sdk.speaker": fake_module}):
            count = detector.enroll_speaker("alice", [loud_noise_frame, loud_noise_frame])

        assert count == 2
        assert hook.calls[0][0] == "alice"
        assert len(hook.calls[0][1]) == 2

    def test_verify_speaker_returns_hook_result(self, loud_noise_frame: bytes) -> None:
        fake_module = ModuleType("violawake_sdk.speaker")

        class SpeakerVerificationHook:
            pass

        class Hook(SpeakerVerificationHook):
            def verify_speaker(self, embedding: np.ndarray) -> dict[str, object]:
                return {"match": True, "embedding_size": len(embedding)}

        fake_module.SpeakerVerificationHook = SpeakerVerificationHook
        hook = Hook()
        detector, _backbone, _session = _build_detector(speaker_verify_fn=hook)

        with patch.dict(sys.modules, {"violawake_sdk.speaker": fake_module}):
            result = detector.verify_speaker(loud_noise_frame)

        assert result == {"match": True, "embedding_size": EMBEDDING_DIM}

    def test_verify_speaker_requires_hook(self, loud_noise_frame: bytes) -> None:
        detector, _backbone, _session = _build_detector()

        with pytest.raises(RuntimeError, match="verify_speaker requires"):
            detector.verify_speaker(loud_noise_frame)


class TestWakewordDetectorCompatibility:
    def test_wrapper_lazy_loads_once_and_proxies_methods(self) -> None:
        detector_instance = MagicMock()
        detector_instance.detect.return_value = True
        detector_instance.process.return_value = 0.77
        detector_instance.stream_mic.return_value = iter([b"frame"])
        detector_instance.get_confidence.return_value = {"raw_score": 0.77}
        type(detector_instance).last_scores = property(lambda self: (0.2, 0.77))

        with patch("violawake_sdk.wake_detector.WakeDetector", return_value=detector_instance) as detector_cls:
            with pytest.deprecated_call():
                compat = WakewordDetector(wake_word="viola", threshold=0.9, cooldown_s=1.5)

            assert compat.detect(b"\x00\x00") is True
            assert compat.process_audio(b"\x00\x00") is True
            assert compat.process(b"\x00\x00") == 0.77
            assert list(compat.stream_mic()) == [b"frame"]
            compat.reset()
            compat.reset_cooldown()
            assert compat.get_confidence() == {"raw_score": 0.77}
            assert compat.last_scores == (0.2, 0.77)

        detector_cls.assert_called_once_with(
            model="temporal_cnn",
            threshold=0.9,
            cooldown_s=1.5,
            providers=None,
            backend="auto",
        )
        detector_instance.detect.assert_any_call(b"\x00\x00", is_playing=False)
        detector_instance.process.assert_called_once_with(b"\x00\x00")
        detector_instance.reset.assert_called_once()
        detector_instance.reset_cooldown.assert_called_once()

    def test_unknown_wakeword_raises_key_error(self) -> None:
        with pytest.deprecated_call():
            with pytest.raises(KeyError, match="Unknown wakeword"):
                WakewordDetector(wake_word="not-real")
