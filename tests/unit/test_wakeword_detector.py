"""Unit tests for the lazy WakewordDetector compatibility wrapper."""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk.wake_detector import WAKE_WORD_ALIASES, WakeDetector, WakewordDetector


class TestWakewordDetector:
    """Test compatibility wrapper behavior without loading ONNX sessions."""

    def test_instantiates_without_loading_models(self) -> None:
        with patch("violawake_sdk.wake_detector.WakeDetector") as mock_detector:
            detector = WakewordDetector()
            assert detector.wake_word == "viola"
            mock_detector.assert_not_called()

    def test_instantiates_with_explicit_wake_word(self) -> None:
        with patch("violawake_sdk.wake_detector.WakeDetector") as mock_detector:
            detector = WakewordDetector(wake_word="viola")
            assert detector.wake_word == "viola"
            mock_detector.assert_not_called()

    def test_wake_detector_accepts_viola_alias(self, tmp_path: Path) -> None:
        (tmp_path / "oww_backbone.onnx").write_bytes(b"backbone")
        (tmp_path / "viola_mlp_oww.onnx").write_bytes(b"head")

        input_meta = MagicMock()
        input_meta.name = "input"

        fake_session = MagicMock()
        fake_session.get_inputs.return_value = [input_meta]

        fake_ort = types.SimpleNamespace(
            InferenceSession=MagicMock(return_value=fake_session),
        )

        with (
            patch.dict("sys.modules", {"onnxruntime": fake_ort}),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
        ):
            detector = WakeDetector(model="viola")

        assert detector._oww_input_name == "input"
        assert detector._mlp_input_name == "input"

    def test_unknown_wakeword_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="unknown"):
            WakewordDetector(wake_word="unknown")

    def test_viola_alias_is_registered(self) -> None:
        assert WAKE_WORD_ALIASES["viola"] == "viola_mlp_oww"

    def test_process_audio_uses_lazy_underlying_detector(self, noise_frame: bytes) -> None:
        mock_detector = MagicMock()
        mock_detector.detect.return_value = True

        with patch("violawake_sdk.wake_detector.WakeDetector", return_value=mock_detector) as detector_cls:
            detector = WakewordDetector(wake_word="viola", threshold=0.91)

            assert detector.process_audio(noise_frame) is True
            detector_cls.assert_called_once_with(
                model="viola_mlp_oww",
                threshold=0.91,
                cooldown_s=2.0,
                providers=None,
            )
            mock_detector.detect.assert_called_once_with(noise_frame, is_playing=False)
