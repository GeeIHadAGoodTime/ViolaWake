from __future__ import annotations

import importlib
import importlib.util
import inspect
import re
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk._exceptions import ModelNotFoundError
from violawake_sdk.models import MODEL_REGISTRY, ModelSpec
from violawake_sdk.vad import VADEngine
from violawake_sdk.wake_detector import WakeDetector

pytestmark = pytest.mark.integration

GITHUB_RELEASE_URL_RE = re.compile(
    r"^https://github\.com/[^/]+/[^/]+/releases/download/v[^/]+/[^/]+$"
)


def _make_fake_ort() -> types.SimpleNamespace:
    input_meta = MagicMock()
    input_meta.name = "input"

    fake_session = MagicMock()
    fake_session.get_inputs.return_value = [input_meta]

    return types.SimpleNamespace(
        InferenceSession=MagicMock(return_value=fake_session),
    )


def test_required_public_imports_work() -> None:
    from violawake_sdk import VADEngine as ExportedVADEngine
    from violawake_sdk import VoicePipeline, WakeDetector
    from violawake_sdk.models import MODEL_REGISTRY, ModelSpec, get_model_path
    from violawake_sdk.tools.evaluate import main as evaluate_main

    assert inspect.isclass(WakeDetector)
    assert inspect.isclass(ExportedVADEngine)
    assert inspect.isclass(VoicePipeline)
    assert callable(get_model_path)
    assert isinstance(MODEL_REGISTRY, dict)
    assert ModelSpec is not None
    assert callable(evaluate_main)


def test_optional_tts_export_if_extra_installed() -> None:
    from violawake_sdk import TTSEngine

    if importlib.util.find_spec("kokoro_onnx") is None:
        pytest.skip("TTS extra not installed")

    assert inspect.isclass(TTSEngine)


def test_optional_stt_export_if_extra_installed() -> None:
    from violawake_sdk import STTEngine

    if importlib.util.find_spec("faster_whisper") is None:
        pytest.skip("STT extra not installed")

    assert inspect.isclass(STTEngine)


def test_confusables_import_if_module_exists() -> None:
    if importlib.util.find_spec("violawake_sdk.tools.confusables") is None:
        pytest.skip("confusables tool not present")

    from violawake_sdk.tools.confusables import generate_confusables

    assert callable(generate_confusables)


def test_model_registry_entries_are_valid() -> None:
    assert MODEL_REGISTRY, "MODEL_REGISTRY should not be empty"

    for model_name, spec in MODEL_REGISTRY.items():
        assert isinstance(spec, ModelSpec), f"{model_name} should map to ModelSpec"
        assert spec.name.strip(), f"{model_name} should have a non-empty name"
        assert spec.url.strip(), f"{model_name} should have a non-empty url"
        assert spec.sha256.strip(), f"{model_name} should have a non-empty sha256"
        assert spec.description.strip(), f"{model_name} should have a non-empty description"
        assert GITHUB_RELEASE_URL_RE.match(spec.url), (
            f"{model_name} URL should use a GitHub Releases download URL: {spec.url}"
        )
        assert 0 < spec.size_bytes < 1_000_000_000, (
            f"{model_name} size_bytes should be > 0 and < 1GB, got {spec.size_bytes}"
        )


def test_wake_detector_invalid_model_path_raises(tmp_path: Path) -> None:
    (tmp_path / "oww_backbone.onnx").write_bytes(b"fake-backbone")
    fake_ort = _make_fake_ort()
    missing_model = tmp_path / "missing_model.onnx"

    with (
        patch.dict("sys.modules", {"onnxruntime": fake_ort}),
        patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
        pytest.raises(ModelNotFoundError, match=re.escape(str(missing_model))),
    ):
        WakeDetector(model=str(missing_model))


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_wake_detector_invalid_threshold_raises(threshold: float) -> None:
    with (
        patch.object(WakeDetector, "_load_session") as load_session,
        pytest.raises(ValueError, match="threshold"),
    ):
        WakeDetector(threshold=threshold)

    load_session.assert_not_called()


def test_vad_engine_webrtc_backend_constructs_without_hardware(silent_frame: bytes) -> None:
    mock_webrtcvad = MagicMock()
    mock_webrtcvad.Vad.return_value.is_speech.return_value = False

    with patch.dict("sys.modules", {"webrtcvad": mock_webrtcvad}):
        vad = VADEngine(backend="webrtc")

    assert vad.backend_name == "webrtc"
    assert vad.process_frame(silent_frame) == 0.0


def test_vad_engine_rms_backend_constructs_without_hardware(silent_frame: bytes) -> None:
    vad = VADEngine(backend="rms")

    assert vad.backend_name == "rms"
    assert vad.process_frame(silent_frame) == 0.0


def test_vad_engine_invalid_backend_raises() -> None:
    with pytest.raises(ValueError, match="invalid_backend"):
        VADEngine(backend="invalid_backend")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "module_name",
    [
        "violawake_sdk.tools.evaluate",
        "violawake_sdk.tools.collect_samples",
        "violawake_sdk.tools.train",
        "violawake_sdk.tools.download_model",
    ],
)
def test_cli_entrypoints_are_importable_and_expose_main(module_name: str) -> None:
    module = importlib.import_module(module_name)

    assert hasattr(module, "main"), f"{module_name} should define main()"
    assert callable(module.main)
