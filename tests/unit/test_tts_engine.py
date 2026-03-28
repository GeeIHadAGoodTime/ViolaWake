"""Unit tests for TTSEngine."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk._exceptions import ModelLoadError, ModelNotFoundError
from violawake_sdk.tts import TTSEngine


def _mock_kokoro_module(kokoro: object | None = None) -> tuple[MagicMock, MagicMock]:
    constructor = MagicMock(return_value=kokoro or MagicMock())
    module = MagicMock()
    module.Kokoro = constructor
    return module, constructor


def test_unknown_voice_raises() -> None:
    with pytest.raises(ValueError, match="Unknown voice"):
        TTSEngine(voice="bad_voice")


def test_valid_init() -> None:
    engine = TTSEngine(voice="af_heart", speed=1.0, sample_rate=16_000)

    assert engine.voice == "af_heart"
    assert engine.speed == 1.0
    assert engine.sample_rate == 16_000


def test_empty_text_returns_zeros() -> None:
    engine = TTSEngine()

    with patch.object(engine, "_get_kokoro") as get_kokoro_mock:
        audio = engine.synthesize("")

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.size == 0
    get_kokoro_mock.assert_not_called()


def test_synthesize_mocked() -> None:
    engine = TTSEngine()
    kokoro = MagicMock()
    kokoro.create.return_value = ([0.1, -0.2, 0.3], 16_000)
    mock_module, constructor = _mock_kokoro_module(kokoro)

    with (
        patch.dict("sys.modules", {"kokoro_onnx": mock_module}),
        patch(
            "violawake_sdk.tts.get_model_path",
            side_effect=[Path("kokoro_v1_0.onnx"), Path("kokoro_voices_v1_0.bin")],
        ),
    ):
        audio = engine.synthesize("hello")

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert np.allclose(audio, np.array([0.1, -0.2, 0.3], dtype=np.float32))
    constructor.assert_called_once_with("kokoro_v1_0.onnx", "kokoro_voices_v1_0.bin")
    kokoro.create.assert_called_once_with(
        "hello",
        voice="af_heart",
        speed=1.0,
        lang="en-us",
    )


def test_model_load_error() -> None:
    engine = TTSEngine()
    mock_module, constructor = _mock_kokoro_module()
    constructor.side_effect = Exception("boom")

    with (
        patch.dict("sys.modules", {"kokoro_onnx": mock_module}),
        patch(
            "violawake_sdk.tts.get_model_path",
            side_effect=[Path("kokoro_v1_0.onnx"), Path("kokoro_voices_v1_0.bin")],
        ),
        pytest.raises(ModelLoadError, match="Failed to load Kokoro model"),
    ):
        engine.synthesize("hello")


def test_model_not_found() -> None:
    engine = TTSEngine()
    mock_module, constructor = _mock_kokoro_module()

    with (
        patch.dict("sys.modules", {"kokoro_onnx": mock_module}),
        patch("violawake_sdk.tts.get_model_path", side_effect=FileNotFoundError("missing")),
        pytest.raises(ModelNotFoundError, match="Kokoro models not found"),
    ):
        engine.synthesize("hello")

    constructor.assert_not_called()


def test_split_sentences() -> None:
    assert TTSEngine._split_sentences("Hello. How are you? Fine!") == [
        "Hello.",
        "How are you?",
        "Fine!",
    ]


def test_chunked_yields() -> None:
    engine = TTSEngine()

    with patch.object(
        engine,
        "synthesize",
        side_effect=[
            np.array([1.0], dtype=np.float32),
            np.array([2.0, 3.0], dtype=np.float32),
        ],
    ) as synthesize_mock:
        chunks = list(engine.synthesize_chunked("Hello. World!"))

    assert len(chunks) == 2
    assert np.array_equal(chunks[0], np.array([1.0], dtype=np.float32))
    assert np.array_equal(chunks[1], np.array([2.0, 3.0], dtype=np.float32))
    assert synthesize_mock.call_count == 2


def test_play_pyaudio_clips_before_int16() -> None:
    """_play_pyaudio should clip audio to [-1, 1] before int16 cast to prevent overflow."""
    engine = TTSEngine()
    # Audio with values outside [-1, 1]
    audio = np.array([-2.0, 0.0, 2.0], dtype=np.float32)

    mock_pa_instance = MagicMock()
    mock_stream = MagicMock()
    mock_pa_instance.open.return_value = mock_stream

    mock_pyaudio = MagicMock()
    mock_pyaudio.PyAudio.return_value = mock_pa_instance
    mock_pyaudio.paInt16 = 8

    with patch.dict("sys.modules", {"pyaudio": mock_pyaudio}):
        engine._play_pyaudio(audio, blocking=True)

    written_bytes = mock_stream.write.call_args[0][0]
    pcm = np.frombuffer(written_bytes, dtype=np.int16)
    # After clipping: -1.0 -> -32767, 0.0 -> 0, 1.0 -> 32767
    assert pcm[0] == -32767
    assert pcm[1] == 0
    assert pcm[2] == 32767


def test_resample_raises_on_missing_scipy() -> None:
    """_resample should raise ImportError with guidance if scipy is not installed."""
    import sys

    with patch.dict(sys.modules, {"scipy": None, "scipy.signal": None}):
        with pytest.raises(ImportError, match="scipy is required"):
            TTSEngine._resample(np.zeros(100, dtype=np.float32), 24000, 16000)


def test_thread_safety() -> None:
    engine = TTSEngine()
    kokoro = MagicMock()
    kokoro.create.return_value = (np.ones(4, dtype=np.float32), 16_000)

    with (
        patch.object(engine, "_load_kokoro", return_value=kokoro) as load_mock,
        ThreadPoolExecutor(max_workers=3) as executor,
    ):
        results = list(executor.map(lambda _: engine.synthesize("hello"), range(3)))

    assert all(isinstance(result, np.ndarray) for result in results)
    assert all(result.shape == (4,) for result in results)
    assert load_mock.call_count == 1


# ===================================================================
# tts_engine.py re-export layer — import and smoke tests
# ===================================================================


class TestTTSEngineReExports:
    """Verify that tts_engine.py re-exports are importable and consistent."""

    def test_import_module(self) -> None:
        import violawake_sdk.tts_engine  # noqa: F401

    def test_import_tts_engine_class(self) -> None:
        from violawake_sdk.tts_engine import TTSEngine as ReExportedTTSEngine
        assert ReExportedTTSEngine is TTSEngine

    def test_import_available_voices(self) -> None:
        from violawake_sdk.tts_engine import AVAILABLE_VOICES
        assert isinstance(AVAILABLE_VOICES, (list, tuple, dict, frozenset, set))

    def test_import_default_voice(self) -> None:
        from violawake_sdk.tts_engine import DEFAULT_VOICE
        assert isinstance(DEFAULT_VOICE, str)

    def test_import_sample_rates(self) -> None:
        from violawake_sdk.tts_engine import TTS_SAMPLE_RATE, TARGET_SAMPLE_RATE
        assert isinstance(TTS_SAMPLE_RATE, int)
        assert isinstance(TARGET_SAMPLE_RATE, int)

    def test_all_exports_present(self) -> None:
        import violawake_sdk.tts_engine as mod
        assert hasattr(mod, "__all__")
        for name in mod.__all__:
            assert hasattr(mod, name), f"__all__ declares '{name}' but it is not defined"

    def test_instantiation_via_reexport(self) -> None:
        from violawake_sdk.tts_engine import TTSEngine as ReExportedTTSEngine
        engine = ReExportedTTSEngine(voice="af_heart")
        assert engine.voice == "af_heart"
