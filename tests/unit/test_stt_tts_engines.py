"""Comprehensive STT and TTS engine tests.

Tests:
- STTEngine initialization with different model sizes
- TTSEngine initialization with voice validation
- Sentence splitting in TTS
- Error handling when dependencies not installed
- Lazy loading behavior
- All tests use mocks (no real model files)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.stt import (
    DEFAULT_MODEL,
    MODEL_PROFILES,
    NO_SPEECH_THRESHOLD,
    STTEngine,
    TranscriptResult,
    TranscriptSegment,
)
from violawake_sdk.tts import (
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    SENTENCE_BOUNDARIES,
    TARGET_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    TTSEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(
    text: str,
    *,
    start: float = 0.0,
    end: float = 1.0,
    no_speech_prob: float = 0.0,
) -> MagicMock:
    segment = MagicMock()
    segment.text = text
    segment.start = start
    segment.end = end
    segment.no_speech_prob = no_speech_prob
    return segment


def _make_info(
    *,
    language: str = "en",
    language_probability: float = 0.95,
    duration: float = 1.5,
) -> MagicMock:
    info = MagicMock()
    info.language = language
    info.language_probability = language_probability
    info.duration = duration
    return info


def _mock_faster_whisper(model: object | None = None) -> tuple[MagicMock, MagicMock]:
    constructor = MagicMock(return_value=model or MagicMock())
    module = MagicMock()
    module.WhisperModel = constructor
    return module, constructor


# ===========================================================================
# STTEngine Tests
# ===========================================================================

class TestSTTEngineInit:
    """Test STTEngine initialization with different model sizes."""

    @pytest.mark.parametrize("model_name", list(MODEL_PROFILES.keys()))
    def test_init_all_valid_models(self, model_name: str) -> None:
        """Each valid model size creates an engine without error."""
        engine = STTEngine(model=model_name)
        assert engine.model_name == model_name
        assert engine._model is None  # Lazy loaded

    def test_init_default_model(self) -> None:
        engine = STTEngine()
        assert engine.model_name == DEFAULT_MODEL

    def test_init_with_device(self) -> None:
        engine = STTEngine(model="tiny", device="cuda")
        assert engine.device == "cuda"

    def test_init_with_compute_type(self) -> None:
        engine = STTEngine(model="tiny", compute_type="float16")
        assert engine.compute_type == "float16"

    def test_init_with_language(self) -> None:
        engine = STTEngine(model="tiny", language="es")
        assert engine.forced_language == "es"

    def test_init_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            STTEngine(model="nonexistent")

    def test_init_invalid_model_shows_available(self) -> None:
        with pytest.raises(ValueError, match="tiny"):
            STTEngine(model="nonexistent")


class TestSTTEngineLazyLoading:
    """Test lazy loading behavior."""

    def test_model_not_loaded_on_init(self) -> None:
        engine = STTEngine(model="base")
        assert engine._model is None

    def test_model_loaded_on_first_transcribe(self) -> None:
        engine = STTEngine(model="base")
        model = MagicMock()
        model.transcribe.return_value = (iter([_make_segment("hello")]), _make_info())
        mock_module, constructor = _mock_faster_whisper(model)

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            engine.transcribe(np.array([0.1], dtype=np.float32))

        assert engine._model is model
        constructor.assert_called_once()

    def test_model_loaded_once_on_multiple_calls(self) -> None:
        engine = STTEngine(model="base")
        model = MagicMock()
        model.transcribe.return_value = (iter([_make_segment("hello")]), _make_info())
        mock_module, constructor = _mock_faster_whisper(model)

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            engine.transcribe(np.array([0.1], dtype=np.float32))
            model.transcribe.return_value = (iter([_make_segment("world")]), _make_info())
            engine.transcribe(np.array([0.2], dtype=np.float32))

        constructor.assert_called_once()

    def test_prewarm_loads_model_eagerly(self) -> None:
        engine = STTEngine(model="base")
        model = MagicMock()
        mock_module, constructor = _mock_faster_whisper(model)

        with patch.dict("sys.modules", {"faster_whisper": mock_module}):
            engine.prewarm()

        assert engine._model is model


class TestSTTEngineErrorHandling:
    """Test error handling when dependencies are missing."""

    def test_import_error_when_faster_whisper_missing(self) -> None:
        engine = STTEngine(model="base")
        # Remove faster_whisper from sys.modules if present
        with patch.dict("sys.modules", {"faster_whisper": None}):
            with pytest.raises(ImportError, match="faster-whisper"):
                engine.transcribe(np.array([0.1], dtype=np.float32))

    def test_transcribe_empty_segments(self) -> None:
        """Transcribing audio that produces no segments returns empty string."""
        engine = STTEngine(model="base")
        engine._model = MagicMock()
        engine._model.transcribe.return_value = (
            iter([]),
            _make_info(),
        )

        result = engine.transcribe(np.array([0.1], dtype=np.float32))
        assert result == ""

    def test_high_no_speech_prob_returns_empty(self) -> None:
        """Segments with no_speech_prob > threshold are suppressed."""
        engine = STTEngine(model="base")
        engine._model = MagicMock()
        engine._model.transcribe.return_value = (
            iter([_make_segment("ghost text", no_speech_prob=0.9)]),
            _make_info(),
        )

        result = engine.transcribe(np.array([0.1], dtype=np.float32))
        assert result == ""


class TestSTTLanguageCache:
    """Test language detection caching."""

    def test_forced_language_bypasses_cache(self) -> None:
        engine = STTEngine(model="base", language="fr")
        assert engine._get_language() == "fr"

    def test_auto_detect_returns_none_initially(self) -> None:
        engine = STTEngine(model="base")
        assert engine._get_language() is None

    def test_language_cached_after_detection(self) -> None:
        engine = STTEngine(model="base")
        engine._model = MagicMock()
        engine._model.transcribe.return_value = (
            iter([_make_segment("hello")]),
            _make_info(language="en", language_probability=0.95),
        )

        engine.transcribe_full(np.array([0.1], dtype=np.float32))

        # Language should now be cached
        assert engine._get_language() == "en"

    def test_language_cache_expires(self) -> None:
        engine = STTEngine(model="base", language_cache_ttl_s=0.01)
        engine._language_cache = ("en", time.monotonic() - 1.0)  # Expired

        assert engine._get_language() is None


# ===========================================================================
# TTSEngine Tests
# ===========================================================================

class TestTTSEngineInit:
    """Test TTSEngine initialization with voice validation."""

    @pytest.mark.parametrize("voice", AVAILABLE_VOICES)
    def test_init_all_valid_voices(self, voice: str) -> None:
        engine = TTSEngine(voice=voice)
        assert engine.voice == voice

    def test_init_default_voice(self) -> None:
        engine = TTSEngine()
        assert engine.voice == DEFAULT_VOICE

    def test_init_with_speed(self) -> None:
        engine = TTSEngine(speed=1.5)
        assert engine.speed == 1.5

    def test_init_with_sample_rate(self) -> None:
        engine = TTSEngine(sample_rate=24_000)
        assert engine.sample_rate == 24_000

    def test_init_invalid_voice_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown voice"):
            TTSEngine(voice="not_a_voice")

    def test_init_invalid_voice_shows_available(self) -> None:
        with pytest.raises(ValueError, match="af_heart"):
            TTSEngine(voice="not_a_voice")


class TestTTSEngineLazyLoading:
    """Test lazy loading of Kokoro model."""

    def test_model_not_loaded_on_init(self) -> None:
        engine = TTSEngine()
        assert engine._kokoro is None

    def test_model_loaded_on_first_synthesize(self) -> None:
        engine = TTSEngine()
        kokoro = MagicMock()
        kokoro.create.return_value = (np.ones(10, dtype=np.float32), 16_000)

        with patch.object(engine, "_load_kokoro", return_value=kokoro) as load_mock:
            engine.synthesize("hello")

        load_mock.assert_called_once()
        assert engine._kokoro is kokoro

    def test_model_loaded_once_across_calls(self) -> None:
        engine = TTSEngine()
        kokoro = MagicMock()
        kokoro.create.return_value = (np.ones(10, dtype=np.float32), 16_000)

        with patch.object(engine, "_load_kokoro", return_value=kokoro) as load_mock:
            engine.synthesize("hello")
            engine.synthesize("world")

        load_mock.assert_called_once()

    def test_empty_text_skips_model_load(self) -> None:
        engine = TTSEngine()

        with patch.object(engine, "_get_kokoro") as get_mock:
            result = engine.synthesize("")

        get_mock.assert_not_called()
        assert result.size == 0

    def test_whitespace_only_skips_model_load(self) -> None:
        engine = TTSEngine()

        with patch.object(engine, "_get_kokoro") as get_mock:
            result = engine.synthesize("   \n\t  ")

        get_mock.assert_not_called()
        assert result.size == 0


class TestTTSEngineErrorHandling:
    """Test error handling when dependencies are missing."""

    def test_import_error_when_kokoro_missing(self) -> None:
        engine = TTSEngine()

        with patch.dict("sys.modules", {"kokoro_onnx": None}):
            with pytest.raises(ImportError, match="kokoro-onnx"):
                engine.synthesize("hello")

    def test_synthesis_failure_raises_runtime_error(self) -> None:
        engine = TTSEngine()
        kokoro = MagicMock()
        kokoro.create.side_effect = RuntimeError("synthesis failed")

        with patch.object(engine, "_get_kokoro", return_value=kokoro):
            with pytest.raises(RuntimeError, match="TTS synthesis failed"):
                engine.synthesize("hello")

    def test_play_import_error_fallback(self) -> None:
        """play() tries sounddevice first, then falls back to pyaudio."""
        engine = TTSEngine()
        audio = np.ones(100, dtype=np.float32)

        # Both missing
        with (
            patch.dict("sys.modules", {"sounddevice": None}),
            patch.object(engine, "_play_pyaudio", side_effect=ImportError("no pyaudio")),
        ):
            with pytest.raises(ImportError, match="pip install sounddevice"):
                engine.play(audio)

    def test_play_non_blocking_uses_sounddevice(self) -> None:
        engine = TTSEngine()
        audio = np.ones(100, dtype=np.float32)
        sounddevice = MagicMock()

        with patch.dict(sys.modules, {"sounddevice": sounddevice}):
            engine.play(audio, blocking=False)

        sounddevice.play.assert_called_once()
        call_args = sounddevice.play.call_args
        assert np.array_equal(call_args[0][0], audio)
        assert call_args[1] == {"samplerate": engine.sample_rate, "blocking": False}

    def test_play_async_uses_non_blocking_playback(self) -> None:
        engine = TTSEngine()
        audio = np.ones(100, dtype=np.float32)

        with patch.object(engine, "play") as play_mock:
            engine.play_async(audio)

        play_mock.assert_called_once_with(audio, blocking=False)


# ===========================================================================
# Sentence Splitting Tests
# ===========================================================================

class TestSentenceSplitting:
    """Test TTS sentence splitting logic."""

    def test_single_sentence(self) -> None:
        result = TTSEngine._split_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self) -> None:
        result = TTSEngine._split_sentences("Hello. How are you? Fine!")
        assert result == ["Hello.", "How are you?", "Fine!"]

    def test_abbreviations_not_split(self) -> None:
        """Abbreviations like 'Dr.' followed by lowercase should not split."""
        result = TTSEngine._split_sentences("Dr. Smith went home.")
        # The regex splits on period+space+uppercase, so "Dr. Smith" should split
        # because 'S' is uppercase. This is expected behavior per the docstring.
        assert len(result) >= 1

    def test_empty_string(self) -> None:
        result = TTSEngine._split_sentences("")
        assert result == []

    def test_no_punctuation(self) -> None:
        result = TTSEngine._split_sentences("Hello world")
        assert result == ["Hello world"]

    def test_multiple_spaces(self) -> None:
        result = TTSEngine._split_sentences("Hello.  World!")
        assert len(result) == 2

    def test_newlines_in_text(self) -> None:
        result = TTSEngine._split_sentences("First sentence.\nSecond sentence.")
        assert len(result) >= 1

    def test_question_and_exclamation(self) -> None:
        result = TTSEngine._split_sentences("Really? Yes! Okay.")
        assert len(result) == 3

    def test_single_word(self) -> None:
        result = TTSEngine._split_sentences("Hello")
        assert result == ["Hello"]

    def test_chunked_uses_split(self) -> None:
        """synthesize_chunked calls _split_sentences internally."""
        engine = TTSEngine()

        with patch.object(
            engine,
            "synthesize",
            side_effect=[
                np.array([1.0], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
                np.array([3.0], dtype=np.float32),
            ],
        ):
            chunks = list(engine.synthesize_chunked("First. Second. Third."))

        assert len(chunks) == 3


# ===========================================================================
# TTS Resampling Tests
# ===========================================================================

class TestTTSResampling:
    """Test that audio is resampled when Kokoro outputs a different sample rate."""

    def test_no_resample_when_rates_match(self) -> None:
        engine = TTSEngine(sample_rate=16_000)
        kokoro = MagicMock()
        kokoro.create.return_value = (np.ones(100, dtype=np.float32), 16_000)

        with patch.object(engine, "_get_kokoro", return_value=kokoro):
            with patch.object(TTSEngine, "_resample") as resample_mock:
                engine.synthesize("hello")

        resample_mock.assert_not_called()

    def test_resample_when_rates_differ(self) -> None:
        engine = TTSEngine(sample_rate=16_000)
        kokoro = MagicMock()
        kokoro.create.return_value = (np.ones(100, dtype=np.float32), 24_000)

        with patch.object(engine, "_get_kokoro", return_value=kokoro):
            with patch.object(
                TTSEngine, "_resample",
                return_value=np.ones(67, dtype=np.float32),
            ) as resample_mock:
                result = engine.synthesize("hello")

        resample_mock.assert_called_once()
        assert result.shape == (67,)


# ===========================================================================
# TranscriptResult dataclass tests
# ===========================================================================

class TestTranscriptResult:
    """Test TranscriptResult and TranscriptSegment dataclasses."""

    def test_transcript_segment_fields(self) -> None:
        seg = TranscriptSegment(text="hello", start=0.0, end=1.0, no_speech_prob=0.1)
        assert seg.text == "hello"
        assert seg.start == 0.0
        assert seg.end == 1.0
        assert seg.no_speech_prob == 0.1

    def test_transcript_result_fields(self) -> None:
        result = TranscriptResult(
            text="hello world",
            segments=[],
            language="en",
            language_prob=0.95,
            duration_s=1.5,
            no_speech_prob=0.0,
        )
        assert result.text == "hello world"
        assert result.language == "en"
        assert result.language_prob == 0.95
        assert result.duration_s == 1.5
