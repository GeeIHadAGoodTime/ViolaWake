"""Unit tests for confusable generation and scoring."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from violawake_sdk.tools.confusables import (
    PHONETIC_SIMILARITY_MAP,
    generate_confusables,
    phonetic_similarity,
    simple_phonetic_key,
)
from violawake_sdk.tools.test_confusables import run_confusable_tests


def test_generate_confusables_returns_non_empty_list() -> None:
    confusables = generate_confusables("viola")

    assert confusables
    assert "crayola" in confusables
    assert "voila" in confusables


def test_generated_words_are_different_from_wake_word() -> None:
    confusables = generate_confusables("viola", count=25)

    assert all(candidate != "viola" for candidate in confusables)


def test_phonetic_similarity_functions() -> None:
    assert simple_phonetic_key("viola") == simple_phonetic_key("voila")
    assert phonetic_similarity("viola", "voila") > phonetic_similarity("viola", "tomato")
    assert phonetic_similarity("viola", "crayola") > 0.45


def test_phonetic_similarity_map_is_non_empty_and_bidirectional() -> None:
    assert PHONETIC_SIMILARITY_MAP

    for source, targets in PHONETIC_SIMILARITY_MAP.items():
        assert targets
        for target in targets:
            assert source in PHONETIC_SIMILARITY_MAP[target]


def test_run_confusable_tests_uses_mocked_detector() -> None:
    class FakeDetector:
        def __init__(self, model: str, threshold: float) -> None:
            self.model = model
            self.threshold = threshold

        def process(self, audio_frame: np.ndarray) -> float:
            return 0.92 if float(np.max(np.abs(audio_frame))) > 0.5 else 0.20

    fake_audio = {
        "crayola": np.ones(640, dtype=np.float32) * 0.8,
        "ebola": np.ones(640, dtype=np.float32) * 0.1,
    }

    with (
        patch(
            "violawake_sdk.tools.test_confusables.generate_confusables",
            return_value=["crayola", "ebola"],
        ),
        patch(
            "violawake_sdk.tools.test_confusables.generate_confusable_tts_audio",
            return_value=fake_audio,
        ),
        patch("violawake_sdk.tools.test_confusables.WakeDetector", FakeDetector),
    ):
        report = run_confusable_tests(
            model_path="fake_model.onnx",
            wake_word="viola",
            threshold=0.80,
        )

    assert report.summary.total_tested == 2
    assert report.summary.false_accepts == 1
    assert report.summary.false_accept_rate == 0.5
    assert report.results[0].word == "crayola"
    assert report.results[0].false_accept is True
    assert report.results[1].passed is True
