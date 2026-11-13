"""Custom words are trained at the positions the desktop scores them, over room tone."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np

from violawake_sdk.tools.train import _shift_positives, _speech_span, _train_temporal_cnn

SR = 16000
CLIP = 24000


def _word(seconds: float = 0.4, at: float = 0.6, total: float = 1.5) -> np.ndarray:
    audio = np.zeros(int(total * SR), dtype=np.float32)
    start = int(at * SR)
    audio[start : start + int(seconds * SR)] = 0.3 * np.sin(np.arange(int(seconds * SR)) / 5.0)
    return audio


def _centre(audio: np.ndarray) -> float:
    start, end = _speech_span(audio)
    return (start + end) / 2 / SR


def test_the_spoken_part_is_found_in_an_untrimmed_recording() -> None:
    start, end = _speech_span(_word(at=0.2, total=1.536))
    assert abs(start / SR - 0.15) < 0.02 and abs(end / SR - 0.65) < 0.02


def test_each_offset_moves_the_word_by_that_much() -> None:
    offsets = (-240, -160, -80, 0, 80, 160, 240)
    out = _shift_positives([_word(at=0.1)], offsets_ms=offsets, clip_samples=CLIP)

    assert len(out) == len(offsets) and all(a.size == CLIP for a in out)
    centres = [_centre(a) for a in out]
    for offset, centre in zip(offsets, centres, strict=True):
        assert abs(centre - (0.75 + offset / 1000)) < 0.03


def test_padding_is_room_tone_not_digital_silence() -> None:
    """Live audio never contains exact zeros; zero-padded TTS clips taught a pattern the mic never shows."""
    (out,) = _shift_positives([_word()], offsets_ms=(0,), clip_samples=CLIP)
    head = out[:2000]
    assert np.count_nonzero(head) == head.size
    assert 0.001 < float(np.sqrt(np.mean(head**2))) < 0.004


def test_a_word_longer_than_the_window_is_cropped_not_dropped() -> None:
    (out,) = _shift_positives([_word(seconds=2.0, at=0.0, total=2.0)], offsets_ms=(0,), clip_samples=CLIP)
    assert out.size == CLIP


def test_the_trainer_takes_the_shift_and_a_minimum_training_length() -> None:
    params = inspect.signature(_train_temporal_cnn).parameters
    assert params["shift_offsets_ms"].default is None
    assert params["min_epochs"].default == 0  # the built-in recipe and the CLI are unchanged


def test_the_service_trains_customer_words_with_both() -> None:
    source = Path(__file__).resolve().parents[2] / "console/backend/app/services/training_service.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    constants = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets)
        if isinstance(target, ast.Name) and target.id.startswith("CUSTOM_WORD_")
    }
    assert constants == {
        "CUSTOM_WORD_SHIFT_OFFSETS_MS": (-240, -160, -80, 0, 80, 160, 240),
        "CUSTOM_WORD_MIN_EPOCHS": 15,
    }
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_train_temporal_cnn"
    ]
    assert calls, "the service no longer calls the temporal trainer"
    for call in calls:
        keywords = {kw.arg: ast.unparse(kw.value) for kw in call.keywords}
        assert keywords.get("shift_offsets_ms") == "CUSTOM_WORD_SHIFT_OFFSETS_MS"
        assert keywords.get("min_epochs") == "CUSTOM_WORD_MIN_EPOCHS"
