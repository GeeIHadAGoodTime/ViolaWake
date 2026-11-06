"""A dead TTS service must be noticed in seconds, and a dead fallback must not claim ready.

Oct-Nov 2026: edge-tts answered every connection with 403. ``_edge_tts_synthesize``
spent three attempts plus backoff on each of ~1,000 phrases, so every job took about
55 minutes to fail, and ``_KokoroFallback.ready()`` printed "fallback ready" because
building a ``TTSEngine`` does not load anything, while kokoro-onnx was not installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from violawake_sdk.tools import train


class _RefusingCommunicate:
    calls = 0

    def __init__(self, text: str, voice: str) -> None:
        pass

    async def stream(self):
        type(self).calls += 1
        raise ConnectionError("403, message='Invalid response status'")
        yield  # pragma: no cover - makes this an async generator


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    train.reset_edge_tts_reporting()
    _RefusingCommunicate.calls = 0
    monkeypatch.setattr(train, "_EDGE_TTS_RETRY_BASE_SECONDS", 0.0)
    monkeypatch.setattr(train._EDGE_TTS_RETRY_RNG, "uniform", lambda *_: 0.0)
    fake = types.ModuleType("edge_tts")
    fake.Communicate = _RefusingCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake)
    yield
    train.reset_edge_tts_reporting()


def test_a_service_refusing_every_voice_is_skipped_after_a_few_phrases(tmp_path) -> None:
    for index in range(1000):
        assert train._edge_tts_synthesize("phrase", "en-US-GuyNeural", tmp_path / f"{index}.wav") is False

    assert _RefusingCommunicate.calls == train._EDGE_TTS_DOWN_AFTER * train._EDGE_TTS_MAX_ATTEMPTS


def test_one_dead_voice_among_working_ones_does_not_trip_it(tmp_path) -> None:
    for index in range(train._EDGE_TTS_DOWN_AFTER * 3):
        train._edge_tts_fail("phrase", "en-US-DeadNeural", "failed after 3 attempts")
        if index % 2:
            train._EDGE_TTS_CONSECUTIVE_FAILURES = 0  # a neighbouring voice succeeded

    assert train._EDGE_TTS_DOWN is False


def test_the_next_job_tries_again(tmp_path) -> None:
    for index in range(train._EDGE_TTS_DOWN_AFTER):
        train._edge_tts_synthesize("phrase", "en-US-GuyNeural", tmp_path / f"{index}.wav")
    assert train._EDGE_TTS_DOWN is True

    train.reset_edge_tts_reporting()

    assert train._EDGE_TTS_DOWN is False


def test_fallback_is_not_ready_when_kokoro_cannot_actually_load(monkeypatch) -> None:
    from violawake_sdk import tts

    def _not_installed(self):
        raise ImportError("kokoro-onnx is not installed")

    monkeypatch.setattr(tts.TTSEngine, "_load_kokoro", _not_installed)

    assert train._KokoroFallback().ready() is False


def test_probe_reports_no_speech_when_nothing_works(monkeypatch, tmp_path) -> None:
    from violawake_sdk import tts

    def _not_installed(self):
        raise ImportError("kokoro-onnx is not installed")

    monkeypatch.setattr(tts.TTSEngine, "_load_kokoro", _not_installed)

    probe = train.probe_speech_synthesis(tmp_dir=tmp_path)

    assert probe.available is False
    assert "403" in (probe.edge_tts_error or "")
    assert _RefusingCommunicate.calls == 1, "the check must not pay for retries"


def test_probe_leaves_a_running_job_breaker_alone(tmp_path) -> None:
    train._EDGE_TTS_DOWN = True
    train._EDGE_TTS_CONSECUTIVE_FAILURES = 99

    train.probe_speech_synthesis(tmp_dir=tmp_path)

    assert train._EDGE_TTS_DOWN is True
    assert train._EDGE_TTS_CONSECUTIVE_FAILURES == 99


def test_edge_tts_is_new_enough_for_microsoft() -> None:
    """edge-tts < 7.3.1 sends Sec-MS-GEC-Version 1-*, which Microsoft answers with 403."""
    import pathlib
    import re

    root = pathlib.Path(train.__file__).resolve().parents[3]
    for name in ("pyproject.toml", "console/backend/requirements.txt"):
        floors = re.findall(r"edge-tts>=([\d.]+)", (root / name).read_text(encoding="utf-8"))
        assert floors, name
        for floor in floors:
            assert tuple(int(part) for part in floor.split(".")) >= (7, 3, 1), name


def test_the_backend_image_installs_the_fallback() -> None:
    import pathlib

    root = pathlib.Path(train.__file__).resolve().parents[3]
    dockerfile = (root / "console/Dockerfile.backend").read_text(encoding="utf-8")
    assert "[training,tts]" in dockerfile
