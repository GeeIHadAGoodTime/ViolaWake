"""
TTS engine module -- re-exports TTSEngine from tts.py.
"""

from __future__ import annotations

from violawake_sdk.tts import (
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    TARGET_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    TTSEngine,
)

__all__ = [
    "TTSEngine",
    "AVAILABLE_VOICES",
    "DEFAULT_VOICE",
    "TTS_SAMPLE_RATE",
    "TARGET_SAMPLE_RATE",
]
