"""
VAD engine module -- re-exports VADEngine from vad.py.
"""
from __future__ import annotations

from violawake_sdk.vad import VADBackend, VADEngine

__all__ = ["VADEngine", "VADBackend"]
