"""Regression test: train_full_pipeline.py's voice pool must not carry a
voice Microsoft has retired from the Edge Read-Aloud service.

Incident (#1768, GlitchTip violawake issues 25/34/38): this script keeps its
own independent copy of the edge-tts voice list, separate from
violawake_sdk.tools.train.EDGE_TTS_VOICES. When that shared list was fixed
to drop en-US-DavisNeural and 6 other retired voices, this script's own
duplicate copy still had all 7 of them -- confirmed dead live via
edge_tts.list_voices() while root-causing the shared bug. Requesting a
retired ShortName completes the edge-tts WebSocket handshake but the server
never returns audio, so this script's un-retried, no-fallback
`_synthesize_one` would silently drop every sample for that voice, forever,
for anyone who runs it.

This pins the fixed voice list and flags the underlying duplication so a
future retirement doesn't quietly regress this file a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import train_full_pipeline  # noqa: E402  (path insert must precede this import)

# Confirmed retired server-side via edge_tts.list_voices() on the prod box,
# 2026-07-15/17 (#1768).
_RETIRED_VOICES = {
    "en-US-DavisNeural",
    "en-US-AmberNeural",
    "en-US-BrandonNeural",
    "en-US-CoraNeural",
    "en-US-ElizabethNeural",
    "en-US-JacobNeural",
    "en-US-MonicaNeural",
}


class TestTrainFullPipelineVoices:
    def test_edge_voices_excludes_known_retired_voices(self) -> None:
        assert not _RETIRED_VOICES & set(train_full_pipeline.EDGE_VOICES)

    def test_edge_voices_has_no_duplicates(self) -> None:
        voices = train_full_pipeline.EDGE_VOICES
        assert len(voices) == len(set(voices))
