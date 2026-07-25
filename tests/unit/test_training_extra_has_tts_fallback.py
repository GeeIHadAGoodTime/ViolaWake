"""Ratchet (#1775/#1768): the training extra must ship the TTS fallback it uses.

The training pipeline synthesizes its positives, its negatives, and the quality
gate's own test material with edge-tts -- a NETWORK dependency on Microsoft's
voice service, which retires voices server-side without notice
(CL-20260717-b117). `_KokoroFallback` is the thing that stops one dead voice from
emptying a sample set, and for the quality gate an empty negative set is grade F
by construction.

That fallback imports `violawake_sdk.tts`, which imports `kokoro-onnx`. But
kokoro-onnx sat only in the `tts` extra while the backend image installs
`/sdk[training]` (console/Dockerfile.backend), so `ready()` returned False in
production and the entire fallback -- the #1768 fix included -- was dead code
wherever it actually mattered. Nothing was red, because the fallback fails
gracefully by design.

Reds if the training extra ever again omits a package the training pipeline's
TTS fallback needs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    tomllib = pytest.importorskip("tomli", reason="tomli required to read pyproject")

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
TRAIN_PY = ROOT / "src" / "violawake_sdk" / "tools" / "train.py"
DOCKERFILE = ROOT / "console" / "Dockerfile.backend"


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _names(reqs: list[str]) -> set[str]:
    """Normalize requirement strings to comparable distribution names."""
    out = set()
    for r in reqs:
        name = re.split(r"[<>=!\[;\s]", r, maxsplit=1)[0].strip()
        if name:
            out.add(name.lower().replace("_", "-"))
    return out


def test_the_training_pipeline_actually_uses_the_kokoro_fallback() -> None:
    """Premise check: if this stops being true the rest of the file is moot."""
    source = TRAIN_PY.read_text(encoding="utf-8")
    assert "_KokoroFallback" in source
    # Used, not merely defined: at least one instantiation site.
    assert source.count("_KokoroFallback()") >= 1, (
        "the Kokoro fallback is defined but never instantiated in the training pipeline"
    )


def test_training_extra_includes_kokoro_so_the_fallback_is_not_inert() -> None:
    """The load-bearing ratchet."""
    training = _names(_extras()["training"])
    assert "kokoro-onnx" in training, (
        "the training pipeline calls _KokoroFallback, which imports "
        "violawake_sdk.tts -> kokoro-onnx, but kokoro-onnx is not in the "
        "`training` extra. The backend image installs /sdk[training], so "
        "_KokoroFallback.ready() returns False in production and every edge-tts "
        "fallback in the training pipeline is dead code -- silently, because "
        "ready() fails gracefully by design."
    )


def test_training_extra_also_still_includes_edge_tts() -> None:
    """The primary engine and its fallback must ship together, or the fallback
    is the only path and the 'fallback' framing is wrong."""
    training = _names(_extras()["training"])
    assert "edge-tts" in training


def test_the_backend_image_installs_the_training_extra() -> None:
    """Pins the assumption this ratchet rests on: if the image ever installs a
    different extra, the dependency check above must move with it."""
    if not DOCKERFILE.exists():  # pragma: no cover - layout change
        pytest.skip("console/Dockerfile.backend not present")
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "[training]" in content, (
        "the backend image no longer installs the `training` extra; re-point "
        "this gate at whichever extra it installs now"
    )
