"""Unit tests for the vad_engine re-export module and lifecycle methods."""

from __future__ import annotations

import violawake_sdk.vad_engine as vad_engine_module
from violawake_sdk.vad import VADBackend as CoreVADBackend
from violawake_sdk.vad import VADEngine as CoreVADEngine
from violawake_sdk.vad_engine import VADBackend, VADEngine


def test_reexports_match_core_symbols() -> None:
    assert VADEngine is CoreVADEngine
    assert VADBackend is CoreVADBackend


def test_module_all_exports_are_present() -> None:
    for name in vad_engine_module.__all__:
        assert hasattr(vad_engine_module, name)


def test_close_releases_backend() -> None:
    vad = VADEngine(backend="rms")

    vad.close()

    assert vad._backend is None


def test_context_manager_closes_backend() -> None:
    with VADEngine(backend="rms") as vad:
        assert vad.backend_name == "rms"

    assert vad._backend is None
