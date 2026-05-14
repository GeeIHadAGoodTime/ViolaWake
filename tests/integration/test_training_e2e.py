"""Integration-style regressions for the temporal-only training surface."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _run_train(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "violawake_sdk.tools.train", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_cli_rejects_legacy_architecture_flag(tmp_path: Path) -> None:
    pos_dir = tmp_path / "positives"
    pos_dir.mkdir()

    result = _run_train(
        [
            "--word", "goldentest",
            "--positives", str(pos_dir),
            "--output", str(tmp_path / "model.onnx"),
            "--architecture", "mlp",
        ]
    )

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr.lower()


def test_legacy_mlp_helper_fails_fast(tmp_path: Path) -> None:
    from violawake_sdk.tools.train import _train_mlp_on_oww

    with pytest.raises(RuntimeError, match="Legacy MLP training has been removed"):
        _train_mlp_on_oww(tmp_path, tmp_path / "model.onnx")
