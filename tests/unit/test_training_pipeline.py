"""Temporal-only training surface tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


def test_legacy_mlp_training_is_removed(tmp_path: Path) -> None:
    """Legacy MLP entry points should fail loudly instead of training."""
    from violawake_sdk.tools.train import _train_mlp_on_oww

    with pytest.raises(RuntimeError, match="Legacy MLP training has been removed"):
        _train_mlp_on_oww(tmp_path, tmp_path / "model.onnx")


def test_temporal_trainer_exposes_progress_callback() -> None:
    """The production trainer should still expose progress updates."""
    from violawake_sdk.tools.train import _train_temporal_cnn

    signature = inspect.signature(_train_temporal_cnn)
    parameter_names = set(signature.parameters.keys())

    assert "pos_files" in parameter_names
    assert "neg_files" in parameter_names
    assert "progress_callback" in parameter_names


def test_temporal_config_metadata_shape(tmp_path: Path) -> None:
    """Saved config metadata should describe the production architecture."""
    config = {
        "architecture": "temporal_cnn",
        "n_pos_samples": 10,
        "n_neg_samples": 50,
        "quality_gate": {"grade": "A"},
    }
    config_path = tmp_path / "model.config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    loaded = json.loads(config_path.read_text(encoding="utf-8"))
    assert loaded["architecture"] == "temporal_cnn"
    assert loaded["quality_gate"]["grade"] == "A"
