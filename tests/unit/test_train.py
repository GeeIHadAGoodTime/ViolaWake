"""Targeted unit tests for violawake_sdk.tools.train CLI behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk.tools import train


def _touch_audio_files(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (directory / f"{idx:03d}.wav").write_bytes(b"wav")


def _path_exists_without_corpus(original_exists):
    def _exists(path: Path) -> bool:
        if "corpus" in {part.lower() for part in path.parts}:
            return False
        return original_exists(path)

    return _exists


class TestTrainHelpers:
    def test_held_out_count_keeps_at_least_one_training_file(self) -> None:
        assert train._held_out_count(0) == 0
        assert train._held_out_count(1) == 0
        assert train._held_out_count(2) == 1
        assert train._held_out_count(10) == 5

    def test_auto_eval_verdict_thresholds(self) -> None:
        assert train._auto_eval_verdict(9.9) == "GOOD (EER < 10%)"
        assert train._auto_eval_verdict(10.0) == "ACCEPTABLE (EER <= 15%)"
        assert train._auto_eval_verdict(20.0) == "WARNING (EER > 15%)"
        assert train._auto_eval_verdict(30.0) == "CRITICAL (EER > 25%)"

    def test_update_auto_eval_config_merges_existing_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / "model.config.json"
        config_path.write_text(json.dumps({"wake_word": "viola"}), encoding="utf-8")

        train._update_auto_eval_config(config_path, {"status": "ok", "eer_percent": 12.5})

        saved = json.loads(config_path.read_text(encoding="utf-8"))
        assert saved["wake_word"] == "viola"
        assert saved["auto_eval"]["status"] == "ok"


class TestTrainMainValidation:
    def test_main_exits_when_positives_dir_is_missing(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

        assert "Positives directory not found" in capsys.readouterr().err

    def test_main_exits_when_negatives_dir_is_missing(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        positives.mkdir()
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(tmp_path / "missing-neg"),
            "--output",
            str(output),
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

        assert "Negatives directory not found" in capsys.readouterr().err

    def test_main_requires_positives_for_mlp_architecture(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--output",
            str(output),
            "--architecture",
            "mlp",
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

        assert "--positives is required for MLP architecture" in capsys.readouterr().err

    def test_main_temporal_exits_with_too_few_positive_files(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        positives.mkdir()
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--output",
            str(output),
            "--no-auto-corpus",
            "--quiet",
        ]

        with patch.object(sys, "argv", argv), pytest.raises(SystemExit, match="1"):
            train.main()

        assert "Provide at least 5 via --positives or enable --auto-corpus" in capsys.readouterr().err

    def test_main_temporal_exits_with_too_few_negative_files(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        positives = tmp_path / "positives"
        negatives = tmp_path / "negatives"
        _touch_audio_files(positives, 5)
        negatives.mkdir()
        output = tmp_path / "model.onnx"
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(negatives),
            "--output",
            str(output),
            "--no-auto-corpus",
            "--quiet",
        ]

        original_exists = Path.exists
        with (
            patch.object(sys, "argv", argv),
            patch("pathlib.Path.exists", new=_path_exists_without_corpus(original_exists)),
            pytest.raises(SystemExit, match="1"),
        ):
            train.main()

        assert "Enable --auto-corpus or provide negatives via --negatives" in capsys.readouterr().err

    def test_main_parses_args_and_invokes_temporal_training(self, tmp_path: Path) -> None:
        positives = tmp_path / "positives"
        negatives = tmp_path / "negatives"
        eval_dir = tmp_path / "eval"
        (eval_dir / "positives").mkdir(parents=True)
        (eval_dir / "negatives").mkdir(parents=True)
        _touch_audio_files(positives, 6)
        _touch_audio_files(negatives, 6)
        output = tmp_path / "model.onnx"
        evaluate_module = ModuleType("violawake_sdk.tools.evaluate")
        evaluate_module.evaluate_onnx_model = MagicMock(
            return_value={
                "architecture": "temporal_cnn",
                "n_positives": 6,
                "n_negatives": 6,
                "eer_approx": 0.08,
                "roc_auc": 0.95,
                "optimal_far": 0.02,
                "optimal_frr": 0.03,
                "optimal_threshold": 0.82,
            }
        )
        argv = [
            "violawake-train",
            "--word",
            "viola",
            "--positives",
            str(positives),
            "--negatives",
            str(negatives),
            "--output",
            str(output),
            "--eval-dir",
            str(eval_dir),
            "--epochs",
            "12",
            "--batch-size",
            "32",
            "--lr",
            "0.002",
            "--patience",
            "4",
            "--no-auto-corpus",
            "--no-augment",
            "--quiet",
        ]

        original_exists = Path.exists
        with (
            patch.object(sys, "argv", argv),
            patch.dict(sys.modules, {"violawake_sdk.tools.evaluate": evaluate_module}),
            patch("pathlib.Path.exists", new=_path_exists_without_corpus(original_exists)),
            patch("violawake_sdk.tools.train._train_temporal_cnn") as train_temporal,
        ):
            train.main()

        train_temporal.assert_called_once()
        kwargs = train_temporal.call_args.kwargs
        assert kwargs["wake_word"] == "viola"
        assert kwargs["epochs"] == 12
        assert kwargs["batch_size"] == 32
        assert kwargs["lr"] == 0.002
        assert kwargs["patience"] == 4
        assert kwargs["augment"] is False
        assert kwargs["verbose"] is False
        assert len(kwargs["pos_files"]) == 6
        assert len(kwargs["neg_files"]) == 6
