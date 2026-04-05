from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="scikit-learn required for evaluation tests")

from violawake_sdk.training.evaluate import (
    _dump_scores_csv,
    compute_confusion_matrix,
    compute_dprime,
    compute_eer,
    detect_architecture,
    evaluate_onnx_model,
    find_optimal_threshold,
)


class _MockInput:
    def __init__(self, shape: list, name: str = "input") -> None:
        self.shape = shape
        self.name = name


class _MockSession:
    def __init__(self, shape: list, name: str = "input") -> None:
        self._input = _MockInput(shape, name)

    def get_inputs(self) -> list[_MockInput]:
        return [self._input]


def _make_test_dir(tmp_path: Path, *, n_pos: int = 3, n_neg: int = 3) -> Path:
    test_dir = tmp_path / "test_data"
    pos_dir = test_dir / "positives"
    neg_dir = test_dir / "negatives"
    pos_dir.mkdir(parents=True)
    neg_dir.mkdir(parents=True)

    for idx in range(n_pos):
        (pos_dir / f"pos_{idx}.wav").touch()
    for idx in range(n_neg):
        (neg_dir / f"neg_{idx}.wav").touch()

    return test_dir


def test_compute_dprime_and_eer_basic() -> None:
    rng = np.random.default_rng(42)
    pos = rng.normal(0.85, 0.05, 100)
    neg = rng.normal(0.15, 0.05, 100)

    d_prime = compute_dprime(pos, neg)
    assert d_prime > 5.0

    fpr = np.array([0.0, 0.1, 1.0])
    tpr = np.array([0.0, 0.9, 1.0])
    eer, idx = compute_eer(fpr, tpr)
    assert 0.0 <= eer <= 0.2
    assert idx in (0, 1, 2)


def test_detect_architecture_from_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()
    model_path.with_suffix(".config.json").write_text(json.dumps({"architecture": "temporal_cnn"}))

    arch = detect_architecture(model_path, _MockSession([1, 40, 100]))
    assert arch == "temporal_oww"


@pytest.mark.parametrize(
    ("shape", "name", "expected"),
    [
        ([1, 96], "input", "mlp_on_oww"),
        (["batch", 96], "input", "mlp_on_oww"),
        ([1, 9, 96], "input", "temporal_oww"),
        ([1, "frames", 96], "input", "temporal_oww"),
        ([1, 40, 100], "input", "cnn"),
        ([1, "frames", 40], "input", "cnn"),
    ],
)
def test_detect_architecture_from_shape(
    tmp_path: Path,
    shape: list,
    name: str,
    expected: str,
) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.touch()

    arch = detect_architecture(model_path, _MockSession(shape, name))
    assert arch == expected


def test_find_optimal_threshold_minimizes_fpr_plus_fnr() -> None:
    pos_scores = np.array([0.81, 0.82, 0.91, 0.95])
    neg_scores = np.array([0.02, 0.08, 0.11, 0.14])

    result = find_optimal_threshold(pos_scores, neg_scores)

    assert 0.14 < result["optimal_threshold"] < 0.81
    assert result["optimal_far"] == 0.0
    assert result["optimal_frr"] == 0.0


def test_compute_confusion_matrix_counts() -> None:
    pos = np.array([0.8, 0.2])
    neg = np.array([0.6, 0.1])

    cm = compute_confusion_matrix(pos, neg, threshold=0.5)

    assert cm["tp"] == 1
    assert cm["fn"] == 1
    assert cm["fp"] == 1
    assert cm["tn"] == 1


def test_dump_scores_csv_uses_correct_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "scores.csv"

    _dump_scores_csv(
        [Path("pos.wav")],
        [0.9],
        [Path("neg.wav")],
        [0.7],
        0.8,
        csv_path,
    )

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["file", "label", "score", "correct"]
    assert rows[1] == ["pos.wav", "positive", "0.900000", "True"]
    assert rows[2] == ["neg.wav", "negative", "0.700000", "True"]


def test_temporal_oww_scorer_uses_seq_len_and_repeat_padding(tmp_path: Path) -> None:
    from violawake_sdk.training.evaluate import _build_temporal_oww_scorer

    mock_session = mock.MagicMock()
    mock_session.get_inputs.return_value = [_MockInput([1, 3, 96], "embeddings")]
    mock_session.run.return_value = [np.array([[0.42]], dtype=np.float32)]

    frame_1 = np.full((96,), 1.0, dtype=np.float32)
    frame_2 = np.full((96,), 2.0, dtype=np.float32)
    mock_preprocessor = mock.MagicMock()
    mock_preprocessor.embed_clips.return_value = np.stack([frame_1, frame_2], axis=0)[
        np.newaxis, :, :
    ]

    mock_oww_model = mock.MagicMock()
    mock_oww_model.preprocessor = mock_preprocessor
    mock_oww_module = mock.MagicMock()
    mock_oww_module.Model.return_value = mock_oww_model

    mock_audio = mock.MagicMock()
    mock_audio.load_audio.return_value = np.ones(16000, dtype=np.float32)
    mock_audio.center_crop.return_value = np.ones(16000, dtype=np.float32)
    mock_constants = mock.MagicMock()
    mock_constants.CLIP_SAMPLES = 16000

    with mock.patch.dict(
        sys.modules,
        {
            "openwakeword": mock.MagicMock(),
            "openwakeword.model": mock_oww_module,
            "violawake_sdk.audio": mock_audio,
            "violawake_sdk._constants": mock_constants,
        },
    ):
        scorer = _build_temporal_oww_scorer(mock_session, "embeddings")

    wav_path = tmp_path / "sample.wav"
    wav_path.touch()
    assert scorer(wav_path) == pytest.approx(0.42)

    run_input = mock_session.run.call_args[0][1]["embeddings"]
    assert run_input.shape == (1, 3, 96)
    np.testing.assert_allclose(run_input[0, 0], frame_1)
    np.testing.assert_allclose(run_input[0, 1], frame_2)
    np.testing.assert_allclose(run_input[0, 2], frame_2)


def test_evaluate_onnx_model_with_sweep_and_csv(tmp_path: Path) -> None:
    test_dir = _make_test_dir(tmp_path, n_pos=3, n_neg=3)
    csv_path = tmp_path / "scores.csv"

    pos_scores = iter([0.91, 0.88, 0.84])
    neg_scores = iter([0.11, 0.09, 0.05])

    def mock_scorer(wav_path: Path) -> float:
        if "positives" in str(wav_path):
            return next(pos_scores)
        return next(neg_scores)

    with mock.patch(
        "violawake_sdk.training.evaluate.build_model_scorer",
        return_value=("temporal_oww", mock_scorer),
    ):
        results = evaluate_onnx_model(
            "fake.onnx",
            test_dir,
            threshold=0.5,
            dump_scores_csv=csv_path,
            sweep=True,
        )

    assert results["threshold_sweep_enabled"] is True
    assert results["optimal_threshold"] != pytest.approx(0.5)
    assert results["optimal_confusion_matrix"]["tp"] == 3
    assert results["optimal_confusion_matrix"]["tn"] == 2

    with open(csv_path) as f:
        rows = list(csv.reader(f))

    assert rows[0] == ["file", "label", "score", "correct"]
    assert len(rows) == 7


def test_evaluate_onnx_model_without_sweep_reuses_requested_threshold(tmp_path: Path) -> None:
    test_dir = _make_test_dir(tmp_path, n_pos=2, n_neg=2)

    pos_scores = iter([0.8, 0.4])
    neg_scores = iter([0.6, 0.1])

    def mock_scorer(wav_path: Path) -> float:
        if "positives" in str(wav_path):
            return next(pos_scores)
        return next(neg_scores)

    with mock.patch(
        "violawake_sdk.training.evaluate.build_model_scorer",
        return_value=("mlp_on_oww", mock_scorer),
    ):
        results = evaluate_onnx_model("fake.onnx", test_dir, threshold=0.5, sweep=False)

    assert results["threshold_sweep_enabled"] is False
    assert results["optimal_threshold"] == pytest.approx(0.5)
    assert results["optimal_confusion_matrix"] == results["confusion_matrix"]


def test_cli_forwards_output_csv_and_sweep(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from violawake_sdk.tools.evaluate import main

    model_path = tmp_path / "model.onnx"
    test_dir = _make_test_dir(tmp_path)
    csv_path = tmp_path / "scores.csv"
    model_path.touch()

    fake_results = {
        "architecture": "temporal_oww",
        "d_prime": 9.0,
        "far_per_hour": 0.0,
        "frr": 0.0,
        "roc_auc": 1.0,
        "n_positives": 3,
        "n_negatives": 3,
        "optimal_threshold": 0.42,
        "optimal_far": 0.0,
        "optimal_frr": 0.0,
        "eer_approx": 0.0,
        "confusion_matrix": {"tp": 3, "fp": 0, "tn": 3, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "optimal_confusion_matrix": {"tp": 3, "fp": 0, "tn": 3, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        "tp_scores": [0.9, 0.8, 0.85],
        "fp_scores": [0.1, 0.2, 0.15],
    }

    with (
        mock.patch("violawake_sdk.tools.evaluate.evaluate_onnx_model", return_value=fake_results) as mocked,
        mock.patch.object(
            sys,
            "argv",
            [
                "violawake-eval",
                "--model",
                str(model_path),
                "--test-dir",
                str(test_dir),
                "--output-csv",
                str(csv_path),
                "--sweep",
            ],
        ),
    ):
        main()

    kwargs = mocked.call_args.kwargs
    assert kwargs["dump_scores_csv"] == str(csv_path)
    assert kwargs["sweep"] is True
    assert "Optimal threshold" in capsys.readouterr().out
