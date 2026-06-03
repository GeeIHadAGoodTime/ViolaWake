from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from benchmark_v2 import reproduce_claims
from violawake_sdk.models import MODEL_REGISTRY


def test_validate_model_metadata_rejects_wrong_sha() -> None:
    spec = MODEL_REGISTRY["temporal_cnn"]
    results = {
        "metadata": {
            "model": {
                "name": spec.name,
                "version": spec.version,
                "sha256": "0" * 64,
                "size_bytes": spec.size_bytes,
            }
        }
    }

    with pytest.raises(ValueError, match="sha256 mismatch"):
        reproduce_claims.validate_model_metadata(results)


def test_score_validation_rejects_positive_row_under_negative_path(tmp_path: Path) -> None:
    score_set = reproduce_claims.ScoreSet(
        system_name="ViolaWake",
        wake_word="viola",
        rows=[
            reproduce_claims.ScoreRow(
                file="benchmark_v2/corpus/negatives/music/song.wav",
                label="positive",
                score=0.99,
                category="positive_viola",
            )
        ],
    )

    with pytest.raises(ValueError, match="corpus/positives/viola"):
        reproduce_claims.validate_score_rows(score_set, tmp_path)


def test_run_benchmark_defaults_are_worktree_relative() -> None:
    module_path = Path("benchmark_v2/run_benchmark.py").resolve()
    spec = importlib.util.spec_from_file_location("run_benchmark_for_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert module.CORPUS_DIR == module.REPO_ROOT / "benchmark_v2" / "corpus"
    assert module.OUTPUT_DIR == module.REPO_ROOT / "benchmark_v2"
    assert module.CORPUS_DIR != Path("J:/CLAUDE/PROJECTS/Wakeword/benchmark_v2/corpus")


def test_reproducer_current_artifacts_pass() -> None:
    report = reproduce_claims.run(
        argparse_namespace(
            benchmark_dir=Path("benchmark_v2"),
            report=None,
            require_audio_files=False,
        )
    )

    assert "EER | 5.49% | 8.24%" in report
    assert "Per-Category FAR/FRR" in report


def argparse_namespace(**kwargs):
    class Namespace:
        pass

    namespace = Namespace()
    for key, value in kwargs.items():
        setattr(namespace, key, value)
    return namespace
