"""CLI entry-point smoke tests.

Covers every ``[project.scripts]`` entry point declared in pyproject.toml
plus the thin wrappers in ``violawake_sdk.cli``.

Strategy:
  - ``--help`` must exit 0 and print usage text (proves argparse is wired).
  - Missing required args must exit non-zero (proves validation works).
  - Valid args with mocked heavy deps must reach the delegation call
    (proves arg-parsing -> business-logic hand-off works).
"""

from __future__ import annotations

import importlib.metadata as metadata
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


EXPECTED_PROJECT_SCRIPTS = {
    "violawake-train": "violawake_sdk.tools.train:main",
    "violawake-eval": "violawake_sdk.tools.evaluate:main",
    "violawake-collect": "violawake_sdk.tools.collect_samples:main",
    "violawake-download": "violawake_sdk.tools.download_model:main",
    "violawake-download-corpus": "violawake_sdk.tools.download_corpus:main",
    "violawake-expand-corpus": "violawake_sdk.tools.expand_corpus:main",
    "violawake-streaming-eval": "violawake_sdk.tools.streaming_eval:main",
    "violawake-test-confusables": "violawake_sdk.tools.test_confusables:main",
    "violawake-contamination-check": "violawake_sdk.tools.contamination_check:main",
    "violawake-generate": "violawake_sdk.tools.generate_samples:main",
}


INSTALLED_SCRIPT_HELP_MARKERS = {
    "violawake-train": ("violawake-train", "--word", "--positives", "--output"),
    "violawake-eval": ("violawake-eval", "--model", "--test-dir"),
    "violawake-collect": ("violawake-collect", "--word", "--output"),
    "violawake-download": ("violawake-download", "--model", "--list"),
    "violawake-download-corpus": ("violawake-download-corpus", "--target-dir"),
    "violawake-expand-corpus": ("violawake-expand-corpus", "--corpus", "--list"),
    "violawake-streaming-eval": ("violawake-streaming-eval", "--audio", "--audio-dir"),
    "violawake-test-confusables": ("violawake-test-confusables", "--wake-word"),
    "violawake-contamination-check": ("violawake-contamination-check", "--train", "--eval"),
    "violawake-generate": ("violawake-generate", "--word", "--output"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cli(module: str, args: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a CLI module via ``python -m`` and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _installed_script_path(script_name: str) -> Path:
    """Return the installed console script path for the current interpreter."""
    scripts_dir = Path(sys.executable).resolve().parent
    suffixes = (".exe", ".cmd", ".bat", "") if os.name == "nt" else ("",)
    for suffix in suffixes:
        candidate = scripts_dir / f"{script_name}{suffix}"
        if candidate.exists():
            return candidate
    return scripts_dir / script_name


class TestInstalledProjectScripts:
    """Smoke the installed console scripts, not just ``python -m`` modules."""

    def test_distribution_metadata_exposes_all_published_cli_scripts(self) -> None:
        dist = metadata.distribution("violawake")
        actual = {
            entry_point.name: entry_point.value
            for entry_point in dist.entry_points
            if entry_point.group == "console_scripts"
        }

        missing = sorted(set(EXPECTED_PROJECT_SCRIPTS) - set(actual))
        wrong_targets = {
            name: actual.get(name)
            for name, expected in EXPECTED_PROJECT_SCRIPTS.items()
            if actual.get(name) != expected
        }

        assert missing == []
        assert wrong_targets == {}

    @pytest.mark.parametrize(
        ("script_name", "markers"),
        sorted(INSTALLED_SCRIPT_HELP_MARKERS.items()),
    )
    def test_installed_script_help_exits_zero(
        self,
        script_name: str,
        markers: tuple[str, ...],
    ) -> None:
        script_path = _installed_script_path(script_name)
        assert script_path.exists(), f"missing installed script: {script_path}"

        result = subprocess.run(
            [str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert result.returncode == 0, result.stderr
        for marker in markers:
            assert marker in result.stdout


# ===================================================================
# violawake-train  (tools.train:main  &  cli.train:main)
# ===================================================================

class TestTrainCLI:
    """Tests for the violawake-train entry point."""

    def test_help_exits_zero(self) -> None:
        result = _run_cli("violawake_sdk.tools.train", ["--help"])
        assert result.returncode == 0
        assert "violawake-train" in result.stdout
        assert "--word" in result.stdout
        assert "--positives" in result.stdout
        assert "--output" in result.stdout
        assert "--architecture" not in result.stdout

    def test_missing_required_args_exits_nonzero(self) -> None:
        result = _run_cli("violawake_sdk.tools.train", [])
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_missing_positives_dir_exits_1(self, tmp_path: Path) -> None:
        """--positives pointing to a non-existent dir must fail gracefully."""
        result = _run_cli("violawake_sdk.tools.train", [
            "--word", "test",
            "--positives", str(tmp_path / "nonexistent"),
            "--output", str(tmp_path / "out.onnx"),
        ])
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_legacy_architecture_flag_is_rejected(self, tmp_path: Path) -> None:
        """Legacy MLP flags should be rejected by the temporal-only CLI."""
        pos_dir = tmp_path / "positives"
        pos_dir.mkdir()
        result = _run_cli("violawake_sdk.tools.train", [
            "--word", "test",
            "--positives", str(pos_dir),
            "--output", str(tmp_path / "out.onnx"),
            "--architecture", "mlp",
        ])
        assert result.returncode != 0
        assert "unrecognized arguments" in result.stderr.lower()

    def test_legacy_mlp_helper_raises_runtime_error(self, tmp_path: Path) -> None:
        """The old helper is kept only to fail loudly."""
        from violawake_sdk.tools.train import _train_mlp_on_oww

        with pytest.raises(RuntimeError, match="Legacy MLP training has been removed"):
            _train_mlp_on_oww(tmp_path, tmp_path / "out.onnx")

    def test_cli_wrapper_help(self) -> None:
        """The cli.train wrapper should also accept --help."""
        result = _run_cli("violawake_sdk.cli.train", ["--help"])
        assert result.returncode == 0
        assert "violawake-train" in result.stdout

    # ---------------------------------------------------------------
    # cli.train:main() — argument parsing and delegation tests
    # ---------------------------------------------------------------

    def test_cli_train_help_exits_zero(self) -> None:
        """cli.train --help must exit 0."""
        with mock.patch("sys.argv", ["violawake-train", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code == 0

    def test_cli_train_missing_required_args_exits_nonzero(self, tmp_path: Path) -> None:
        """cli.train mirrors the production trainer's required args."""
        with mock.patch("sys.argv", ["violawake-train", "--output", str(tmp_path / "out.onnx")]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code != 0

    def test_cli_train_missing_output_exits_nonzero(self, tmp_path: Path) -> None:
        """cli.train without --output must exit non-zero."""
        pos_dir = tmp_path / "positives"
        pos_dir.mkdir()
        with mock.patch("sys.argv", ["violawake-train", "--word", "test", "--positives", str(pos_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code != 0

    def test_cli_train_nonexistent_positive_dir_exits_1(self, tmp_path: Path) -> None:
        """cli.train with a nonexistent --positives dir must exit 1."""
        with mock.patch("sys.argv", ["violawake-train",
                                      "--word", "test",
                                      "--positives", str(tmp_path / "nope"),
                                      "--output", str(tmp_path / "out.onnx")]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code == 1

    def test_cli_train_delegates_to_tools_main(self) -> None:
        """cli.train is now a thin alias for the temporal trainer."""
        with mock.patch("violawake_sdk.tools.train.main") as mock_main:
            from violawake_sdk.cli.train import main

            main()
            mock_main.assert_called_once_with()

    def test_negatives_dir_not_found_exits_1(self, tmp_path: Path) -> None:
        pos_dir = tmp_path / "positives"
        pos_dir.mkdir()
        test_args = [
            "--word", "test",
            "--positives", str(pos_dir),
            "--output", str(tmp_path / "out.onnx"),
            "--negatives", str(tmp_path / "nonexistent_neg"),
        ]
        with mock.patch("sys.argv", ["violawake-train", *test_args]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.tools.train import main
                main()
            assert exc_info.value.code == 1


# ===================================================================
# violawake-eval  (tools.evaluate:main  &  cli.evaluate:main)
# ===================================================================

class TestEvalCLI:
    """Tests for the violawake-eval entry point."""

    def test_help_exits_zero(self) -> None:
        result = _run_cli("violawake_sdk.tools.evaluate", ["--help"])
        assert result.returncode == 0
        assert "violawake-eval" in result.stdout
        assert "--model" in result.stdout
        assert "--test-dir" in result.stdout

    def test_missing_required_args_exits_nonzero(self) -> None:
        result = _run_cli("violawake_sdk.tools.evaluate", [])
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_model_not_found_exits_1(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        test_args = [
            "--model", str(tmp_path / "nonexistent.onnx"),
            "--test-dir", str(test_dir),
        ]
        with mock.patch("sys.argv", ["violawake-eval", *test_args]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.tools.evaluate import main
                main()
            assert exc_info.value.code == 1

    def test_test_dir_not_found_exits_1(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        test_args = [
            "--model", str(model_file),
            "--test-dir", str(tmp_path / "nonexistent_dir"),
        ]
        with mock.patch("sys.argv", ["violawake-eval", *test_args]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.tools.evaluate import main
                main()
            assert exc_info.value.code == 1

    def test_valid_args_reach_evaluate_function(self, tmp_path: Path) -> None:
        """With valid paths, CLI should call evaluate_onnx_model."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        test_args = [
            "--model", str(model_file),
            "--test-dir", str(test_dir),
            "--threshold", "0.6",
        ]
        with mock.patch("sys.argv", ["violawake-eval", *test_args]):
            with mock.patch(
                "violawake_sdk.tools.evaluate.evaluate_onnx_model",
                create=True,
            ) as mock_eval:
                mock_eval.return_value = {
                    "architecture": "mlp_on_oww",
                    "d_prime": 15.0,
                    "far_per_hour": 0.1,
                    "frr": 0.02,
                    "roc_auc": 0.99,
                    "n_positives": 50,
                    "n_negatives": 200,
                    "optimal_threshold": 0.5,
                    "optimal_far": 0.001,
                    "optimal_frr": 0.01,
                    "eer_approx": 0.007,
                    "confusion_matrix": {
                        "tp": 49, "fp": 1, "fn": 1, "tn": 199,
                        "precision": 0.98, "recall": 0.98, "f1": 0.98,
                    },
                    "tp_scores": [0.9] * 50,
                    "fp_scores": [0.1] * 200,
                }
                # The import inside main() uses a different path; we need to
                # patch at the point of import.
                with mock.patch.dict("sys.modules", {
                    "violawake_sdk.training.evaluate": mock.MagicMock(
                        evaluate_onnx_model=mock_eval
                    ),
                }):
                    from violawake_sdk.tools.evaluate import main
                    main()
                    mock_eval.assert_called_once()

    def test_cli_wrapper_help(self) -> None:
        """The cli.evaluate wrapper re-exports the same main."""
        result = _run_cli("violawake_sdk.cli.evaluate", ["--help"])
        assert result.returncode == 0
        assert "violawake-eval" in result.stdout


# ===================================================================
# violawake-download  (tools.download_model:main  &  cli.download:main)
# ===================================================================

class TestDownloadCLI:
    """Tests for the violawake-download entry point."""

    def test_help_exits_zero(self) -> None:
        result = _run_cli("violawake_sdk.tools.download_model", ["--help"])
        assert result.returncode == 0
        assert "violawake-download" in result.stdout
        assert "--model" in result.stdout
        assert "--list" in result.stdout

    def test_list_models(self) -> None:
        """--list should print available models and exit 0."""
        fake_registry = {
            "test_model": mock.MagicMock(size_bytes=10_000_000, description="A test model"),
        }
        with mock.patch("sys.argv", ["violawake-download", "--list"]):
            with mock.patch.dict("sys.modules", {
                "violawake_sdk.models": mock.MagicMock(
                    MODEL_REGISTRY=fake_registry,
                    download_model=mock.MagicMock(),
                    list_cached_models=mock.MagicMock(return_value=[]),
                ),
            }):
                from violawake_sdk.tools.download_model import main as dl_main
                # Re-import to pick up the patched module
                import importlib
                import violawake_sdk.tools.download_model as dl_mod
                importlib.reload(dl_mod)
                dl_mod.main()
                # If it didn't raise, --list worked

    def test_list_cached_models(self) -> None:
        """--list-cached should print cached models and exit 0."""
        cached = [("test_model", Path("/tmp/test.onnx"), 10.0)]
        with mock.patch("sys.argv", ["violawake-download", "--list-cached"]):
            with mock.patch.dict("sys.modules", {
                "violawake_sdk.models": mock.MagicMock(
                    MODEL_REGISTRY={},
                    download_model=mock.MagicMock(),
                    list_cached_models=mock.MagicMock(return_value=cached),
                ),
            }):
                import importlib
                import violawake_sdk.tools.download_model as dl_mod
                importlib.reload(dl_mod)
                dl_mod.main()

    def test_unknown_model_exits_1(self) -> None:
        """Requesting an unknown model name should exit 1."""
        fake_registry = {
            "real_model": mock.MagicMock(size_bytes=10_000_000, description="A model"),
        }
        with mock.patch("sys.argv", ["violawake-download", "--model", "nonexistent"]):
            with mock.patch.dict("sys.modules", {
                "violawake_sdk.models": mock.MagicMock(
                    MODEL_REGISTRY=fake_registry,
                    download_model=mock.MagicMock(),
                    list_cached_models=mock.MagicMock(return_value=[]),
                ),
            }):
                import importlib
                import violawake_sdk.tools.download_model as dl_mod
                importlib.reload(dl_mod)
                with pytest.raises(SystemExit) as exc_info:
                    dl_mod.main()
                assert exc_info.value.code == 1

    def test_download_falls_back_to_builtin_downloader_without_download_extra(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Core installs should still support the documented model download command."""
        from violawake_sdk.models import ModelSpec
        import violawake_sdk.models as models
        import violawake_sdk.tools.download_model as dl_mod

        fake_spec = ModelSpec(
            name="temporal_cnn",
            url="https://example.com/temporal_cnn.onnx",
            sha256="a" * 64,
            size_bytes=4,
            description="test model",
        )
        downloaded_path = tmp_path / "temporal_cnn.onnx"

        def missing_download_extra(*args: object, **kwargs: object) -> Path:
            raise ImportError("requests is required for model downloading")

        def builtin_download(model_name: str, spec: ModelSpec) -> Path:
            downloaded_path.write_bytes(b"fake")
            return downloaded_path

        monkeypatch.setattr(models, "MODEL_REGISTRY", {"temporal_cnn": fake_spec})
        monkeypatch.setattr(models, "download_model", missing_download_extra)
        monkeypatch.setattr(models, "_auto_download_model", mock.Mock(side_effect=builtin_download))
        monkeypatch.setattr(models, "get_model_dir", lambda: tmp_path)
        monkeypatch.setattr(sys, "argv", ["violawake-download", "--model", "temporal_cnn"])

        dl_mod.main()

        models._auto_download_model.assert_called_once_with("temporal_cnn", fake_spec)
        assert f"Done. Models cached to {tmp_path}" in capsys.readouterr().out

    def test_cli_wrapper_help(self) -> None:
        """The cli.download wrapper re-exports the same main."""
        result = _run_cli("violawake_sdk.cli.download", ["--help"])
        assert result.returncode == 0
        assert "violawake-download" in result.stdout


# ===================================================================
# violawake-download-corpus  (tools.download_corpus:main)
# ===================================================================

class TestDownloadCorpusCLI:
    """Tests for the violawake-download-corpus entry point."""

    def test_help_exits_zero(self) -> None:
        result = _run_cli("violawake_sdk.tools.download_corpus", ["--help"])
        assert result.returncode == 0
        assert "violawake-download-corpus" in result.stdout
        assert "--target-dir" in result.stdout


# ===================================================================
# violawake-collect  (tools.collect_samples:main)
# ===================================================================

class TestCollectCLI:
    """Tests for the violawake-collect entry point."""

    def test_help_exits_zero(self) -> None:
        result = _run_cli("violawake_sdk.tools.collect_samples", ["--help"])
        assert result.returncode == 0
        assert "violawake-collect" in result.stdout
        assert "--word" in result.stdout
        assert "--output" in result.stdout

    def test_missing_required_args_exits_nonzero(self) -> None:
        result = _run_cli("violawake_sdk.tools.collect_samples", [])
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_argument_parsing(self, tmp_path: Path) -> None:
        """Verify all arguments are parsed correctly."""
        out_dir = tmp_path / "samples"
        test_args = [
            "--word", "hello",
            "--output", str(out_dir),
            "--count", "3",
            "--duration", "1.0",
            "--delay", "0.5",
            "--sample-rate", "16000",
        ]
        # The actual recording requires microphone hardware, so mock the
        # audio capture boundary and verify the parsed options drive output.
        with mock.patch("sys.argv", ["violawake-collect", *test_args]):
            with mock.patch(
                "violawake_sdk.tools.collect_samples._record_clip",
                return_value=b"\x00\x00" * 160,
            ):
                from violawake_sdk.tools.collect_samples import main as collect_main
                collect_main()
                # If we reach here, argparse worked and the CLI handled the interrupt

        assert (out_dir / "sample_0001.wav").exists()

    def test_zero_recorded_samples_exits_1(self, tmp_path: Path) -> None:
        """A failed recording session must not look successful to scripts."""
        out_dir = tmp_path / "samples"
        test_args = [
            "--word", "hello",
            "--output", str(out_dir),
            "--count", "1",
            "--duration", "0.01",
            "--delay", "0",
        ]
        with mock.patch("sys.argv", ["violawake-collect", *test_args]):
            with mock.patch("violawake_sdk.tools.collect_samples._record_clip", return_value=None):
                from violawake_sdk.tools.collect_samples import main as collect_main

                with pytest.raises(SystemExit) as exc_info:
                    collect_main()

        assert exc_info.value.code == 1
        assert not list(out_dir.glob("sample_*.wav"))


# ===================================================================
# cli/__init__.py module import
# ===================================================================

class TestCLIPackage:
    """Test that the cli package is importable."""

    def test_cli_package_imports(self) -> None:
        import violawake_sdk.cli
        assert hasattr(violawake_sdk.cli, "__doc__")

    def test_cli_train_imports(self) -> None:
        from violawake_sdk.cli import train
        assert hasattr(train, "main")

    def test_cli_download_imports(self) -> None:
        from violawake_sdk.cli import download
        assert hasattr(download, "main")

    def test_cli_evaluate_imports(self) -> None:
        from violawake_sdk.cli import evaluate
        assert hasattr(evaluate, "main")
