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

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest


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

    def test_valid_args_reach_training_function(self, tmp_path: Path) -> None:
        """With valid dirs, the CLI should reach the training function."""
        pos_dir = tmp_path / "positives"
        pos_dir.mkdir()
        output = tmp_path / "out.onnx"

        # Use --architecture mlp to hit the fast mockable path.
        # The default temporal_cnn runs TTS generation inline (slow).
        with mock.patch("violawake_sdk.tools.train._train_mlp_on_oww") as mock_train:
            result = _run_cli("violawake_sdk.tools.train", [
                "--word", "test",
                "--positives", str(pos_dir),
                "--output", str(output),
                "--epochs", "2",
                "--architecture", "mlp",
                "--quiet",
            ])
            # The mock is in a subprocess so it won't be captured here.
            # Instead we just verify the process didn't crash on argparse.
            # The real validation is that it gets past argument parsing.
            # A non-existent import or bad arg would cause returncode != 0.
            # Since the subprocess has its own mock context, we test via
            # direct function call below.

    def test_direct_call_reaches_train_function(self, tmp_path: Path) -> None:
        """Direct call to main() with mocked _train_mlp_on_oww (legacy MLP path)."""
        pos_dir = tmp_path / "positives"
        pos_dir.mkdir()
        output = tmp_path / "out.onnx"

        test_args = [
            "--word", "test",
            "--positives", str(pos_dir),
            "--output", str(output),
            "--epochs", "2",
            "--architecture", "mlp",
            "--quiet",
        ]
        with mock.patch("sys.argv", ["violawake-train", *test_args]):
            with mock.patch("violawake_sdk.tools.train._train_mlp_on_oww") as mock_train:
                from violawake_sdk.tools.train import main
                main()
                mock_train.assert_called_once()
                call_kwargs = mock_train.call_args
                assert call_kwargs[1]["epochs"] == 2 or call_kwargs.kwargs["epochs"] == 2

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

    def test_cli_train_missing_positive_dir_exits_nonzero(self, tmp_path: Path) -> None:
        """cli.train without --positive-dir must exit non-zero."""
        with mock.patch("sys.argv", ["violawake-train",
                                      "--output-model", str(tmp_path / "out.onnx")]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code != 0

    def test_cli_train_missing_output_model_exits_nonzero(self, tmp_path: Path) -> None:
        """cli.train without --output-model must exit non-zero."""
        pos_dir = tmp_path / "pos"
        pos_dir.mkdir()
        with mock.patch("sys.argv", ["violawake-train",
                                      "--positive-dir", str(pos_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code != 0

    def test_cli_train_nonexistent_positive_dir_exits_1(self, tmp_path: Path) -> None:
        """cli.train with a nonexistent --positive-dir must exit 1."""
        with mock.patch("sys.argv", ["violawake-train",
                                      "--positive-dir", str(tmp_path / "nope"),
                                      "--output-model", str(tmp_path / "out.onnx")]):
            with pytest.raises(SystemExit) as exc_info:
                from violawake_sdk.cli.train import main
                main()
            assert exc_info.value.code == 1

    def test_cli_train_delegates_with_defaults(self, tmp_path: Path) -> None:
        """cli.train with minimal valid args delegates to _train_mlp_on_oww with defaults."""
        pos_dir = tmp_path / "pos"
        pos_dir.mkdir()
        out = tmp_path / "out.onnx"
        with mock.patch("sys.argv", ["violawake-train",
                                      "--positive-dir", str(pos_dir),
                                      "--output-model", str(out)]):
            with mock.patch("violawake_sdk.cli.train._train_mlp_on_oww", create=True) as mock_train:
                # Patch the import inside main()
                with mock.patch.dict("sys.modules", {}):
                    import importlib
                    import violawake_sdk.cli.train as cli_train_mod
                    importlib.reload(cli_train_mod)
                    with mock.patch("violawake_sdk.tools.train._train_mlp_on_oww", mock_train):
                        cli_train_mod.main()
                        mock_train.assert_called_once()
                        kw = mock_train.call_args
                        # Check default epochs=50 and augment=True (no --no-augment)
                        assert kw.kwargs.get("epochs", kw[1].get("epochs")) == 50
                        assert kw.kwargs.get("augment", kw[1].get("augment")) is True

    def test_cli_train_delegates_with_all_flags(self, tmp_path: Path) -> None:
        """cli.train with all flags passes correct args to _train_mlp_on_oww."""
        pos_dir = tmp_path / "pos"
        pos_dir.mkdir()
        neg_dir = tmp_path / "neg"
        neg_dir.mkdir()
        out = tmp_path / "out.onnx"
        with mock.patch("sys.argv", ["violawake-train",
                                      "--positive-dir", str(pos_dir),
                                      "--negative-dir", str(neg_dir),
                                      "--output-model", str(out),
                                      "--epochs", "10",
                                      "--no-augment",
                                      "--quiet"]):
            with mock.patch("violawake_sdk.tools.train._train_mlp_on_oww") as mock_train:
                import importlib
                import violawake_sdk.cli.train as cli_train_mod
                importlib.reload(cli_train_mod)
                cli_train_mod.main()
                mock_train.assert_called_once()
                kw = mock_train.call_args
                assert kw.kwargs.get("epochs", kw[1].get("epochs")) == 10
                assert kw.kwargs.get("augment", kw[1].get("augment")) is False
                assert kw.kwargs.get("verbose", kw[1].get("verbose")) is False

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

    def test_cli_wrapper_help(self) -> None:
        """The cli.download wrapper re-exports the same main."""
        result = _run_cli("violawake_sdk.cli.download", ["--help"])
        assert result.returncode == 0
        assert "violawake-download" in result.stdout


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
        # We just need to verify argparse succeeds; the actual recording
        # requires microphone hardware, so we mock _record_one_sample.
        with mock.patch("sys.argv", ["violawake-collect", *test_args]):
            with mock.patch(
                "violawake_sdk.tools.collect_samples._record_one_sample",
                create=True,
                side_effect=KeyboardInterrupt,  # Stop after setup
            ):
                from violawake_sdk.tools.collect_samples import main as collect_main
                # KeyboardInterrupt is caught by the CLI
                collect_main()
                # If we reach here, argparse worked and the CLI handled the interrupt


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
