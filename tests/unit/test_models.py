"""Unit tests for the model registry and cache utilities.

No actual downloads — all HTTP calls are mocked.
No actual model files — only path resolution and validation logic is tested.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk._exceptions import ModelNotFoundError
from violawake_sdk.models import (
    MODEL_REGISTRY,
    SIZE_TOLERANCE_FRACTION,
    ModelSpec,
    _auto_download_model,
    _format_size,
    _is_auto_download_disabled,
    _verify_sha256,
    check_registry_integrity,
    download_model,
    get_model_dir,
    get_model_path,
    list_cached_models,
)


class TestModelRegistry:
    """Test that MODEL_REGISTRY is well-formed."""

    def test_registry_not_empty(self) -> None:
        assert len(MODEL_REGISTRY) > 0

    def test_all_models_have_required_fields(self) -> None:
        required_attrs = ["name", "url", "sha256", "size_bytes", "description"]
        for model_name, spec in MODEL_REGISTRY.items():
            for attr in required_attrs:
                assert hasattr(spec, attr), f"Model '{model_name}' missing '{attr}'"

    def test_all_urls_are_https(self) -> None:
        for model_name, spec in MODEL_REGISTRY.items():
            assert spec.url.startswith("https://"), (
                f"Model '{model_name}' URL should be HTTPS: {spec.url}"
            )

    def test_viola_mlp_oww_is_in_registry(self) -> None:
        assert "viola_mlp_oww" in MODEL_REGISTRY

    def test_oww_backbone_is_in_registry(self) -> None:
        assert "oww_backbone" in MODEL_REGISTRY

    def test_temporal_models_are_in_registry(self) -> None:
        assert "temporal_cnn" in MODEL_REGISTRY
        assert "temporal_convgru" in MODEL_REGISTRY

    def test_deprecated_r3_model_is_in_registry(self) -> None:
        assert "r3_10x_s42" in MODEL_REGISTRY

    def test_model_sizes_are_positive(self) -> None:
        for model_name, spec in MODEL_REGISTRY.items():
            assert spec.size_bytes > 0, f"Model '{model_name}' has invalid size {spec.size_bytes}"


class TestGetModelDir:
    """Test get_model_dir() path resolution."""

    def test_default_dir_is_under_home(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            model_dir = get_model_dir()
            assert str(Path.home()) in str(model_dir)

    def test_env_var_override(self, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom_models"
        with patch.dict("os.environ", {"VIOLAWAKE_MODEL_DIR": str(custom_dir)}):
            model_dir = get_model_dir()
            assert model_dir == custom_dir
            assert model_dir.exists()

    def test_creates_directory_if_missing(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "doesnt_exist" / "nested"
        with patch.dict("os.environ", {"VIOLAWAKE_MODEL_DIR": str(new_dir)}):
            result = get_model_dir()
            assert result.exists()


class TestGetModelPath:
    """Test get_model_path() cache lookup."""

    def test_returns_path_when_cached(self, tmp_path: Path) -> None:
        """get_model_path returns the correct path when model file exists."""
        # Create a fake model file in the cache
        model_file = tmp_path / "viola_mlp_oww.onnx"
        model_file.write_bytes(b"fake model")

        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            path = get_model_path("viola_mlp_oww")
            assert path == model_file

    def test_raises_file_not_found_when_not_cached(self, tmp_path: Path) -> None:
        """get_model_path raises FileNotFoundError if model not in cache and auto-download disabled."""
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="viola_mlp_oww"):
                get_model_path("viola_mlp_oww", auto_download=False)

    def test_raises_key_error_for_unknown_model(self) -> None:
        """get_model_path raises KeyError for unregistered model names."""
        with pytest.raises(KeyError, match="nonexistent_model"):
            get_model_path("nonexistent_model")

    def test_error_message_includes_download_hint(self, tmp_path: Path) -> None:
        """FileNotFoundError message should suggest how to fix."""
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="violawake-download"):
                get_model_path("viola_mlp_oww", auto_download=False)


class TestVerifySHA256:
    """Test SHA-256 file verification."""

    def test_correct_hash_passes(self, tmp_path: Path) -> None:
        content = b"test model content"
        file_path = tmp_path / "test.onnx"
        file_path.write_bytes(content)
        correct_hash = hashlib.sha256(content).hexdigest()
        # Should not raise
        _verify_sha256(file_path, correct_hash, "test_model")

    def test_wrong_hash_raises_and_deletes(self, tmp_path: Path) -> None:
        content = b"test model content"
        file_path = tmp_path / "test.onnx"
        file_path.write_bytes(content)
        wrong_hash = "a" * 64  # wrong hash

        with pytest.raises(ValueError, match="SHA-256"):
            _verify_sha256(file_path, wrong_hash, "test_model")

        # File should be deleted after failed verification
        assert not file_path.exists()

    def test_placeholder_hash_skips_verification(self, tmp_path: Path) -> None:
        """Dev builds with PLACEHOLDER hashes should skip verification."""
        file_path = tmp_path / "test.onnx"
        file_path.write_bytes(b"any content")
        # Should not raise
        _verify_sha256(file_path, "PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT", "test_model")
        # File should still exist
        assert file_path.exists()


class TestListCachedModels:
    """Test list_cached_models() utility."""

    def test_empty_cache(self, tmp_path: Path) -> None:
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            cached = list_cached_models()
            assert cached == []

    def test_lists_cached_models(self, tmp_path: Path) -> None:
        # Create some fake cached models
        (tmp_path / "viola_mlp_oww.onnx").write_bytes(b"x" * 1000)

        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            cached = list_cached_models()
            names = [m[0] for m in cached]
            assert "viola_mlp_oww" in names

    def test_returns_size_in_mb(self, tmp_path: Path) -> None:
        # Write ~1MB file
        (tmp_path / "viola_mlp_oww.onnx").write_bytes(b"x" * 1_000_000)

        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            cached = list_cached_models()
            for name, path, size_mb in cached:
                if name == "viola_mlp_oww":
                    assert abs(size_mb - 1.0) < 0.1


class TestCheckRegistryIntegrity:
    """Test check_registry_integrity() catches placeholder hashes."""

    def test_raises_on_placeholder_hashes(self) -> None:
        """The real registry has placeholders, so this must raise."""
        with pytest.raises(RuntimeError, match="placeholder SHA-256"):
            check_registry_integrity()

    def test_lists_all_placeholder_models(self) -> None:
        """Error message should name every offending model."""
        with pytest.raises(RuntimeError) as exc_info:
            check_registry_integrity()
        msg = str(exc_info.value)
        # These deprecated models have placeholder hashes in the registry.
        # oww_backbone, kokoro_v1_0, kokoro_voices_v1_0 now have real hashes.
        for name in ("viola_mlp_oww", "viola_cnn_v4"):
            assert name in msg, f"Expected '{name}' in error but got: {msg}"

    def test_passes_when_all_hashes_are_real(self) -> None:
        """Registry with only real hashes should pass."""
        clean_registry = {
            "model_a": ModelSpec(
                name="model_a",
                url="https://example.com/a.onnx",
                sha256="a" * 64,
                size_bytes=1000,
                description="test",
            ),
        }
        with patch("violawake_sdk.models.MODEL_REGISTRY", clean_registry):
            # Should not raise
            check_registry_integrity()


class TestSizeValidationTolerance:
    """Test that SIZE_TOLERANCE_FRACTION is correctly set and enforced."""

    def test_tolerance_is_five_percent(self) -> None:
        assert SIZE_TOLERANCE_FRACTION == 0.05

    def test_file_within_tolerance_passes(self, tmp_path: Path) -> None:
        """A file within 5% of declared size should pass size validation."""
        declared_size = 100_000
        # 3% over — within 5% tolerance
        actual_content = b"x" * 103_000
        correct_sha = hashlib.sha256(actual_content).hexdigest()

        spec = ModelSpec(
            name="test_model",
            url="https://example.com/test.onnx",
            sha256=correct_sha,
            size_bytes=declared_size,
            description="test",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": str(len(actual_content))}
        mock_resp.iter_content.return_value = [actual_content]
        mock_resp.raise_for_status.return_value = None

        mock_tqdm_cls = MagicMock()
        mock_tqdm_ctx = MagicMock()
        mock_tqdm_cls.return_value.__enter__ = MagicMock(return_value=mock_tqdm_ctx)
        mock_tqdm_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"test_model": spec}),
            patch("violawake_sdk.models._PACKAGE_MANAGED_MODELS", set()),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": MagicMock(get=MagicMock(return_value=mock_resp))}),
            patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=mock_tqdm_cls)}),
        ):
            path = download_model("test_model", force=True, verify=True)
        assert path.exists()

    def test_file_outside_tolerance_raises(self, tmp_path: Path) -> None:
        """A file >5% different from declared size should fail."""
        declared_size = 100_000
        # 10% over — outside 5% tolerance
        actual_content = b"x" * 110_000
        correct_sha = hashlib.sha256(actual_content).hexdigest()

        spec = ModelSpec(
            name="test_model",
            url="https://example.com/test.onnx",
            sha256=correct_sha,
            size_bytes=declared_size,
            description="test",
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-length": str(len(actual_content))}
        mock_resp.iter_content.return_value = [actual_content]
        mock_resp.raise_for_status.return_value = None

        mock_tqdm_cls = MagicMock()
        mock_tqdm_ctx = MagicMock()
        mock_tqdm_cls.return_value.__enter__ = MagicMock(return_value=mock_tqdm_ctx)
        mock_tqdm_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"test_model": spec}),
            patch("violawake_sdk.models._PACKAGE_MANAGED_MODELS", set()),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": MagicMock(get=MagicMock(return_value=mock_resp))}),
            patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=mock_tqdm_cls)}),
        ):
            with pytest.raises(ValueError, match="Size validation failed"):
                download_model("test_model", force=True, verify=True)


# ---------------------------------------------------------------------------
# Auto-download helpers
# ---------------------------------------------------------------------------


class TestFormatSize:
    """Test _format_size() human-readable output."""

    def test_bytes(self) -> None:
        assert _format_size(500) == "500 B"

    def test_kilobytes(self) -> None:
        assert _format_size(2_500) == "2.5 KB"

    def test_megabytes(self) -> None:
        assert _format_size(2_100_000) == "2.1 MB"

    def test_gigabytes(self) -> None:
        assert _format_size(1_500_000_000) == "1.5 GB"


class TestIsAutoDownloadDisabled:
    """Test _is_auto_download_disabled() env var checks."""

    def test_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", raising=False)
        assert _is_auto_download_disabled() is False

    def test_set_to_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", "1")
        assert _is_auto_download_disabled() is True

    def test_set_to_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", "true")
        assert _is_auto_download_disabled() is True

    def test_set_to_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", "yes")
        assert _is_auto_download_disabled() is True

    def test_set_to_0(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", "0")
        assert _is_auto_download_disabled() is False

    def test_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", "")
        assert _is_auto_download_disabled() is False


class TestAutoDownloadInGetModelPath:
    """Test that get_model_path triggers auto-download when model is missing."""

    def test_auto_download_triggered_when_not_cached(self, tmp_path: Path) -> None:
        """get_model_path auto-downloads when model is missing and auto_download=True."""
        fake_model = tmp_path / "temporal_cnn.onnx"

        with (
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch("violawake_sdk.models._auto_download_model", return_value=fake_model) as mock_dl,
        ):
            result = get_model_path("temporal_cnn", auto_download=True)
            mock_dl.assert_called_once()
            assert result == fake_model

    def test_auto_download_disabled_via_param(self, tmp_path: Path) -> None:
        """get_model_path raises FileNotFoundError when auto_download=False."""
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError):
                get_model_path("temporal_cnn", auto_download=False)

    def test_auto_download_disabled_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """get_model_path raises FileNotFoundError when VIOLAWAKE_NO_AUTO_DOWNLOAD=1."""
        monkeypatch.setenv("VIOLAWAKE_NO_AUTO_DOWNLOAD", "1")
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError):
                get_model_path("temporal_cnn", auto_download=True)

    def test_auto_download_not_triggered_when_cached(self, tmp_path: Path) -> None:
        """get_model_path returns cached path without auto-downloading."""
        cached = tmp_path / "temporal_cnn.onnx"
        cached.write_bytes(b"cached model")

        with (
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch("violawake_sdk.models._auto_download_model") as mock_dl,
        ):
            result = get_model_path("temporal_cnn")
            mock_dl.assert_not_called()
            assert result == cached


class TestAutoDownloadModel:
    """Test _auto_download_model() directly."""

    def test_downloads_and_verifies(self, tmp_path: Path) -> None:
        """Successful auto-download with real hash verification."""
        import hashlib
        import io

        content = b"fake-model-content-for-auto-download"
        correct_sha = hashlib.sha256(content).hexdigest()

        spec = ModelSpec(
            name="test_model",
            url="https://example.com/test_model.onnx",
            sha256=correct_sha,
            size_bytes=len(content),
            description="test",
        )

        mock_response = MagicMock()
        mock_response.read.side_effect = [content, b""]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            path = _auto_download_model("test_model", spec)

        assert path.exists()
        assert path.read_bytes() == content
        assert path.name == "test_model.onnx"

    def test_placeholder_hash_downloads_with_warning(self, tmp_path: Path, capfd) -> None:
        """Placeholder hash models download but print a warning."""
        content = b"placeholder-model"

        spec = ModelSpec(
            name="placeholder_model",
            url="https://example.com/placeholder_model.onnx",
            sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
            size_bytes=len(content),
            description="test placeholder",
        )

        mock_response = MagicMock()
        mock_response.read.side_effect = [content, b""]
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        with (
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch("urllib.request.urlopen", return_value=mock_response),
        ):
            path = _auto_download_model("placeholder_model", spec)

        assert path.exists()
        stderr_output = capfd.readouterr().err
        assert "WARNING" in stderr_output
        assert "placeholder" in stderr_output.lower()

    def test_network_error_raises_runtime_error(self, tmp_path: Path) -> None:
        """Network failure raises RuntimeError with helpful message."""
        import urllib.error

        spec = ModelSpec(
            name="fail_model",
            url="https://example.com/fail.onnx",
            sha256="a" * 64,
            size_bytes=1000,
            description="test failure",
        )

        with (
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")),
        ):
            with pytest.raises(RuntimeError, match="Auto-download.*failed"):
                _auto_download_model("fail_model", spec)
