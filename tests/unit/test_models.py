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
    ModelSpec,
    _verify_sha256,
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
        """get_model_path raises FileNotFoundError if model not in cache."""
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="viola_mlp_oww"):
                get_model_path("viola_mlp_oww")

    def test_raises_key_error_for_unknown_model(self) -> None:
        """get_model_path raises KeyError for unregistered model names."""
        with pytest.raises(KeyError, match="nonexistent_model"):
            get_model_path("nonexistent_model")

    def test_error_message_includes_download_hint(self, tmp_path: Path) -> None:
        """FileNotFoundError message should suggest how to fix."""
        with patch("violawake_sdk.models.get_model_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="violawake-download"):
                get_model_path("viola_mlp_oww")


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
