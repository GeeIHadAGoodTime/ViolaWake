"""Unit tests for model download and SHA-256 verification.

Mocks ``requests.get`` and ``tqdm`` so no real HTTP calls or model files
are needed. Tests the download_model() and _verify_sha256() functions.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from violawake_sdk.models import (
    MODEL_REGISTRY,
    _verify_sha256,
    download_model,
    get_model_dir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_model_content() -> bytes:
    """Return deterministic fake model bytes."""
    return b"fake-onnx-model-content-12345"


def _sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response with streaming support."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = {"content-length": str(len(content))}
    resp.iter_content.return_value = [content]
    resp.raise_for_status.side_effect = (
        None if status_code == 200 else _http_error(status_code)
    )
    return resp


def _http_error(status_code: int) -> Exception:
    """Build a requests-like HTTPError."""
    import requests
    resp = MagicMock()
    resp.status_code = status_code
    return requests.HTTPError(response=resp)


# ---------------------------------------------------------------------------
# Successful download + SHA-256 verification
# ---------------------------------------------------------------------------

class TestSuccessfulDownload:
    """Test happy path: download, write, verify SHA-256."""

    def test_download_and_verify(self, tmp_path: Path) -> None:
        content = _fake_model_content()
        correct_sha = _sha256_of(content)

        # Patch the registry entry to use our known SHA
        fake_spec = MODEL_REGISTRY["viola_mlp_oww"]
        patched_spec = fake_spec.__class__(
            name=fake_spec.name,
            url=fake_spec.url,
            sha256=correct_sha,
            size_bytes=len(content),
            description=fake_spec.description,
            version=fake_spec.version,
        )

        mock_resp = _mock_response(content)
        mock_tqdm_cls = MagicMock()
        mock_tqdm_ctx = MagicMock()
        mock_tqdm_cls.return_value.__enter__ = MagicMock(return_value=mock_tqdm_ctx)
        mock_tqdm_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"viola_mlp_oww": patched_spec, "viola": patched_spec}),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": MagicMock(get=MagicMock(return_value=mock_resp))}),
            patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=mock_tqdm_cls)}),
        ):
            path = download_model("viola_mlp_oww", force=True, verify=True)

        assert path.exists()
        assert path.read_bytes() == content

    def test_download_skips_if_cached(self, tmp_path: Path) -> None:
        """download_model returns cached path without re-downloading."""
        content = _fake_model_content()
        correct_sha = _sha256_of(content)

        fake_spec = MODEL_REGISTRY["viola_mlp_oww"]
        patched_spec = fake_spec.__class__(
            name=fake_spec.name,
            url=fake_spec.url,
            sha256=correct_sha,
            size_bytes=len(content),
            description=fake_spec.description,
            version=fake_spec.version,
        )

        # Pre-populate cache
        cached_file = tmp_path / "viola_mlp_oww.onnx"
        cached_file.write_bytes(content)

        mock_requests_mod = MagicMock()
        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"viola_mlp_oww": patched_spec, "viola": patched_spec}),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": mock_requests_mod}),
            patch.dict("sys.modules", {"tqdm": MagicMock()}),
        ):
            path = download_model("viola_mlp_oww", force=False, verify=True)
            # Should NOT have called requests.get
            mock_requests_mod.get.assert_not_called()

        assert path == cached_file


# ---------------------------------------------------------------------------
# HTTP 404
# ---------------------------------------------------------------------------

class TestDownloadFailure:
    """Test failed HTTP download raises appropriate error."""

    def test_http_404(self, tmp_path: Path) -> None:
        import requests as real_requests

        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status.side_effect = real_requests.HTTPError(
            "404 Not Found", response=mock_resp,
        )

        mock_requests_mod = MagicMock()
        mock_requests_mod.get.return_value = mock_resp
        mock_requests_mod.HTTPError = real_requests.HTTPError

        # Use a spec with a real hash so the placeholder guard doesn't block
        spec = MODEL_REGISTRY["temporal_cnn"]

        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"temporal_cnn": spec, "viola": spec}),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": mock_requests_mod}),
            patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=MagicMock())}),
        ):
            with pytest.raises(real_requests.HTTPError):
                download_model("temporal_cnn", force=True)


# ---------------------------------------------------------------------------
# SHA-256 mismatch
# ---------------------------------------------------------------------------

class TestSHA256Mismatch:
    """SHA-256 mismatch should delete the file and raise ValueError."""

    def test_sha256_mismatch_deletes_file(self, tmp_path: Path) -> None:
        bad_content = b"corrupted-model-data"
        model_path = tmp_path / "test_model.onnx"
        model_path.write_bytes(bad_content)

        expected_sha = "a" * 64  # obviously wrong

        with pytest.raises(ValueError, match="SHA-256 verification failed"):
            _verify_sha256(model_path, expected_sha, "test_model")

        # File should be deleted
        assert not model_path.exists()

    def test_correct_sha256_passes(self, tmp_path: Path) -> None:
        content = _fake_model_content()
        model_path = tmp_path / "test_model.onnx"
        model_path.write_bytes(content)
        correct_sha = _sha256_of(content)

        # Should not raise
        _verify_sha256(model_path, correct_sha, "test_model")
        assert model_path.exists()


# ---------------------------------------------------------------------------
# Placeholder hash warning
# ---------------------------------------------------------------------------

class TestPlaceholderHash:
    """Placeholder SHA-256 should warn but not fail."""

    def test_placeholder_emits_warning(self, tmp_path: Path) -> None:
        content = _fake_model_content()
        model_path = tmp_path / "test_model.onnx"
        model_path.write_bytes(content)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _verify_sha256(model_path, "PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT", "test_model")

        # Should have emitted a UserWarning
        assert len(w) >= 1
        assert any("PLACEHOLDER" in str(warning.message) or "SHA-256" in str(warning.message) for warning in w)

        # File should NOT be deleted
        assert model_path.exists()

    def test_placeholder_does_not_raise(self, tmp_path: Path) -> None:
        content = _fake_model_content()
        model_path = tmp_path / "test_model.onnx"
        model_path.write_bytes(content)

        # Should not raise ValueError
        _verify_sha256(model_path, "PLACEHOLDER_something", "test_model")


# ---------------------------------------------------------------------------
# Custom VIOLAWAKE_MODEL_DIR
# ---------------------------------------------------------------------------

class TestCustomModelDir:
    """get_model_dir() respects VIOLAWAKE_MODEL_DIR env var."""

    def test_custom_model_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_dir = tmp_path / "custom_models"
        monkeypatch.setenv("VIOLAWAKE_MODEL_DIR", str(custom_dir))

        result = get_model_dir()
        assert result == custom_dir
        assert result.exists()

    def test_default_model_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VIOLAWAKE_MODEL_DIR", raising=False)
        result = get_model_dir()
        assert "violawake" in str(result).lower()


# ---------------------------------------------------------------------------
# Unknown model name
# ---------------------------------------------------------------------------

class TestUnknownModel:
    """download_model with unknown model name raises KeyError."""

    def test_unknown_model_name(self) -> None:
        from violawake_sdk._exceptions import ModelNotFoundError

        with (
            patch.dict("sys.modules", {"requests": MagicMock()}),
            patch.dict("sys.modules", {"tqdm": MagicMock()}),
        ):
            with pytest.raises(ModelNotFoundError, match="Unknown model"):
                download_model("nonexistent_model_xyz")


class TestPackageManagedModel:
    """Package-managed backbone entries should not be downloaded from the registry."""

    def test_oww_backbone_download_is_rejected(self) -> None:
        with (
            patch.dict("sys.modules", {"requests": MagicMock()}),
            patch.dict("sys.modules", {"tqdm": MagicMock()}),
        ):
            with pytest.raises(ValueError, match="openwakeword"):
                download_model("oww_backbone")


# ---------------------------------------------------------------------------
# Placeholder hash guard in download_model
# ---------------------------------------------------------------------------

class TestPlaceholderDownloadGuard:
    """download_model refuses models with placeholder hashes unless skip_verify=True."""

    def test_placeholder_hash_blocks_download(self) -> None:
        """Models with placeholder hashes should raise RuntimeError by default."""
        with (
            patch.dict("sys.modules", {"requests": MagicMock()}),
            patch.dict("sys.modules", {"tqdm": MagicMock()}),
        ):
            with pytest.raises(RuntimeError, match="placeholder SHA-256"):
                download_model("viola_mlp_oww", force=True)

    def test_placeholder_hash_allowed_with_skip_verify(self, tmp_path: Path) -> None:
        """skip_verify=True should bypass the placeholder guard."""
        content = _fake_model_content()

        mock_resp = _mock_response(content)
        mock_tqdm_cls = MagicMock()
        mock_tqdm_ctx = MagicMock()
        mock_tqdm_cls.return_value.__enter__ = MagicMock(return_value=mock_tqdm_ctx)
        mock_tqdm_cls.return_value.__exit__ = MagicMock(return_value=False)

        # Use a spec with placeholder hash but matching size
        spec = MODEL_REGISTRY["viola_mlp_oww"]
        patched_spec = spec.__class__(
            name=spec.name,
            url=spec.url,
            sha256=spec.sha256,  # keep the placeholder
            size_bytes=len(content),
            description=spec.description,
            version=spec.version,
        )

        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"viola_mlp_oww": patched_spec, "viola": patched_spec}),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": MagicMock(get=MagicMock(return_value=mock_resp))}),
            patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=mock_tqdm_cls)}),
        ):
            # Should NOT raise because skip_verify=True
            path = download_model("viola_mlp_oww", force=True, verify=True, skip_verify=True)
        assert path.exists()

    def test_real_hash_model_not_blocked(self, tmp_path: Path) -> None:
        """Models with real hashes should download without skip_verify."""
        content = _fake_model_content()
        correct_sha = _sha256_of(content)

        mock_resp = _mock_response(content)
        mock_tqdm_cls = MagicMock()
        mock_tqdm_ctx = MagicMock()
        mock_tqdm_cls.return_value.__enter__ = MagicMock(return_value=mock_tqdm_ctx)
        mock_tqdm_cls.return_value.__exit__ = MagicMock(return_value=False)

        spec = MODEL_REGISTRY["temporal_cnn"]
        patched_spec = spec.__class__(
            name=spec.name,
            url=spec.url,
            sha256=correct_sha,
            size_bytes=len(content),
            description=spec.description,
            version=spec.version,
        )

        with (
            patch("violawake_sdk.models.MODEL_REGISTRY", {"temporal_cnn": patched_spec, "viola": patched_spec}),
            patch("violawake_sdk.models.get_model_dir", return_value=tmp_path),
            patch.dict("sys.modules", {"requests": MagicMock(get=MagicMock(return_value=mock_resp))}),
            patch.dict("sys.modules", {"tqdm": MagicMock(tqdm=mock_tqdm_cls)}),
        ):
            path = download_model("temporal_cnn", force=True, verify=True)
        assert path.exists()
