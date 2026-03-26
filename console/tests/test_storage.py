from __future__ import annotations

import sys
from pathlib import Path

import pytest

backend_dir = Path(__file__).resolve().parents[1] / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app import storage


@pytest.fixture(autouse=True)
def _storage_roots(tmp_path, monkeypatch) -> None:
    """Point storage settings at pytest-managed directories."""
    upload_dir = tmp_path / "recordings"
    models_dir = tmp_path / "models"
    upload_dir.mkdir()
    models_dir.mkdir()

    monkeypatch.setattr(storage.settings, "upload_dir", upload_dir)
    monkeypatch.setattr(storage.settings, "models_dir", models_dir)
    monkeypatch.setattr(storage, "_storage", None)


@pytest.fixture
def local_storage() -> storage.LocalStorageBackend:
    return storage.LocalStorageBackend()


def test_build_recording_key() -> None:
    assert storage.build_recording_key(42, "jarvis", "sample.wav") == "recordings/42/jarvis/sample.wav"


def test_build_model_key() -> None:
    assert storage.build_model_key(42, "wakeword.onnx") == "models/42/wakeword.onnx"


def test_build_companion_config_identifier() -> None:
    assert (
        storage.build_companion_config_identifier("models/42/wakeword.onnx")
        == "models/42/wakeword.config.json"
    )


def test_local_storage_upload_download_roundtrip(local_storage: storage.LocalStorageBackend) -> None:
    key = storage.build_recording_key(7, "viola", "clip.wav")
    payload = b"fake wav bytes"

    url = local_storage.upload(key, payload, "audio/wav")

    assert url == "/api/files/recordings/7/viola/clip.wav"
    assert local_storage.download(key) == payload


def test_local_storage_exists_before_and_after_upload(local_storage: storage.LocalStorageBackend) -> None:
    key = storage.build_model_key(7, "model.onnx")

    assert local_storage.exists(key) is False

    local_storage.upload(key, b"model bytes", "application/octet-stream")

    assert local_storage.exists(key) is True


def test_local_storage_delete_removes_file(local_storage: storage.LocalStorageBackend) -> None:
    key = storage.build_model_key(9, "model.onnx")
    local_storage.upload(key, b"model bytes", "application/octet-stream")

    assert local_storage.delete(key) is True
    assert local_storage.exists(key) is False


def test_normalize_storage_key_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="Storage key cannot be empty"):
        storage._normalize_storage_key("")


def test_normalize_storage_key_rejects_path_traversal() -> None:
    with pytest.raises(ValueError, match="Invalid storage key"):
        storage._normalize_storage_key("recordings/7/../clip.wav")


def test_normalize_storage_key_rejects_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Unsupported storage key prefix"):
        storage._normalize_storage_key("artifacts/7/file.bin")


def test_resolve_path_rejects_absolute_paths_outside_managed_dirs(
    local_storage: storage.LocalStorageBackend,
    tmp_path: Path,
) -> None:
    outside_path = tmp_path / "outside.bin"

    with pytest.raises(ValueError, match="outside managed storage roots"):
        local_storage._resolve_path(str(outside_path))


def test_presigned_url_returns_api_files_path(local_storage: storage.LocalStorageBackend) -> None:
    key = storage.build_recording_key(11, "viola", "clip.wav")

    assert local_storage.presigned_url(key) == "/api/files/recordings/11/viola/clip.wav"
