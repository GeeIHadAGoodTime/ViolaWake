"""Storage backends for recordings and trained models."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import quote

from app.config import settings

logger = logging.getLogger("violawake.storage")

_RECORDINGS_PREFIX = "recordings"
_MODELS_PREFIX = "models"
_VALID_PREFIXES = {_RECORDINGS_PREFIX, _MODELS_PREFIX}
_storage: StorageBackend | None = None


class StorageBackend(Protocol):
    """Protocol implemented by all storage backends."""

    def upload(self, key: str, data: bytes, content_type: str) -> str:
        """Store bytes at the given key and return an access URL."""

    def download(self, key: str) -> bytes:
        """Download bytes for the given key."""

    def delete(self, key: str) -> bool:
        """Delete the object at the given key if it exists."""

    def exists(self, key: str) -> bool:
        """Return whether the key exists."""

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Return a time-limited or internal-access URL for the key."""


def build_recording_key(user_id: int, wake_word: str, filename: str) -> str:
    """Build a storage key for a user recording."""
    return PurePosixPath(_RECORDINGS_PREFIX, str(user_id), wake_word, filename).as_posix()


def build_model_key(user_id: int, filename: str) -> str:
    """Build a storage key for a trained model artifact."""
    return PurePosixPath(_MODELS_PREFIX, str(user_id), filename).as_posix()


def build_companion_config_identifier(identifier: str) -> str:
    """Return the companion config identifier for a model key or legacy path."""
    legacy_path = _as_legacy_path(identifier)
    if legacy_path is not None:
        return str(legacy_path.with_suffix(".config.json"))

    key = _normalize_storage_key(identifier)
    return PurePosixPath(key).with_suffix(".config.json").as_posix()


def get_storage() -> StorageBackend:
    """Return the configured storage backend singleton."""
    global _storage

    if _storage is None:
        if _r2_is_configured():
            logger.info("Using Cloudflare R2 bucket %s at %s", settings.r2_bucket, settings.r2_endpoint)
            _storage = R2StorageBackend()
        else:
            logger.info(
                "Using local storage under %s and %s",
                settings.upload_dir,
                settings.models_dir,
            )
            _storage = LocalStorageBackend()
    return _storage


class LocalStorageBackend:
    """Filesystem-backed storage for development."""

    def upload(self, key: str, data: bytes, content_type: str) -> str:
        """Write a file to the local filesystem."""
        del content_type

        path = self._resolve_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        logger.debug("Stored local object %s at %s", key, path)
        return self.presigned_url(key)

    def download(self, key: str) -> bytes:
        """Read a file from the local filesystem."""
        path = self._resolve_path(key)
        return path.read_bytes()

    def delete(self, key: str) -> bool:
        """Delete a local file if it exists."""
        path = self._resolve_path(key)
        if not path.exists():
            return False

        path.unlink()
        logger.debug("Deleted local object %s at %s", key, path)
        return True

    def exists(self, key: str) -> bool:
        """Return whether the local file exists."""
        path = self._resolve_path(key)
        return path.exists()

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Return the internal file route for a local storage key."""
        del expires

        route_key = self._key_for_identifier(key)
        return f"/api/files/{quote(route_key, safe='/')}"

    def _resolve_path(self, identifier: str) -> Path:
        legacy_path = _as_legacy_path(identifier)
        if legacy_path is not None:
            return _validate_legacy_path(legacy_path)

        key = _normalize_storage_key(identifier)
        parts = PurePosixPath(key).parts
        prefix = parts[0]
        relative_parts = parts[1:]

        if prefix == _RECORDINGS_PREFIX:
            if len(relative_parts) < 3:
                raise ValueError(f"Invalid recording key: {identifier}")
            return settings.upload_dir.joinpath(*relative_parts)

        if prefix == _MODELS_PREFIX:
            if len(relative_parts) < 2:
                raise ValueError(f"Invalid model key: {identifier}")
            return settings.models_dir.joinpath(*relative_parts)

        raise ValueError(f"Unsupported storage key prefix: {prefix}")

    def _key_for_identifier(self, identifier: str) -> str:
        legacy_path = _as_legacy_path(identifier)
        if legacy_path is not None:
            return _key_from_local_path(_validate_legacy_path(legacy_path))
        return _normalize_storage_key(identifier)


class R2StorageBackend:
    """Cloudflare R2 storage using the S3-compatible API."""

    def __init__(self) -> None:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as exc:
            raise RuntimeError("boto3 is required when Cloudflare R2 storage is configured") from exc

        self.bucket = settings.r2_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=BotoConfig(signature_version="s3v4"),
        )

    def upload(self, key: str, data: bytes, content_type: str) -> str:
        """Upload an object to R2."""
        normalized_key = _normalize_storage_key(key)
        self.client.put_object(
            Bucket=self.bucket,
            Key=normalized_key,
            Body=data,
            ContentType=content_type,
        )
        logger.debug("Uploaded R2 object %s to bucket %s", normalized_key, self.bucket)
        return self.presigned_url(normalized_key)

    def download(self, key: str) -> bytes:
        """Download an object from R2."""
        normalized_key = _normalize_storage_key(key)
        response = self.client.get_object(Bucket=self.bucket, Key=normalized_key)
        body = response["Body"]
        try:
            return body.read()
        finally:
            body.close()

    def delete(self, key: str) -> bool:
        """Delete an object from R2 if it exists."""
        normalized_key = _normalize_storage_key(key)
        existed = self.exists(normalized_key)
        self.client.delete_object(Bucket=self.bucket, Key=normalized_key)
        logger.debug("Deleted R2 object %s from bucket %s", normalized_key, self.bucket)
        return existed

    def exists(self, key: str) -> bool:
        """Return whether an object exists in R2."""
        from botocore.exceptions import ClientError

        normalized_key = _normalize_storage_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=normalized_key)
            return True
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def presigned_url(self, key: str, expires: int = 3600) -> str:
        """Generate a signed GET URL for an R2 object."""
        normalized_key = _normalize_storage_key(key)
        return self.client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": normalized_key},
            ExpiresIn=expires,
        )


def _normalize_storage_key(key: str) -> str:
    """Normalize and validate a storage key."""
    candidate = key.replace("\\", "/").strip().strip("/")
    if not candidate:
        raise ValueError("Storage key cannot be empty")

    parts = PurePosixPath(candidate).parts
    if not parts:
        raise ValueError("Storage key cannot be empty")
    if parts[0] not in _VALID_PREFIXES:
        raise ValueError(f"Unsupported storage key prefix: {parts[0]}")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Invalid storage key: {key}")
    if len(parts) < 3:
        raise ValueError(f"Incomplete storage key: {key}")

    return PurePosixPath(*parts).as_posix()


def _r2_is_configured() -> bool:
    """Return whether R2 credentials are configured."""
    return all((
        settings.r2_endpoint.strip(),
        settings.r2_access_key_id.strip(),
        settings.r2_secret_access_key.strip(),
    ))


def _as_legacy_path(identifier: str) -> Path | None:
    """Return a legacy absolute path identifier if one was provided."""
    path = Path(identifier)
    if path.is_absolute():
        return path
    return None


def _validate_legacy_path(path: Path) -> Path:
    """Ensure a legacy filesystem path is inside a managed storage root."""
    resolved = path.resolve(strict=False)
    for base_dir in (settings.upload_dir.resolve(strict=False), settings.models_dir.resolve(strict=False)):
        try:
            resolved.relative_to(base_dir)
            return resolved
        except ValueError:
            continue
    raise ValueError(f"Path is outside managed storage roots: {path}")


def _key_from_local_path(path: Path) -> str:
    """Convert a managed local path into a storage key."""
    resolved = _validate_legacy_path(path)
    upload_dir = settings.upload_dir.resolve(strict=False)
    models_dir = settings.models_dir.resolve(strict=False)

    try:
        relative = resolved.relative_to(upload_dir)
        return PurePosixPath(_RECORDINGS_PREFIX, *relative.parts).as_posix()
    except ValueError:
        pass

    relative = resolved.relative_to(models_dir)
    return PurePosixPath(_MODELS_PREFIX, *relative.parts).as_posix()
