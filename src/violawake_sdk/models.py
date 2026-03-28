"""Model registry, download, and cache management.

Models are distributed via GitHub Releases (not PyPI — too large).
This module handles:
  - Declaring the model registry (name, URL, SHA-256, size)
  - Downloading models on demand with progress and verification
  - Caching models in ~/.violawake/models/ (or VIOLAWAKE_MODEL_DIR)

See ADR-005 for the full rationale behind this distribution approach.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model cache directory
DEFAULT_MODEL_DIR = Path.home() / ".violawake" / "models"


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a downloadable model."""

    name: str
    url: str
    sha256: str
    size_bytes: int
    description: str
    version: str = "latest"


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# Update this table when releasing new model versions.
# SHA-256 values are filled in during the release process by tools/update_model_registry.py.
# ──────────────────────────────────────────────────────────────────────────────
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "temporal_cnn": ModelSpec(
        name="temporal_cnn",
        url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_cnn.onnx",
        sha256="9c0b12c68593cfdb3d320a3b34667913b18d63e89eb01247d6332d7839ac9efe",
        size_bytes=102378,
        description="Temporal CNN on OWW embeddings — production default, best live recall + lowest FP",
        version="0.1.0",
    ),
    # DEPRECATED: viola_mlp_oww was never uploaded to GitHub Releases.
    # Kept in registry for backward compatibility (tests reference it).
    # Do NOT list in user-facing documentation.
    "viola_mlp_oww": ModelSpec(
        name="viola_mlp_oww",
        url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/viola_mlp_oww.onnx",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=2_100_000,
        description="DEPRECATED — MLP on OWW embeddings, never released. Use temporal_cnn instead.",
        version="0.1.0",
    ),
    "oww_backbone": ModelSpec(
        name="oww_backbone",
        # Not directly downloadable — bundled inside the openwakeword package.
        # URL is a reference only; download is blocked by _PACKAGE_MANAGED_MODELS.
        url="https://github.com/dscripka/openWakeWord/tree/main/openwakeword/resources",
        sha256="70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
        size_bytes=1_326_578,
        description="OpenWakeWord embedding backbone — installed with openwakeword package, not separately downloadable",
        version="0.6.0",
    ),
    # DEPRECATED: viola_cnn_v4 was never uploaded to GitHub Releases.
    # Kept in registry for backward compatibility.
    "viola_cnn_v4": ModelSpec(
        name="viola_cnn_v4",
        url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/viola_cnn_v4.onnx",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=1_800_000,
        description="DEPRECATED — CNN v4, never released. Use temporal_cnn instead.",
        version="0.1.0",
    ),
    # Kokoro TTS models hosted upstream at thewh1teagle/kokoro-onnx (Apache 2.0).
    # These are large (325MB + 28MB) so they're not bundled in the PyPI package —
    # they auto-download on first TTSEngine use.
    "kokoro_v1_0": ModelSpec(
        name="kokoro_v1_0",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        sha256="7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
        size_bytes=325_532_387,
        description="Kokoro-82M TTS model — Apache 2.0 licensed, 24kHz output",
        version="1.0",
    ),
    "kokoro_voices_v1_0": ModelSpec(
        name="kokoro_voices_v1_0",
        url="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
        sha256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
        size_bytes=28_214_398,
        description="Kokoro voice embeddings — required for TTS",
        version="1.0",
    ),
    "temporal_convgru": ModelSpec(
        name="temporal_convgru",
        url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/temporal_convgru.onnx",
        sha256="5990edf8c6228c45a53e08a270ec4ad9350c9133091ebd95d023a5745578e6f8",
        size_bytes=80780,
        description="Temporal Conv-GRU on OWW embeddings — reserve model",
        version="0.1.0",
    ),
    "r3_10x_s42": ModelSpec(
        name="r3_10x_s42",
        url="https://github.com/GeeIHadAGoodTime/ViolaWake/releases/download/v0.1.0/r3_10x_s42.onnx",
        sha256="98d028fe36ed10c51791da91fcead6c7c0dd3149049da02a0436ed008da7a362",
        size_bytes=34010,
        description="MLP on OWW embeddings — DEPRECATED: fails live mic test (max score 0.50)",
        version="0.1.0",
    ),
}
# Alias: "viola" resolves to "temporal_cnn" — the current production default model.
# This allows callers to use the generic name without coupling to a specific architecture.
MODEL_REGISTRY["viola"] = MODEL_REGISTRY["temporal_cnn"]

_PACKAGE_MANAGED_MODELS = {"oww_backbone"}

# Size validation tolerance: downloaded file size must be within this fraction of
# the declared size_bytes.  5% accommodates minor compression/header differences
# while still catching truncated or wildly wrong files.
SIZE_TOLERANCE_FRACTION = 0.05


def check_registry_integrity(*, strict: bool = True) -> list[str]:
    """Verify that no model in MODEL_REGISTRY has a placeholder SHA-256 hash.

    Args:
        strict: If True (default), raise RuntimeError when placeholders are
            found.  If False, just return the list without raising — useful
            for reporting in CI without aborting immediately.

    Returns:
        List of model names that still have placeholder hashes.  Empty list
        means the registry is fully populated and ready for release.

    Raises:
        RuntimeError: If ``strict`` is True and any model has a placeholder
            hash.  The error message lists every offending model so CI can
            surface the full list.

    This is intended to be called from CI pipelines and release scripts to
    prevent shipping unverified models.  ``download_model()`` also calls this
    for the specific model being downloaded (unless ``skip_verify=True``).
    """
    placeholder_models = sorted(set(
        name
        for name, spec in MODEL_REGISTRY.items()
        if "placeholder" in spec.sha256.lower()
    ))
    if placeholder_models and strict:
        raise RuntimeError(
            f"Registry integrity check failed — {len(placeholder_models)} model(s) have "
            f"placeholder SHA-256 hashes: {', '.join(placeholder_models)}. "
            f"Run tools/update_model_registry.py to populate real hashes before release."
        )
    return placeholder_models


def get_model_dir() -> Path:
    """Return the model cache directory, creating it if needed.

    Override via VIOLAWAKE_MODEL_DIR environment variable.
    """
    model_dir = Path(os.environ.get("VIOLAWAKE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _is_auto_download_disabled() -> bool:
    """Check if auto-download is disabled via environment variable."""
    return os.environ.get("VIOLAWAKE_NO_AUTO_DOWNLOAD", "").strip() in ("1", "true", "yes")


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string (e.g. '2.1 MB')."""
    if size_bytes >= 1_000_000_000:
        return f"{size_bytes / 1e9:.1f} GB"
    if size_bytes >= 1_000_000:
        return f"{size_bytes / 1e6:.1f} MB"
    if size_bytes >= 1_000:
        return f"{size_bytes / 1e3:.1f} KB"
    return f"{size_bytes} B"


def _auto_download_model(model_name: str, spec: ModelSpec) -> Path:
    """Auto-download a model on first use, with progress to stderr.

    This is a lightweight download path that does NOT require tqdm or
    requests as hard dependencies — it uses urllib from the standard library.
    For full download features (progress bars, pinning), use ``download_model()``.

    Models with placeholder hashes are refused (raises RuntimeError).

    Args:
        model_name: Name from MODEL_REGISTRY.
        spec: The ModelSpec for this model.

    Returns:
        Path to the downloaded model file.

    Raises:
        RuntimeError: If the download fails.
    """
    import tempfile
    import urllib.request
    import urllib.error

    url_suffix = Path(spec.url).suffix
    ext = url_suffix or ".onnx"
    model_dir = get_model_dir()
    model_path = model_dir / f"{spec.name}{ext}"

    has_placeholder = "placeholder" in spec.sha256.lower()
    size_str = _format_size(spec.size_bytes)

    # Print progress to stderr so it doesn't interfere with piped output
    print(
        f"Downloading model '{model_name}' ({size_str})...",
        end="",
        flush=True,
        file=sys.stderr,
    )

    if has_placeholder:
        print(" REFUSED", file=sys.stderr, flush=True)
        raise RuntimeError(
            f"Model '{model_name}' has a placeholder SHA-256 hash and cannot be "
            f"verified. This model was never released. Use 'temporal_cnn' instead."
        )

    # Reject non-HTTPS URLs to prevent MITM attacks on model downloads
    if not spec.url.startswith("https://"):
        raise ValueError(
            f"Refusing to download model '{model_name}' from non-HTTPS URL: "
            f"{spec.url}. Only HTTPS URLs are allowed for model downloads."
        )

    tmp_fd = None
    tmp_path = None
    try:
        response = urllib.request.urlopen(spec.url, timeout=60)  # noqa: S310 — URLs validated as HTTPS above

        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(model_dir),
            prefix=f".{model_name}_autodownload_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        os.chmod(tmp_path_str, 0o600)

        with os.fdopen(tmp_fd, "wb") as f:
            tmp_fd = None  # os.fdopen takes ownership
            chunk_size = 65_536
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        # Atomic rename
        tmp_path.replace(model_path)
        tmp_path = None

    except Exception as e:
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        print(" FAILED", file=sys.stderr, flush=True)
        raise RuntimeError(
            f"Auto-download of model '{model_name}' failed: {e}. "
            f"Download manually with: violawake-download --model {model_name}"
        ) from e

    print(" done.", file=sys.stderr, flush=True)

    # Verify SHA-256 (placeholder hashes are blocked above, so this always runs)
    _verify_sha256(model_path, spec.sha256, model_name)

    # Size validation (same logic as download_model)
    actual_size = model_path.stat().st_size
    if spec.size_bytes and abs(actual_size - spec.size_bytes) > max(1024, spec.size_bytes * SIZE_TOLERANCE_FRACTION):
        model_path.unlink(missing_ok=True)
        raise ValueError(
            f"Size validation failed for auto-downloaded '{model_name}'. "
            f"Expected ~{spec.size_bytes} bytes, got {actual_size} bytes. "
            f"File deleted — retry or download manually."
        )

    logger.info(
        "Auto-downloaded model '%s' to %s (%.1f MB)",
        model_name, model_path, model_path.stat().st_size / 1e6,
    )
    return model_path


def get_model_path(model_name: str, *, auto_download: bool = True) -> Path:
    """Return the local path for a model, optionally downloading on first use.

    When the model is not cached and ``auto_download`` is True (the default),
    the model is downloaded automatically from the registry.  Auto-download
    can be disabled globally with the ``VIOLAWAKE_NO_AUTO_DOWNLOAD=1``
    environment variable, or per-call with ``auto_download=False``.

    Args:
        model_name: Name from MODEL_REGISTRY (without .onnx extension).
        auto_download: If True (default), download the model if not cached.
            Set to False to get the old raise-on-missing behavior.

    Returns:
        Path to the cached model file.

    Raises:
        FileNotFoundError: If model is not in cache and auto-download is
            disabled or unavailable.
        KeyError: If model_name is not in MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

    if model_name in _PACKAGE_MANAGED_MODELS:
        raise FileNotFoundError(
            f"Model '{model_name}' is provided by the openwakeword package, not the ViolaWake "
            "model cache. Install with: pip install openwakeword"
        )

    spec = MODEL_REGISTRY[model_name]
    # Determine file extension from URL
    url_suffix = Path(spec.url).suffix
    if not url_suffix:
        logger.warning(
            "URL for model '%s' has no file extension; defaulting to .onnx",
            model_name,
        )
    ext = url_suffix or ".onnx"
    model_path = get_model_dir() / f"{spec.name}{ext}"

    if not model_path.exists():
        if auto_download and not _is_auto_download_disabled():
            return _auto_download_model(model_name, spec)

        raise FileNotFoundError(
            f"Model '{model_name}' not found in cache at {model_path}. "
            f"Run: violawake-download --model {model_name}"
        )

    return model_path


def download_model(
    model_name: str,
    force: bool = False,
    verify: bool = True,
    use_pinning: bool = False,
    skip_verify: bool = False,
) -> Path:
    """Download a model from the registry to the local cache.

    Args:
        model_name: Name from MODEL_REGISTRY.
        force: Re-download even if already cached.
        verify: Verify SHA-256 after download (recommended).
        use_pinning: Enable optional certificate pinning / TOFU checks.
        skip_verify: If True, allow downloading models with placeholder hashes
            without raising RuntimeError.  **Only for development builds.**

    Returns:
        Path to the downloaded model file.

    Raises:
        KeyError: If model_name is not in MODEL_REGISTRY.
        ValueError: If SHA-256 verification fails.
        RuntimeError: If model has a placeholder hash and skip_verify is False.
    """
    try:
        import requests  # lazy import — optional [download] extra
    except ImportError:
        raise ImportError(
            "requests is required for model downloading. "
            "Install with: pip install violawake[download]"
        ) from None

    try:
        from tqdm import tqdm  # lazy import — optional [download] extra
    except ImportError:
        raise ImportError(
            "tqdm is required for model downloading. "
            "Install with: pip install violawake[download]"
        ) from None

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

    if model_name in _PACKAGE_MANAGED_MODELS:
        raise ValueError(
            f"Model '{model_name}' is installed with the openwakeword package and is not "
            "downloaded via violawake-download. Install or upgrade with: pip install openwakeword"
        )

    spec = MODEL_REGISTRY[model_name]

    # Block downloads of models with placeholder hashes unless explicitly overridden.
    # This prevents users from silently downloading unverified models.
    if not skip_verify and "placeholder" in spec.sha256.lower():
        raise RuntimeError(
            f"Model '{model_name}' has a placeholder SHA-256 hash and cannot be "
            f"verified. Either run tools/update_model_registry.py to populate the "
            f"real hash, or pass skip_verify=True to download without verification "
            f"(development only)."
        )

    # Reject non-HTTPS URLs to prevent MITM attacks on model downloads
    if not spec.url.startswith("https://"):
        raise ValueError(
            f"Refusing to download model '{model_name}' from non-HTTPS URL: "
            f"{spec.url}. Only HTTPS URLs are allowed for model downloads."
        )

    url_suffix = Path(spec.url).suffix
    if not url_suffix:
        logger.warning(
            "URL for model '%s' has no file extension; defaulting to .onnx",
            model_name,
        )
    ext = url_suffix or ".onnx"
    model_path = get_model_dir() / f"{spec.name}{ext}"

    if model_path.exists() and not force:
        logger.info("Model already cached: %s", model_path)
        if verify:
            _verify_sha256(model_path, spec.sha256, model_name)
        return model_path

    logger.info("Downloading model '%s' from %s", model_name, spec.url)

    # G3: Atomic write — download to a temp file, rename on success.
    # This prevents partial/corrupt files from being left in the cache
    # if the download is interrupted or fails mid-stream.
    import tempfile

    # Certificate pinning available but requires real SPKI pins. See security/cert_pinning.py
    if use_pinning:
        from violawake_sdk.security.cert_pinning import pinned_download

        response = pinned_download(
            spec.url,
            model_path,
            verify_pin=True,
            timeout=30,
        )
    else:
        response = requests.get(spec.url, stream=True, timeout=30)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", spec.size_bytes))
    chunk_size = 8_192

    tmp_fd = None
    tmp_path = None
    try:
        # Create temp file in same directory as dest (ensures same filesystem for rename)
        # Use mode 0o600 to prevent world-readable temp files
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(model_path.parent),
            prefix=f".{model_name}_download_",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_path_str)
        os.chmod(tmp_path_str, 0o600)

        with (
            os.fdopen(tmp_fd, "wb") as f,
            tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {model_name}",
            ) as progress,
        ):
            tmp_fd = None  # os.fdopen takes ownership of the fd
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))

        # Atomic rename: tmp -> dest
        tmp_path.replace(model_path)
        tmp_path = None  # Prevent cleanup since rename succeeded
    except Exception:
        # G3: Clean up temp file on error
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except OSError:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

    # Verify file size matches spec (catches truncated downloads)
    actual_size = model_path.stat().st_size
    if spec.size_bytes and abs(actual_size - spec.size_bytes) > max(1024, spec.size_bytes * SIZE_TOLERANCE_FRACTION):
        model_path.unlink(missing_ok=True)
        raise ValueError(
            f"Size validation failed for '{model_name}'. "
            f"Expected ~{spec.size_bytes} bytes, got {actual_size} bytes. "
            f"File deleted — re-run download."
        )

    if verify:
        _verify_sha256(model_path, spec.sha256, model_name)

    logger.info("Model downloaded and cached: %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)
    return model_path


def _verify_sha256(model_path: Path, expected_sha256: str, model_name: str) -> None:
    """Verify the SHA-256 of a downloaded model file."""
    if expected_sha256.startswith("PLACEHOLDER"):
        import warnings

        warnings.warn(
            f"SHA-256 verification skipped for '{model_name}': hash not yet set "
            f"(pre-release build). Model integrity CANNOT be verified. "
            f"Do NOT ship to production without real hashes. "
            f"Run: python tools/update_model_registry.py --model {model_name}",
            UserWarning,
            stacklevel=3,
        )
        logger.warning(
            "SHA-256 verification SKIPPED for '%s' — placeholder hash (pre-release build). "
            "Run tools/update_model_registry.py to populate hashes before release.",
            model_name,
        )
        return

    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    actual = sha256.hexdigest()
    if actual != expected_sha256:
        model_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 verification failed for '{model_name}'. "
            f"Expected: {expected_sha256[:16]}... Got: {actual[:16]}... "
            f"File deleted — re-run download."
        )

    logger.debug("SHA-256 verified: %s", model_name)


def list_cached_models() -> list[tuple[str, Path, float]]:
    """List all models currently in the local cache.

    Returns:
        List of (model_name, path, size_mb) tuples.
    """
    model_dir = get_model_dir()
    result = []
    seen_paths: set[Path] = set()
    for model_name in MODEL_REGISTRY:
        if model_name in _PACKAGE_MANAGED_MODELS:
            continue
        spec = MODEL_REGISTRY[model_name]
        ext = Path(spec.url).suffix or ".onnx"
        path = model_dir / f"{spec.name}{ext}"
        if path.exists() and path not in seen_paths:
            size_mb = path.stat().st_size / 1e6
            result.append((model_name, path, size_mb))
            seen_paths.add(path)
    return result
