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
    "viola_mlp_oww": ModelSpec(
        name="viola_mlp_oww",
        url="https://github.com/youorg/violawake/releases/download/v0.1.0/viola_mlp_oww.onnx",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=2_100_000,
        description="ViolaWake MLP on OWW embeddings — d-prime 15.10 (default, recommended)",
        version="0.1.0",
    ),
    "oww_backbone": ModelSpec(
        name="oww_backbone",
        url="https://github.com/youorg/violawake/releases/download/v0.1.0/oww_backbone.onnx",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=10_000_000,
        description="OpenWakeWord audio embedding backbone — required for MLP models",
        version="0.1.0",
    ),
    "viola_cnn_v4": ModelSpec(
        name="viola_cnn_v4",
        url="https://github.com/youorg/violawake/releases/download/v0.1.0/viola_cnn_v4.onnx",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=1_800_000,
        description="ViolaWake CNN v4 — d-prime 8.2 (no OWW dependency, lightweight)",
        version="0.1.0",
    ),
    "kokoro_v1_0": ModelSpec(
        name="kokoro_v1_0",
        url="https://github.com/youorg/violawake/releases/download/v0.1.0/kokoro-v1.0.onnx",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=330_000_000,
        description="Kokoro-82M TTS model — Apache 2.0 licensed, 24kHz output",
        version="0.1.0",
    ),
    "kokoro_voices_v1_0": ModelSpec(
        name="kokoro_voices_v1_0",
        url="https://github.com/youorg/violawake/releases/download/v0.1.0/voices-v1.0.bin",
        sha256="PLACEHOLDER_SHA256_FILLED_BY_RELEASE_SCRIPT",
        size_bytes=8_000_000,
        description="Kokoro voice embeddings — required for TTS",
        version="0.1.0",
    ),
}


def get_model_dir() -> Path:
    """Return the model cache directory, creating it if needed.

    Override via VIOLAWAKE_MODEL_DIR environment variable.
    """
    model_dir = Path(os.environ.get("VIOLAWAKE_MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_model_path(model_name: str) -> Path:
    """Return the local path for a model, raising FileNotFoundError if not cached.

    Does NOT download automatically — use ``download_model()`` for that.

    Args:
        model_name: Name from MODEL_REGISTRY (without .onnx extension).

    Returns:
        Path to the cached model file.

    Raises:
        FileNotFoundError: If model is not in cache.
        KeyError: If model_name is not in MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

    spec = MODEL_REGISTRY[model_name]
    # Determine file extension from URL
    ext = Path(spec.url).suffix or ".onnx"
    model_path = get_model_dir() / f"{model_name}{ext}"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model '{model_name}' not found in cache at {model_path}. "
            f"Run: violawake-download --model {model_name}"
        )

    return model_path


def download_model(
    model_name: str,
    force: bool = False,
    verify: bool = True,
) -> Path:
    """Download a model from the registry to the local cache.

    Args:
        model_name: Name from MODEL_REGISTRY.
        force: Re-download even if already cached.
        verify: Verify SHA-256 after download (recommended).

    Returns:
        Path to the downloaded model file.

    Raises:
        KeyError: If model_name is not in MODEL_REGISTRY.
        ValueError: If SHA-256 verification fails.
    """
    import requests  # lazy import
    from tqdm import tqdm  # lazy import

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown model '{model_name}'. Available: {available}")

    spec = MODEL_REGISTRY[model_name]
    ext = Path(spec.url).suffix or ".onnx"
    model_path = get_model_dir() / f"{model_name}{ext}"

    if model_path.exists() and not force:
        logger.info("Model already cached: %s", model_path)
        if verify:
            _verify_sha256(model_path, spec.sha256, model_name)
        return model_path

    logger.info("Downloading model '%s' from %s", model_name, spec.url)

    response = requests.get(spec.url, stream=True, timeout=30)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", spec.size_bytes))
    chunk_size = 8_192

    with (
        open(model_path, "wb") as f,
        tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {model_name}",
        ) as progress,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                progress.update(len(chunk))

    if verify:
        _verify_sha256(model_path, spec.sha256, model_name)

    logger.info("Model downloaded and cached: %s (%.1f MB)", model_path, model_path.stat().st_size / 1e6)
    return model_path


def _verify_sha256(model_path: Path, expected_sha256: str, model_name: str) -> None:
    """Verify the SHA-256 of a downloaded model file."""
    if expected_sha256.startswith("PLACEHOLDER"):
        logger.warning(
            "Skipping SHA-256 verification for %s — placeholder hash (dev build)",
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
    for model_name in MODEL_REGISTRY:
        spec = MODEL_REGISTRY[model_name]
        ext = Path(spec.url).suffix or ".onnx"
        path = model_dir / f"{model_name}{ext}"
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            result.append((model_name, path, size_mb))
    return result
