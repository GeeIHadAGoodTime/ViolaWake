"""Inference backend abstraction for ViolaWake.

Supports multiple inference runtimes (ONNX Runtime, TFLite) behind a
unified interface.  The ``get_backend()`` factory selects a backend
by name or auto-detects the best available runtime.

Public API::

    from violawake_sdk.backends import get_backend, InferenceBackend

    backend = get_backend("auto")           # onnx > tflite
    session = backend.load("model.onnx")    # returns a BackendSession
    out = session.run(input_array)          # numpy in, numpy out
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from violawake_sdk.backends.base import InferenceBackend

logger = logging.getLogger(__name__)

# Re-export public names
from violawake_sdk.backends.base import BackendSession, InferenceBackend  # noqa: E402, F811


def get_backend(
    name: str = "auto",
    providers: list[str] | None = None,
) -> InferenceBackend:
    """Return an inference backend instance by name.

    Args:
        name: Backend selector.  One of:
            - ``"onnx"`` -- ONNX Runtime (requires ``onnxruntime``).
            - ``"tflite"`` -- TFLite Runtime (requires ``tflite-runtime``
              or ``tensorflow``).
            - ``"auto"`` -- Try ONNX Runtime first, fall back to TFLite.
        providers: ONNX Runtime execution providers (ignored for TFLite).
            Default: ``["CPUExecutionProvider"]``.

    Returns:
        An ``InferenceBackend`` instance ready to load models.

    Raises:
        ImportError: If no suitable runtime is installed.
        ValueError: If *name* is not a recognised backend.
    """
    if name == "onnx":
        return _make_onnx(providers)
    if name == "tflite":
        return _make_tflite()
    if name == "auto":
        return _auto_select(providers)
    raise ValueError(f"Unknown backend {name!r}.  Choose from: 'onnx', 'tflite', 'auto'.")


def _make_onnx(providers: list[str] | None = None) -> InferenceBackend:
    from violawake_sdk.backends.onnx_backend import OnnxBackend

    return OnnxBackend(providers=providers or ["CPUExecutionProvider"])


def _make_tflite() -> InferenceBackend:
    from violawake_sdk.backends.tflite_backend import TFLiteBackend

    return TFLiteBackend()


def _auto_select(providers: list[str] | None = None) -> InferenceBackend:
    """Try ONNX Runtime first, then TFLite, raising if neither works."""
    try:
        backend = _make_onnx(providers)
        logger.debug("Auto-selected ONNX Runtime backend")
        return backend
    except ImportError:
        pass

    try:
        backend = _make_tflite()
        logger.debug("Auto-selected TFLite backend")
        return backend
    except ImportError:
        pass

    raise ImportError(
        "No inference backend available. Install onnxruntime: pip install "
        "violawake  OR  tflite-runtime: pip install violawake[tflite]"
    )


__all__ = [
    "BackendSession",
    "InferenceBackend",
    "get_backend",
]
