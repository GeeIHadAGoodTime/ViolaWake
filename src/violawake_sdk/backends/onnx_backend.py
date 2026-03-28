"""ONNX Runtime inference backend.

Thin wrapper around ``onnxruntime.InferenceSession`` that conforms to
the ``InferenceBackend`` / ``BackendSession`` protocol so that the
existing ONNX-based code paths require zero changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from violawake_sdk._exceptions import ModelLoadError
from violawake_sdk.backends.base import (
    BackendSession,
    InferenceBackend,
    SessionInput,
    SessionOutput,
)

logger = logging.getLogger(__name__)


class OnnxSession(BackendSession):
    """Wraps an ``onnxruntime.InferenceSession``."""

    def __init__(self, ort_session: Any) -> None:
        self._session = ort_session

    # -- metadata ------------------------------------------------------

    def get_inputs(self) -> list[SessionInput]:
        return [
            SessionInput(
                name=inp.name,
                shape=list(inp.shape),
                type=inp.type,
            )
            for inp in self._session.get_inputs()
        ]

    def get_outputs(self) -> list[SessionOutput]:
        return [
            SessionOutput(
                name=out.name,
                shape=list(out.shape),
                type=out.type,
            )
            for out in self._session.get_outputs()
        ]

    # -- inference -----------------------------------------------------

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        return self._session.run(output_names, input_feed)


class OnnxBackend(InferenceBackend):
    """Backend powered by ``onnxruntime``."""

    def __init__(self, providers: list[str] | None = None) -> None:
        # Import eagerly so ``get_backend("onnx")`` fails fast when
        # onnxruntime is not installed.
        import onnxruntime  # noqa: F401

        self._providers = providers or ["CPUExecutionProvider"]

    @property
    def name(self) -> str:
        return "onnx"

    def load(self, model_path: str | Path, **kwargs: Any) -> OnnxSession:
        import onnxruntime as ort

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        providers = kwargs.get("providers", self._providers)
        try:
            session = ort.InferenceSession(str(model_path), providers=providers)
        except Exception as e:
            raise ModelLoadError(f"ONNX Runtime failed to load {model_path}: {e}") from e

        logger.debug("OnnxBackend loaded: %s", model_path)
        return OnnxSession(session)

    def is_available(self) -> bool:
        try:
            import onnxruntime  # noqa: F401

            return True
        except ImportError:
            return False
