"""Abstract base classes for inference backends.

Every backend (ONNX, TFLite, ...) implements ``InferenceBackend`` to
create ``BackendSession`` objects that wrap a loaded model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BackendSession(ABC):
    """A loaded model ready for inference.

    Wraps runtime-specific session objects behind a uniform API that
    mirrors the subset of ``onnxruntime.InferenceSession`` used by
    ``WakeDetector``.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @abstractmethod
    def get_inputs(self) -> list[SessionInput]:
        """Return metadata for each model input tensor."""
        ...

    @abstractmethod
    def get_outputs(self) -> list[SessionOutput]:
        """Return metadata for each model output tensor."""
        ...

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @abstractmethod
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Execute the model.

        Signature intentionally matches ``onnxruntime.InferenceSession.run``
        so existing call-sites work without changes.

        Args:
            output_names: List of output tensor names to return, or ``None``
                          for all outputs.
            input_feed:   Mapping from input tensor name to numpy array.

        Returns:
            List of numpy arrays, one per requested output.
        """
        ...


class SessionInput:
    """Descriptor for a model input tensor."""

    __slots__ = ("name", "shape", "type")

    def __init__(self, name: str, shape: list[int | str], type: str) -> None:
        self.name = name
        self.shape = shape
        self.type = type

    def __repr__(self) -> str:
        return f"SessionInput(name={self.name!r}, shape={self.shape}, type={self.type!r})"


class SessionOutput:
    """Descriptor for a model output tensor."""

    __slots__ = ("name", "shape", "type")

    def __init__(self, name: str, shape: list[int | str], type: str) -> None:
        self.name = name
        self.shape = shape
        self.type = type

    def __repr__(self) -> str:
        return f"SessionOutput(name={self.name!r}, shape={self.shape}, type={self.type!r})"


class InferenceBackend(ABC):
    """Factory that creates ``BackendSession`` objects from model files."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name (e.g. ``'onnx'``, ``'tflite'``)."""
        ...

    @abstractmethod
    def load(
        self,
        model_path: str | Path,
        **kwargs: Any,
    ) -> BackendSession:
        """Load a model file and return an inference session.

        Args:
            model_path: Path to the model file (``.onnx`` or ``.tflite``).
            **kwargs:   Backend-specific options.

        Returns:
            A ``BackendSession`` wrapping the loaded model.

        Raises:
            FileNotFoundError: If *model_path* does not exist.
            violawake_sdk._exceptions.ModelLoadError: If the model cannot
                be loaded by the runtime.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the underlying runtime library is importable."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
