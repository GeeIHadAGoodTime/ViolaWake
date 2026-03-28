"""TFLite Runtime inference backend.

Drop-in alternative to ONNX Runtime for environments where
``tflite-runtime`` (or full ``tensorflow``) is available but
``onnxruntime`` is not.  Typical on edge devices (Raspberry Pi,
Coral, mobile).

Model files must be in ``.tflite`` format.  Use the
``convert_onnx_to_tflite()`` utility at the bottom of this module
to convert existing ``.onnx`` models.

Usage::

    from violawake_sdk.backends import get_backend

    backend = get_backend("tflite")
    session = backend.load("model.tflite")
    out = session.run(None, {"input": np.zeros((1, 96), dtype=np.float32)})
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

# ──────────────────────────────────────────────────────────────────────────────
# TFLite dtype mapping
# ──────────────────────────────────────────────────────────────────────────────

# TFLite uses integer dtype constants.  Map them to numpy dtypes and
# human-readable ONNX-style type strings for SessionInput/SessionOutput.

_TFLITE_DTYPE_TO_NP: dict[int, np.dtype[Any]] = {
    0: np.dtype(np.float32),   # kTfLiteFloat32
    1: np.dtype(np.int32),     # kTfLiteInt32
    2: np.dtype(np.uint8),     # kTfLiteUInt8
    3: np.dtype(np.int64),     # kTfLiteInt64
    7: np.dtype(np.float16),   # kTfLiteFloat16
    9: np.dtype(np.int8),      # kTfLiteInt8
    11: np.dtype(np.float64),  # kTfLiteFloat64
}

_TFLITE_DTYPE_TO_STR: dict[int, str] = {
    0: "tensor(float)",
    1: "tensor(int32)",
    2: "tensor(uint8)",
    3: "tensor(int64)",
    7: "tensor(float16)",
    9: "tensor(int8)",
    11: "tensor(float64)",
}


def _resolve_dtype(raw_dtype: Any) -> tuple[np.dtype[Any], str]:
    """Resolve a TFLite dtype to (numpy_dtype, type_string).

    The TFLite Python API returns ``detail["dtype"]`` as a numpy dtype
    object (e.g. ``numpy.float32``), while the C API uses integer
    constants.  This helper handles both cases.

    Args:
        raw_dtype: Either a numpy dtype / type (from the Python API)
                   or an integer constant (from the C API).

    Returns:
        Tuple of (numpy dtype, ONNX-style type string).
    """
    # Case 1: already a numpy dtype or type (Python TFLite API)
    try:
        as_dtype = np.dtype(raw_dtype)
        # Check if it's a recognised numpy type (not just an int misinterpreted)
        if isinstance(raw_dtype, (np.dtype, type)) and issubclass(
            as_dtype.type, np.generic
        ):
            type_str = f"tensor({as_dtype.name})"
            return as_dtype, type_str
    except (TypeError, AttributeError):
        pass

    # Case 2: integer constant (TFLite C API)
    np_dt = _TFLITE_DTYPE_TO_NP.get(raw_dtype)
    type_str = _TFLITE_DTYPE_TO_STR.get(raw_dtype) or (
        f"tensor({np_dt.name})" if np_dt is not None
        else f"tensor(unknown_{raw_dtype})"
    )
    if np_dt is None:
        np_dt = np.dtype(np.float32)  # safe fallback
    return np_dt, type_str


def _get_tflite_interpreter_class() -> type:
    """Import and return the TFLite ``Interpreter`` class.

    Tries ``tflite_runtime`` first (lightweight, ~5 MB), then falls
    back to the full ``tensorflow`` package.

    Raises:
        ImportError: If neither package is installed.
    """
    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass

    try:
        from tensorflow.lite.python.interpreter import Interpreter

        return Interpreter
    except ImportError:
        pass

    raise ImportError(
        "TFLite backend runtime is not available. Install one of:\n"
        "  pip install violawake[tflite]\n"
        "  pip install tflite-runtime>=2.14"
    )


# ──────────────────────────────────────────────────────────────────────────────
# TFLiteSession
# ──────────────────────────────────────────────────────────────────────────────

class TFLiteSession(BackendSession):
    """Wraps a TFLite ``Interpreter`` with the same API as ``OnnxSession``.

    TFLite models do not carry tensor *names* (they use integer indices),
    so this class synthesises names of the form ``input_0``, ``output_0``
    and builds an index-to-name lookup so that callers can pass names in
    ``input_feed`` dicts, just like with ONNX Runtime.
    """

    def __init__(self, interpreter: Any, model_path: Path) -> None:
        self._interpreter = interpreter
        self._model_path = model_path

        # Cache input / output details once
        self._input_details = interpreter.get_input_details()
        self._output_details = interpreter.get_output_details()

        # Build name -> index maps.
        # TFLite does expose a ``name`` field; use it if non-empty,
        # otherwise synthesise ``input_0`` etc.
        self._input_name_to_idx: dict[str, int] = {}
        self._input_meta: list[SessionInput] = []
        for i, detail in enumerate(self._input_details):
            name = detail.get("name") or f"input_{i}"
            _, type_str = _resolve_dtype(detail["dtype"])
            shape = [int(d) for d in detail["shape"]]
            self._input_name_to_idx[name] = i
            self._input_meta.append(SessionInput(name=name, shape=shape, type=type_str))

        self._output_name_to_idx: dict[str, int] = {}
        self._output_meta: list[SessionOutput] = []
        for i, detail in enumerate(self._output_details):
            name = detail.get("name") or f"output_{i}"
            _, type_str = _resolve_dtype(detail["dtype"])
            shape = [int(d) for d in detail["shape"]]
            self._output_name_to_idx[name] = i
            self._output_meta.append(SessionOutput(name=name, shape=shape, type=type_str))

    # -- metadata ------------------------------------------------------

    def get_inputs(self) -> list[SessionInput]:
        return list(self._input_meta)

    def get_outputs(self) -> list[SessionOutput]:
        return list(self._output_meta)

    # -- inference -----------------------------------------------------

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        """Run inference, matching the ONNX Runtime ``session.run()`` signature.

        Args:
            output_names: Output tensor names to return, or ``None`` for all.
            input_feed:   ``{tensor_name: numpy_array}`` mapping.

        Returns:
            List of numpy arrays in the same order as *output_names*
            (or all outputs if ``None``).
        """
        # ----------------------------------------------------------
        # 1.  Resize input tensors if the feed shape differs from
        #     the compiled shape (common for variable-length audio).
        # ----------------------------------------------------------
        needs_allocate = False
        for name, array in input_feed.items():
            idx = self._resolve_input_index(name)
            detail = self._input_details[idx]
            expected_shape = tuple(detail["shape"])
            actual_shape = tuple(array.shape)
            if expected_shape != actual_shape:
                self._interpreter.resize_tensor_input(
                    detail["index"], list(actual_shape)
                )
                needs_allocate = True

        if needs_allocate:
            try:
                self._interpreter.allocate_tensors()
            except Exception as e:
                raise ModelLoadError(
                    f"TFLite failed to allocate tensors after resize "
                    f"(model={self._model_path}): {e}"
                ) from e
            # Refresh detail caches after resize
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()

        # ----------------------------------------------------------
        # 2.  Set input tensors
        # ----------------------------------------------------------
        for name, array in input_feed.items():
            idx = self._resolve_input_index(name)
            detail = self._input_details[idx]
            # Resolve expected dtype: handles both numpy dtype objects
            # (Python TFLite API) and integer constants (C API).
            expected_np_dtype, _ = _resolve_dtype(detail["dtype"])
            # Cast if needed (e.g. float64 -> float32)
            if array.dtype != expected_np_dtype:
                array = array.astype(expected_np_dtype)
            self._interpreter.set_tensor(detail["index"], array)

        # ----------------------------------------------------------
        # 3.  Run
        # ----------------------------------------------------------
        self._interpreter.invoke()

        # ----------------------------------------------------------
        # 4.  Collect outputs
        # ----------------------------------------------------------
        if output_names is None:
            # All outputs in order.
            # .copy() is required because TFLite's get_tensor() returns a
            # view into the interpreter's internal buffer, which is
            # overwritten on the next invoke() call.
            return [
                self._interpreter.get_tensor(d["index"]).copy()
                for d in self._output_details
            ]

        results: list[np.ndarray] = []
        for oname in output_names:
            oidx = self._resolve_output_index(oname)
            detail = self._output_details[oidx]
            # .copy() — same reason as above: detach from interpreter buffer
            results.append(self._interpreter.get_tensor(detail["index"]).copy())
        return results

    # -- helpers -------------------------------------------------------

    def _resolve_input_index(self, name: str) -> int:
        """Map an input tensor name to its detail-list index."""
        if name in self._input_name_to_idx:
            return self._input_name_to_idx[name]
        # Fallback: try interpreting as integer index
        try:
            idx = int(name)
            if 0 <= idx < len(self._input_details):
                return idx
        except ValueError:
            pass
        available = list(self._input_name_to_idx.keys())
        raise KeyError(
            f"Input tensor {name!r} not found.  Available: {available}"
        )

    def _resolve_output_index(self, name: str) -> int:
        """Map an output tensor name to its detail-list index."""
        if name in self._output_name_to_idx:
            return self._output_name_to_idx[name]
        try:
            idx = int(name)
            if 0 <= idx < len(self._output_details):
                return idx
        except ValueError:
            pass
        available = list(self._output_name_to_idx.keys())
        raise KeyError(
            f"Output tensor {name!r} not found.  Available: {available}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# TFLiteBackend
# ──────────────────────────────────────────────────────────────────────────────

class TFLiteBackend(InferenceBackend):
    """Backend powered by ``tflite-runtime`` (or ``tensorflow.lite``)."""

    def __init__(self) -> None:
        # Fail fast if the runtime is missing
        self._interpreter_cls = _get_tflite_interpreter_class()

    @property
    def name(self) -> str:
        return "tflite"

    def load(self, model_path: str | Path, **kwargs: Any) -> TFLiteSession:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        num_threads: int = kwargs.get("num_threads", 2)
        try:
            interpreter = self._interpreter_cls(
                model_path=str(model_path),
                num_threads=num_threads,
            )
            interpreter.allocate_tensors()
        except Exception as e:
            raise ModelLoadError(
                f"TFLite failed to load {model_path}: {e}"
            ) from e

        logger.debug("TFLiteBackend loaded: %s (threads=%d)", model_path, num_threads)
        return TFLiteSession(interpreter, model_path)

    def is_available(self) -> bool:
        try:
            _get_tflite_interpreter_class()
            return True
        except ImportError:
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Model conversion utility:  ONNX -> TFLite
# ──────────────────────────────────────────────────────────────────────────────

def convert_onnx_to_tflite(
    onnx_path: str | Path,
    tflite_path: str | Path | None = None,
    quantize: bool = False,
) -> Path:
    """Convert an ONNX model to TFLite format.

    Requires ``onnx`` and ``onnx-tf`` (or ``onnx2tf``) and ``tensorflow``.

    The conversion pipeline is::

        .onnx  -->  TF SavedModel  -->  .tflite

    Args:
        onnx_path:   Path to the source ``.onnx`` file.
        tflite_path: Destination path for the ``.tflite`` file.
                     Defaults to the same directory / stem as *onnx_path*.
        quantize:    If ``True``, apply dynamic-range (int8 weights)
                     quantization to reduce model size by ~4x.

    Returns:
        Path to the written ``.tflite`` file.

    Raises:
        ImportError: If conversion dependencies are missing.
        RuntimeError: If conversion fails.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if tflite_path is None:
        tflite_path = onnx_path.with_suffix(".tflite")
    else:
        tflite_path = Path(tflite_path)

    # ---- Step 1: ONNX -> TF SavedModel ----
    saved_model_dir = _onnx_to_saved_model(onnx_path)

    # ---- Step 2: SavedModel -> TFLite ----
    _saved_model_to_tflite(saved_model_dir, tflite_path, quantize=quantize)

    # Clean up temporary saved model
    import shutil

    try:
        shutil.rmtree(saved_model_dir)
    except OSError as e:
        logger.warning("Failed to clean up temp SavedModel dir %s: %s", saved_model_dir, e)

    logger.info(
        "Converted %s -> %s (%.1f KB%s)",
        onnx_path.name,
        tflite_path.name,
        tflite_path.stat().st_size / 1024,
        ", quantized" if quantize else "",
    )
    return tflite_path


def _onnx_to_saved_model(onnx_path: Path) -> Path:
    """Convert ONNX to a TensorFlow SavedModel directory.

    Tries ``onnx2tf`` first (better operator coverage for recent opsets),
    then falls back to ``onnx_tf``.
    """
    saved_model_dir = onnx_path.parent / f"_tf_saved_{onnx_path.stem}"

    # Strategy 1: onnx2tf (recommended, handles more opsets)
    try:
        import onnx2tf  # noqa: F401

        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(saved_model_dir),
            non_verbose=True,
        )
        logger.debug("ONNX -> SavedModel via onnx2tf: %s", saved_model_dir)
        return saved_model_dir
    except ImportError:
        pass
    except Exception as e:
        logger.warning("onnx2tf conversion failed, trying onnx-tf: %s", e)

    # Strategy 2: onnx-tf
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        raise ImportError(
            "ONNX-to-TFLite conversion requires one of:\n"
            "  pip install onnx2tf          # recommended\n"
            "  pip install onnx onnx-tf     # alternative"
        ) from None

    try:
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(str(saved_model_dir))
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert {onnx_path} to SavedModel: {e}"
        ) from e

    logger.debug("ONNX -> SavedModel via onnx-tf: %s", saved_model_dir)
    return saved_model_dir


def _saved_model_to_tflite(
    saved_model_dir: Path,
    tflite_path: Path,
    quantize: bool = False,
) -> None:
    """Convert a TF SavedModel to a ``.tflite`` flatbuffer."""
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError(
            "tensorflow is required for TFLite conversion.  Install with:\n"
            "  pip install tensorflow"
        ) from None

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        raise RuntimeError(
            f"TFLite conversion failed for {saved_model_dir}: {e}"
        ) from e

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    logger.debug("SavedModel -> TFLite: %s", tflite_path)
