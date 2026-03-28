"""Tests for the TFLite inference backend (C4).

All tests mock tflite-runtime so they run without the actual package installed.
Tests verify:
  - TFLiteSession wraps the interpreter correctly
  - Input/output metadata is exposed with correct names and shapes
  - run() handles tensor resizing, dtype casting, and multi-output
  - TFLiteBackend.load() validates paths and creates sessions
  - get_backend("auto") fallback chain works
  - get_backend("tflite") raises ImportError when runtime is missing
  - ONNX-to-TFLite conversion utility interface
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: build a realistic mock tflite interpreter
# ---------------------------------------------------------------------------


def _make_mock_interpreter(
    input_details: list[dict] | None = None,
    output_details: list[dict] | None = None,
    output_data: list[np.ndarray] | None = None,
) -> MagicMock:
    """Create a mock TFLite Interpreter with realistic input/output details.

    Args:
        input_details: List of input detail dicts.  Defaults to a single
            float32 input named ``"input_0"`` with shape ``[1, 96]``.
        output_details: List of output detail dicts.  Defaults to a single
            float32 output named ``"output_0"`` with shape ``[1, 1]``.
        output_data: List of numpy arrays returned by ``get_tensor()``.
            Defaults to ``[np.array([[0.85]], dtype=np.float32)]``.
    """
    if input_details is None:
        input_details = [
            {
                "name": "input_0",
                "index": 0,
                "shape": np.array([1, 96]),
                "dtype": np.float32,
            },
        ]
    if output_details is None:
        output_details = [
            {
                "name": "output_0",
                "index": 1,
                "shape": np.array([1, 1]),
                "dtype": np.float32,
            },
        ]
    if output_data is None:
        output_data = [np.array([[0.85]], dtype=np.float32)]

    interp = MagicMock()
    interp.get_input_details.return_value = input_details
    interp.get_output_details.return_value = output_details
    interp.allocate_tensors.return_value = None
    interp.invoke.return_value = None
    interp.resize_tensor_input.return_value = None
    interp.set_tensor.return_value = None

    # Map output index -> data
    _tensor_store: dict[int, np.ndarray] = {}
    for detail, data in zip(output_details, output_data):
        _tensor_store[detail["index"]] = data

    def _get_tensor(index: int) -> np.ndarray:
        return _tensor_store.get(index, np.zeros(1, dtype=np.float32))

    interp.get_tensor.side_effect = _get_tensor
    return interp


# ---------------------------------------------------------------------------
# TFLiteSession tests
# ---------------------------------------------------------------------------


class TestTFLiteSession:
    """Tests for TFLiteSession (the session wrapper)."""

    def _make_session(self, **kwargs):  # type: ignore[no-untyped-def]
        from violawake_sdk.backends.tflite_backend import TFLiteSession

        interp = _make_mock_interpreter(**kwargs)
        return TFLiteSession(interp, Path("/fake/model.tflite")), interp

    def test_get_inputs_metadata(self) -> None:
        session, _ = self._make_session()
        inputs = session.get_inputs()
        assert len(inputs) == 1
        assert inputs[0].name == "input_0"
        assert inputs[0].shape == [1, 96]
        assert "float32" in inputs[0].type

    def test_get_outputs_metadata(self) -> None:
        session, _ = self._make_session()
        outputs = session.get_outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "output_0"
        assert outputs[0].shape == [1, 1]

    def test_run_basic_inference(self) -> None:
        session, interp = self._make_session()
        inp = np.random.randn(1, 96).astype(np.float32)
        result = session.run(None, {"input_0": inp})

        assert len(result) == 1
        np.testing.assert_allclose(result[0], [[0.85]], atol=1e-6)
        interp.set_tensor.assert_called_once()
        interp.invoke.assert_called_once()

    def test_run_with_named_outputs(self) -> None:
        session, _ = self._make_session()
        inp = np.random.randn(1, 96).astype(np.float32)
        result = session.run(["output_0"], {"input_0": inp})
        assert len(result) == 1

    def test_run_resizes_on_shape_mismatch(self) -> None:
        """When the input shape differs from compiled shape, resize is called."""
        session, interp = self._make_session()
        # Feed a different shape than [1, 96]
        inp = np.random.randn(1, 128).astype(np.float32)
        session.run(None, {"input_0": inp})

        interp.resize_tensor_input.assert_called_once_with(0, [1, 128])
        # allocate_tensors called again after resize (plus once at construction)
        assert interp.allocate_tensors.call_count >= 1

    def test_run_casts_dtype(self) -> None:
        """Float64 input is cast to float32 to match model expectation."""
        session, interp = self._make_session()
        inp = np.random.randn(1, 96).astype(np.float64)
        session.run(None, {"input_0": inp})

        # The set_tensor call should receive float32
        call_args = interp.set_tensor.call_args
        actual_arr = call_args[0][1]
        assert actual_arr.dtype == np.float32

    def test_numpy_dtype_objects_in_details(self) -> None:
        """TFLite Python API returns numpy dtype objects, not integer constants."""
        # The Python TFLite API returns detail["dtype"] as np.float32 (a numpy type),
        # NOT an integer constant like 0.  Verify our session handles this correctly.
        input_details = [
            {
                "name": "input_0",
                "index": 0,
                "shape": np.array([1, 96]),
                "dtype": np.float32,  # numpy type, not int
            },
        ]
        output_details = [
            {
                "name": "output_0",
                "index": 1,
                "shape": np.array([1, 1]),
                "dtype": np.float32,  # numpy type, not int
            },
        ]
        session, interp = self._make_session(
            input_details=input_details,
            output_details=output_details,
        )
        inputs = session.get_inputs()
        assert "float32" in inputs[0].type

        # Run inference with float64 -> should cast to float32
        inp = np.random.randn(1, 96).astype(np.float64)
        session.run(None, {"input_0": inp})
        call_args = interp.set_tensor.call_args
        assert call_args[0][1].dtype == np.float32

    def test_integer_dtype_constants_in_details(self) -> None:
        """TFLite C API returns integer dtype constants."""
        input_details = [
            {
                "name": "input_0",
                "index": 0,
                "shape": np.array([1, 96]),
                "dtype": 0,  # kTfLiteFloat32 integer constant
            },
        ]
        output_details = [
            {
                "name": "output_0",
                "index": 1,
                "shape": np.array([1, 1]),
                "dtype": 0,  # kTfLiteFloat32 integer constant
            },
        ]
        session, _ = self._make_session(
            input_details=input_details,
            output_details=output_details,
        )
        inputs = session.get_inputs()
        assert "float" in inputs[0].type

    def test_run_unknown_input_name_raises(self) -> None:
        session, _ = self._make_session()
        inp = np.random.randn(1, 96).astype(np.float32)
        with pytest.raises(KeyError, match="not_a_real_input"):
            session.run(None, {"not_a_real_input": inp})

    def test_run_unknown_output_name_raises(self) -> None:
        session, _ = self._make_session()
        inp = np.random.randn(1, 96).astype(np.float32)
        with pytest.raises(KeyError, match="bad_output"):
            session.run(["bad_output"], {"input_0": inp})

    def test_multi_input_multi_output(self) -> None:
        """Session with 2 inputs and 2 outputs."""
        input_details = [
            {"name": "audio", "index": 0, "shape": np.array([1, 16000]),
             "dtype": np.float32},
            {"name": "length", "index": 1, "shape": np.array([1]),
             "dtype": np.int32},
        ]
        output_details = [
            {"name": "mel", "index": 2, "shape": np.array([1, 100, 32]),
             "dtype": np.float32},
            {"name": "confidence", "index": 3, "shape": np.array([1]),
             "dtype": np.float32},
        ]
        output_data = [
            np.random.randn(1, 100, 32).astype(np.float32),
            np.array([0.9], dtype=np.float32),
        ]
        session, _ = self._make_session(
            input_details=input_details,
            output_details=output_details,
            output_data=output_data,
        )

        inputs = session.get_inputs()
        assert len(inputs) == 2
        assert inputs[0].name == "audio"
        assert inputs[1].name == "length"

        feed = {
            "audio": np.random.randn(1, 16000).astype(np.float32),
            "length": np.array([16000], dtype=np.int32),
        }
        results = session.run(None, feed)
        assert len(results) == 2
        assert results[0].shape == (1, 100, 32)
        assert results[1].shape == (1,)


# ---------------------------------------------------------------------------
# TFLiteBackend tests
# ---------------------------------------------------------------------------


class TestTFLiteBackend:
    """Tests for TFLiteBackend (the factory)."""

    def test_name_property(self) -> None:
        with patch(
            "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
            return_value=MagicMock,
        ):
            from violawake_sdk.backends.tflite_backend import TFLiteBackend

            backend = TFLiteBackend()
            assert backend.name == "tflite"

    def test_is_available_true(self) -> None:
        with patch(
            "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
            return_value=MagicMock,
        ):
            from violawake_sdk.backends.tflite_backend import TFLiteBackend

            backend = TFLiteBackend()
            assert backend.is_available() is True

    def test_load_nonexistent_file_raises(self) -> None:
        with patch(
            "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
            return_value=MagicMock,
        ):
            from violawake_sdk.backends.tflite_backend import TFLiteBackend

            backend = TFLiteBackend()
            with pytest.raises(FileNotFoundError):
                backend.load("/nonexistent/model.tflite")

    def test_load_creates_session(self, tmp_path: Path) -> None:
        """load() returns a TFLiteSession when file exists."""
        model_file = tmp_path / "test_model.tflite"
        model_file.write_bytes(b"fake_tflite_content")

        mock_interp = _make_mock_interpreter()
        mock_cls = MagicMock(return_value=mock_interp)

        with patch(
            "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
            return_value=mock_cls,
        ):
            from violawake_sdk.backends.tflite_backend import TFLiteBackend

            backend = TFLiteBackend()
            session = backend.load(model_file)

        assert session is not None
        assert len(session.get_inputs()) == 1
        mock_cls.assert_called_once_with(
            model_path=str(model_file), num_threads=2
        )

    def test_load_model_load_error(self, tmp_path: Path) -> None:
        """load() raises ModelLoadError when interpreter init fails."""
        from violawake_sdk._exceptions import ModelLoadError

        model_file = tmp_path / "broken.tflite"
        model_file.write_bytes(b"broken")

        mock_cls = MagicMock(side_effect=RuntimeError("bad flatbuffer"))

        with patch(
            "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
            return_value=mock_cls,
        ):
            from violawake_sdk.backends.tflite_backend import TFLiteBackend

            backend = TFLiteBackend()
            with pytest.raises(ModelLoadError, match="TFLite failed"):
                backend.load(model_file)

    def test_missing_runtime_error_message(self) -> None:
        from violawake_sdk.backends.tflite_backend import _get_tflite_interpreter_class

        with patch.dict(
            "sys.modules",
            {
                "tflite_runtime": None,
                "tflite_runtime.interpreter": None,
                "tensorflow": None,
                "tensorflow.lite": None,
                "tensorflow.lite.python": None,
                "tensorflow.lite.python.interpreter": None,
            },
        ):
            with pytest.raises(ImportError) as exc_info:
                _get_tflite_interpreter_class()

        message = str(exc_info.value)
        assert "pip install violawake[tflite]" in message
        assert "pip install tflite-runtime>=2.14" in message


# ---------------------------------------------------------------------------
# Backend auto-detection tests
# ---------------------------------------------------------------------------


class TestGetBackend:
    """Tests for the get_backend() factory."""

    def test_get_backend_onnx(self) -> None:
        from violawake_sdk.backends import get_backend

        # onnxruntime is installed in the test env
        backend = get_backend("onnx")
        assert backend.name == "onnx"

    def test_get_backend_tflite_when_available(self) -> None:
        mock_cls = MagicMock()
        with patch(
            "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
            return_value=mock_cls,
        ):
            from violawake_sdk.backends import get_backend

            backend = get_backend("tflite")
            assert backend.name == "tflite"

    def test_get_backend_auto_prefers_onnx(self) -> None:
        from violawake_sdk.backends import get_backend

        backend = get_backend("auto")
        # onnxruntime is installed, so auto should pick it
        assert backend.name == "onnx"

    def test_get_backend_auto_falls_back_to_tflite(self) -> None:
        """When onnxruntime is not importable, auto falls back to tflite."""
        mock_cls = MagicMock()

        with (
            patch.dict("sys.modules", {"onnxruntime": None}),
            patch(
                "violawake_sdk.backends.tflite_backend._get_tflite_interpreter_class",
                return_value=mock_cls,
            ),
        ):
            from violawake_sdk.backends import _auto_select

            backend = _auto_select()
            assert backend.name == "tflite"

    def test_get_backend_unknown_raises(self) -> None:
        from violawake_sdk.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("pytorch")

    def test_get_backend_auto_no_backend_available_message(self) -> None:
        expected = (
            "No inference backend available. Install onnxruntime: pip install "
            "violawake  OR  tflite-runtime: pip install violawake[tflite]"
        )

        with (
            patch("violawake_sdk.backends._make_onnx", side_effect=ImportError),
            patch("violawake_sdk.backends._make_tflite", side_effect=ImportError),
        ):
            from violawake_sdk.backends import _auto_select

            with pytest.raises(ImportError) as exc_info:
                _auto_select()

        assert str(exc_info.value) == expected


# ---------------------------------------------------------------------------
# OnnxBackend tests (sanity check that wrapper works)
# ---------------------------------------------------------------------------


class TestOnnxBackend:
    """Basic tests for the ONNX backend wrapper."""

    def test_name(self) -> None:
        from violawake_sdk.backends.onnx_backend import OnnxBackend

        backend = OnnxBackend()
        assert backend.name == "onnx"

    def test_is_available(self) -> None:
        from violawake_sdk.backends.onnx_backend import OnnxBackend

        backend = OnnxBackend()
        assert backend.is_available() is True

    def test_load_nonexistent_raises(self) -> None:
        from violawake_sdk.backends.onnx_backend import OnnxBackend

        backend = OnnxBackend()
        with pytest.raises(FileNotFoundError):
            backend.load("/nonexistent/model.onnx")


# ---------------------------------------------------------------------------
# Conversion utility tests
# ---------------------------------------------------------------------------


class TestConvertOnnxToTflite:
    """Tests for the ONNX-to-TFLite conversion utility."""

    def test_nonexistent_source_raises(self) -> None:
        from violawake_sdk.backends.tflite_backend import convert_onnx_to_tflite

        with pytest.raises(FileNotFoundError):
            convert_onnx_to_tflite("/nonexistent/model.onnx")

    def test_missing_deps_raises_import_error(self, tmp_path: Path) -> None:
        """When onnx2tf and onnx-tf are both missing, ImportError is raised."""
        from violawake_sdk.backends.tflite_backend import convert_onnx_to_tflite

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake_onnx")

        with (
            patch.dict("sys.modules", {"onnx2tf": None, "onnx_tf": None, "onnx": None}),
            pytest.raises(ImportError, match="onnx2tf|onnx-tf"),
        ):
            convert_onnx_to_tflite(onnx_file)

    def test_default_output_path(self, tmp_path: Path) -> None:
        """Default tflite_path has the same stem as the onnx file."""
        from violawake_sdk.backends.tflite_backend import convert_onnx_to_tflite

        onnx_file = tmp_path / "my_model.onnx"
        onnx_file.write_bytes(b"fake_onnx")

        fake_tflite_bytes = b"fake_tflite_flatbuffer"
        expected_out = tmp_path / "my_model.tflite"

        # Mock the entire conversion chain
        mock_onnx2tf = MagicMock()
        mock_tf = MagicMock()
        mock_converter = MagicMock()
        mock_converter.convert.return_value = fake_tflite_bytes
        mock_tf.lite.TFLiteConverter.from_saved_model.return_value = mock_converter

        with (
            patch.dict("sys.modules", {
                "onnx2tf": mock_onnx2tf,
                "tensorflow": mock_tf,
                "tensorflow.lite": mock_tf.lite,
                "tensorflow.lite.python": MagicMock(),
                "tensorflow.lite.python.interpreter": MagicMock(),
            }),
            patch("shutil.rmtree"),
        ):
            result = convert_onnx_to_tflite(onnx_file)

        assert result == expected_out
        assert expected_out.read_bytes() == fake_tflite_bytes


# ---------------------------------------------------------------------------
# Integration: TFLite session matches ONNX session interface
# ---------------------------------------------------------------------------


class TestInterfaceCompatibility:
    """Verify TFLiteSession and OnnxSession expose the same methods."""

    def test_session_interface_methods(self) -> None:
        from violawake_sdk.backends.base import BackendSession

        required_methods = {"get_inputs", "get_outputs", "run"}
        for method_name in required_methods:
            assert hasattr(BackendSession, method_name), (
                f"BackendSession missing method: {method_name}"
            )

    def test_backend_interface_methods(self) -> None:
        from violawake_sdk.backends.base import InferenceBackend

        required = {"name", "load", "is_available"}
        for attr in required:
            assert hasattr(InferenceBackend, attr), (
                f"InferenceBackend missing: {attr}"
            )

    def test_session_input_output_have_name_and_shape(self) -> None:
        from violawake_sdk.backends.base import SessionInput, SessionOutput

        inp = SessionInput(name="x", shape=[1, 96], type="tensor(float)")
        assert inp.name == "x"
        assert inp.shape == [1, 96]

        out = SessionOutput(name="y", shape=[1, 1], type="tensor(float)")
        assert out.name == "y"
        assert out.shape == [1, 1]


# ---------------------------------------------------------------------------
# WakeDetector backend parameter tests
# ---------------------------------------------------------------------------


class TestWakeDetectorBackendParam:
    """Tests for WakeDetector's ``backend`` parameter integration."""

    def test_invalid_backend_raises_value_error(self) -> None:
        """Passing an unrecognised backend name raises immediately."""
        from violawake_sdk.wake_detector import WakeDetector

        with pytest.raises(ValueError, match="backend must be one of"):
            WakeDetector(backend="pytorch")

    def _make_detector(self, backend_name: str = "onnx") -> None:
        """Helper: create a WakeDetector with fully mocked backend + model resolution."""
        from violawake_sdk.wake_detector import WakeDetector

        mock_session = MagicMock()
        input_meta = MagicMock()
        input_meta.name = "input"
        input_meta.shape = [1, 96]
        mock_session.get_inputs.return_value = [input_meta]

        mock_backend = MagicMock()
        mock_backend.name = backend_name
        mock_backend.load.return_value = mock_session

        fake_path = Path("/fake/model.onnx")

        with (
            patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
            patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock()),
        ):
            detector = WakeDetector(backend=backend_name)
        return detector, mock_backend

    def test_backend_param_accepts_onnx(self) -> None:
        """backend='onnx' is accepted (validation only, model load mocked)."""
        detector, _ = self._make_detector("onnx")
        assert detector._backend.name == "onnx"

    def test_backend_param_accepts_tflite(self) -> None:
        """backend='tflite' is accepted."""
        detector, _ = self._make_detector("tflite")
        assert detector._backend.name == "tflite"

    def test_backend_param_accepts_auto(self) -> None:
        """backend='auto' is the default and is accepted."""
        detector, _ = self._make_detector("auto")
        assert detector._backend.name == "auto"

    def test_backend_passed_to_get_backend(self) -> None:
        """The backend string is forwarded to get_backend() correctly."""
        from violawake_sdk.wake_detector import WakeDetector

        mock_session = MagicMock()
        input_meta = MagicMock()
        input_meta.name = "input"
        input_meta.shape = [1, 96]
        mock_session.get_inputs.return_value = [input_meta]

        mock_backend = MagicMock()
        mock_backend.name = "tflite"
        mock_backend.load.return_value = mock_session

        fake_path = Path("/fake/model.onnx")

        with (
            patch(
                "violawake_sdk.wake_detector.get_backend", return_value=mock_backend
            ) as mock_get,
            patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock()),
        ):
            WakeDetector(backend="tflite")
            mock_get.assert_called_once_with(
                "tflite", providers=["CPUExecutionProvider"]
            )

    def test_load_session_uses_backend_load(self) -> None:
        """_load_session delegates to self._backend.load(), not onnxruntime."""
        _, mock_backend = self._make_detector("onnx")
        assert mock_backend.load.call_count == 1

    def test_resolve_model_path_existing_file(self, tmp_path: Path) -> None:
        """_resolve_model_path returns the path if the file exists."""
        from violawake_sdk.wake_detector import WakeDetector

        model_file = tmp_path / "my_model.onnx"
        model_file.write_bytes(b"fake")

        result = WakeDetector._resolve_model_path(str(model_file))
        assert result == model_file

    def test_resolve_model_path_tflite_extension(self, tmp_path: Path) -> None:
        """_resolve_model_path accepts .tflite files."""
        from violawake_sdk.wake_detector import WakeDetector

        model_file = tmp_path / "my_model.tflite"
        model_file.write_bytes(b"fake")

        result = WakeDetector._resolve_model_path(str(model_file))
        assert result == model_file

    def test_resolve_model_path_missing_onnx_raises(self) -> None:
        """_resolve_model_path raises ModelNotFoundError for missing .onnx."""
        from violawake_sdk._exceptions import ModelNotFoundError
        from violawake_sdk.wake_detector import WakeDetector

        with pytest.raises(ModelNotFoundError, match="Model file not found"):
            WakeDetector._resolve_model_path("/nonexistent/model.onnx")

    def test_resolve_model_path_missing_tflite_raises(self) -> None:
        """_resolve_model_path raises ModelNotFoundError for missing .tflite."""
        from violawake_sdk._exceptions import ModelNotFoundError
        from violawake_sdk.wake_detector import WakeDetector

        with pytest.raises(ModelNotFoundError, match="Model file not found"):
            WakeDetector._resolve_model_path("/nonexistent/model.tflite")

    def test_tflite_backend_prefers_tflite_sibling(self, tmp_path: Path) -> None:
        """When backend is tflite and .onnx is resolved, .tflite sibling is used."""
        from violawake_sdk.wake_detector import WakeDetector

        # Create both .onnx and .tflite files
        onnx_file = tmp_path / "backbone.onnx"
        onnx_file.write_bytes(b"onnx_data")
        tflite_file = tmp_path / "backbone.tflite"
        tflite_file.write_bytes(b"tflite_data")

        mock_session = MagicMock()
        input_meta = MagicMock()
        input_meta.name = "input"
        input_meta.shape = [1, 96]
        mock_session.get_inputs.return_value = [input_meta]

        mock_backend = MagicMock()
        mock_backend.name = "tflite"
        mock_backend.load.return_value = mock_session

        with (
            patch("violawake_sdk.wake_detector.get_backend", return_value=mock_backend),
            patch.object(
                WakeDetector, "_resolve_model_path", return_value=onnx_file
            ),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock()),
        ):
            WakeDetector(backend="tflite")

        # Verify that load was called with the .tflite sibling, not .onnx
        load_calls = mock_backend.load.call_args_list
        for call in load_calls:
            loaded_path = call[0][0]
            assert str(loaded_path).endswith(".tflite"), (
                f"Expected .tflite path, got {loaded_path}"
            )


class TestWakewordDetectorBackendParam:
    """Tests for WakewordDetector's ``backend`` parameter pass-through."""

    def test_backend_param_stored(self) -> None:
        """backend is stored on the WakewordDetector instance."""
        from violawake_sdk.wake_detector import WakewordDetector

        det = WakewordDetector(backend="tflite")
        assert det.backend == "tflite"

    def test_backend_default_is_auto(self) -> None:
        """Default backend is 'auto'."""
        from violawake_sdk.wake_detector import WakewordDetector

        det = WakewordDetector()
        assert det.backend == "auto"

    def test_backend_forwarded_to_wake_detector(self) -> None:
        """backend is passed through to WakeDetector on lazy init."""
        from violawake_sdk.wake_detector import WakeDetector, WakewordDetector

        mock_session = MagicMock()
        input_meta = MagicMock()
        input_meta.name = "input"
        input_meta.shape = [1, 96]
        mock_session.get_inputs.return_value = [input_meta]

        mock_backend = MagicMock()
        mock_backend.name = "tflite"
        mock_backend.load.return_value = mock_session

        fake_path = Path("/fake/model.onnx")

        with (
            patch(
                "violawake_sdk.wake_detector.get_backend", return_value=mock_backend
            ) as mock_get,
            patch.object(WakeDetector, "_resolve_model_path", return_value=fake_path),
            patch.object(WakeDetector, "_create_oww_backbone", return_value=MagicMock()),
        ):
            det = WakewordDetector(backend="tflite")
            det._get_detector()
            mock_get.assert_called_once_with(
                "tflite", providers=["CPUExecutionProvider"]
            )
