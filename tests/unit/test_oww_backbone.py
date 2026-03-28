"""Tests for oww_backbone: _RingBuffer, OpenWakeWordBackbone, and helpers."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from violawake_sdk.oww_backbone import (
    EMBEDDING_DIM,
    MEL_FRAMES_PER_EMBEDDING,
    OWW_CHUNK_SAMPLES,
    SAMPLE_RATE,
    _MAX_RAW_SAMPLES,
    _RingBuffer,
    OpenWakeWordBackbone,
    OpenWakeWordBackbonePaths,
    resolve_openwakeword_backbone_paths,
)
from violawake_sdk._exceptions import ModelNotFoundError


# ──────────────────────────────────────────────────────────────────────────────
# _RingBuffer tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRingBufferBasic:
    """Basic extend / tail operations."""

    def test_empty_buffer_tail(self):
        rb = _RingBuffer(100)
        result = rb.tail(10)
        assert result.shape == (0,)
        assert result.dtype == np.int16

    def test_empty_buffer_count(self):
        rb = _RingBuffer(100)
        assert rb.count == 0

    def test_extend_and_tail(self):
        rb = _RingBuffer(100)
        data = np.arange(20, dtype=np.int16)
        rb.extend(data)
        assert rb.count == 20
        np.testing.assert_array_equal(rb.tail(20), data)

    def test_tail_fewer_than_available(self):
        rb = _RingBuffer(100)
        data = np.arange(50, dtype=np.int16)
        rb.extend(data)
        np.testing.assert_array_equal(rb.tail(10), data[-10:])

    def test_tail_more_than_available(self):
        """Requesting more than stored returns all stored samples."""
        rb = _RingBuffer(100)
        data = np.arange(30, dtype=np.int16)
        rb.extend(data)
        np.testing.assert_array_equal(rb.tail(999), data)

    def test_extend_empty_array(self):
        rb = _RingBuffer(100)
        rb.extend(np.array([], dtype=np.int16))
        assert rb.count == 0

    def test_multiple_extends(self):
        rb = _RingBuffer(100)
        a = np.arange(10, dtype=np.int16)
        b = np.arange(10, 25, dtype=np.int16)
        rb.extend(a)
        rb.extend(b)
        expected = np.arange(25, dtype=np.int16)
        np.testing.assert_array_equal(rb.tail(25), expected)
        assert rb.count == 25


class TestRingBufferWraparound:
    """Wraparound and capacity-limit behavior."""

    def test_wraparound_preserves_order(self):
        """After wrapping, tail() returns samples in chronological order."""
        cap = 20
        rb = _RingBuffer(cap)
        # Write 15, then 10 more — forces wrap.
        rb.extend(np.arange(15, dtype=np.int16))
        rb.extend(np.arange(15, 25, dtype=np.int16))
        # Buffer holds last 20: [5..24]
        expected = np.arange(5, 25, dtype=np.int16)
        np.testing.assert_array_equal(rb.tail(20), expected)
        assert rb.count == cap

    def test_wraparound_partial_tail(self):
        cap = 10
        rb = _RingBuffer(cap)
        rb.extend(np.arange(8, dtype=np.int16))
        rb.extend(np.arange(8, 14, dtype=np.int16))
        # Buffer holds last 10: [4..13]
        np.testing.assert_array_equal(rb.tail(5), np.arange(9, 14, dtype=np.int16))

    def test_exact_capacity_fill(self):
        cap = 16
        rb = _RingBuffer(cap)
        data = np.arange(cap, dtype=np.int16)
        rb.extend(data)
        np.testing.assert_array_equal(rb.tail(cap), data)
        assert rb.count == cap

    def test_overwrite_entire_buffer(self):
        """Extending with data >= capacity keeps only the tail."""
        cap = 10
        rb = _RingBuffer(cap)
        rb.extend(np.arange(5, dtype=np.int16))  # partial fill
        big = np.arange(100, 120, dtype=np.int16)  # 20 > cap
        rb.extend(big)
        expected = big[-cap:]
        np.testing.assert_array_equal(rb.tail(cap), expected)
        assert rb.count == cap

    def test_many_small_extends_match_deque(self):
        """Ring buffer produces the same tail as a maxlen deque."""
        cap = 50
        rb = _RingBuffer(cap)
        dq: deque[int] = deque(maxlen=cap)
        rng = np.random.default_rng(99)

        for _ in range(30):
            chunk_size = rng.integers(1, 20)
            chunk = rng.integers(-32768, 32767, size=chunk_size, dtype=np.int16)
            rb.extend(chunk)
            dq.extend(chunk.tolist())

        deque_arr = np.array(list(dq), dtype=np.int16)
        np.testing.assert_array_equal(rb.tail(rb.count), deque_arr)


class TestRingBufferEdgeCases:
    """Edge cases: capacity 1, dtype, etc."""

    def test_capacity_one(self):
        rb = _RingBuffer(1)
        rb.extend(np.array([42], dtype=np.int16))
        np.testing.assert_array_equal(rb.tail(1), np.array([42], dtype=np.int16))
        rb.extend(np.array([99], dtype=np.int16))
        np.testing.assert_array_equal(rb.tail(1), np.array([99], dtype=np.int16))

    def test_dtype_is_int16(self):
        rb = _RingBuffer(10)
        rb.extend(np.array([1, 2, 3], dtype=np.int16))
        assert rb.tail(3).dtype == np.int16


# ──────────────────────────────────────────────────────────────────────────────
# resolve_openwakeword_backbone_paths tests
# ──────────────────────────────────────────────────────────────────────────────

class TestResolveBackbonePaths:

    def test_raises_when_openwakeword_not_installed(self):
        with patch("violawake_sdk.oww_backbone.importlib.util.find_spec", return_value=None):
            with pytest.raises(ModelNotFoundError, match="openwakeword"):
                resolve_openwakeword_backbone_paths()

    def test_raises_when_models_missing(self, tmp_path):
        spec = MagicMock()
        spec.submodule_search_locations = [str(tmp_path)]
        (tmp_path / "resources" / "models").mkdir(parents=True)
        with patch("violawake_sdk.oww_backbone.importlib.util.find_spec", return_value=spec):
            with pytest.raises(ModelNotFoundError, match="missing"):
                resolve_openwakeword_backbone_paths("onnx")

    def test_resolves_onnx_paths(self, tmp_path):
        models_dir = tmp_path / "resources" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "melspectrogram.onnx").write_bytes(b"fake")
        (models_dir / "embedding_model.onnx").write_bytes(b"fake")
        spec = MagicMock()
        spec.submodule_search_locations = [str(tmp_path)]
        with patch("violawake_sdk.oww_backbone.importlib.util.find_spec", return_value=spec):
            paths = resolve_openwakeword_backbone_paths("onnx")
        assert paths.melspectrogram.name == "melspectrogram.onnx"
        assert paths.embedding_model.name == "embedding_model.onnx"

    def test_resolves_tflite_paths(self, tmp_path):
        models_dir = tmp_path / "resources" / "models"
        models_dir.mkdir(parents=True)
        (models_dir / "melspectrogram.tflite").write_bytes(b"fake")
        (models_dir / "embedding_model.tflite").write_bytes(b"fake")
        spec = MagicMock()
        spec.submodule_search_locations = [str(tmp_path)]
        with patch("violawake_sdk.oww_backbone.importlib.util.find_spec", return_value=spec):
            paths = resolve_openwakeword_backbone_paths("tflite")
        assert paths.melspectrogram.name == "melspectrogram.tflite"


# ──────────────────────────────────────────────────────────────────────────────
# OpenWakeWordBackbone tests (mocked ONNX sessions)
# ──────────────────────────────────────────────────────────────────────────────

def _make_mock_backend(tmp_path):
    """Create a mock backend whose sessions return plausible shapes."""
    models_dir = tmp_path / "resources" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "melspectrogram.onnx").write_bytes(b"fake")
    (models_dir / "embedding_model.onnx").write_bytes(b"fake")

    spec_mock = MagicMock()
    spec_mock.submodule_search_locations = [str(tmp_path)]

    mel_session = MagicMock()
    mel_session.get_inputs.return_value = [MagicMock(name="input")]
    # Return 8 mel frames of 32 bins per 1280-sample chunk (matches MEL_STRIDE=8).
    mel_session.run.return_value = [np.ones((1, 8, 32), dtype=np.float32)]

    emb_session = MagicMock()
    emb_session.get_inputs.return_value = [MagicMock(name="input")]
    emb_session.run.return_value = [np.ones((1, EMBEDDING_DIM), dtype=np.float32) * 0.5]

    backend = MagicMock()
    backend.name = "onnx"
    backend.load.side_effect = [mel_session, emb_session]

    return backend, spec_mock


@pytest.fixture
def backbone(tmp_path):
    backend, spec_mock = _make_mock_backend(tmp_path)
    with patch("violawake_sdk.oww_backbone.importlib.util.find_spec", return_value=spec_mock):
        bb = OpenWakeWordBackbone(backend)
    return bb


class TestBackbonePushAudio:

    def test_no_embedding_before_full_chunk(self, backbone):
        """Frames smaller than OWW_CHUNK_SAMPLES should not produce an embedding."""
        frame = np.zeros(320, dtype=np.int16)  # 20 ms
        produced, emb = backbone.push_audio(frame)
        assert produced is False

    def test_embedding_after_full_chunk(self, backbone):
        """Exactly OWW_CHUNK_SAMPLES (1280) should produce an embedding."""
        frame = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16)
        produced, emb = backbone.push_audio(frame)
        assert produced is True
        assert emb is not None
        assert emb.shape == (EMBEDDING_DIM,)

    def test_bytes_input(self, backbone):
        frame_bytes = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16).tobytes()
        produced, emb = backbone.push_audio(frame_bytes)
        assert produced is True

    def test_float32_input(self, backbone):
        frame = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.float32)
        produced, emb = backbone.push_audio(frame)
        assert produced is True

    def test_multiple_small_frames_accumulate(self, backbone):
        """Four 320-sample frames = 1280 = one chunk."""
        frame = np.zeros(320, dtype=np.int16)
        for _ in range(3):
            produced, _ = backbone.push_audio(frame)
            assert produced is False
        produced, emb = backbone.push_audio(frame)
        assert produced is True
        assert emb.shape == (EMBEDDING_DIM,)

    def test_remainder_carried_across_calls(self, backbone):
        """Non-aligned frames carry remainder to next call."""
        # 500 samples: 0 full chunks, 500 remainder
        backbone.push_audio(np.zeros(500, dtype=np.int16))
        # 780 more: 500+780=1280 = exactly 1 chunk
        produced, emb = backbone.push_audio(np.zeros(780, dtype=np.int16))
        assert produced is True

    def test_large_frame_multiple_chunks(self, backbone):
        """A single large frame spanning multiple chunks."""
        frame = np.zeros(OWW_CHUNK_SAMPLES * 3, dtype=np.int16)
        produced, emb = backbone.push_audio(frame)
        assert produced is True

    def test_last_embedding_persists(self, backbone):
        frame = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16)
        backbone.push_audio(frame)
        emb = backbone.last_embedding
        assert emb is not None
        assert emb.shape == (EMBEDDING_DIM,)

    def test_last_embedding_is_copy(self, backbone):
        frame = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16)
        backbone.push_audio(frame)
        a = backbone.last_embedding
        b = backbone.last_embedding
        assert a is not b
        np.testing.assert_array_equal(a, b)


class TestBackboneReset:

    def test_reset_clears_embedding(self, backbone):
        frame = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16)
        backbone.push_audio(frame)
        assert backbone.last_embedding is not None
        backbone.reset()
        assert backbone.last_embedding is None

    def test_reset_clears_accumulation(self, backbone):
        backbone.push_audio(np.zeros(500, dtype=np.int16))
        backbone.reset()
        # After reset, 1280 samples should produce embedding (no leftover 500).
        produced, emb = backbone.push_audio(np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16))
        assert produced is True


class TestBackboneBufferState:

    def test_raw_buffer_grows(self, backbone):
        frame = np.zeros(OWW_CHUNK_SAMPLES, dtype=np.int16)
        backbone.push_audio(frame)
        assert backbone._raw_data_buffer.count == OWW_CHUNK_SAMPLES

    def test_raw_buffer_capped_at_max(self, backbone):
        """Buffer never exceeds _MAX_RAW_SAMPLES."""
        # Push more than 10s of audio
        big = np.zeros(_MAX_RAW_SAMPLES + OWW_CHUNK_SAMPLES, dtype=np.int16)
        backbone.push_audio(big)
        assert backbone._raw_data_buffer.count <= _MAX_RAW_SAMPLES


class TestToPcmInt16:

    def test_int16_passthrough(self):
        data = np.array([1, 2, 3], dtype=np.int16)
        result = OpenWakeWordBackbone._to_pcm_int16(data)
        np.testing.assert_array_equal(result, data)

    def test_float_normalized(self):
        data = np.array([0.5, -0.5], dtype=np.float32)
        result = OpenWakeWordBackbone._to_pcm_int16(data)
        assert result.dtype == np.int16
        assert result[0] > 0
        assert result[1] < 0

    def test_float_large_range(self):
        data = np.array([10000.0, -10000.0], dtype=np.float32)
        result = OpenWakeWordBackbone._to_pcm_int16(data)
        assert result.dtype == np.int16

    def test_nan_replaced(self):
        data = np.array([np.nan, 0.5], dtype=np.float32)
        result = OpenWakeWordBackbone._to_pcm_int16(data)
        assert result[0] == 0

    def test_int32_clipped(self):
        data = np.array([100000, -100000], dtype=np.int32)
        result = OpenWakeWordBackbone._to_pcm_int16(data)
        assert result[0] == 32767
        assert result[1] == -32768

    def test_unsupported_dtype_raises(self):
        data = np.array(["a", "b"])
        with pytest.raises(ValueError, match="Unsupported"):
            OpenWakeWordBackbone._to_pcm_int16(data)

    def test_bytes_input(self):
        raw = np.array([100, -200], dtype=np.int16).tobytes()
        result = OpenWakeWordBackbone._to_pcm_int16(raw)
        np.testing.assert_array_equal(result, np.array([100, -200], dtype=np.int16))
