"""Unit tests for K6: Streaming audio source abstraction.

Tests FileSource, CallbackSource, and AudioSource protocol compliance
without requiring hardware (no microphone, no network).
"""

from __future__ import annotations

import tempfile
import types
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from violawake_sdk.audio_source import (
    FRAME_BYTES,
    FRAME_SAMPLES,
    AudioSource,
    CallbackSource,
    FileSource,
    MicrophoneSource,
)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int = 16000,
    *,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    """Write a WAV file with int16 PCM data."""
    if sample_width == 1:
        data = ((samples + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    else:
        data = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    if channels > 1:
        data = np.repeat(data[:, None], channels, axis=1).reshape(-1)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


class TestAudioSourceProtocol:
    """Test that concrete sources implement the AudioSource protocol."""

    def test_callback_source_is_audio_source(self) -> None:
        source = CallbackSource()
        assert isinstance(source, AudioSource)

    def test_file_source_is_audio_source(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        try:
            _write_wav(path, np.zeros(320, dtype=np.float32))
            source = FileSource(path)
            assert isinstance(source, AudioSource)
        finally:
            path.unlink(missing_ok=True)


class TestCallbackSource:
    """Test CallbackSource push/pull model."""

    def test_start_stop(self) -> None:
        source = CallbackSource()
        source.start()
        source.stop()

    def test_push_and_read_exact_frame(self) -> None:
        source = CallbackSource(timeout=1.0)
        source.start()
        # Push exactly one frame of int16 PCM bytes
        frame_data = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
        source.push_audio(frame_data)
        result = source.read_frame()
        assert result is not None
        assert len(result) == FRAME_BYTES

    def test_push_numpy_float32(self) -> None:
        source = CallbackSource(timeout=1.0)
        source.start()
        audio = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        source.push_audio(audio)
        result = source.read_frame()
        assert result is not None
        assert len(result) == FRAME_BYTES

    def test_push_numpy_int16(self) -> None:
        source = CallbackSource(timeout=1.0)
        source.start()
        audio = np.zeros(FRAME_SAMPLES, dtype=np.int16)
        source.push_audio(audio)
        result = source.read_frame()
        assert result is not None
        assert len(result) == FRAME_BYTES

    def test_read_timeout_returns_none(self) -> None:
        source = CallbackSource(timeout=0.01)
        source.start()
        result = source.read_frame()
        assert result is None

    def test_read_when_stopped_returns_none(self) -> None:
        source = CallbackSource(timeout=0.01)
        # Not started
        assert source.read_frame() is None

    def test_push_when_stopped_is_noop(self) -> None:
        source = CallbackSource()
        # Not started, push should be silently ignored
        source.push_audio(np.zeros(320, dtype=np.int16).tobytes())

    def test_reassembly_of_partial_frames(self) -> None:
        source = CallbackSource(timeout=1.0)
        source.start()
        # Push half a frame
        half = np.zeros(FRAME_SAMPLES // 2, dtype=np.int16).tobytes()
        source.push_audio(half)
        # read should block/timeout since not enough data for a frame
        # Push the other half
        source.push_audio(half)
        result = source.read_frame()
        assert result is not None
        assert len(result) == FRAME_BYTES

    def test_large_push_yields_multiple_frames(self) -> None:
        source = CallbackSource(timeout=1.0)
        source.start()
        # Push 3 frames worth of data
        data = np.zeros(FRAME_SAMPLES * 3, dtype=np.int16).tobytes()
        source.push_audio(data)
        # Should be able to read 3 frames
        for _ in range(3):
            result = source.read_frame()
            assert result is not None
            assert len(result) == FRAME_BYTES

    def test_queue_overflow_drops_oldest(self) -> None:
        source = CallbackSource(timeout=0.1, max_queue_size=2)
        source.start()
        # Push 5 frames (max_queue_size=2, so 3 should be dropped)
        for i in range(5):
            data = np.full(FRAME_SAMPLES, i, dtype=np.int16).tobytes()
            source.push_audio(data)
        # Should still be able to read (at least 1 frame in the queue)
        result = source.read_frame()
        assert result is not None

    def test_stop_drains_queue(self) -> None:
        source = CallbackSource(timeout=0.1)
        source.start()
        source.push_audio(np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes())
        source.stop()
        # After stop, read should return None
        assert source.read_frame() is None


class TestFileSource:
    """Test FileSource reading from WAV files."""

    def test_read_single_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            # Create a WAV with exactly one frame (320 samples)
            _write_wav(path, np.zeros(FRAME_SAMPLES, dtype=np.float32))
            source = FileSource(path)
            source.start()
            frame = source.read_frame()
            assert frame is not None
            assert len(frame) == FRAME_BYTES
            # Next read should return None (exhausted)
            assert source.read_frame() is None
            source.stop()

    def test_read_multiple_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            # Create 5 frames worth of audio
            _write_wav(path, np.zeros(FRAME_SAMPLES * 5, dtype=np.float32))
            source = FileSource(path)
            source.start()
            frames = []
            while True:
                frame = source.read_frame()
                if frame is None:
                    break
                frames.append(frame)
            assert len(frames) == 5
            source.stop()

    def test_loop_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            _write_wav(path, np.zeros(FRAME_SAMPLES, dtype=np.float32))
            source = FileSource(path, loop=True)
            source.start()
            # Should loop: read more than file length
            for _ in range(10):
                frame = source.read_frame()
                assert frame is not None
                assert len(frame) == FRAME_BYTES
            source.stop()

    def test_nonexistent_file_raises(self) -> None:
        source = FileSource("/nonexistent/file.wav")
        with pytest.raises(FileNotFoundError):
            source.start()

    def test_start_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            _write_wav(path, np.zeros(320, dtype=np.float32))
            source = FileSource(path)
            source.start()
            source.stop()
            # After stop, read returns None
            assert source.read_frame() is None

    def test_partial_frame_padded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.wav"
            # Create audio with 160 samples (half a frame)
            _write_wav(path, np.zeros(160, dtype=np.float32))
            source = FileSource(path)
            source.start()
            frame = source.read_frame()
            # Should be padded to full frame size
            assert frame is not None
            assert len(frame) == FRAME_BYTES
            source.stop()

    @pytest.mark.parametrize("sample_rate", [8000, 22050, 48000])
    def test_rejects_wrong_wav_sample_rate(self, sample_rate: int) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "wrong-rate.wav"
            _write_wav(path, np.zeros(FRAME_SAMPLES, dtype=np.float32), sample_rate=sample_rate)
            source = FileSource(path)
            with pytest.raises(ValueError, match="audio contract"):
                source.start()

    def test_rejects_stereo_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stereo.wav"
            _write_wav(path, np.zeros(FRAME_SAMPLES, dtype=np.float32), channels=2)
            source = FileSource(path)
            with pytest.raises(ValueError, match="audio contract"):
                source.start()

    def test_rejects_non_16bit_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "eight-bit.wav"
            _write_wav(
                path,
                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                sample_width=1,
            )
            source = FileSource(path)
            with pytest.raises(ValueError, match="audio contract"):
                source.start()

    def test_read_non_wav_via_soundfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.flac"
            path.write_bytes(b"fake-flac")

            class FakeSoundFile:
                def __init__(self, *_args, **_kwargs) -> None:
                    self.samplerate = 16000
                    self.channels = 1
                    self._cursor = 0
                    self._audio = np.zeros((FRAME_SAMPLES, 1), dtype=np.int16)

                def read(self, frames: int, dtype: str = "int16", always_2d: bool = True) -> np.ndarray:
                    assert dtype == "int16"
                    assert always_2d is True
                    chunk = self._audio[self._cursor:self._cursor + frames]
                    self._cursor += len(chunk)
                    return chunk

                def seek(self, position: int) -> None:
                    self._cursor = position

                def close(self) -> None:
                    return None

            fake_soundfile = types.ModuleType("soundfile")
            fake_soundfile.SoundFile = FakeSoundFile

            with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
                source = FileSource(path)
                source.start()
                frame = source.read_frame()
                assert frame is not None
                assert len(frame) == FRAME_BYTES
                assert source.read_frame() is None
                source.stop()

    def test_rejects_non_wav_wrong_sample_rate_via_soundfile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.flac"
            path.write_bytes(b"fake-flac")

            class FakeSoundFile:
                def __init__(self, *_args, **_kwargs) -> None:
                    self.samplerate = 48000
                    self.channels = 1

                def close(self) -> None:
                    return None

            fake_soundfile = types.ModuleType("soundfile")
            fake_soundfile.SoundFile = FakeSoundFile

            with patch.dict("sys.modules", {"soundfile": fake_soundfile}):
                source = FileSource(path)
                with pytest.raises(ValueError, match="audio contract"):
                    source.start()

    def test_non_wav_without_soundfile_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.flac"
            path.write_bytes(b"fake-flac")

            source = FileSource(path)
            with (
                patch.dict("sys.modules", {"soundfile": None}),
                pytest.raises(ImportError, match="pip install soundfile"),
            ):
                source.start()


class TestNetworkSource:
    """Test NetworkSource (protocol validation only, no actual network)."""

    def test_invalid_protocol_raises(self) -> None:
        from violawake_sdk.audio_source import NetworkSource
        source = NetworkSource(protocol="http")
        with pytest.raises(ValueError, match="Unsupported protocol"):
            source.start()


class TestMicrophoneSourceContract:
    """MicrophoneSource must not open non-contract audio streams."""

    @pytest.mark.parametrize("sample_rate", [8000, 22050, 48000])
    def test_rejects_wrong_sample_rate(self, sample_rate: int) -> None:
        with pytest.raises(ValueError, match="audio contract"):
            MicrophoneSource(sample_rate=sample_rate)

    def test_rejects_wrong_frame_samples(self) -> None:
        with pytest.raises(ValueError, match="20ms frame"):
            MicrophoneSource(frame_samples=960)
