"""K6: Streaming audio source abstraction.

Provides an AudioSource protocol with concrete implementations for:
- Microphone (PyAudio)
- File (WAV via ``wave``; FLAC/non-WAV optional via ``soundfile``)
- Network stream (TCP/UDP)
- Callback-based (push model)

WakeDetector.from_source() factory integrates any AudioSource.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pyaudio

import contextlib

import numpy as np

from violawake_sdk._constants import SAMPLE_RATE

logger = logging.getLogger(__name__)

# Standard frame size: 20ms at 16kHz = 320 samples = 640 bytes (int16)
FRAME_SAMPLES = 320
FRAME_BYTES = FRAME_SAMPLES * 2  # int16 = 2 bytes per sample


@runtime_checkable
class AudioSource(Protocol):
    """Protocol for streaming audio sources.

    Any class implementing these three methods can be used with
    WakeDetector.from_source().
    """

    def read_frame(self) -> bytes | None:
        """Read one 20ms frame of 16kHz mono int16 PCM audio.

        Returns:
            640 bytes of audio data, or None if the source is exhausted.
        """
        ...

    def start(self) -> None:
        """Start the audio source (open device, connect, etc.)."""
        ...

    def stop(self) -> None:
        """Stop the audio source and release resources."""
        ...


class MicrophoneSource:
    """Audio source from the system microphone via PyAudio.

    Args:
        device_index: PyAudio device index. None = system default.
        sample_rate: Sample rate in Hz. Default 16000.
        frame_samples: Samples per frame. Default 320 (20ms).
    """

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_samples: int = FRAME_SAMPLES,
    ) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._frame_samples = frame_samples
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def start(self) -> None:
        """Open the microphone stream."""
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio is required for MicrophoneSource. "
                "Install with: pip install violawake[audio]"
            ) from None

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._sample_rate,
            input=True,
            frames_per_buffer=self._frame_samples,
            input_device_index=self._device_index,
        )
        logger.info("MicrophoneSource started (device=%s)", self._device_index)

    def read_frame(self) -> bytes | None:
        """Read one frame from the microphone."""
        if self._stream is None:
            return None
        try:
            return self._stream.read(self._frame_samples, exception_on_overflow=False)  # type: ignore[union-attr]
        except Exception as e:
            logger.warning("Microphone read error: %s", e)
            return None

    def stop(self) -> None:
        """Close the microphone stream."""
        if self._stream is not None:
            try:
                self._stream.stop_stream()  # type: ignore[union-attr]
                self._stream.close()  # type: ignore[union-attr]
            except OSError:
                pass
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()  # type: ignore[union-attr]
            self._pa = None
        logger.info("MicrophoneSource stopped")


class FileSource:
    """Audio source that reads from a local audio file.

    WAV files only (16-bit PCM, 16kHz mono) are supported via Python's
    standard-library ``wave`` module. If optional ``soundfile`` is installed,
    non-WAV files such as FLAC can also be decoded and streamed.

    Reads the file in 20ms chunks, simulating real-time audio input.
    Returns None when the file is exhausted.

    Args:
        path: Path to a WAV file (16-bit PCM, 16kHz mono). Non-WAV formats
              such as FLAC require the optional ``soundfile`` package.
        loop: If True, restart from beginning when file ends. Default False.
    """

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self._path = Path(path)
        self._loop = loop
        self._wf: wave.Wave_read | None = None
        self._sf: object | None = None
        self._buffer = b""

    def start(self) -> None:
        """Open the audio file."""
        if not self._path.exists():
            raise FileNotFoundError(f"Audio file not found: {self._path}")

        self._buffer = b""

        if self._path.suffix.lower() == ".wav":
            self._wf = wave.open(str(self._path), "rb")  # noqa: SIM115
            sr = self._wf.getframerate()
            ch = self._wf.getnchannels()
            sw = self._wf.getsampwidth()
            if sr != SAMPLE_RATE or ch != 1 or sw != 2:
                logger.warning(
                    "FileSource: %s is %dHz/%dch/%dbit; expected 16kHz/1ch/16bit. "
                    "Audio may not be interpreted correctly.",
                    self._path.name,
                    sr,
                    ch,
                    sw * 8,
                )
        else:
            try:
                import soundfile as sf
            except ImportError:
                raise ImportError(
                    "soundfile is required for non-WAV audio files such as FLAC. "
                    "Install with: pip install soundfile"
                ) from None

            self._sf = sf.SoundFile(str(self._path), "r")
            sr = self._sf.samplerate  # type: ignore[attr-defined]
            ch = self._sf.channels  # type: ignore[attr-defined]
            if sr != SAMPLE_RATE or ch != 1:
                logger.warning(
                    "FileSource: %s is %dHz/%dch; expected 16kHz/1ch. "
                    "Audio will be decoded but should be resampled externally.",
                    self._path.name,
                    sr,
                    ch,
                )

        logger.info("FileSource started: %s", self._path.name)

    def read_frame(self) -> bytes | None:
        """Read one 20ms frame from the file."""
        if self._wf is None and self._sf is None:
            return None

        while len(self._buffer) < FRAME_BYTES:
            data = self._read_chunk()
            if not data:
                if self._loop:
                    self._rewind()
                    data = self._read_chunk()
                    if not data:
                        return None
                else:
                    if self._buffer:
                        # Pad remaining with silence
                        padded = self._buffer + b"\x00" * (FRAME_BYTES - len(self._buffer))
                        self._buffer = b""
                        return padded
                    return None
            self._buffer += data

        frame = self._buffer[:FRAME_BYTES]
        self._buffer = self._buffer[FRAME_BYTES:]
        return frame

    def _read_chunk(self) -> bytes:
        if self._wf is not None:
            return self._wf.readframes(FRAME_SAMPLES)
        if self._sf is None:
            return b""

        data = self._sf.read(FRAME_SAMPLES, dtype="int16", always_2d=True)  # type: ignore[attr-defined]
        if data.size == 0:
            return b""
        if data.shape[1] > 1:
            data = data.astype(np.int32).mean(axis=1).astype(np.int16)
        else:
            data = data[:, 0]
        return np.ascontiguousarray(data, dtype=np.int16).tobytes()

    def _rewind(self) -> None:
        if self._wf is not None:
            self._wf.rewind()
        elif self._sf is not None:
            self._sf.seek(0)  # type: ignore[attr-defined]

    def stop(self) -> None:
        """Close the audio file."""
        if self._wf is not None:
            self._wf.close()
            self._wf = None
        if self._sf is not None:
            self._sf.close()  # type: ignore[attr-defined]
            self._sf = None
        self._buffer = b""
        logger.info("FileSource stopped")


class NetworkSource:
    """Audio source from a TCP or UDP network stream.

    Expects raw 16kHz mono int16 PCM data on the wire (no framing protocol).

    .. warning:: **Security: No Authentication or Encryption**

       This class opens a plain TCP/UDP socket with NO authentication,
       encryption, or sender validation:

       - **TCP mode** binds to the specified address and accepts the first
         incoming connection without verifying the client's identity.
       - **UDP mode** receives datagrams from ANY sender on the bound
         port without validation.

       **Intended for trusted local networks only** (loopback or a
       physically-secured LAN). Do NOT bind to ``0.0.0.0`` or a public
       interface in untrusted environments.

       For production deployments requiring security, consider:

       - Adding application-level authentication (shared secret, token).
       - Using Unix domain sockets (eliminates network exposure entirely).
       - Wrapping the connection with TLS.

    Args:
        host: Hostname or IP to listen on. Default "127.0.0.1" (loopback only).
        port: Port number to listen on.
        protocol: "tcp" or "udp". Default "tcp".
        timeout: Socket receive timeout in seconds. Default 5.0.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        protocol: str = "tcp",
        timeout: float = 5.0,
    ) -> None:
        self._host = host
        self._port = port
        self._protocol = protocol.lower()
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._conn: socket.socket | None = None
        self._buffer = b""

    def start(self) -> None:
        """Start listening for connections (TCP) or datagrams (UDP).

        Note: For TCP mode, start() blocks until a client connects or timeout
        is reached.
        """
        if self._protocol == "tcp":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self._host, self._port))
            self._sock.listen(1)
            self._sock.settimeout(self._timeout)
            logger.info("NetworkSource (TCP) listening on %s:%d", self._host, self._port)
            # Accept first connection
            try:
                self._conn, addr = self._sock.accept()
                self._conn.settimeout(self._timeout)
                logger.info("NetworkSource: client connected from %s", addr)
            except TimeoutError:
                logger.warning("NetworkSource: no client connected within timeout")
        elif self._protocol == "udp":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((self._host, self._port))
            self._sock.settimeout(self._timeout)
            logger.info("NetworkSource (UDP) listening on %s:%d", self._host, self._port)
        else:
            raise ValueError(f"Unsupported protocol: {self._protocol}. Use 'tcp' or 'udp'.")

    def read_frame(self) -> bytes | None:
        """Read one 20ms frame from the network stream."""
        if self._sock is None:
            return None
        try:
            while len(self._buffer) < FRAME_BYTES:
                if self._protocol == "tcp":
                    if self._conn is None:
                        return None
                    data = self._conn.recv(FRAME_BYTES - len(self._buffer))
                    if not data:
                        return None  # Connection closed
                else:
                    data, _ = self._sock.recvfrom(4096)
                self._buffer += data
        except TimeoutError:
            return None
        except Exception as e:
            logger.warning("NetworkSource read error: %s", e)
            return None

        frame = self._buffer[:FRAME_BYTES]
        self._buffer = self._buffer[FRAME_BYTES:]
        return frame

    def stop(self) -> None:
        """Close network sockets."""
        if self._conn is not None:
            with contextlib.suppress(OSError):
                self._conn.close()
            self._conn = None
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None
        self._buffer = b""
        logger.info("NetworkSource stopped")


class CallbackSource:
    """Audio source that receives audio via push callbacks.

    Call ``push_audio(data)`` from your audio pipeline to feed frames.
    ``read_frame()`` blocks until a frame is available (with timeout).

    Args:
        timeout: Maximum seconds to wait for a frame in read_frame(). Default 1.0.
        max_queue_size: Maximum queued frames before dropping. Default 100.
    """

    def __init__(
        self,
        timeout: float = 1.0,
        max_queue_size: int = 100,
    ) -> None:
        self._timeout = timeout
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=max_queue_size)
        self._buffer = b""
        self._buffer_lock = threading.Lock()
        self._running = False

    def start(self) -> None:
        """Mark the source as active."""
        self._running = True
        logger.info("CallbackSource started")

    def push_audio(self, data: bytes | np.ndarray) -> None:
        """Push audio data into the source.

        Args:
            data: Raw int16 PCM bytes, or numpy array (int16 or float32).
                  Any size is accepted; frames are reassembled internally.
        """
        if not self._running:
            return

        if isinstance(data, np.ndarray):
            if data.dtype == np.float32 or data.dtype == np.float64:
                # Use 32768.0 for symmetric float-to-int16 range: [-1.0, 1.0) -> [-32768, 32767]
                data = (data * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
            elif data.dtype == np.int16:
                data = data.tobytes()
            else:
                data = data.astype(np.int16).tobytes()

        # Buffer incoming data and enqueue complete frames
        with self._buffer_lock:
            self._buffer += data
            while len(self._buffer) >= FRAME_BYTES:
                frame = self._buffer[:FRAME_BYTES]
                self._buffer = self._buffer[FRAME_BYTES:]
                try:
                    self._queue.put_nowait(frame)
                except queue.Full:
                    # Drop oldest frame to make room
                    with contextlib.suppress(queue.Empty):
                        self._queue.get_nowait()
                    with contextlib.suppress(queue.Full):
                        self._queue.put_nowait(frame)

    def read_frame(self) -> bytes | None:
        """Read one 20ms frame, blocking up to timeout seconds."""
        if not self._running:
            return None
        try:
            return self._queue.get(timeout=self._timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        """Stop accepting audio and drain the queue."""
        self._running = False
        # Drain
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._buffer_lock:
            self._buffer = b""
        logger.info("CallbackSource stopped")
