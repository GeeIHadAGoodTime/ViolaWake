"""OpenWakeWord backbone resolution and streaming embedding extraction."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from violawake_sdk._exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from violawake_sdk.backends.base import InferenceBackend

SAMPLE_RATE = 16_000
MEL_FRAMES_PER_EMBEDDING = 76
MEL_STRIDE = 8
EMBEDDING_DIM = 96
OWW_CHUNK_SAMPLES = 1_280
_OWW_MELSPEC_CONTEXT_SAMPLES = 160 * 3
# 10 seconds of raw audio at 16 kHz — rolling window for mel context.
_MAX_RAW_SAMPLES = SAMPLE_RATE * 10
# 10 seconds of mel frames at ~97 frames/sec — bounds spectrogram buffer growth.
_MAX_MELSPEC_FRAMES = 10 * 97


class _RingBuffer:
    """Fixed-capacity ring buffer backed by a pre-allocated numpy int16 array.

    Avoids the Python-object overhead of ``deque.extend(array.tolist())`` by
    keeping samples in contiguous numpy memory and using a write pointer that
    wraps around.
    """

    __slots__ = ("_buf", "_capacity", "_write_pos", "_count")

    def __init__(self, capacity: int) -> None:
        self._buf = np.zeros(capacity, dtype=np.int16)
        self._capacity = capacity
        self._write_pos = 0
        self._count = 0

    @property
    def count(self) -> int:
        """Number of samples currently stored."""
        return self._count

    def extend(self, data: np.ndarray) -> None:
        """Append *data* (int16 ndarray) to the ring buffer."""
        n = data.shape[0]
        if n == 0:
            return

        if n >= self._capacity:
            # Data larger than buffer — keep only the tail.
            self._buf[:] = data[-self._capacity:]
            self._write_pos = 0
            self._count = self._capacity
            return

        end = self._write_pos + n
        if end <= self._capacity:
            self._buf[self._write_pos:end] = data
        else:
            first = self._capacity - self._write_pos
            self._buf[self._write_pos:] = data[:first]
            self._buf[:n - first] = data[first:]

        self._write_pos = end % self._capacity
        self._count = min(self._count + n, self._capacity)

    def tail(self, n: int) -> np.ndarray:
        """Return the last *n* samples in chronological order.

        If fewer than *n* samples have been written, returns all available
        samples.
        """
        n = min(n, self._count)
        if n == 0:
            return np.empty(0, dtype=np.int16)

        start = (self._write_pos - n) % self._capacity
        if start + n <= self._capacity:
            return self._buf[start:start + n].copy()

        # Wraps around — two slices.
        return np.concatenate((self._buf[start:], self._buf[:self._write_pos]))


@dataclass(frozen=True)
class OpenWakeWordBackbonePaths:
    """Resolved OpenWakeWord backbone asset paths."""

    melspectrogram: Path
    embedding_model: Path


def resolve_openwakeword_backbone_paths(backend_name: str = "onnx") -> OpenWakeWordBackbonePaths:
    """Resolve the installed OpenWakeWord backbone assets for the chosen backend."""
    spec = importlib.util.find_spec("openwakeword")
    if spec is None or not spec.submodule_search_locations:
        raise ModelNotFoundError(
            "OpenWakeWord is required for wake word detection. "
            "Install with: pip install openwakeword"
        )

    extension = ".tflite" if backend_name == "tflite" else ".onnx"
    resources_dir = Path(spec.submodule_search_locations[0]) / "resources" / "models"
    melspectrogram = resources_dir / f"melspectrogram{extension}"
    embedding_model = resources_dir / f"embedding_model{extension}"

    missing = [path.name for path in (melspectrogram, embedding_model) if not path.exists()]
    if missing:
        missing_str = ", ".join(missing)
        raise ModelNotFoundError(
            "OpenWakeWord backbone files are missing from the installed package: "
            f"{missing_str}. Reinstall or upgrade openwakeword."
        )

    return OpenWakeWordBackbonePaths(
        melspectrogram=melspectrogram,
        embedding_model=embedding_model,
    )


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_openwakeword_backbone_hashes(backend_name: str = "onnx") -> dict[str, str]:
    """Return SHA-256 hashes for the installed OpenWakeWord backbone files."""
    paths = resolve_openwakeword_backbone_paths(backend_name)
    return {
        "oww_mel_sha256": _sha256_file(paths.melspectrogram),
        "oww_emb_sha256": _sha256_file(paths.embedding_model),
    }


class OpenWakeWordBackbone:
    """Streaming wrapper around OpenWakeWord's mel + embedding backbone."""

    def __init__(self, backend: InferenceBackend) -> None:
        paths = resolve_openwakeword_backbone_paths(backend.name)
        self._melspec_session = backend.load(paths.melspectrogram)
        self._embedding_session = backend.load(paths.embedding_model)
        self._melspec_input_name = self._melspec_session.get_inputs()[0].name
        self._embedding_input_name = self._embedding_session.get_inputs()[0].name
        self.reset()

    @property
    def last_embedding(self) -> np.ndarray | None:
        """Return the most recent 96-d embedding, if any."""
        if self._last_embedding is None:
            return None
        return self._last_embedding.copy()

    def reset(self) -> None:
        """Reset streaming buffers and cached embeddings."""
        self._raw_data_buffer = _RingBuffer(_MAX_RAW_SAMPLES)
        self._melspectrogram_buffer = np.ones(
            (MEL_FRAMES_PER_EMBEDDING, 32), dtype=np.float32,
        )
        self._accumulated_samples = 0
        self._raw_data_remainder = np.empty(0, dtype=np.int16)
        self._last_embedding: np.ndarray | None = None

    def push_audio(self, audio_frame: bytes | np.ndarray) -> tuple[bool, np.ndarray | None]:
        """Buffer audio and return the newest embedding when one is produced."""
        pcm_i16 = self._to_pcm_int16(audio_frame)

        if self._raw_data_remainder.size:
            pcm_i16 = np.concatenate((self._raw_data_remainder, pcm_i16))
            self._raw_data_remainder = np.empty(0, dtype=np.int16)

        remainder = (self._accumulated_samples + pcm_i16.shape[0]) % OWW_CHUNK_SAMPLES
        data_to_buffer = pcm_i16[:-remainder] if remainder else pcm_i16
        self._buffer_raw_data(data_to_buffer)
        self._accumulated_samples += data_to_buffer.shape[0]
        if remainder:
            self._raw_data_remainder = pcm_i16[-remainder:].copy()

        new_embeddings: list[np.ndarray] = []
        if self._accumulated_samples >= OWW_CHUNK_SAMPLES and self._accumulated_samples % OWW_CHUNK_SAMPLES == 0:
            self._streaming_melspectrogram(self._accumulated_samples)

            n_chunks = self._accumulated_samples // OWW_CHUNK_SAMPLES
            # Iterate chunks in reverse (newest-first) so each chunk_index maps
            # to a negative mel offset: chunk 0 = buffer tail, chunk N = N*MEL_STRIDE
            # frames back. end_index converts the offset to an absolute slice bound.
            for chunk_index in range(n_chunks - 1, -1, -1):
                offset = -MEL_STRIDE * chunk_index
                end_index = offset if offset != 0 else len(self._melspectrogram_buffer)
                window = self._melspectrogram_buffer[
                    -MEL_FRAMES_PER_EMBEDDING + end_index:end_index
                ]
                if window.shape[0] == MEL_FRAMES_PER_EMBEDDING:
                    embedding = self._predict_embedding(window)
                    self._last_embedding = embedding
                    new_embeddings.append(embedding)
                else:
                    logger.debug(
                        "Dropped embedding: insufficient mel context (%d/%d frames)",
                        window.shape[0],
                        MEL_FRAMES_PER_EMBEDDING,
                    )

            self._accumulated_samples = 0

        if new_embeddings:
            return True, new_embeddings[-1]
        return False, self.last_embedding

    def _buffer_raw_data(self, pcm_i16: np.ndarray) -> None:
        self._raw_data_buffer.extend(pcm_i16)

    def _streaming_melspectrogram(self, n_samples: int) -> None:
        raw_window = self._raw_data_buffer.tail(n_samples + _OWW_MELSPEC_CONTEXT_SAMPLES)
        melspectrogram = self._predict_melspectrogram(raw_window)
        self._melspectrogram_buffer = np.vstack((self._melspectrogram_buffer, melspectrogram))[-_MAX_MELSPEC_FRAMES:, :]

    def _predict_melspectrogram(self, pcm_i16: np.ndarray) -> np.ndarray:
        batch = pcm_i16.astype(np.float32)[None, :]
        outputs = self._melspec_session.run(None, {self._melspec_input_name: batch})
        spec = np.squeeze(outputs[0]).astype(np.float32)
        # Matches OpenWakeWord's log-mel normalization: divide by 10.0 and add 2.0
        # to shift into the expected input range for the embedding model.
        return spec / 10.0 + 2.0

    def _predict_embedding(self, melspectrogram: np.ndarray) -> np.ndarray:
        batch = melspectrogram.astype(np.float32)[None, :, :, None]
        outputs = self._embedding_session.run(None, {self._embedding_input_name: batch})
        return np.asarray(outputs[0]).reshape(EMBEDDING_DIM).astype(np.float32)

    @staticmethod
    def _to_pcm_int16(audio_frame: bytes | np.ndarray) -> np.ndarray:
        if isinstance(audio_frame, bytes):
            return np.frombuffer(audio_frame, dtype=np.int16).copy()

        pcm = np.asarray(audio_frame)
        if pcm.dtype == np.int16:
            return pcm

        if np.issubdtype(pcm.dtype, np.floating):
            pcm = np.where(np.isfinite(pcm), pcm, 0.0)
            peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
            if peak <= 1.5:
                pcm = np.clip(pcm, -1.0, 1.0) * 32767.0
            else:
                pcm = np.clip(pcm, -32768.0, 32767.0)
            return pcm.astype(np.int16)

        if np.issubdtype(pcm.dtype, np.integer):
            return np.clip(pcm, -32768, 32767).astype(np.int16)

        raise ValueError(f"Unsupported audio dtype for OpenWakeWord backbone: {pcm.dtype}")
