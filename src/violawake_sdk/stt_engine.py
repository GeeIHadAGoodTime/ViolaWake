"""WAV-file-level STT wrapper for STTEngine.

Provides :class:`STTFileEngine` which loads a WAV file and runs transcription,
and a module-level convenience function :func:`transcribe_wav_file`.

Usage::

    from violawake_sdk.stt_engine import STTFileEngine, transcribe_wav_file

    # Class API
    engine = STTFileEngine(model="base")
    text = engine.transcribe_wav("recording.wav")

    # Convenience function
    text = transcribe_wav_file("recording.wav", model="base")

Requirements:
    - scipy (for WAV reading) — included in core dependencies
    - numpy (for array conversion) — included in core dependencies
    - faster-whisper — install with: pip install 'violawake[stt]'
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile  # type: ignore[import]

from violawake_sdk.stt import STTEngine

REQUIRED_SAMPLE_RATE = 16_000


class STTFileEngine:
    """WAV-file transcription wrapper around :class:`STTEngine`.

    Reads a WAV file, converts it to the float32 format expected by
    ``STTEngine.transcribe()``, and returns the transcribed text.

    Example::

        engine = STTFileEngine(model="base")
        text = engine.transcribe_wav("my_recording.wav")
        print(text)  # "what's the weather today"
    """

    def __init__(
        self,
        model: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        """Initialise the engine.

        Args:
            model: Whisper model size (tiny/base/small/medium/large-v3).
            device: Compute device — "cpu" or "cuda".
            compute_type: CTranslate2 precision ("int8", "float16", "float32").
            language: Force a language code (e.g. "en") or None for auto-detect.
        """
        self._stt = STTEngine(
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
        )

    def transcribe_wav(self, path: str | Path) -> str:
        """Transcribe a WAV file and return the text.

        Args:
            path: Path to a mono or stereo WAV file sampled at 16 kHz.

        Returns:
            Transcribed text as a string. Empty string when no speech is detected.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the WAV sample rate is not 16 000 Hz.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"WAV file not found: {path}")

        sample_rate, audio_data = scipy.io.wavfile.read(str(path))

        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f"WAV sample rate must be {REQUIRED_SAMPLE_RATE} Hz, "
                f"got {sample_rate} Hz. "
                f"Resample the file before passing it to STTFileEngine."
            )

        # Convert to float32 in [-1.0, 1.0]
        if audio_data.dtype == np.int16:
            audio_float32 = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.float32:
            audio_float32 = audio_data
        else:
            # Handle int32, uint8, etc. by normalising to [-1, 1]
            info = np.iinfo(str(audio_data.dtype)) if np.issubdtype(audio_data.dtype, np.integer) else None
            if info is not None:
                audio_float32 = audio_data.astype(np.float32) / max(abs(info.min), info.max)
            else:
                audio_float32 = audio_data.astype(np.float32)

        return self._stt.transcribe(audio_float32)


def transcribe_wav_file(path: str | Path, model: str = "base") -> str:
    """Convenience function: transcribe a WAV file in one call.

    Creates a temporary :class:`STTFileEngine` with default settings and
    returns the transcribed text.

    Args:
        path: Path to a 16 kHz mono or stereo WAV file.
        model: Whisper model size (default "base").

    Returns:
        Transcribed text as a string.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the WAV sample rate is not 16 000 Hz.
    """
    engine = STTFileEngine(model=model)
    return engine.transcribe_wav(path)
