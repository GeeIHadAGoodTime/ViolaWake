"""Recording upload and listing routes."""

from __future__ import annotations

import re
import struct
import wave
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.models import Recording, User
from app.schemas import RecordingResponse, RecordingUploadResponse
from app.storage import build_recording_key, get_storage

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

# Limits
MIN_DURATION_S = 0.5
MAX_DURATION_S = 5.0
TARGET_SAMPLE_RATE = 16000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


def _validate_wav(file_bytes: bytes) -> tuple[int, int, int, float]:
    """Validate WAV file and return (sample_rate, channels, sample_width, duration_s).

    Raises:
        HTTPException on invalid WAV data.
    """
    # Check RIFF header
    if len(file_bytes) < 44:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File too small to be a valid WAV")

    if file_bytes[:4] != b"RIFF" or file_bytes[8:12] != b"WAVE":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not a valid WAV file")

    # Parse fmt chunk — walk chunks to find it
    pos = 12
    fmt_found = False
    sample_rate = 0
    channels = 0
    sample_width = 0
    n_frames = 0

    while pos < len(file_bytes) - 8:
        chunk_id = file_bytes[pos:pos + 4]
        chunk_size = struct.unpack_from("<I", file_bytes, pos + 4)[0]

        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Malformed fmt chunk")
            audio_format = struct.unpack_from("<H", file_bytes, pos + 8)[0]
            if audio_format not in (1, 3):  # 1=PCM, 3=IEEE float
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported WAV format ({audio_format}). Only PCM and IEEE float are supported.",
                )
            channels = struct.unpack_from("<H", file_bytes, pos + 10)[0]
            sample_rate = struct.unpack_from("<I", file_bytes, pos + 12)[0]
            bits_per_sample = struct.unpack_from("<H", file_bytes, pos + 22)[0]
            sample_width = bits_per_sample // 8
            fmt_found = True

        elif chunk_id == b"data":
            data_size = chunk_size
            if fmt_found and sample_width > 0 and channels > 0:
                n_frames = data_size // (sample_width * channels)

        pos += 8 + chunk_size
        # Chunks are word-aligned
        if chunk_size % 2 != 0:
            pos += 1

    if not fmt_found:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="WAV file missing fmt chunk")

    if sample_rate == 0 or channels == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid WAV header values")

    duration_s = n_frames / sample_rate if sample_rate > 0 else 0.0

    if duration_s < MIN_DURATION_S:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recording too short ({duration_s:.2f}s). Minimum is {MIN_DURATION_S}s.",
        )
    if duration_s > MAX_DURATION_S:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recording too long ({duration_s:.2f}s). Maximum is {MAX_DURATION_S}s.",
        )

    return sample_rate, channels, sample_width, duration_s


def _ensure_mono_16k(file_bytes: bytes, orig_sr: int, channels: int) -> tuple[bytes, float]:
    """Convert WAV to mono 16kHz if needed. Returns (wav_bytes, duration_s)."""
    import io

    import numpy as np

    # Read raw audio data
    with io.BytesIO(file_bytes) as buf:
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            ch = wf.getnchannels()

    # Decode to float32
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        # Fallback: return as-is
        return file_bytes, len(raw) / (sw * ch * sr)

    # Convert to mono if stereo
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)

    # Resample if not 16kHz
    if sr != TARGET_SAMPLE_RATE:
        from scipy.signal import resample

        new_length = int(len(samples) * TARGET_SAMPLE_RATE / sr)
        samples = resample(samples, new_length)

    duration_s = len(samples) / TARGET_SAMPLE_RATE

    # Encode back to 16-bit PCM WAV
    samples_int16 = np.clip(samples * 32767, -32768, 32767).astype(np.int16)
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(samples_int16.tobytes())

    return out_buf.getvalue(), duration_s


@router.post("/upload", response_model=RecordingUploadResponse)
async def upload_recording(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),
    wake_word: str = Form(...),
) -> RecordingUploadResponse:
    """Upload a WAV recording for a wake word."""
    if not wake_word.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="wake_word is required")

    wake_word = wake_word.strip().lower()

    # Only allow safe characters in wake_word for filesystem safety
    wake_word = re.sub(r'[^a-z0-9 _-]', '', wake_word)
    if not wake_word:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Wake word contains no valid characters")

    # Read file content
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(content)} bytes). Maximum is {MAX_FILE_SIZE} bytes (10 MB).",
        )

    # Validate WAV
    sample_rate, channels, sample_width, duration_s = _validate_wav(content)

    # Convert to mono 16kHz if necessary
    needs_conversion = sample_rate != TARGET_SAMPLE_RATE or channels != 1
    if needs_conversion:
        content, duration_s = _ensure_mono_16k(content, sample_rate, channels)

    storage = get_storage()
    count_result = await db.execute(
        select(func.count())
        .select_from(Recording)
        .where(
            Recording.user_id == current_user.id,
            Recording.wake_word == wake_word,
        )
    )
    idx = int(count_result.scalar_one()) + 1
    filename = f"{wake_word}_{idx:04d}.wav"
    storage_key = build_recording_key(current_user.id, wake_word, filename)

    # Avoid collision
    while storage.exists(storage_key):
        idx += 1
        filename = f"{wake_word}_{idx:04d}.wav"
        storage_key = build_recording_key(current_user.id, wake_word, filename)

    storage.upload(storage_key, content, "audio/wav")

    # Create DB record
    recording = Recording(
        user_id=current_user.id,
        wake_word=wake_word,
        filename=filename,
        file_path=storage_key,
        duration_s=round(duration_s, 3),
        sample_rate=TARGET_SAMPLE_RATE,
    )
    db.add(recording)
    await db.flush()

    return RecordingUploadResponse(
        recording_id=recording.id,
        filename=filename,
        wake_word=wake_word,
        duration_s=round(duration_s, 3),
    )


@router.get("", response_model=list[RecordingResponse])
async def list_recordings(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    wake_word: str | None = Query(default=None),
) -> list[RecordingResponse]:
    """List all recordings for the current user, optionally filtered by wake word."""
    stmt = select(Recording).where(Recording.user_id == current_user.id)
    if wake_word:
        stmt = stmt.where(Recording.wake_word == wake_word.strip().lower())
    stmt = stmt.order_by(Recording.created_at.desc())

    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        RecordingResponse(
            id=r.id,
            wake_word=r.wake_word,
            filename=r.filename,
            duration_s=r.duration_s,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.delete(
    "/{recording_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_recording(
    recording_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Delete a recording. Only the owner may delete their recordings."""
    result = await db.execute(
        select(Recording).where(
            Recording.id == recording_id,
            Recording.user_id == current_user.id,
        )
    )
    recording = result.scalar_one_or_none()
    if recording is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recording not found",
        )

    get_storage().delete(recording.file_path)

    await db.delete(recording)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
