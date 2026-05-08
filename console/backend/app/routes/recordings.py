"""Recording upload and listing routes."""

from __future__ import annotations

import io
import re
import tempfile
import uuid
import wave
from pathlib import Path
from typing import Annotated

import numpy as np

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_verified_user
from app.database import get_db
from app.models import Recording, User
from app.rate_limit import RECORDING_UPLOAD_LIMIT, consume_rate_limit, key_by_user, set_rate_limit_user
from app.schemas import (
    RecordingBulkUploadItem,
    RecordingBulkUploadResponse,
    RecordingCountResponse,
    RecordingResponse,
    RecordingUploadResponse,
)
from app.storage import build_recording_key, get_storage

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0
TARGET_SAMPLE_RATE = 16000
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_FILES_PER_REQUEST = 50
MAX_RECORDINGS_PER_USER = 500
MIN_RMS_ENERGY = 10.0
MIN_RMS_FLOAT = MIN_RMS_ENERGY / 32767.0
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
RECORDING_UPLOAD_SCOPE = "recordings-upload"


async def _verified_user_with_rate_key(
    request: Request,
    current_user: Annotated[User, Depends(get_verified_user)],
) -> User:
    """Resolve the user and stash the ID on request.state for rate limiting."""
    set_rate_limit_user(request, current_user.id)
    return current_user


def _normalize_wake_word(wake_word: str) -> str:
    if not wake_word.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="wake_word is required")

    normalized = wake_word.strip().lower()
    normalized = re.sub(r"[^a-z0-9 _-]", "", normalized)
    if not normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Wake word contains no valid characters")
    return normalized


def _safe_upload_filename(file: UploadFile) -> str:
    return Path(file.filename or "upload").name or "upload"


def _validate_extension(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Upload one of: {allowed}.",
        )
    return extension


def _raise_duration_error(duration_s: float) -> None:
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


def _encode_mono_wav(samples: np.ndarray) -> bytes:
    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    out_buf = io.BytesIO()
    with wave.open(out_buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SAMPLE_RATE)
        wf.writeframes(samples_int16.tobytes())
    return out_buf.getvalue()


def _load_with_soundfile(file_bytes: bytes) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
    samples = np.asarray(data, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        import librosa

        samples = librosa.resample(samples, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)
    return np.asarray(samples, dtype=np.float32)


def _load_with_librosa(file_bytes: bytes, extension: str) -> np.ndarray:
    import librosa

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_file.write(file_bytes)
            temp_path = Path(temp_file.name)
        samples, _sample_rate = librosa.load(str(temp_path), sr=TARGET_SAMPLE_RATE, mono=True)
        return np.asarray(samples, dtype=np.float32)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _decode_audio_to_wav(file_bytes: bytes, extension: str) -> tuple[bytes, float]:
    try:
        try:
            samples = _load_with_soundfile(file_bytes)
        except Exception:
            samples = _load_with_librosa(file_bytes, extension)
    except Exception as exc:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode audio file. Supported formats: {allowed}.",
        ) from exc

    if samples.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file contains no samples.")
    if not np.all(np.isfinite(samples)):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Audio file contains invalid samples.")

    duration_s = float(len(samples) / TARGET_SAMPLE_RATE)
    _raise_duration_error(duration_s)

    rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
    if rms < MIN_RMS_FLOAT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Recording appears to be silent. Please check the audio and try again.",
        )

    return _encode_mono_wav(samples), duration_s


async def _read_audio_upload(file: UploadFile) -> tuple[str, str, bytes]:
    original_filename = _safe_upload_filename(file)
    extension = _validate_extension(original_filename)

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large ({len(content)} bytes). Maximum is {MAX_FILE_SIZE} bytes (5 MB).",
        )

    return original_filename, extension, content


async def _active_recording_count(db: AsyncSession, user_id: int) -> int:
    result = await db.execute(
        select(func.count())
        .select_from(Recording)
        .where(
            Recording.user_id == user_id,
            Recording.deleted_at.is_(None),
        )
    )
    return int(result.scalar_one())


async def _store_upload(
    *,
    db: AsyncSession,
    user_id: int,
    wake_word: str,
    file: UploadFile,
) -> RecordingUploadResponse:
    _original_filename, extension, content = await _read_audio_upload(file)
    wav_bytes, duration_s = _decode_audio_to_wav(content, extension)

    storage = get_storage()
    filename = f"{wake_word}_{uuid.uuid4().hex[:8]}.wav"
    storage_key = build_recording_key(user_id, wake_word, filename)
    storage.upload(storage_key, wav_bytes, "audio/wav")

    recording = Recording(
        user_id=user_id,
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


def _consume_upload_budget(request: Request, response: Response, file_count: int) -> None:
    consume_rate_limit(
        request,
        limit_value=RECORDING_UPLOAD_LIMIT,
        key=key_by_user(request),
        scope=RECORDING_UPLOAD_SCOPE,
        cost=file_count,
        response=response,
    )


def _error_text(exc: HTTPException) -> str:
    detail = exc.detail
    return str(detail) if detail else "Upload failed"


@router.post("/upload", response_model=RecordingUploadResponse)
async def upload_recording(
    request: Request,
    response: Response,
    current_user: Annotated[User, Depends(_verified_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),  # noqa: B008
    wake_word: str = Form(...),
) -> RecordingUploadResponse:
    """Upload one audio recording for a wake word."""
    normalized_wake_word = _normalize_wake_word(wake_word)
    _consume_upload_budget(request, response, 1)

    total_recordings = await _active_recording_count(db, current_user.id)
    if total_recordings >= MAX_RECORDINGS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Recording limit reached. Delete old recordings to upload new ones.",
        )

    return await _store_upload(
        db=db,
        user_id=current_user.id,
        wake_word=normalized_wake_word,
        file=file,
    )


@router.post("/bulk-upload", response_model=RecordingBulkUploadResponse)
async def bulk_upload_recordings(
    request: Request,
    response: Response,
    current_user: Annotated[User, Depends(_verified_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    files: list[UploadFile] = File(..., alias="file"),  # noqa: B008
    wake_word: str = Form(...),
) -> RecordingBulkUploadResponse:
    """Upload multiple audio files for a wake word with per-file outcomes."""
    normalized_wake_word = _normalize_wake_word(wake_word)
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum is {MAX_FILES_PER_REQUEST} files per request.",
        )

    _consume_upload_budget(request, response, len(files))

    total_recordings = await _active_recording_count(db, current_user.id)
    remaining_slots = MAX_RECORDINGS_PER_USER - total_recordings
    successful_uploads = 0
    results: list[RecordingBulkUploadItem] = []

    for file in files:
        original_filename = _safe_upload_filename(file)
        if successful_uploads >= remaining_slots:
            results.append(
                RecordingBulkUploadItem(
                    filename=original_filename,
                    status="error",
                    error="Recording limit reached. Delete old recordings to upload new ones.",
                )
            )
            continue

        try:
            upload = await _store_upload(
                db=db,
                user_id=current_user.id,
                wake_word=normalized_wake_word,
                file=file,
            )
        except HTTPException as exc:
            results.append(
                RecordingBulkUploadItem(
                    filename=original_filename,
                    status="error",
                    error=_error_text(exc),
                )
            )
            continue

        successful_uploads += 1
        results.append(
            RecordingBulkUploadItem(
                filename=original_filename,
                status="success",
                recording_id=upload.recording_id,
                wake_word=upload.wake_word,
                duration_s=upload.duration_s,
            )
        )

    failed = len(results) - successful_uploads
    return RecordingBulkUploadResponse(
        results=results,
        uploaded=successful_uploads,
        failed=failed,
    )


@router.get("/count", response_model=RecordingCountResponse)
async def count_recordings(
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    wake_word: str | None = Query(default=None),
) -> RecordingCountResponse:
    """Count active recordings for the current user, optionally filtered by wake word."""
    stmt = select(func.count()).select_from(Recording).where(
        Recording.user_id == current_user.id,
        Recording.deleted_at.is_(None),
    )
    normalized_wake_word: str | None = None
    if wake_word:
        normalized_wake_word = wake_word.strip().lower()
        stmt = stmt.where(Recording.wake_word == normalized_wake_word)

    result = await db.execute(stmt)
    return RecordingCountResponse(
        wake_word=normalized_wake_word,
        count=int(result.scalar_one()),
    )


@router.get("", response_model=list[RecordingResponse])
async def list_recordings(
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    wake_word: str | None = Query(default=None),
) -> list[RecordingResponse]:
    """List all recordings for the current user, optionally filtered by wake word."""
    stmt = select(Recording).where(
        Recording.user_id == current_user.id,
        Recording.deleted_at.is_(None),
    )
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
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Delete a recording. Only the owner may delete their recordings."""
    result = await db.execute(
        select(Recording).where(
            Recording.id == recording_id,
            Recording.user_id == current_user.id,
            Recording.deleted_at.is_(None),
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
