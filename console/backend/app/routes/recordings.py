"""Recording upload and listing routes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import io
import json
import logging
import math
import os
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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
from fastapi.responses import JSONResponse
from scipy import signal
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_verified_user
from app.config import settings
from app.database import get_db
from app.models import Recording, Subscription, TrainedModel, User
from app.rate_limit import (
    RECORDING_UPLOAD_LIMIT,
    consume_rate_limit,
    key_by_user,
    set_rate_limit_user,
)
from app.schemas import (
    RecordingBulkUploadItem,
    RecordingBulkUploadResponse,
    RecordingCountResponse,
    RecordingResponse,
    RecordingUploadResponse,
)
from app.storage import build_recording_key, get_storage

logger = logging.getLogger("violawake.recordings")

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

MIN_DURATION_S = 0.3
MAX_DURATION_S = 30.0
TARGET_SAMPLE_RATE = 16000
MAX_FILE_SIZE = 5 * 1024 * 1024
MAX_FILES_PER_REQUEST = 50
MAX_RECORDINGS_PER_USER = 500
MIN_RMS_ENERGY = 10.0
MIN_RMS_FLOAT = MIN_RMS_ENERGY / 32767.0
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac"}
RECORDING_UPLOAD_SCOPE = "recordings-upload"

STORAGE_DAILY_LIMITS_BYTES = {
    "free": 50 * 1024 * 1024,
    "developer": 500 * 1024 * 1024,
    "business": 500 * 1024 * 1024,
    "enterprise": 500 * 1024 * 1024,
}
STORAGE_LIFETIME_LIMITS_BYTES = {
    "free": 200 * 1024 * 1024,
    "developer": 2 * 1024 * 1024 * 1024,
    "business": 2 * 1024 * 1024 * 1024,
    "enterprise": 2 * 1024 * 1024 * 1024,
}
GLOBAL_VOLUME_MAX_USED_BYTES = 50 * 1024 * 1024 * 1024
GLOBAL_VOLUME_MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
GLOBAL_VOLUME_CACHE_SECONDS = 60
AUDIT_LOG_MAX_BYTES = 100 * 1024 * 1024
DECODE_TIMEOUT_SECONDS = 30

# Header precheck: at most 30s of stereo 16 kHz decoded sample values. Tighter
# memory caps need subprocess isolation; deferred to Tier 3 - see SECURITY.md.
MAX_DECODED_SAMPLE_VALUES = int(MAX_DURATION_S * TARGET_SAMPLE_RATE * 2)

_audit_log_lock = threading.Lock()
_volume_cache_lock = threading.Lock()
_volume_cache: tuple[tuple[str, int, int], float, dict[str, int | str]] | None = None
_user_quota_locks: dict[int, asyncio.Lock] = {}
_user_quota_locks_guard = asyncio.Lock()


class UploadRejected(HTTPException):
    """HTTP error with the upload cap response body required by clients."""

    def __init__(
        self,
        status_code: int,
        detail: str,
        *,
        limit: int | None = None,
        current: int | None = None,
        retry_after_seconds: int | None = None,
        headers: dict[str, str] | None = None,
        decode_status: str | None = None,
        audit_size_bytes: int | None = None,
        audit_magic_bytes: bytes | None = None,
    ) -> None:
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        payload: dict[str, int | str] = {"detail": detail}
        if limit is not None:
            payload["limit"] = limit
        if current is not None:
            payload["current"] = current
        if retry_after_seconds is not None:
            payload["retry_after_seconds"] = retry_after_seconds
        self.payload = payload
        self.decode_status = decode_status
        self.audit_size_bytes = audit_size_bytes
        self.audit_magic_bytes = audit_magic_bytes


async def upload_rejected_exception_handler(
    request: Request,
    exc: UploadRejected,
) -> JSONResponse:
    """Return upload-specific errors without FastAPI's nested detail wrapper."""
    del request
    return JSONResponse(status_code=exc.status_code, content=exc.payload, headers=exc.headers)


@dataclass(slots=True)
class RawUpload:
    filename_claimed: str
    mimetype_claimed: str | None
    content: bytes
    magic: bytes
    detected_extension: str


@dataclass(slots=True)
class PreparedUpload:
    filename_claimed: str
    filename_stored: str
    storage_key: str
    mimetype_claimed: str | None
    magic_hex: str
    wav_bytes: bytes
    duration_s: float
    size_bytes: int


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


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.client.host if request.client else ""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _consume_upload_budget(request: Request, response: Response, file_count: int) -> None:
    consume_rate_limit(
        request,
        limit_value=RECORDING_UPLOAD_LIMIT,
        key=key_by_user(request),
        scope=RECORDING_UPLOAD_SCOPE,
        cost=file_count,
        response=response,
    )


def _log_cap_hit(
    *,
    cap: str,
    limit: int,
    current: int,
    user_id: int | None = None,
    retry_after_seconds: int | None = None,
) -> None:
    logger.warning(
        "Upload storage cap hit",
        extra={
            "event_data": {
                "source": "upload",
                "cap": cap,
                "user_id": user_id,
                "limit": limit,
                "current": current,
                "retry_after_seconds": retry_after_seconds,
            }
        },
    )


def _raise_cap_hit(
    *,
    status_code: int,
    detail: str,
    limit: int,
    current: int,
    user_id: int | None = None,
    cap: str,
    retry_after_seconds: int | None = None,
) -> None:
    # Status-code choices for upload caps:
    # 413 = per-file/decoded-size cap, 429 = rolling daily or velocity budget,
    # 507 = per-user lifetime storage, 503 = shared service volume capacity.
    _log_cap_hit(
        cap=cap,
        limit=limit,
        current=current,
        user_id=user_id,
        retry_after_seconds=retry_after_seconds,
    )
    raise UploadRejected(
        status_code,
        detail,
        limit=limit,
        current=current,
        retry_after_seconds=retry_after_seconds,
        decode_status="cap_rejected",
    )


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


async def _user_quota_lock(user_id: int) -> asyncio.Lock:
    async with _user_quota_locks_guard:
        lock = _user_quota_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            _user_quota_locks[user_id] = lock
        return lock


async def _subscription_tier(db: AsyncSession, user_id: int) -> str:
    result = await db.execute(
        select(Subscription.tier, Subscription.status).where(Subscription.user_id == user_id)
    )
    row = result.first()
    if row is None:
        return "free"
    tier, subscription_status = row
    if subscription_status != "active":
        return "free"
    return str(tier or "free")


def _limit_for_tier(tier: str, limits: dict[str, int]) -> int:
    return limits.get(tier, limits["developer"] if tier != "free" else limits["free"])


async def _daily_uploaded_bytes(db: AsyncSession, user_id: int) -> int:
    cutoff = _utcnow() - timedelta(hours=24)
    uploaded_at = func.coalesce(Recording.uploaded_at, Recording.created_at)
    result = await db.execute(
        select(func.coalesce(func.sum(func.coalesce(Recording.size_bytes, 0)), 0)).where(
            Recording.user_id == user_id,
            uploaded_at >= cutoff,
        )
    )
    return int(result.scalar_one() or 0)


async def _daily_retry_after_seconds(db: AsyncSession, user_id: int) -> int:
    cutoff = _utcnow() - timedelta(hours=24)
    uploaded_at = func.coalesce(Recording.uploaded_at, Recording.created_at)
    result = await db.execute(
        select(func.min(uploaded_at)).where(
            Recording.user_id == user_id,
            uploaded_at >= cutoff,
        )
    )
    oldest = result.scalar_one_or_none()
    if oldest is None:
        return 24 * 60 * 60
    if isinstance(oldest, str):
        try:
            oldest = datetime.fromisoformat(oldest)
        except ValueError:
            return 24 * 60 * 60
    if oldest.tzinfo is None:
        oldest = oldest.replace(tzinfo=timezone.utc)
    return max(1, math.ceil((oldest + timedelta(hours=24) - _utcnow()).total_seconds()))


async def _lifetime_storage_bytes(db: AsyncSession, user_id: int) -> int:
    recordings_result = await db.execute(
        select(func.coalesce(func.sum(func.coalesce(Recording.size_bytes, 0)), 0)).where(
            Recording.user_id == user_id,
            Recording.deleted_at.is_(None),
        )
    )
    models_result = await db.execute(
        select(func.coalesce(func.sum(func.coalesce(TrainedModel.size_bytes, 0)), 0)).where(
            TrainedModel.user_id == user_id,
        )
    )
    return int(recordings_result.scalar_one() or 0) + int(models_result.scalar_one() or 0)


async def _enforce_user_storage_caps(
    db: AsyncSession,
    user_id: int,
    incoming_bytes: int,
) -> None:
    tier = await _subscription_tier(db, user_id)
    daily_limit = _limit_for_tier(tier, STORAGE_DAILY_LIMITS_BYTES)
    lifetime_limit = _limit_for_tier(tier, STORAGE_LIFETIME_LIMITS_BYTES)

    daily_current = await _daily_uploaded_bytes(db, user_id)
    daily_after = daily_current + incoming_bytes
    if daily_after > daily_limit:
        retry_after = await _daily_retry_after_seconds(db, user_id)
        _raise_cap_hit(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Daily recording upload storage limit reached.",
            limit=daily_limit,
            current=daily_after,
            user_id=user_id,
            cap="daily",
            retry_after_seconds=retry_after,
        )

    lifetime_current = await _lifetime_storage_bytes(db, user_id)
    lifetime_after = lifetime_current + incoming_bytes
    if lifetime_after > lifetime_limit:
        _raise_cap_hit(
            status_code=507,
            detail="Recording storage limit reached. Delete old recordings or trained models before uploading more.",
            limit=lifetime_limit,
            current=lifetime_after,
            user_id=user_id,
            cap="lifetime",
        )


def _configured_volume_path() -> Path:
    configured = getattr(settings, "upload_volume_path", Path("/app/data"))
    path = Path(configured)
    if path.exists():
        return path
    return settings.data_dir


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += os.stat(Path(root, filename)).st_size
            except OSError:
                continue
    return total


def _free_space_bytes(path: Path) -> int:
    if hasattr(os, "statvfs"):
        stats = os.statvfs(path)
        return int(stats.f_bavail * stats.f_frsize)
    return int(shutil.disk_usage(path).free)


def _global_volume_status() -> dict[str, int | str]:
    global _volume_cache

    volume_path = _configured_volume_path().resolve(strict=False)
    max_used = int(getattr(settings, "upload_global_max_used_bytes", GLOBAL_VOLUME_MAX_USED_BYTES))
    min_free = int(getattr(settings, "upload_global_min_free_bytes", GLOBAL_VOLUME_MIN_FREE_BYTES))
    cache_key = (str(volume_path), max_used, min_free)
    now = time.monotonic()

    with _volume_cache_lock:
        if _volume_cache is not None:
            cached_key, checked_at, cached_status = _volume_cache
            if cached_key == cache_key and now - checked_at < GLOBAL_VOLUME_CACHE_SECONDS:
                return cached_status

    volume_path.mkdir(parents=True, exist_ok=True)
    status_payload: dict[str, int | str] = {
        "path": str(volume_path),
        "used_bytes": _directory_size_bytes(volume_path),
        "free_bytes": _free_space_bytes(volume_path),
        "max_used_bytes": max_used,
        "min_free_bytes": min_free,
    }

    with _volume_cache_lock:
        _volume_cache = (cache_key, now, status_payload)

    return status_payload


def _enforce_global_volume_capacity() -> None:
    volume = _global_volume_status()
    used_bytes = int(volume["used_bytes"])
    free_bytes = int(volume["free_bytes"])
    max_used = int(volume["max_used_bytes"])
    min_free = int(volume["min_free_bytes"])

    if free_bytes < min_free:
        _raise_cap_hit(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recording upload service capacity reached. Please try again later.",
            limit=min_free,
            current=free_bytes,
            cap="global_free_space",
        )
    if used_bytes > max_used:
        _raise_cap_hit(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recording upload service capacity reached. Please try again later.",
            limit=max_used,
            current=used_bytes,
            cap="global_used",
        )


def _audit_log_path() -> Path:
    return _configured_volume_path() / "logs" / "uploads.jsonl"


def _rotate_audit_log_if_needed(path: Path) -> None:
    if not path.exists() or path.stat().st_size < AUDIT_LOG_MAX_BYTES:
        return
    rotated = path.with_suffix(path.suffix + ".1")
    if rotated.exists():
        rotated.unlink()
    path.replace(rotated)


def _append_upload_audit(
    *,
    request: Request,
    user_id: int,
    filename_claimed: str,
    filename_stored: str | None,
    size_bytes: int,
    mimetype_claimed: str | None,
    magic_bytes: bytes | str,
    decode_status: str,
    recording_id_if_ok: int | None,
    wake_word: str,
) -> None:
    path = _audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    magic_hex = magic_bytes[:12].hex() if isinstance(magic_bytes, bytes) else magic_bytes
    payload = {
        "ts": _utcnow().isoformat(),
        "user_id": user_id,
        "ip": _client_ip(request),
        "filename_claimed": filename_claimed,
        "filename_stored": filename_stored,
        "size_bytes": size_bytes,
        "mimetype_claimed": mimetype_claimed,
        "magic_bytes_hex_first_12": magic_hex,
        "decode_status": decode_status,
        "recording_id_if_ok": recording_id_if_ok,
        "wake_word": wake_word,
    }

    line = json.dumps(payload, separators=(",", ":"), default=str)
    with _audit_log_lock:
        _rotate_audit_log_if_needed(path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


async def _read_audio_upload(file: UploadFile, user_id: int | None = None) -> RawUpload:
    filename = _safe_upload_filename(file)
    first_bytes = await file.read(12)
    if not first_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file")

    magic = first_bytes[:12]
    detected_extension = _validate_magic_and_extension(filename, magic)
    content = bytearray(first_bytes)
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > MAX_FILE_SIZE:
            _log_cap_hit(
                cap="per_file",
                limit=MAX_FILE_SIZE,
                current=len(content),
                user_id=user_id,
            )
            raise UploadRejected(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "File too large. Maximum recording upload size is 5 MB.",
                limit=MAX_FILE_SIZE,
                current=len(content),
                decode_status="cap_rejected",
                audit_size_bytes=len(content),
                audit_magic_bytes=magic,
            )

    file_bytes = bytes(content)
    return RawUpload(
        filename_claimed=filename,
        mimetype_claimed=file.content_type,
        content=file_bytes,
        magic=magic,
        detected_extension=detected_extension,
    )


def _validate_magic_and_extension(filename: str, magic: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise UploadRejected(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"Unsupported audio format. Upload WAV or FLAC only ({allowed}).",
            decode_status="format_rejected",
            audit_size_bytes=len(magic),
            audit_magic_bytes=magic,
        )

    if len(magic) >= 12 and magic[:4] == b"RIFF" and magic[8:12] == b"WAVE":
        detected = ".wav"
    elif len(magic) >= 4 and magic[:4] == b"fLaC":
        detected = ".flac"
    else:
        raise UploadRejected(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Unsupported audio content. Upload WAV or FLAC only.",
            decode_status="format_rejected",
            audit_size_bytes=len(magic),
            audit_magic_bytes=magic,
        )

    if extension != detected:
        raise UploadRejected(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Audio file extension does not match the file header.",
            decode_status="format_rejected",
            audit_size_bytes=len(magic),
            audit_magic_bytes=magic,
        )
    return detected


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


def _precheck_audio_header(file_bytes: bytes, user_id: int | None = None) -> None:
    import soundfile as sf

    try:
        info = sf.info(io.BytesIO(file_bytes))
    except Exception as exc:
        raise UploadRejected(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Could not read WAV/FLAC audio header.",
            decode_status="format_rejected",
        ) from exc

    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise UploadRejected(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Audio file header is missing sample metadata.",
            decode_status="format_rejected",
        )

    decoded_sample_values = int(info.frames) * int(info.channels)
    duration_s = float(info.frames / info.samplerate)
    if decoded_sample_values > MAX_DECODED_SAMPLE_VALUES or duration_s > MAX_DURATION_S:
        _log_cap_hit(
            cap="decoded_header",
            limit=MAX_DECODED_SAMPLE_VALUES,
            current=decoded_sample_values,
            user_id=user_id,
        )
        raise UploadRejected(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "Audio file expands beyond the 30 second stereo 16 kHz decode limit.",
            limit=MAX_DECODED_SAMPLE_VALUES,
            current=decoded_sample_values,
            decode_status="bomb_rejected",
        )


def _decode_and_encode_canonical_wav(file_bytes: bytes) -> tuple[bytes, float]:
    import soundfile as sf

    data, sample_rate = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=True)
    samples = np.asarray(data, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("Audio file contains no samples.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("Audio file contains invalid samples.")

    mono = samples.mean(axis=1)
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // divisor
        down = int(sample_rate) // divisor
        mono = signal.resample_poly(mono, up, down).astype(np.float32, copy=False)

    duration_s = float(len(mono) / TARGET_SAMPLE_RATE)
    _raise_duration_error(duration_s)

    rms = float(np.sqrt(np.mean(mono.astype(np.float32) ** 2)))
    if rms < MIN_RMS_FLOAT:
        raise ValueError("Recording appears to be silent. Please check the audio and try again.")

    out = io.BytesIO()
    sf.write(out, np.clip(mono, -1.0, 1.0), TARGET_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return out.getvalue(), duration_s


def _decode_with_watchdog(file_bytes: bytes) -> tuple[bytes, float]:
    # Thread timeout bounds wall-clock decode time. A tighter memory cap needs
    # subprocess isolation and OS limits; that is deferred to Tier 3 in SECURITY.md.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="upload-decode")
    future = executor.submit(_decode_and_encode_canonical_wav, file_bytes)
    try:
        wav_bytes, duration_s = future.result(timeout=DECODE_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise UploadRejected(
            status.HTTP_400_BAD_REQUEST,
            "Audio decode timed out.",
            decode_status="timeout",
        ) from exc
    except HTTPException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    except Exception as exc:
        executor.shutdown(wait=False, cancel_futures=True)
        logger.warning(
            "Recording upload decode failed: %s",
            exc,
            extra={"event_data": {"source": "upload", "decode_status": "decode_error"}},
        )
        raise UploadRejected(
            status.HTTP_400_BAD_REQUEST,
            "Could not decode audio file as WAV or FLAC.",
            decode_status="decode_error",
        ) from exc
    else:
        executor.shutdown(wait=True)
        return wav_bytes, duration_s


async def _prepare_upload(
    *,
    request: Request,
    user_id: int,
    wake_word: str,
    file: UploadFile,
) -> PreparedUpload:
    filename_claimed = _safe_upload_filename(file)
    mimetype_claimed = file.content_type
    raw: RawUpload | None = None
    magic: bytes = b""
    size_bytes = 0

    try:
        raw = await _read_audio_upload(file, user_id=user_id)
        filename_claimed = raw.filename_claimed
        mimetype_claimed = raw.mimetype_claimed
        magic = raw.magic
        size_bytes = len(raw.content)
        _precheck_audio_header(raw.content, user_id=user_id)
        wav_bytes, duration_s = _decode_with_watchdog(raw.content)
    except UploadRejected as exc:
        if size_bytes == 0 and exc.audit_size_bytes is not None:
            size_bytes = exc.audit_size_bytes
        if not magic and exc.audit_magic_bytes is not None:
            magic = exc.audit_magic_bytes
        _append_upload_audit(
            request=request,
            user_id=user_id,
            filename_claimed=filename_claimed,
            filename_stored=None,
            size_bytes=size_bytes,
            mimetype_claimed=mimetype_claimed,
            magic_bytes=magic,
            decode_status=exc.decode_status or "decode_error",
            recording_id_if_ok=None,
            wake_word=wake_word,
        )
        raise
    except HTTPException as exc:
        _append_upload_audit(
            request=request,
            user_id=user_id,
            filename_claimed=filename_claimed,
            filename_stored=None,
            size_bytes=size_bytes,
            mimetype_claimed=mimetype_claimed,
            magic_bytes=magic,
            decode_status="decode_error",
            recording_id_if_ok=None,
            wake_word=wake_word,
        )
        raise exc

    stored_filename = f"{uuid.uuid4()}.wav"
    return PreparedUpload(
        filename_claimed=filename_claimed,
        filename_stored=stored_filename,
        storage_key=build_recording_key(user_id, wake_word, stored_filename),
        mimetype_claimed=mimetype_claimed,
        magic_hex=raw.magic.hex() if raw else "",
        wav_bytes=wav_bytes,
        duration_s=duration_s,
        # Quotas account for the larger of ingress bytes and canonical storage
        # bytes. That keeps the daily cap meaningful for near-5 MB uploads while
        # still covering FLAC uploads that expand when stored as canonical WAV.
        size_bytes=max(len(raw.content) if raw else 0, len(wav_bytes)),
    )


def _error_text(exc: HTTPException) -> str:
    detail = exc.detail
    return str(detail) if detail else "Upload failed"


async def _store_prepared_upload(
    *,
    request: Request,
    db: AsyncSession,
    user_id: int,
    wake_word: str,
    prepared: PreparedUpload,
) -> RecordingUploadResponse:
    storage = get_storage()
    storage.upload(prepared.storage_key, prepared.wav_bytes, "audio/wav")

    recording = Recording(
        user_id=user_id,
        wake_word=wake_word,
        filename=prepared.filename_stored,
        display_name=prepared.filename_claimed,
        file_path=prepared.storage_key,
        duration_s=round(prepared.duration_s, 3),
        sample_rate=TARGET_SAMPLE_RATE,
        size_bytes=prepared.size_bytes,
        uploaded_at=_utcnow(),
    )
    db.add(recording)
    await db.flush()

    _append_upload_audit(
        request=request,
        user_id=user_id,
        filename_claimed=prepared.filename_claimed,
        filename_stored=prepared.filename_stored,
        size_bytes=prepared.size_bytes,
        mimetype_claimed=prepared.mimetype_claimed,
        magic_bytes=prepared.magic_hex,
        decode_status="ok",
        recording_id_if_ok=recording.id,
        wake_word=wake_word,
    )

    return RecordingUploadResponse(
        recording_id=recording.id,
        filename=prepared.filename_stored,
        wake_word=wake_word,
        duration_s=round(prepared.duration_s, 3),
    )


def _audit_cap_rejection_for_prepared(
    *,
    request: Request,
    user_id: int,
    wake_word: str,
    prepared_uploads: list[PreparedUpload],
    decode_status: str,
) -> None:
    for prepared in prepared_uploads:
        _append_upload_audit(
            request=request,
            user_id=user_id,
            filename_claimed=prepared.filename_claimed,
            filename_stored=None,
            size_bytes=prepared.size_bytes,
            mimetype_claimed=prepared.mimetype_claimed,
            magic_bytes=prepared.magic_hex,
            decode_status=decode_status,
            recording_id_if_ok=None,
            wake_word=wake_word,
        )


def _audit_pre_read_rejection_for_files(
    *,
    request: Request,
    user_id: int,
    wake_word: str,
    files: list[UploadFile],
    decode_status: str,
) -> None:
    for file in files:
        _append_upload_audit(
            request=request,
            user_id=user_id,
            filename_claimed=_safe_upload_filename(file),
            filename_stored=None,
            size_bytes=0,
            mimetype_claimed=file.content_type,
            magic_bytes=b"",
            decode_status=decode_status,
            recording_id_if_ok=None,
            wake_word=wake_word,
        )


@router.post("/upload", response_model=RecordingUploadResponse)
async def upload_recording(
    request: Request,
    response: Response,
    current_user: Annotated[User, Depends(_verified_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    file: UploadFile = File(...),  # noqa: B008
    wake_word: str = Form(...),
) -> RecordingUploadResponse:
    """Upload one WAV or FLAC recording; stores canonical 16 kHz mono PCM WAV."""
    normalized_wake_word = _normalize_wake_word(wake_word)
    try:
        _consume_upload_budget(request, response, 1)
        _enforce_global_volume_capacity()
    except HTTPException:
        _audit_pre_read_rejection_for_files(
            request=request,
            user_id=current_user.id,
            wake_word=normalized_wake_word,
            files=[file],
            decode_status="cap_rejected",
        )
        raise

    lock = await _user_quota_lock(current_user.id)
    async with lock:
        total_recordings = await _active_recording_count(db, current_user.id)
        if total_recordings >= MAX_RECORDINGS_PER_USER:
            _log_cap_hit(
                cap="recording_count",
                limit=MAX_RECORDINGS_PER_USER,
                current=total_recordings,
                user_id=current_user.id,
            )
            _append_upload_audit(
                request=request,
                user_id=current_user.id,
                filename_claimed=_safe_upload_filename(file),
                filename_stored=None,
                size_bytes=0,
                mimetype_claimed=file.content_type,
                magic_bytes=b"",
                decode_status="cap_rejected",
                recording_id_if_ok=None,
                wake_word=normalized_wake_word,
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Recording limit reached. Delete old recordings to upload new ones.",
            )

        prepared = await _prepare_upload(
            request=request,
            user_id=current_user.id,
            wake_word=normalized_wake_word,
            file=file,
        )
        try:
            await _enforce_user_storage_caps(db, current_user.id, prepared.size_bytes)
        except UploadRejected as exc:
            _audit_cap_rejection_for_prepared(
                request=request,
                user_id=current_user.id,
                wake_word=normalized_wake_word,
                prepared_uploads=[prepared],
                decode_status=exc.decode_status or "cap_rejected",
            )
            raise

        return await _store_prepared_upload(
            request=request,
            db=db,
            user_id=current_user.id,
            wake_word=normalized_wake_word,
            prepared=prepared,
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
    """Upload multiple WAV/FLAC files for a wake word with per-file outcomes."""
    normalized_wake_word = _normalize_wake_word(wake_word)
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file is required")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum is {MAX_FILES_PER_REQUEST} files per request.",
        )

    try:
        _consume_upload_budget(request, response, len(files))
        _enforce_global_volume_capacity()
    except HTTPException:
        _audit_pre_read_rejection_for_files(
            request=request,
            user_id=current_user.id,
            wake_word=normalized_wake_word,
            files=files,
            decode_status="cap_rejected",
        )
        raise

    lock = await _user_quota_lock(current_user.id)
    async with lock:
        total_recordings = await _active_recording_count(db, current_user.id)
        remaining_slots = MAX_RECORDINGS_PER_USER - total_recordings
        prepared_uploads: list[tuple[int, PreparedUpload]] = []
        results: list[RecordingBulkUploadItem | None] = [None] * len(files)

        for index, file in enumerate(files):
            original_filename = _safe_upload_filename(file)
            if len(prepared_uploads) >= remaining_slots:
                _log_cap_hit(
                    cap="recording_count",
                    limit=MAX_RECORDINGS_PER_USER,
                    current=total_recordings + len(prepared_uploads),
                    user_id=current_user.id,
                )
                results[index] = RecordingBulkUploadItem(
                    filename=original_filename,
                    status="error",
                    error="Recording limit reached. Delete old recordings to upload new ones.",
                )
                _append_upload_audit(
                    request=request,
                    user_id=current_user.id,
                    filename_claimed=original_filename,
                    filename_stored=None,
                    size_bytes=0,
                    mimetype_claimed=file.content_type,
                    magic_bytes=b"",
                    decode_status="cap_rejected",
                    recording_id_if_ok=None,
                    wake_word=normalized_wake_word,
                )
                continue

            try:
                prepared = await _prepare_upload(
                    request=request,
                    user_id=current_user.id,
                    wake_word=normalized_wake_word,
                    file=file,
                )
            except HTTPException as exc:
                if isinstance(exc, UploadRejected) and exc.status_code in {
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    507,
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                }:
                    raise
                results[index] = RecordingBulkUploadItem(
                    filename=original_filename,
                    status="error",
                    error=_error_text(exc),
                )
                continue

            prepared_uploads.append((index, prepared))

        incoming_bytes = sum(prepared.size_bytes for _, prepared in prepared_uploads)
        try:
            await _enforce_user_storage_caps(db, current_user.id, incoming_bytes)
        except UploadRejected as exc:
            _audit_cap_rejection_for_prepared(
                request=request,
                user_id=current_user.id,
                wake_word=normalized_wake_word,
                prepared_uploads=[prepared for _, prepared in prepared_uploads],
                decode_status=exc.decode_status or "cap_rejected",
            )
            raise

        uploaded = 0
        for index, prepared in prepared_uploads:
            upload = await _store_prepared_upload(
                request=request,
                db=db,
                user_id=current_user.id,
                wake_word=normalized_wake_word,
                prepared=prepared,
            )
            uploaded += 1
            results[index] = RecordingBulkUploadItem(
                filename=prepared.filename_claimed,
                status="success",
                recording_id=upload.recording_id,
                wake_word=upload.wake_word,
                duration_s=upload.duration_s,
            )

        ordered_results = [
            result
            for result in results
            if result is not None
        ]
        failed = sum(1 for result in ordered_results if result.status == "error")
        return RecordingBulkUploadResponse(
            results=ordered_results,
            uploaded=uploaded,
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
