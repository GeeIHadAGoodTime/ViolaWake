"""Authenticated local file serving routes."""

from __future__ import annotations

import logging
import mimetypes
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import decode_token
from app.database import get_db
from app.models import User
from app.storage import LocalStorageBackend, get_storage

router = APIRouter(prefix="/api/files", tags=["files"])
logger = logging.getLogger("violawake.files")


@router.get("/{key:path}")
async def get_file(
    key: str,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    token: str | None = Query(default=None),
) -> Response:
    """Serve a locally stored file after validating ownership."""
    storage = get_storage()
    if not isinstance(storage, LocalStorageBackend):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Local file route is unavailable for object storage",
        )

    current_user = await _resolve_file_user(request, token, db)
    _validate_user_access(key, current_user.id)

    try:
        content = storage.download(key)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file key",
        ) from exc

    filename = PurePosixPath(key).name
    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    headers: dict[str, str] = {}
    if key.startswith("models/"):
        headers["Content-Disposition"] = 'attachment; filename="%s"' % filename

    return Response(content=content, media_type=media_type, headers=headers)


def _validate_user_access(key: str, user_id: int) -> None:
    """Ensure the requested key belongs to the authenticated user."""
    normalized = key.replace("\\", "/").strip().strip("/")
    parts = PurePosixPath(normalized).parts
    if len(parts) < 3 or parts[0] not in {"recordings", "models"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found",
        )

    if parts[1] != str(user_id):
        logger.warning("User %s attempted to access file %s", user_id, key)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this file",
        )


async def _resolve_file_user(
    request: Request,
    token: str | None,
    db: AsyncSession,
) -> User:
    """Resolve the authenticated user for local file downloads."""
    user_id: int | None = None

    if token:
        user_id = decode_token(token)
    else:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            user_id = decode_token(auth_header.removeprefix("Bearer ").strip())

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user
