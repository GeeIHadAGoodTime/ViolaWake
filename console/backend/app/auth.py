"""JWT authentication and password hashing."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models import User

security = HTTPBearer()

EMAIL_VERIFICATION_TOKEN_HOURS = 48
PASSWORD_RESET_TOKEN_HOURS = 2


def _prep_password(password: str) -> bytes:
    """Prepare password for bcrypt.

    bcrypt has a 72-byte limit. For longer passwords, pre-hash with SHA-256
    to stay within that limit while preserving full entropy.
    """
    pw_bytes = password.encode("utf-8")
    if len(pw_bytes) > 72:
        pw_bytes = hashlib.sha256(pw_bytes).hexdigest().encode("utf-8")
    return pw_bytes


def hash_password(password: str) -> str:
    """Hash a plaintext password with bcrypt."""
    return bcrypt.hashpw(_prep_password(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its bcrypt hash."""
    try:
        return bcrypt.checkpw(_prep_password(plain_password), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: int) -> str:
    """Create a JWT access token for the given user ID."""
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.access_token_expire_hours)
    payload = {
        "sub": str(user_id),
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def _create_action_token(user_id: int, purpose: str, expires_in: timedelta) -> str:
    """Create a purpose-scoped JWT for user actions such as verify/reset."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "purpose": purpose,
        "exp": now + expires_in,
        "iat": now,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_email_verification_token(user_id: int) -> str:
    """Create a JWT email verification token for the given user ID."""
    return _create_action_token(
        user_id,
        purpose="verify_email",
        expires_in=timedelta(hours=EMAIL_VERIFICATION_TOKEN_HOURS),
    )


def create_password_reset_token(user_id: int) -> str:
    """Create a JWT password reset token for the given user ID."""
    return _create_action_token(
        user_id,
        purpose="reset_password",
        expires_in=timedelta(hours=PASSWORD_RESET_TOKEN_HOURS),
    )


def decode_token(token: str) -> int:
    """Decode a JWT token and return the user ID.

    Raises:
        HTTPException: If the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id_str: str | None = payload.get("sub")
        if user_id_str is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
            )
        return int(user_id_str)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        ) from e


def decode_action_token(token: str, expected_purpose: str) -> int:
    """Decode a purpose-scoped JWT token and return the user ID."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id_str: str | None = payload.get("sub")
        purpose: str | None = payload.get("purpose")
        if user_id_str is None or purpose is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token",
            )
        if purpose != expected_purpose:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid token purpose",
            )
        return int(user_id_str)
    except HTTPException:
        raise
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid token: {e}",
        ) from e


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """FastAPI dependency: extract and validate the current user from JWT.

    Returns the User ORM object for the authenticated user.
    """
    user_id = decode_token(credentials.credentials)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user
