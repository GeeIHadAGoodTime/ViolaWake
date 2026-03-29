"""JWT authentication and password hashing."""

from __future__ import annotations

import hashlib
import secrets
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
from app.models import Team, TeamMember, User

security = HTTPBearer()

EMAIL_VERIFICATION_TOKEN_HOURS = 48
PASSWORD_RESET_TOKEN_HOURS = 2
DOWNLOAD_TOKEN_SECONDS = 60
TEAM_INVITE_TOKEN_HOURS = 72
_download_token_jtis: dict[str, datetime] = {}


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


def _prune_download_tokens(now: datetime | None = None) -> None:
    """Drop expired one-time download token identifiers."""
    current_time = now or datetime.now(timezone.utc)
    expired = [
        jti
        for jti, expires_at in _download_token_jtis.items()
        if expires_at <= current_time
    ]
    for jti in expired:
        _download_token_jtis.pop(jti, None)


def reset_download_tokens() -> None:
    """Clear tracked one-time download tokens. Used by test fixtures."""
    _download_token_jtis.clear()


def create_download_token(
    user_id: int,
    *,
    action: str,
    resource_id: int,
) -> str:
    """Create a short-lived, single-use token for browser SSE/download flows."""
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=DOWNLOAD_TOKEN_SECONDS)
    jti = secrets.token_urlsafe(16)
    payload = {
        "sub": str(user_id),
        "purpose": "download_access",
        "action": action,
        "resource_id": str(resource_id),
        "one_time": True,
        "jti": jti,
        "exp": expires_at,
        "iat": now,
    }
    _prune_download_tokens(now)
    _download_token_jtis[jti] = expires_at
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


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


def create_team_invite_token(inviter_id: int, team_id: int, email: str, role: str) -> str:
    """Create a signed JWT invitation token for a team invite."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(inviter_id),
        "purpose": "team_invite",
        "team_id": team_id,
        "email": email,
        "role": role,
        "exp": now + timedelta(hours=TEAM_INVITE_TOKEN_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_team_invite_token(token: str) -> dict:
    """Decode a team invite token.

    Returns a dict with keys: inviter_id, team_id, email, role.

    Raises:
        HTTPException: If the token is invalid, expired, or wrong purpose.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        purpose = payload.get("purpose")
        if purpose != "team_invite":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid invite token",
            )
        return {
            "inviter_id": int(payload["sub"]),
            "team_id": int(payload["team_id"]),
            "email": payload["email"],
            "role": payload["role"],
        }
    except HTTPException:
        raise
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid invite token: {e}",
        ) from e


def decode_download_token(
    token: str,
    *,
    expected_action: str,
    expected_resource_id: int,
) -> int:
    """Decode and consume a one-time download token."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id_str = payload.get("sub")
        purpose = payload.get("purpose")
        action = payload.get("action")
        resource_id = payload.get("resource_id")
        one_time = payload.get("one_time")
        jti = payload.get("jti")

        if (
            user_id_str is None
            or purpose != "download_access"
            or action != expected_action
            or resource_id != str(expected_resource_id)
            or one_time is not True
            or not isinstance(jti, str)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid download token",
            )

        _prune_download_tokens()
        expires_at = _download_token_jtis.pop(jti, None)
        if expires_at is None or expires_at <= datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Download token has expired or was already used",
            )

        return int(user_id_str)
    except HTTPException:
        raise
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid download token: {e}",
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


async def get_verified_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Require an authenticated user with a verified email address."""
    if not current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verify your email address to access recording, training, and billing features.",
        )
    return current_user


def make_get_team_member(required_roles: list[str] | None = None):
    """Return a FastAPI dependency that verifies team membership and optionally requires a specific role.

    Usage::

        # Any member
        member = Depends(make_get_team_member())

        # Admin or owner only
        admin = Depends(make_get_team_member(["owner", "admin"]))
    """
    async def _get_team_member(
        team_id: int,
        current_user: Annotated[User, Depends(get_verified_user)],
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> TeamMember:
        result = await db.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id,
                TeamMember.user_id == current_user.id,
                TeamMember.joined_at.is_not(None),
            )
        )
        member = result.scalar_one_or_none()
        if member is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not a member of this team",
            )
        if required_roles and member.role not in required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient team role",
            )
        return member

    return _get_team_member
