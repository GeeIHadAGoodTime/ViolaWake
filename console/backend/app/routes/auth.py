"""Auth routes: register, login, me."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    create_access_token,
    create_email_verification_token,
    create_password_reset_token,
    decode_action_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.email_service import get_email_service
from app.models import User
from app.schemas import (
    AuthResponse,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    ResetPasswordRequest,
    UserDetailResponse,
    UserResponse,
    VerifyEmailRequest,
)

logger = logging.getLogger("violawake.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# In-memory rate limiter (no Redis needed for a local-first app)
# ---------------------------------------------------------------------------
_rate_store: dict[str, list[float]] = {}


def _check_rate_limit(key: str, max_attempts: int, window_seconds: int) -> None:
    """Raise 429 if *key* has exceeded *max_attempts* within *window_seconds*.

    Old timestamps outside the window are pruned on each call so the store
    does not grow unbounded.
    """
    now = time.monotonic()
    cutoff = now - window_seconds

    timestamps = _rate_store.get(key, [])
    # Prune expired entries
    timestamps = [t for t in timestamps if t > cutoff]
    _rate_store[key] = timestamps

    if len(timestamps) >= max_attempts:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
        )

    timestamps.append(now)


def reset_rate_limits() -> None:
    """Clear all rate-limit state. Used by test fixtures."""
    _rate_store.clear()


def _client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind a reverse proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: RegisterRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Register a new user account."""
    # Rate limit: 100 registrations per hour per IP (generous for dev/test; tighten in production)
    _check_rate_limit(f"register:{_client_ip(request)}", max_attempts=100, window_seconds=3600)
    # Check if email already taken
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
    )
    db.add(user)
    await db.flush()  # Populate user.id

    verification_token = create_email_verification_token(user.id)
    await get_email_service().send_verification_email(
        to=user.email,
        token=verification_token,
        name=user.name,
    )

    token = create_access_token(user.id)
    return AuthResponse(
        token=token,
        user=UserResponse(id=user.id, email=user.email, name=user.name),
    )


@router.post("/login", response_model=AuthResponse)
async def login(
    body: LoginRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Authenticate and receive a JWT token."""
    # Rate limit: 5 login attempts per minute per IP
    _check_rate_limit(f"login:{_client_ip(request)}", max_attempts=5, window_seconds=60)
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_access_token(user.id)
    return AuthResponse(
        token=token,
        user=UserResponse(id=user.id, email=user.email, name=user.name),
    )


@router.get("/me", response_model=UserDetailResponse)
async def me(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserDetailResponse:
    """Return the currently authenticated user's profile."""
    return UserDetailResponse(
        id=current_user.id,
        email=current_user.email,
        name=current_user.name,
        created_at=current_user.created_at,
    )


@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(
    body: VerifyEmailRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Verify a user's email address from a signed token."""
    _check_rate_limit(f"verify-email:{_client_ip(request)}", max_attempts=20, window_seconds=300)
    user_id = decode_action_token(body.token, expected_purpose="verify_email")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not user.email_verified:
        user.email_verified = True
        await db.flush()
        await get_email_service().send_welcome(to=user.email, name=user.name)
        logger.info("Verified email for user %s", user.id)

    return MessageResponse(message="Email verified successfully")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    body: ForgotPasswordRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Send a password reset email when the account exists."""
    _check_rate_limit(f"forgot-password:{_client_ip(request)}", max_attempts=5, window_seconds=300)

    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if user is not None:
        reset_token = create_password_reset_token(user.id)
        await get_email_service().send_password_reset(
            to=user.email,
            token=reset_token,
            name=user.name,
        )

    return MessageResponse(message="If an account exists for that email, a reset link has been sent.")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    body: ResetPasswordRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Reset a user's password from a signed token."""
    _check_rate_limit(f"reset-password:{_client_ip(request)}", max_attempts=10, window_seconds=300)
    user_id = decode_action_token(body.token, expected_purpose="reset_password")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    user.password_hash = hash_password(body.password)
    await db.flush()
    logger.info("Reset password for user %s", user.id)
    return MessageResponse(message="Password reset successfully")
