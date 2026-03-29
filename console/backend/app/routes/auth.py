"""Auth routes: register, login, me."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    DOWNLOAD_TOKEN_SECONDS,
    create_access_token,
    create_download_token,
    create_email_verification_token,
    create_password_reset_token,
    decode_action_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.email_service import get_email_service
from app.job_queue import init_job_queue
from app.models import Recording, Subscription, TrainedModel, TrainingJob, UsageRecord, User
from app.rate_limit import (
    FORGOT_PASSWORD_LIMIT,
    LOGIN_LIMIT,
    REGISTER_LIMIT,
    RESET_PASSWORD_LIMIT,
    VERIFY_EMAIL_LIMIT,
    limiter,
    reset_rate_limits as reset_shared_rate_limits,
)
from app.schemas import (
    AuthResponse,
    ChangePasswordRequest,
    DownloadTokenRequest,
    DownloadTokenResponse,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    ResetPasswordRequest,
    UserDetailResponse,
    UserResponse,
    VerifyEmailRequest,
)
from app.storage import build_companion_config_identifier, get_storage

logger = logging.getLogger("violawake.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


def reset_rate_limits() -> None:
    """Clear all rate-limit state. Used by test fixtures."""
    reset_shared_rate_limits()


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(REGISTER_LIMIT)
async def register(
    request: Request,
    body: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Register a new user account."""
    # Check if email already taken
    result = await db.execute(select(User).where(User.email == body.email))
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    email_svc = get_email_service()

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
    )

    if not email_svc.enabled:
        logger.warning("Email service disabled — auto-verifying email for %s", body.email)
        user.email_verified = True

    db.add(user)
    await db.flush()  # Populate user.id

    if email_svc.enabled:
        verification_token = create_email_verification_token(user.id)
        sent = await email_svc.send_verification_email(
            to=user.email,
            token=verification_token,
            name=user.name,
        )
        if not sent:
            logger.warning("Verification email failed — auto-verifying %s", body.email)
            user.email_verified = True
            await db.flush()

    token = create_access_token(user.id)
    return AuthResponse(
        token=token,
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            email_verified=user.email_verified,
        ),
    )


@router.post("/login", response_model=AuthResponse)
@limiter.limit(LOGIN_LIMIT)
async def login(
    request: Request,
    body: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Authenticate and receive a JWT token."""
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
        user=UserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            email_verified=user.email_verified,
        ),
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
        email_verified=current_user.email_verified,
        created_at=current_user.created_at,
    )


@router.post("/download-token", response_model=DownloadTokenResponse)
async def issue_download_token(
    body: DownloadTokenRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> DownloadTokenResponse:
    """Issue a short-lived one-time token for SSE or model download URLs."""
    if body.action == "model_download":
        result = await db.execute(
            select(TrainedModel.id).where(
                TrainedModel.id == body.resource_id,
                TrainedModel.user_id == current_user.id,
            )
        )
        model_id = result.scalar_one_or_none()
        if model_id is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found",
            )
    else:
        job = await (await init_job_queue()).get_job(body.resource_id)
        if job is None or job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Training job not found",
            )

    token = create_download_token(
        current_user.id,
        action=body.action,
        resource_id=body.resource_id,
    )
    return DownloadTokenResponse(
        token=token,
        expires_in_seconds=DOWNLOAD_TOKEN_SECONDS,
    )


@router.post("/verify-email", response_model=MessageResponse)
@limiter.limit(VERIFY_EMAIL_LIMIT)
async def verify_email(
    request: Request,
    body: VerifyEmailRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Verify a user's email address from a signed token."""
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
@limiter.limit(FORGOT_PASSWORD_LIMIT)
async def forgot_password(
    request: Request,
    body: ForgotPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Send a password reset email when the account exists."""

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
@limiter.limit(RESET_PASSWORD_LIMIT)
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Reset a user's password from a signed token."""
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


@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    body: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Change the authenticated user's password."""
    if not verify_password(body.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )
    current_user.password_hash = hash_password(body.new_password)
    await db.flush()
    logger.info("Changed password for user %s", current_user.id)
    return MessageResponse(message="Password changed successfully")


@router.delete("/account", response_model=MessageResponse)
async def delete_account(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Delete the authenticated user's account and associated data."""
    storage = get_storage()
    queue = await init_job_queue()

    recording_paths = (
        await db.execute(select(Recording.file_path).where(Recording.user_id == current_user.id))
    ).scalars().all()
    model_paths = (
        await db.execute(select(TrainedModel.file_path).where(TrainedModel.user_id == current_user.id))
    ).scalars().all()

    await queue.delete_jobs_for_user(current_user.id)

    for file_path in recording_paths:
        storage.delete(file_path)

    for file_path in model_paths:
        storage.delete(file_path)
        storage.delete(build_companion_config_identifier(file_path))

    await db.execute(delete(UsageRecord).where(UsageRecord.user_id == current_user.id))
    await db.execute(delete(Subscription).where(Subscription.user_id == current_user.id))
    await db.execute(delete(TrainingJob).where(TrainingJob.user_id == current_user.id))
    await db.execute(delete(TrainedModel).where(TrainedModel.user_id == current_user.id))
    await db.execute(delete(Recording).where(Recording.user_id == current_user.id))
    await db.execute(delete(User).where(User.id == current_user.id))

    logger.info("Deleted account for user %s", current_user.id)
    return MessageResponse(message="Account and associated data deleted.")
