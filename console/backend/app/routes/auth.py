"""Auth routes: register, login, me."""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    DOWNLOAD_TOKEN_SECONDS,
    create_access_token,
    create_download_token,
    create_email_verification_token,
    create_password_reset_token,
    decode_action_token,
    decode_reset_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.config import settings
from app.database import get_db
from app.email_service import get_email_service
from app.job_queue import init_job_queue
from app.models import Recording, Subscription, TrainedModel, User
from app.rate_limit import (
    CHANGE_PASSWORD_LIMIT,
    FORGOT_PASSWORD_LIMIT,
    LOGIN_LIMIT,
    REGISTER_LIMIT,
    RESEND_VERIFICATION_LIMIT,
    RESET_PASSWORD_LIMIT,
    VERIFY_EMAIL_LIMIT,
    key_by_user,
    limiter,
    set_rate_limit_user,
)
from app.rate_limit import (
    reset_rate_limits as reset_shared_rate_limits,
)
from app.schemas import (
    AuthResponse,
    ChangePasswordRequest,
    DeleteAccountRequest,
    DownloadTokenRequest,
    DownloadTokenResponse,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    UserDetailResponse,
    UserResponse,
    VerifyEmailRequest,
)

# Pre-computed dummy bcrypt hash used to make login timing constant
# regardless of whether the account exists (prevents email enumeration).
_DUMMY_HASH = bcrypt.hashpw(b"dummy-timing-pad", bcrypt.gensalt())

logger = logging.getLogger("violawake.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])

_LOGIN_LOCKOUT_THRESHOLD = 5
_LOGIN_LOCKOUT_DURATION = timedelta(minutes=15)


def reset_rate_limits() -> None:
    """Clear all rate-limit state. Used by test fixtures."""
    reset_shared_rate_limits()


def _registration_pending_response(body: RegisterRequest) -> AuthResponse:
    """Return a generic success response for email-based registration flows."""
    return AuthResponse(
        token="",
        user=UserResponse(
            id=0,
            email=body.email,
            name=body.name,
            email_verified=False,
        ),
    )


async def _current_user_with_rate_key(
    request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Resolve the current user and stash the ID for per-user rate limiting."""
    set_rate_limit_user(request, current_user.id)
    return current_user


def _client_ip(request: Request) -> str:
    """Resolve the client IP, honoring trusted_proxy_count for X-Forwarded-For.

    With trusted_proxy_count=N (N>0), returns the Nth-from-right entry of
    X-Forwarded-For (after the N proxies we trust). With count=0 (default),
    ignores X-Forwarded-For entirely and returns the direct client host.
    """
    proxies = settings.trusted_proxy_count
    if proxies > 0:
        xff = request.headers.get("x-forwarded-for")
        if xff:
            parts = [p.strip() for p in xff.split(",") if p.strip()]
            if len(parts) >= proxies:
                return parts[-proxies]
    return request.client.host


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(REGISTER_LIMIT)
async def register(
    request: Request,
    body: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Register a new user account."""
    result = await db.execute(select(User).where(User.email == body.email))
    email_svc = get_email_service()
    existing_user = result.scalar_one_or_none()
    if existing_user is not None:
        await email_svc.send_existing_account_notice(
            to=existing_user.email,
            name=existing_user.name,
        )
        logger.info("Duplicate registration attempt for existing email %s", body.email)
        return _registration_pending_response(body)

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

    return _registration_pending_response(body)


@router.post("/login", response_model=AuthResponse)
@limiter.limit(LOGIN_LIMIT)
async def login(
    request: Request,
    body: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> AuthResponse:
    """Authenticate and receive a JWT token."""
    now = datetime.now(timezone.utc)
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if user is not None and user.locked_until is not None:
        locked_until = user.locked_until
        if locked_until.tzinfo is None:
            locked_until = locked_until.replace(tzinfo=timezone.utc)
        if locked_until > now:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Account temporarily locked. Try again later.",
            )

    if user is None:
        # Run a dummy bcrypt check so the response time is indistinguishable
        # from a real password verification (prevents email enumeration).
        bcrypt.checkpw(b"dummy", _DUMMY_HASH)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not verify_password(body.password, user.password_hash):
        user.failed_login_count += 1
        if user.failed_login_count >= _LOGIN_LOCKOUT_THRESHOLD:
            user.locked_until = now + _LOGIN_LOCKOUT_DURATION
            user.failed_login_count = 0
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user.failed_login_count = 0
    user.locked_until = None
    await db.commit()

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


async def _apply_email_verification(token: str, db: AsyncSession) -> User:
    """Decode a verify-email token and mark the user verified (idempotent).

    Shared by the JSON POST endpoint (used by the SPA) and the GET endpoint
    (used by the link in the verification email). Raises HTTPException on an
    invalid/expired token or unknown user.
    """
    user_id = decode_action_token(token, expected_purpose="verify_email")

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

    return user


@router.get("/verify-email")
@limiter.limit(VERIFY_EMAIL_LIMIT)
async def verify_email_link(
    request: Request,
    token: str,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RedirectResponse:
    """Verify an email from the link in the verification email, then redirect.

    The verification email links directly here (a GET on the backend) rather
    than to the client-rendered ``/verify-email`` SPA page. Verifying
    server-side means a successful click no longer depends on the frontend's
    CDN routing rendering the SPA route and running JavaScript — the historical
    failure where the link 308-redirected away from the SPA route left every
    real signup unverified. On success (or an already-verified account) the
    browser is sent to the console signed-in; on a bad/expired token it is sent
    to login with an error flag so the user can request a fresh link.
    """
    console = settings.console_base_url.rstrip("/")
    try:
        await _apply_email_verification(token, db)
    except HTTPException:
        return RedirectResponse(
            url=f"{console}/login?verify_error=1",
            status_code=status.HTTP_303_SEE_OTHER,
        )
    return RedirectResponse(
        url=f"{console}/login?verified=1",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@router.post("/verify-email", response_model=MessageResponse)
@limiter.limit(VERIFY_EMAIL_LIMIT)
async def verify_email(
    request: Request,
    body: VerifyEmailRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Verify a user's email address from a signed token (JSON API for the SPA)."""
    await _apply_email_verification(body.token, db)
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


@router.post("/resend-verification", response_model=MessageResponse)
@limiter.limit(RESEND_VERIFICATION_LIMIT)
async def resend_verification(
    request: Request,
    body: ResendVerificationRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Re-send an email-verification link when an unverified account exists.

    Verification tokens expire after 48h (EMAIL_VERIFICATION_TOKEN_HOURS) and
    there was previously no way to request a fresh one, stranding any user
    whose link expired (notably the 2026-05-08→07-03 deep-link outage cohort).

    Privacy-preserving: always returns the same generic response so the
    endpoint can't enumerate which emails have accounts or which are already
    verified. Only sends when the account exists, is still unverified, and the
    email service is enabled.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if user is not None and not user.email_verified:
        email_svc = get_email_service()
        if email_svc.enabled:
            verification_token = create_email_verification_token(user.id)
            await email_svc.send_verification_email(
                to=user.email,
                token=verification_token,
                name=user.name,
            )
            logger.info("Re-sent verification email for user %s", user.id)

    return MessageResponse(
        message="If an unverified account exists for that email, a new verification link has been sent.",
    )


@router.post("/reset-password", response_model=MessageResponse)
@limiter.limit(RESET_PASSWORD_LIMIT)
async def reset_password(
    request: Request,
    body: ResetPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Reset a user's password from a signed, single-use token."""
    user_id = decode_reset_token(body.token)

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
@limiter.limit(CHANGE_PASSWORD_LIMIT, key_func=key_by_user)
async def change_password(
    request: Request,
    body: ChangePasswordRequest,
    current_user: Annotated[User, Depends(_current_user_with_rate_key)],
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
    body: DeleteAccountRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Soft-delete the authenticated user's account and associated data.

    Requires the user's current password for confirmation.
    """
    if not verify_password(body.password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password",
        )

    queue = await init_job_queue()
    await queue.delete_jobs_for_user(current_user.id)
    original_email = current_user.email
    original_name = current_user.name
    now = datetime.now(timezone.utc)
    scheduled_hard_delete_at = now + timedelta(days=30)

    # Delete the Stripe customer before scrubbing local identity fields.
    sub_result = await db.execute(
        select(Subscription).where(Subscription.user_id == current_user.id)
    )
    subscription = sub_result.scalar_one_or_none()
    if subscription and subscription.stripe_customer_id and settings.billing_enabled:
        try:
            import stripe
            stripe.api_key = settings.stripe_secret_key
            stripe.Customer.delete(subscription.stripe_customer_id)
        except Exception:
            logger.warning(
                "Failed to cancel Stripe subscription %s for user %s — proceeding with deletion",
                subscription.stripe_customer_id,
                current_user.id,
            )

    if subscription is not None:
        subscription.tier = "free"
        subscription.status = "canceled"
        subscription.stripe_customer_id = None
        subscription.stripe_subscription_id = None
        subscription.current_period_end = None

    await db.execute(
        update(Recording)
        .where(Recording.user_id == current_user.id, Recording.deleted_at.is_(None))
        .values(deleted_at=now)
    )
    await db.execute(
        update(TrainedModel)
        .where(TrainedModel.user_id == current_user.id, TrainedModel.deleted_at.is_(None))
        .values(deleted_at=now)
    )
    # Training jobs live only in the async job queue's SQLite store; they are
    # cancelled and hard-deleted above via ``queue.delete_jobs_for_user`` (issue
    # #1438 — the Postgres ``training_jobs`` table was never written and is retired).

    current_user.email = f"deleted-user-{current_user.id}@deleted.violawake.local"
    current_user.name = "Deleted user"
    current_user.password_hash = hash_password(secrets.token_urlsafe(32))
    current_user.email_verified = False
    current_user.failed_login_count = 0
    current_user.locked_until = None
    current_user.deleted_at = now
    current_user.scheduled_hard_delete_at = scheduled_hard_delete_at

    await get_email_service().send_account_deleted(
        to=original_email,
        name=original_name,
        scheduled_hard_delete_at=scheduled_hard_delete_at.date().isoformat(),
    )

    logger.info("Deleted account for user %s", current_user.id)
    return MessageResponse(message="Account deleted. Data is scheduled for permanent deletion in 30 days.")
