"""Training quota enforcement.

ViolaWake Console is a free service; these limits exist to protect shared
training capacity, not to sell an upgrade. Legacy tier names are kept because
existing subscription rows still carry them.
"""

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_verified_user
from app.database import get_db
from app.models import Subscription, UsageRecord, User

logger = logging.getLogger("violawake.quota")

# ---------------------------------------------------------------------------
# Tier limits — single source of truth
# ---------------------------------------------------------------------------

TIER_LIMITS: dict[str, int | None] = {
    "free": 3,
    "developer": 20,
    "business": None,      # unlimited
    "enterprise": None,    # unlimited / custom
}


def _current_period_start() -> datetime:
    """Return the first instant of the current UTC month (usage period)."""
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _current_period_end() -> datetime:
    """Return the first instant of the next UTC month (end of usage period)."""
    now = datetime.now(timezone.utc)
    if now.month == 12:
        return now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


async def _get_or_create_subscription(db: AsyncSession, user_id: int) -> Subscription:
    """Return the user's subscription row, creating a free-tier one if absent."""
    result = await db.execute(
        select(Subscription).where(Subscription.user_id == user_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        sub = Subscription(user_id=user_id, tier="free", status="active")
        db.add(sub)
        await db.flush()
    return sub


async def _get_usage_count(db: AsyncSession, user_id: int) -> int:
    """Return the number of training jobs started in the current usage period."""
    period_start = _current_period_start()
    result = await db.execute(
        select(UsageRecord).where(
            UsageRecord.user_id == user_id,
            UsageRecord.action == "training_job",
            UsageRecord.period_start == period_start,
        )
    )
    record = result.scalar_one_or_none()
    return record.count if record else 0


# ---------------------------------------------------------------------------
# Public helper: record_usage (called by training service)
# ---------------------------------------------------------------------------

async def record_usage(db: AsyncSession, user_id: int, action: str = "training_job") -> None:
    """Increment the usage counter for the current usage period.

    Creates the UsageRecord row if it does not exist yet.

    Uses atomic SQL to avoid a read-modify-write race where two concurrent
    requests could both pass the quota check before either increments.
    """
    period_start = _current_period_start()

    # Attempt atomic UPDATE first: SET count = count + 1 in SQL, not Python.
    result = await db.execute(
        update(UsageRecord)
        .where(
            UsageRecord.user_id == user_id,
            UsageRecord.action == action,
            UsageRecord.period_start == period_start,
        )
        .values(count=UsageRecord.count + 1)
    )

    if result.rowcount == 0:
        # No existing row — insert one.  If a concurrent request races us and
        # inserts between our UPDATE and this INSERT, the unique constraint
        # (user_id, action, period_start) will raise IntegrityError; retry
        # with the atomic UPDATE which is now guaranteed to match.
        try:
            db.add(UsageRecord(
                user_id=user_id,
                action=action,
                period_start=period_start,
                count=1,
            ))
            await db.flush()
        except IntegrityError:
            await db.rollback()
            await db.execute(
                update(UsageRecord)
                .where(
                    UsageRecord.user_id == user_id,
                    UsageRecord.action == action,
                    UsageRecord.period_start == period_start,
                )
                .values(count=UsageRecord.count + 1)
            )
            await db.flush()
    else:
        await db.flush()


async def refund_usage(
    db: AsyncSession,
    user_id: int,
    period_start: datetime,
    action: str = "training_job",
) -> None:
    """Credit back one previously-charged attempt for a SPECIFIC usage period.

    Charge-at-submit (``record_usage`` above) is a deliberate premise -- a
    submission consumes training compute -- but it breaks when a job dies on
    OUR infrastructure before it consumes that compute (#4207). The refund is
    the exception to the rule, and it must be exact in two ways:

    * **Period-correct.** The credit lands on ``period_start`` -- the period the
      charge was actually made in, carried on the job row -- never the current
      period. A 07-31 submit that fails on 08-01 must decrement July's counter,
      not mint a bonus attempt in August.
    * **Clamped, never negative.** ``count > 0`` is part of the WHERE clause, so
      a refund can only lower a positive counter. If no row exists for that
      period (e.g. the charge was never actually recorded), or the counter is
      already zero, the statement matches nothing and does nothing -- it can
      never create a row or push a counter below zero, so it can never manufacture
      quota the user did not have.

    Idempotency across "however many paths ask" is the CALLER's guard (the job
    row's one-shot ``usage_refunded`` flag), not this function's: this decrements
    unconditionally when it matches, so calling it twice for one charge would
    over-credit. It is deliberately the narrow arithmetic primitive; the job
    queue owns the once-only claim.
    """
    await db.execute(
        update(UsageRecord)
        .where(
            UsageRecord.user_id == user_id,
            UsageRecord.action == action,
            UsageRecord.period_start == period_start,
            UsageRecord.count > 0,
        )
        .values(count=UsageRecord.count - 1)
    )
    await db.flush()


# ---------------------------------------------------------------------------
# Public helper: check_training_quota (dependency for training route)
# ---------------------------------------------------------------------------

async def check_training_quota(
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """FastAPI dependency that enforces the training quota for the current user.

    Returns the User if quota is available; raises HTTP 403 if the monthly
    limit has been reached.
    """
    sub = await _get_or_create_subscription(db, current_user.id)
    limit = TIER_LIMITS.get(sub.tier)

    # None means unlimited
    if limit is None:
        return current_user

    used = await _get_usage_count(db, current_user.id)

    # Warn at 80% of limit (best-effort, non-blocking).
    if limit and used == int(limit * 0.8):
        try:
            from app.email_service import get_email_service

            email_svc = get_email_service()
            if email_svc.enabled:
                import asyncio

                asyncio.create_task(
                    email_svc.send_quota_warning(
                        to=current_user.email,
                        used=used,
                        limit=limit,
                    )
                )
        except Exception:
            pass  # non-critical — don't block training

    if used >= limit:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Monthly training limit reached ({used}/{limit}). "
                "The limit resets at the start of next month."
            ),
        )

    return current_user
