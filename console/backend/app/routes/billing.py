"""Billing routes: Stripe checkout, webhook, subscription management, usage."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_verified_user
from app.config import settings
from app.database import get_db
from app.models import ProcessedStripeEvent, Subscription, UsageRecord, User
from app.rate_limit import CHECKOUT_LIMIT, PORTAL_LIMIT, key_by_user, limiter, set_rate_limit_user
from app.schemas import (
    BillingPortalResponse,
    CheckoutRequest,
    CheckoutResponse,
    SubscriptionResponse,
    UsageResponse,
)

logger = logging.getLogger("violawake.billing")

router = APIRouter(prefix="/api/billing", tags=["billing"])

# ---------------------------------------------------------------------------
# Tier limits — single source of truth
# ---------------------------------------------------------------------------

TIER_LIMITS: dict[str, int | None] = {
    "free": 3,
    "developer": 20,
    "business": None,      # unlimited
    "enterprise": None,    # unlimited / custom
}

TIER_PRICE_MAP: dict[str, str] = {
    "developer": "stripe_price_developer",
    "business": "stripe_price_business",
}

# ---------------------------------------------------------------------------
# Webhook idempotency: persist Stripe event IDs so redeliveries after a backend
# restart do not re-run billing side effects.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_billing() -> None:
    """Raise 503 if Stripe is not configured."""
    if not settings.billing_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Billing is not configured. Set VIOLAWAKE_STRIPE_SECRET_KEY to enable.",
        )


def _get_stripe():
    """Return the stripe module, configured with the secret key.

    Deferred import so the app starts even when stripe is not installed
    (e.g., in development without billing).
    """
    import stripe
    stripe.api_key = settings.stripe_secret_key
    return stripe


async def _verified_user_with_rate_key(
    request: Request,
    current_user: Annotated[User, Depends(get_verified_user)],
) -> User:
    """Resolve the verified user and stash the ID for per-user rate limiting."""
    set_rate_limit_user(request, current_user.id)
    return current_user


def _current_period_start() -> datetime:
    """Return the first instant of the current UTC month (billing period)."""
    now = datetime.now(timezone.utc)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def _current_period_end() -> datetime:
    """Return the first instant of the next UTC month (end of billing period)."""
    now = datetime.now(timezone.utc)
    if now.month == 12:
        return now.replace(year=now.year + 1, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    return now.replace(month=now.month + 1, day=1, hour=0, minute=0, second=0, microsecond=0)


def _price_id_for_tier(tier: str) -> str:
    """Resolve the Stripe Price ID for a tier, or raise if not configured."""
    attr = TIER_PRICE_MAP.get(tier)
    if attr is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tier '{tier}' is not available for checkout.",
        )
    price_id = getattr(settings, attr, "")
    if not price_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Stripe price for '{tier}' tier is not configured.",
        )
    return price_id


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
    """Return the number of training jobs started in the current billing period."""
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


async def _get_or_create_stripe_customer(
    stripe, db: AsyncSession, user: User, sub: Subscription,
) -> str:
    """Ensure a Stripe Customer exists for this user. Return the customer ID."""
    if sub.stripe_customer_id:
        return sub.stripe_customer_id

    customer = stripe.Customer.create(
        email=user.email,
        name=user.name,
        metadata={"violawake_user_id": str(user.id)},
    )
    sub.stripe_customer_id = customer.id
    await db.flush()
    return customer.id


async def _cleanup_processed_stripe_events(db: AsyncSession) -> None:
    """Delete processed Stripe event IDs older than 30 days."""
    bind = db.get_bind()
    dialect_name = getattr(getattr(bind, "dialect", None), "name", "")
    if dialect_name == "postgresql":
        await db.execute(
            text(
                "DELETE FROM processed_stripe_events "
                "WHERE processed_at < NOW() - INTERVAL '30 days'"
            )
        )
        return

    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    await db.execute(
        delete(ProcessedStripeEvent).where(ProcessedStripeEvent.processed_at < cutoff)
    )


async def _record_stripe_event_if_new(db: AsyncSession, event_id: str | None) -> bool:
    """Return True when this Stripe event ID has not already been processed."""
    await _cleanup_processed_stripe_events(db)
    if not event_id:
        return True

    result = await db.execute(
        text(
            "INSERT INTO processed_stripe_events (event_id) "
            "VALUES (:event_id) "
            "ON CONFLICT (event_id) DO NOTHING"
        ),
        {"event_id": event_id},
    )
    return result.rowcount != 0


# ---------------------------------------------------------------------------
# Public helper: record_usage (called by training service)
# ---------------------------------------------------------------------------

async def record_usage(db: AsyncSession, user_id: int, action: str = "training_job") -> None:
    """Increment the usage counter for the current billing period.

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
                        tier=sub.tier,
                    )
                )
        except Exception:
            pass  # non-critical — don't block training

    if used >= limit:
        tier_name = sub.tier.capitalize()
        if sub.tier == "free":
            upgrade_msg = "Upgrade to Developer for 20 models/month."
        elif sub.tier == "developer":
            upgrade_msg = "Upgrade to Business for unlimited models."
        else:
            upgrade_msg = "Contact sales for a custom plan."

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Monthly training limit reached ({used}/{limit}). "
                f"You are on the {tier_name} plan. {upgrade_msg}"
            ),
            headers={"X-Upgrade-URL": "/pricing"},
        )

    return current_user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/checkout", response_model=CheckoutResponse)
@limiter.limit(CHECKOUT_LIMIT, key_func=key_by_user)
async def create_checkout_session(
    request: Request,
    body: CheckoutRequest,
    current_user: Annotated[User, Depends(_verified_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> CheckoutResponse:
    """Create a Stripe Checkout Session for upgrading to a paid tier."""
    _require_billing()
    stripe = _get_stripe()

    price_id = _price_id_for_tier(body.tier)
    sub = await _get_or_create_subscription(db, current_user.id)
    customer_id = await _get_or_create_stripe_customer(stripe, db, current_user, sub)

    # Prevent checkout if already on the requested tier (or higher)
    tier_rank = {"free": 0, "developer": 1, "business": 2, "enterprise": 3}
    if tier_rank.get(sub.tier, 0) >= tier_rank.get(body.tier, 0) and sub.status == "active":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You are already on the {sub.tier} plan (or higher).",
        )

    # Mitigation for shared-Stripe-account branding (the account display name
    # is "Viola Voice Assistant" because NOVVIOLA shares this Stripe account).
    # The header brand cannot be overridden per-session without Stripe Connect,
    # but `subscription_data.description` (shown on receipts) and
    # `custom_text.submit.message` (shown above the pay button) make it
    # unambiguous that the user is subscribing to ViolaWake.
    tier_label = body.tier.capitalize()
    subscription_data: dict = {
        "metadata": {
            "violawake_user_id": str(current_user.id),
            "tier": body.tier,
        },
        "description": f"ViolaWake {tier_label} subscription (violawake.com)",
    }

    # Add free trial period if configured (VIOLAWAKE_TRIAL_DAYS, default 14, 0 to disable)
    if settings.trial_days > 0:
        subscription_data["trial_period_days"] = settings.trial_days
        logger.info(
            "Adding %d-day free trial to checkout for user %d, tier=%s",
            settings.trial_days, current_user.id, body.tier,
        )

    session = stripe.checkout.Session.create(
        customer=customer_id,
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=f"{settings.console_base_url}/billing?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{settings.console_base_url}/pricing",
        metadata={
            "violawake_user_id": str(current_user.id),
            "tier": body.tier,
        },
        subscription_data=subscription_data,
        custom_text={
            "submit": {
                "message": (
                    "You're subscribing to ViolaWake (violawake.com). "
                    "ViolaWake bills via the same Stripe account as Viola Voice Assistant; "
                    "your charge will appear under that account name."
                )
            },
        },
    )

    logger.info(
        "Checkout session created for user %d, tier=%s, session=%s",
        current_user.id, body.tier, session.id,
    )
    return CheckoutResponse(checkout_url=session.url)


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    stripe_signature: Annotated[str | None, Header(alias="stripe-signature")] = None,
) -> dict:
    """Handle Stripe webhook events.

    This endpoint is called directly by Stripe -- it is NOT authenticated
    via JWT. Instead, the webhook signature is verified using the shared
    secret.
    """
    _require_billing()
    stripe = _get_stripe()

    if not settings.stripe_webhook_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Webhook secret not configured.",
        )

    body = await request.body()

    if not stripe_signature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing stripe-signature header.",
        )

    try:
        event = stripe.Webhook.construct_event(
            payload=body,
            sig_header=stripe_signature,
            secret=settings.stripe_webhook_secret,
        )
    except stripe.error.SignatureVerificationError:
        logger.warning("Webhook signature verification failed")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid webhook signature.",
        )

    # Deduplicate: Stripe may deliver the same event more than once.
    event_id = event.get("id")
    if not await _record_stripe_event_if_new(db, event_id):
        logger.debug("Duplicate webhook event ignored: %s", event_id)
        return {"status": "ok"}

    event_type = event["type"]
    data = event["data"]["object"]
    logger.info("Stripe webhook received: %s (id=%s)", event_type, event_id)

    if event_type == "checkout.session.completed":
        await _handle_checkout_completed(db, data)
    elif event_type == "customer.subscription.updated":
        await _handle_subscription_updated(db, data)
    elif event_type == "customer.subscription.deleted":
        await _handle_subscription_deleted(db, data)
    elif event_type == "invoice.payment_failed":
        await _handle_payment_failed(db, data)
    else:
        logger.debug("Ignoring unhandled webhook event type: %s", event_type)

    return {"status": "ok"}


@router.get("/subscription", response_model=SubscriptionResponse)
async def get_subscription(
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> SubscriptionResponse:
    """Return the current user's subscription tier, status, and usage."""
    sub = await _get_or_create_subscription(db, current_user.id)
    used = await _get_usage_count(db, current_user.id)
    limit = TIER_LIMITS.get(sub.tier)

    # Fetch trial status from Stripe if subscription exists
    trial_active = False
    trial_end = None
    if sub.stripe_subscription_id and settings.billing_enabled:
        try:
            stripe = _get_stripe()
            stripe_sub = stripe.Subscription.retrieve(sub.stripe_subscription_id)
            if stripe_sub.status == "trialing" and stripe_sub.trial_end:
                trial_active = True
                trial_end = datetime.fromtimestamp(stripe_sub.trial_end, tz=timezone.utc)
        except Exception:
            logger.exception(
                "Failed to fetch trial status for subscription %s",
                sub.stripe_subscription_id,
            )

    return SubscriptionResponse(
        tier=sub.tier,
        status=sub.status,
        current_period_end=sub.current_period_end,
        trial_active=trial_active,
        trial_end=trial_end,
        usage=UsageResponse(
            models_used=used,
            models_limit=limit,
            period_start=_current_period_start(),
            period_end=_current_period_end(),
        ),
    )


@router.post("/portal", response_model=BillingPortalResponse)
@limiter.limit(PORTAL_LIMIT, key_func=key_by_user)
async def create_billing_portal(
    request: Request,
    current_user: Annotated[User, Depends(_verified_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> BillingPortalResponse:
    """Create a Stripe Billing Portal session for managing the subscription."""
    _require_billing()
    stripe = _get_stripe()

    sub = await _get_or_create_subscription(db, current_user.id)
    if not sub.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No billing account found. Subscribe to a plan first.",
        )

    session = stripe.billing_portal.Session.create(
        customer=sub.stripe_customer_id,
        return_url=f"{settings.console_base_url}/billing",
    )

    return BillingPortalResponse(url=session.url)


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> UsageResponse:
    """Return the current month's usage vs the tier limit."""
    sub = await _get_or_create_subscription(db, current_user.id)
    used = await _get_usage_count(db, current_user.id)
    limit = TIER_LIMITS.get(sub.tier)

    return UsageResponse(
        models_used=used,
        models_limit=limit,
        period_start=_current_period_start(),
        period_end=_current_period_end(),
    )


# ---------------------------------------------------------------------------
# Webhook event handlers
# ---------------------------------------------------------------------------

def _tier_from_price_id(price_id: str) -> str:
    """Resolve a Stripe Price ID back to a ViolaWake tier name."""
    if price_id == settings.stripe_price_developer:
        return "developer"
    if price_id == settings.stripe_price_business:
        return "business"
    logger.warning("Unknown Stripe price ID: %s — defaulting to 'developer'", price_id)
    return "developer"


async def _handle_checkout_completed(db: AsyncSession, session: dict) -> None:
    """Handle checkout.session.completed: create/update subscription."""
    metadata = session.get("metadata", {})
    user_id_str = metadata.get("violawake_user_id")
    tier = metadata.get("tier")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    if not user_id_str:
        logger.error("checkout.session.completed missing violawake_user_id in metadata: %s", session.get("id"))
        return

    user_id = int(user_id_str)
    sub = await _get_or_create_subscription(db, user_id)
    sub.stripe_customer_id = customer_id
    sub.stripe_subscription_id = subscription_id
    sub.tier = tier or "developer"
    sub.status = "active"

    # Fetch the subscription object to get current_period_end
    if subscription_id:
        try:
            stripe = _get_stripe()
            stripe_sub = stripe.Subscription.retrieve(subscription_id)
            sub.current_period_end = datetime.fromtimestamp(
                stripe_sub.current_period_end, tz=timezone.utc,
            )
        except Exception:
            logger.exception("Failed to fetch subscription %s for period end", subscription_id)

    await db.flush()
    logger.info(
        "Subscription activated: user=%d tier=%s stripe_sub=%s",
        user_id, sub.tier, subscription_id,
    )


async def _handle_subscription_updated(db: AsyncSession, subscription: dict) -> None:
    """Handle customer.subscription.updated: tier change, renewal, etc."""
    stripe_sub_id = subscription.get("id")
    result = await db.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == stripe_sub_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        logger.warning("subscription.updated for unknown stripe_subscription_id=%s", stripe_sub_id)
        return

    # Update status
    stripe_status = subscription.get("status", "active")
    status_map = {
        "active": "active",
        "past_due": "past_due",
        "canceled": "canceled",
        "unpaid": "past_due",
        "incomplete": "past_due",
        "incomplete_expired": "canceled",
        "trialing": "active",
        "paused": "canceled",
    }
    sub.status = status_map.get(stripe_status, "active")

    # Update period end
    period_end = subscription.get("current_period_end")
    if period_end:
        sub.current_period_end = datetime.fromtimestamp(period_end, tz=timezone.utc)

    # Update tier from the subscription's price
    items = subscription.get("items", {}).get("data", [])
    if items:
        price_id = items[0].get("price", {}).get("id", "")
        if price_id:
            sub.tier = _tier_from_price_id(price_id)

    # Also check metadata for explicit tier override
    meta_tier = subscription.get("metadata", {}).get("tier")
    if meta_tier and meta_tier in TIER_LIMITS:
        sub.tier = meta_tier

    await db.flush()
    logger.info(
        "Subscription updated: user=%d tier=%s status=%s",
        sub.user_id, sub.tier, sub.status,
    )


async def _handle_subscription_deleted(db: AsyncSession, subscription: dict) -> None:
    """Handle customer.subscription.deleted: downgrade to free."""
    stripe_sub_id = subscription.get("id")
    result = await db.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == stripe_sub_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        logger.warning("subscription.deleted for unknown stripe_subscription_id=%s", stripe_sub_id)
        return

    old_tier = sub.tier
    sub.tier = "free"
    sub.status = "canceled"
    sub.stripe_subscription_id = None
    sub.current_period_end = None
    await db.flush()
    logger.info(
        "Subscription deleted: user=%d downgraded from %s to free",
        sub.user_id, old_tier,
    )


async def _handle_payment_failed(db: AsyncSession, invoice: dict) -> None:
    """Handle invoice.payment_failed: mark subscription as past_due."""
    stripe_sub_id = invoice.get("subscription")
    if not stripe_sub_id:
        logger.debug("invoice.payment_failed without subscription ID, ignoring")
        return

    result = await db.execute(
        select(Subscription).where(Subscription.stripe_subscription_id == stripe_sub_id)
    )
    sub = result.scalar_one_or_none()
    if sub is None:
        logger.warning("payment_failed for unknown stripe_subscription_id=%s", stripe_sub_id)
        return

    sub.status = "past_due"
    await db.flush()
    logger.info(
        "Payment failed: user=%d subscription=%s set to past_due",
        sub.user_id, stripe_sub_id,
    )
