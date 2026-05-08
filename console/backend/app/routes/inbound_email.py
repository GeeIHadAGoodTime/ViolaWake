"""Inbound support email webhook routes."""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from email.utils import parseaddr
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.email_service import get_email_service
from app.models import InboundEmailAutoReply

router = APIRouter(prefix="/api/email", tags=["email"])

_BLOCKED_LOCAL_PARTS = {"mailer-daemon", "postmaster", "no-reply", "noreply"}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _candidate_values(payload: Any) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    candidates: list[Any] = []
    for key in ("from", "sender", "reply_to", "replyTo", "mail_from", "mailFrom"):
        if key in payload:
            candidates.append(payload[key])
    message = payload.get("message")
    if isinstance(message, dict):
        candidates.extend(_candidate_values(message))
    return candidates


def _email_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        _, address = parseaddr(value)
        return address.strip().lower() or None
    if isinstance(value, dict):
        for key in ("email", "address", "value"):
            address = _email_from_value(value.get(key))
            if address:
                return address
    if isinstance(value, list):
        for item in value:
            address = _email_from_value(item)
            if address:
                return address
    return None


def _extract_sender_email(payload: dict[str, Any]) -> str:
    for value in _candidate_values(payload):
        address = _email_from_value(value)
        if not address:
            continue
        local_part = address.split("@", 1)[0].lower()
        if local_part in _BLOCKED_LOCAL_PARTS:
            raise HTTPException(
                status_code=status.HTTP_202_ACCEPTED,
                detail="Auto-reply skipped for automated sender.",
            )
        return address
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Inbound email payload is missing a sender address.",
    )


def _require_inbound_secret(secret: str | None) -> None:
    expected = settings.email_inbound_webhook_secret.strip()
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found")
    if not secrets.compare_digest(secret or "", expected):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")


@router.post("/inbound")
async def inbound_email_webhook(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    x_violawake_email_secret: str | None = Header(default=None),
) -> dict[str, Any]:
    """Receive inbound support email webhook events and send a deduped auto-reply."""
    _require_inbound_secret(x_violawake_email_secret)

    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Inbound email payload must be a JSON object.",
        )

    sender_email = _extract_sender_email(payload)
    cutoff = _utcnow() - timedelta(hours=settings.support_autoreply_window_hours)

    existing_result = await db.execute(
        select(InboundEmailAutoReply)
        .where(
            InboundEmailAutoReply.sender_email == sender_email,
            InboundEmailAutoReply.created_at >= cutoff,
        )
        .order_by(desc(InboundEmailAutoReply.created_at))
        .limit(1)
    )
    existing = existing_result.scalar_one_or_none()
    if existing is not None:
        return {
            "status": "deduped",
            "sent": False,
            "ticket_reference": existing.ticket_reference,
        }

    ticket_reference = f"VW-{secrets.token_hex(4).upper()}"
    db.add(
        InboundEmailAutoReply(
            sender_email=sender_email,
            ticket_reference=ticket_reference,
            created_at=_utcnow(),
        )
    )
    await db.flush()

    sent = await get_email_service().send_support_autoreply(
        to=sender_email,
        ticket_reference=ticket_reference,
    )
    return {
        "status": "sent" if sent else "email_unavailable",
        "sent": sent,
        "ticket_reference": ticket_reference,
    }
