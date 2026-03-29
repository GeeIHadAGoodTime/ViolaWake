"""Compatibility routes for legacy training endpoints."""

import asyncio
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.auth import decode_download_token, decode_token, get_verified_user
from app.database import get_db
from app.job_queue import init_job_queue
from app.models import User
from app.rate_limit import TRAINING_SUBMIT_LIMIT, key_by_user, limiter, set_rate_limit_user
from app.routes.billing import check_training_quota
from app.routes.jobs import get_owned_job_or_404, submit_training_job
from app.schemas import JobSubmitRequest, TrainingStartResponse, TrainingStatusResponse

router = APIRouter(prefix="/api/training", tags=["training"])


async def _quota_user_with_rate_key(
    request: Request,
    current_user: Annotated[User, Depends(check_training_quota)],
) -> User:
    """Resolve the user via training-quota check and stash ID for rate limiting."""
    set_rate_limit_user(request, current_user.id)
    return current_user


def _legacy_status(status_value: str) -> str:
    """Map queue status names to the legacy training API values."""
    if status_value == "pending":
        return "queued"
    return status_value


@router.post("/start", response_model=TrainingStartResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(TRAINING_SUBMIT_LIMIT, key_func=key_by_user)
async def start_training(
    request: Request,
    body: JobSubmitRequest,
    current_user: Annotated[User, Depends(_quota_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TrainingStartResponse:
    """Submit a training job through the persistent queue."""
    job_response = await submit_training_job(body, current_user, db)
    return TrainingStartResponse(job_id=job_response.job_id, status=job_response.status)


@router.get("/status/{job_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    job_id: int,
    current_user: Annotated[User, Depends(get_verified_user)],
) -> TrainingStatusResponse:
    """Get the current status of a queued training job."""
    job = await get_owned_job_or_404(job_id, current_user)
    return TrainingStatusResponse(
        job_id=job.id,
        status=_legacy_status(job.status.value),
        progress=job.progress_pct,
        d_prime=job.d_prime,
        model_id=job.model_id,
        error=job.error,
    )


async def _resolve_sse_user(
    request: Request,
    token: str | None,
    db: AsyncSession,
    *,
    job_id: int,
) -> User:
    """Resolve the authenticated user for SSE endpoints."""
    user_id: int | None = None
    if token:
        user_id = decode_download_token(
            token,
            expected_action="training_stream",
            expected_resource_id=job_id,
        )
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
    if not user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verify your email address to access recording, training, and billing features.",
        )
    return user


@router.get("/stream/{job_id}")
async def stream_training(
    job_id: int,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    token: str | None = Query(default=None),
) -> EventSourceResponse:
    """Stream queued training progress via Server-Sent Events."""
    current_user = await _resolve_sse_user(request, token, db, job_id=job_id)
    job = await get_owned_job_or_404(job_id, current_user)
    queue_manager = await init_job_queue()

    if job.status.value in {"completed", "failed", "cancelled"}:

        async def _final_event():
            yield {
                "event": "training",
                "data": json.dumps({
                    "status": _legacy_status(job.status.value),
                    "progress": job.progress_pct,
                    "epoch": 0,
                    "total_epochs": job.epochs,
                    "train_loss": 0.0,
                    "val_loss": 0.0,
                    "d_prime": job.d_prime,
                    "model_id": job.model_id,
                    "message": job.error if job.status.value != "completed" else "Training complete.",
                    "error": job.error,
                }),
            }

        return EventSourceResponse(_final_event())

    queue = queue_manager.subscribe(job_id)

    async def _event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if "status" in event:
                        event = {**event, "status": _legacy_status(str(event["status"]))}
                    yield {
                        "event": "training",
                        "data": json.dumps(event),
                    }
                    if event.get("status") in {"completed", "failed", "cancelled"}:
                        break
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            queue_manager.unsubscribe(job_id, queue)

    return EventSourceResponse(_event_generator())
