"""Async training job queue routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.database import get_db
from app.job_queue import Job, QueueFullError, init_job_queue
from app.models import Recording, User
from app.routes.billing import check_training_quota, record_usage
from app.schemas import (
    JobCircuitBreakerResponse,
    JobResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    MessageResponse,
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def serialize_job(job: Job) -> JobResponse:
    """Convert a queue job dataclass into an API response."""
    return JobResponse(
        job_id=job.id,
        user_id=job.user_id,
        wake_word=job.wake_word,
        status=job.status.value,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
        progress_pct=job.progress_pct,
        d_prime=job.d_prime,
        model_id=job.model_id,
    )


async def validate_training_request(
    body: JobSubmitRequest,
    current_user: User,
    db: AsyncSession,
) -> tuple[str, list[int], int]:
    """Validate the submitted recordings for a new training job."""
    wake_word = body.wake_word.strip().lower()
    result = await db.execute(
        select(Recording).where(
            Recording.id.in_(body.recording_ids),
            Recording.user_id == current_user.id,
        )
    )
    recordings = result.scalars().all()

    if len(recordings) != len(body.recording_ids):
        found_ids = {recording.id for recording in recordings}
        missing = [recording_id for recording_id in body.recording_ids if recording_id not in found_ids]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Recordings not found or not owned by you: %s" % missing,
        )

    wrong_word = [recording.id for recording in recordings if recording.wake_word != wake_word]
    if wrong_word:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Recordings %s do not match wake word '%s'" % (wrong_word, body.wake_word),
        )

    if len(recordings) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 5 recordings. Got %s." % len(recordings),
        )

    return wake_word, list(body.recording_ids), body.epochs


async def submit_training_job(
    body: JobSubmitRequest,
    current_user: User,
    db: AsyncSession,
) -> JobSubmitResponse:
    """Validate and enqueue a training job."""
    wake_word, recording_ids, epochs = await validate_training_request(body, current_user, db)
    queue = await init_job_queue()

    try:
        job_id = await queue.submit_job(
            user_id=current_user.id,
            wake_word=wake_word,
            recording_ids=recording_ids,
            epochs=epochs,
        )
    except QueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    await record_usage(db, current_user.id, action="training_job")
    return JobSubmitResponse(job_id=job_id, status="queued")


async def get_owned_job_or_404(job_id: int, current_user: User) -> Job:
    """Return an owned job or raise 404."""
    job = await (await init_job_queue()).get_job(job_id)
    if job is None or job.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")
    return job


@router.post("", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    body: JobSubmitRequest,
    current_user: Annotated[User, Depends(check_training_quota)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobSubmitResponse:
    """Submit a new training job."""
    return await submit_training_job(body, current_user, db)


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
) -> list[JobResponse]:
    """List the current user's training jobs."""
    jobs = await (await init_job_queue()).list_jobs(current_user.id)
    return [serialize_job(job) for job in jobs]


@router.post("/resume", response_model=MessageResponse)
async def resume_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
) -> MessageResponse:
    """Manually resume a user's paused queue after circuit breaker activation."""
    await (await init_job_queue()).resume_user(current_user.id)
    return MessageResponse(message="Training queue resumed")


@router.get("/circuit-breaker/state", response_model=JobCircuitBreakerResponse)
async def get_circuit_breaker_state(
    current_user: Annotated[User, Depends(get_current_user)],
) -> JobCircuitBreakerResponse:
    """Return the current user's circuit breaker state."""
    breaker = await (await init_job_queue()).get_circuit_breaker(current_user.id)
    return JobCircuitBreakerResponse(
        consecutive_failures=breaker.consecutive_failures,
        paused=breaker.paused,
        next_attempt_at=breaker.next_attempt_at,
        last_failure_at=breaker.last_failure_at,
        pause_reason=breaker.pause_reason,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
) -> JobResponse:
    """Return one training job."""
    job = await get_owned_job_or_404(job_id, current_user)
    return serialize_job(job)


@router.delete("/{job_id}", response_model=MessageResponse)
async def cancel_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
) -> MessageResponse:
    """Cancel a pending or running training job."""
    await get_owned_job_or_404(job_id, current_user)
    cancelled = await (await init_job_queue()).cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training job can no longer be cancelled",
        )
    return MessageResponse(message="Training job cancellation requested")
