"""Async training job queue routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import (
    get_service_or_verified_user,
    is_service_user,
    resolve_queue_partition,
)
from app.database import get_db
from app.job_queue import Job, QueueFullError, TooManyPendingJobsError, init_job_queue
from app.models import Recording, User
from app.rate_limit import TRAINING_SUBMIT_LIMIT, key_by_user, limiter, set_rate_limit_user
from app.quota import _current_period_start, check_training_quota, record_usage
from app.schemas import (
    JobCircuitBreakerResponse,
    JobResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    MessageResponse,
)
from app.tenancy import QueuePartition

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


async def _quota_user_with_rate_key(
    request: Request,
    current_user: Annotated[User, Depends(check_training_quota)],
) -> User:
    """Resolve the user via training-quota check and stash ID for rate limiting."""
    set_rate_limit_user(request, current_user.id)
    return current_user


async def _service_or_quota_user_with_rate_key(
    request: Request,
    candidate_user: Annotated[User, Depends(get_service_or_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """Privileged service key OR end-user with quota check.

    Service-key callers (Viola backend) bypass per-user training quotas:
    upstream Viola handles its own per-tenant metering. End-user callers go
    through the standard ``check_training_quota`` path.

    The rate-limit key is the caller's PARTITION, not the account. Keyed on
    the account, ``TRAINING_SUBMIT_LIMIT`` (5/hour) would be five trainings
    per hour for the service caller's entire install base.
    """
    if is_service_user(candidate_user):
        set_rate_limit_user(request, resolve_queue_partition(request, candidate_user))
        return candidate_user
    # Mirror check_training_quota inline so we can reuse the already-resolved
    # User and avoid double DB lookups.
    quota_user = await check_training_quota(candidate_user, db)
    set_rate_limit_user(request, quota_user.id)
    return quota_user


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
            Recording.deleted_at.is_(None),
        )
    )
    recordings = result.scalars().all()

    if len(recordings) != len(body.recording_ids):
        found_ids = {recording.id for recording in recordings}
        missing = [recording_id for recording_id in body.recording_ids if recording_id not in found_ids]
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recordings not found or not owned by you: {missing}",
        )

    wrong_word = [recording.id for recording in recordings if recording.wake_word != wake_word]
    if wrong_word:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recordings {wrong_word} do not match wake word '{body.wake_word}'",
        )

    if len(recordings) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Need at least 5 recordings. Got {len(recordings)}.",
        )

    return wake_word, list(body.recording_ids), body.epochs


async def submit_training_job(
    body: JobSubmitRequest,
    current_user: User,
    db: AsyncSession,
    partition: QueuePartition,
) -> JobSubmitResponse:
    """Validate and enqueue a training job in *partition*."""
    wake_word, recording_ids, epochs = await validate_training_request(body, current_user, db)
    queue = await init_job_queue()

    # A paused breaker (FAILURE_THRESHOLD real failures, next_attempt_at=NULL)
    # only clears via resume_user -- the dispatcher skips its jobs silently and
    # they can never auto-run. Charging an attempt for a job that can never
    # dispatch is exactly the #4207 harm, so refuse BEFORE record_usage fires.
    # Only `paused` blocks here; a plain backoff (next_attempt_at in the future)
    # is temporary and the job WILL run once it elapses, so those still enqueue.
    breaker = await queue.get_circuit_breaker(partition)
    if breaker.paused:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Training is paused after repeated failures and must be resumed "
                "before you can submit again. No training attempt was used."
            ),
            headers={"X-Training-Paused": "1"},
        )

    try:
        job_id = await queue.submit_job(
            partition=partition,
            wake_word=wake_word,
            recording_ids=recording_ids,
            epochs=epochs,
        )
    except TooManyPendingJobsError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except QueueFullError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    # Charge the attempt, and stamp the job with the account + period the charge
    # landed in. The failure handler reads those back to refund the SAME period
    # (so a month-boundary failure cannot mint a bonus attempt) exactly once
    # (via the row's one-shot usage_refunded flag) if the job dies on our
    # infrastructure. Stamp first so the worker can never observe a charged job
    # that is not yet marked refundable.
    period_start = _current_period_start()
    await queue.mark_usage_charged(job_id, user_id=current_user.id, period_start=period_start)
    await record_usage(db, current_user.id, action="training_job")
    return JobSubmitResponse(job_id=job_id, status="queued")


async def get_owned_job_or_404(job_id: int, partition: QueuePartition) -> Job:
    """Return a job owned by *partition* or raise 404.

    Ownership is the whole partition, not just the account. On the service
    path every upstream user's job sits under one account, so an account-only
    check would let any one of them read or cancel any other's training job.
    """
    job = await (await init_job_queue()).get_job(job_id)
    if job is None or job.partition != partition:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")
    return job


@router.post("", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit(TRAINING_SUBMIT_LIMIT, key_func=key_by_user)
async def create_job(
    request: Request,
    body: JobSubmitRequest,
    current_user: Annotated[User, Depends(_service_or_quota_user_with_rate_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobSubmitResponse:
    """Submit a new training job."""
    partition = resolve_queue_partition(request, current_user)
    return await submit_training_job(body, current_user, db, partition)


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    request: Request,
    current_user: Annotated[User, Depends(get_service_or_verified_user)],
) -> list[JobResponse]:
    """List the caller's training jobs."""
    partition = resolve_queue_partition(request, current_user)
    jobs = await (await init_job_queue()).list_jobs(partition)
    return [serialize_job(job) for job in jobs]


@router.post("/resume", response_model=MessageResponse)
async def resume_jobs(
    request: Request,
    current_user: Annotated[User, Depends(get_service_or_verified_user)],
) -> MessageResponse:
    """Manually resume the caller's paused queue after a breaker trip.

    Reachable from the service path too, which is the point: a service-key
    caller could previously neither trip its own breaker in isolation nor
    clear it, because this endpoint demanded an end-user JWT it has no way to
    present. That is how one paused queue stranded pending work for 27h+ with
    no API able to release it. It now resumes exactly the caller's partition.
    """
    partition = resolve_queue_partition(request, current_user)
    await (await init_job_queue()).resume_user(partition)
    return MessageResponse(message="Training queue resumed")


@router.get("/circuit-breaker/state", response_model=JobCircuitBreakerResponse)
async def get_circuit_breaker_state(
    request: Request,
    current_user: Annotated[User, Depends(get_service_or_verified_user)],
) -> JobCircuitBreakerResponse:
    """Return the caller's circuit breaker state."""
    partition = resolve_queue_partition(request, current_user)
    breaker = await (await init_job_queue()).get_circuit_breaker(partition)
    return JobCircuitBreakerResponse(
        consecutive_failures=breaker.consecutive_failures,
        paused=breaker.paused,
        next_attempt_at=breaker.next_attempt_at,
        last_failure_at=breaker.last_failure_at,
        pause_reason=breaker.pause_reason,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    request: Request,
    job_id: int,
    current_user: Annotated[User, Depends(get_service_or_verified_user)],
) -> JobResponse:
    """Return one training job."""
    job = await get_owned_job_or_404(job_id, resolve_queue_partition(request, current_user))
    return serialize_job(job)


@router.delete("/{job_id}", response_model=MessageResponse)
async def cancel_job(
    request: Request,
    job_id: int,
    current_user: Annotated[User, Depends(get_service_or_verified_user)],
) -> MessageResponse:
    """Cancel a pending or running training job."""
    await get_owned_job_or_404(job_id, resolve_queue_partition(request, current_user))
    cancelled = await (await init_job_queue()).cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Training job can no longer be cancelled",
        )
    return MessageResponse(message="Training job cancellation requested")
