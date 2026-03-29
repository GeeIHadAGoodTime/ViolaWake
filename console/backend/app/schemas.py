"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field


# ── Auth ─────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=255)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class VerifyEmailRequest(BaseModel):
    token: str = Field(min_length=1)


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=1)
    password: str = Field(min_length=8, max_length=128)


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    email_verified: bool

    model_config = {"from_attributes": True}


class UserDetailResponse(UserResponse):
    created_at: datetime


class AuthResponse(BaseModel):
    token: str
    user: UserResponse


class DownloadTokenRequest(BaseModel):
    action: Literal["model_download", "training_stream"]
    resource_id: int = Field(ge=1)


class DownloadTokenResponse(BaseModel):
    token: str
    expires_in_seconds: int


class MessageResponse(BaseModel):
    message: str


# ── Recordings ───────────────────────────────────────────────────────────────

class RecordingResponse(BaseModel):
    id: int
    wake_word: str
    filename: str
    duration_s: float
    created_at: datetime

    model_config = {"from_attributes": True}


class RecordingUploadResponse(BaseModel):
    recording_id: int
    filename: str
    wake_word: str
    duration_s: float


# ── Training ─────────────────────────────────────────────────────────────────

class TrainingStartRequest(BaseModel):
    wake_word: str = Field(min_length=1, max_length=100)
    recording_ids: list[int] = Field(min_length=1)
    epochs: int = Field(default=80, ge=5, le=500)


class TrainingStartResponse(BaseModel):
    job_id: int
    status: str


class TrainingStatusResponse(BaseModel):
    job_id: int
    status: str
    progress: float
    d_prime: float | None = None
    model_id: int | None = None
    error: str | None = None

    model_config = {"from_attributes": True}


class TrainingSSEEvent(BaseModel):
    status: str
    progress: float
    epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    d_prime: float | None = None
    model_id: int | None = None
    message: str = ""
    error: str | None = None


# Jobs

class JobSubmitRequest(TrainingStartRequest):
    pass


class JobSubmitResponse(BaseModel):
    job_id: int
    status: str


class JobResponse(BaseModel):
    job_id: int
    user_id: int
    wake_word: str
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    progress_pct: float
    d_prime: float | None = None
    model_id: int | None = None


class JobCircuitBreakerResponse(BaseModel):
    consecutive_failures: int
    paused: bool
    next_attempt_at: datetime | None = None
    last_failure_at: datetime | None = None
    pause_reason: str | None = None


# ── Models ───────────────────────────────────────────────────────────────────

class TrainedModelResponse(BaseModel):
    id: int
    wake_word: str
    d_prime: float | None = None
    created_at: datetime
    size_bytes: int

    model_config = {"from_attributes": True}


class ModelConfigResponse(BaseModel):
    d_prime: float | None = None
    far_per_hour: float | None = None
    frr: float | None = None
    training_config: dict = {}


class ModelPerformanceResponse(BaseModel):
    model_name: str
    cohen_d: float | None = None
    threshold: float | None = None
    file_size: int
    created_at: datetime
    positive_scores: list[float] = Field(default_factory=list)
    negative_scores: list[float] = Field(default_factory=list)
    evaluation_available: bool = False
    evaluation_data: dict[str, Any] = Field(default_factory=dict)


# ── Teams ────────────────────────────────────────────────────────────────────

class TeamMemberRole(str, Enum):
    owner = "owner"
    admin = "admin"
    member = "member"


class TeamCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class TeamInviteRequest(BaseModel):
    email: EmailStr
    role: TeamMemberRole = TeamMemberRole.member


class TeamMemberResponse(BaseModel):
    user_id: int
    email: str
    name: str
    role: str
    invited_at: datetime
    joined_at: datetime | None = None

    model_config = {"from_attributes": True}


class TeamResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    owner_id: int
    members: list[TeamMemberResponse] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class TeamListItemResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    owner_id: int
    member_count: int = 0

    model_config = {"from_attributes": True}


# ── Billing ─────────────────────────────────────────────────────────────────

class BillingTier(str, Enum):
    free = "free"
    developer = "developer"
    business = "business"
    enterprise = "enterprise"


class CheckoutRequest(BaseModel):
    tier: str = Field(pattern="^(developer|business)$", description="Target subscription tier")


class CheckoutResponse(BaseModel):
    checkout_url: str


class UsageResponse(BaseModel):
    models_used: int
    models_limit: int | None = None  # None means unlimited
    period_start: datetime
    period_end: datetime


class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    current_period_end: datetime | None = None
    trial_active: bool = False
    trial_end: datetime | None = None
    usage: UsageResponse

    model_config = {"from_attributes": True}


class BillingPortalResponse(BaseModel):
    url: str
