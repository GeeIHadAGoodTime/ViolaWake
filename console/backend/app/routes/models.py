"""Trained model routes: list, download, config."""

import json
import logging
from numbers import Real
from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import decode_download_token, decode_token, get_current_user
from app.database import get_db
from app.models import TrainedModel, User
from app.schemas import (
    ModelConfigResponse,
    ModelPerformanceResponse,
    TrainedModelResponse,
)
from app.storage import LocalStorageBackend, build_companion_config_identifier, get_storage

router = APIRouter(prefix="/api/models", tags=["models"])
logger = logging.getLogger("violawake.models")

_COHEN_D_KEYS = (
    "cohen_d",
    "cohens_d",
    "cohen-d",
    "d_prime",
    "dprime",
)
_THRESHOLD_KEYS = (
    "threshold",
    "score_threshold",
    "detection_threshold",
    "decision_threshold",
)
_POSITIVE_SCORE_KEYS = (
    "positive_scores",
    "positive_score_distribution",
    "positive_eval_scores",
    "pos_scores",
)
_NEGATIVE_SCORE_KEYS = (
    "negative_scores",
    "negative_score_distribution",
    "negative_eval_scores",
    "neg_scores",
)
_EVALUATION_KEYS = {
    *_COHEN_D_KEYS,
    *_THRESHOLD_KEYS,
    *_POSITIVE_SCORE_KEYS,
    *_NEGATIVE_SCORE_KEYS,
    "far_per_hour",
    "frr",
}


def _normalize_key(key: str) -> str:
    return key.strip().lower().replace("-", "_").replace(" ", "_")


def _flatten_numeric_list(value: object) -> list[float] | None:
    if not isinstance(value, list):
        return None

    values: list[float] = []
    stack = list(value)
    while stack:
        item = stack.pop(0)
        if isinstance(item, list):
            stack = list(item) + stack
            continue
        if isinstance(item, Real) and not isinstance(item, bool):
            values.append(float(item))
    return values or None


def _find_value(payload: object, keys: tuple[str, ...]) -> object | None:
    normalized_keys = {_normalize_key(key) for key in keys}

    def _walk(node: object) -> object | None:
        if isinstance(node, dict):
            for raw_key, value in node.items():
                if _normalize_key(str(raw_key)) in normalized_keys:
                    return value
            for value in node.values():
                found = _walk(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for value in node:
                found = _walk(value)
                if found is not None:
                    return found
        return None

    return _walk(payload)


def _extract_float(payload: object, keys: tuple[str, ...]) -> float | None:
    value = _find_value(payload, keys)
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value)
    return None


def _extract_scores(payload: object, keys: tuple[str, ...]) -> list[float]:
    value = _find_value(payload, keys)
    scores = _flatten_numeric_list(value)
    return scores or []


def _collect_evaluation_data(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}

    evaluation_data: dict = {}
    for key, value in payload.items():
        normalized = _normalize_key(str(key))
        if normalized in _EVALUATION_KEYS:
            evaluation_data[normalized] = value

    for container_key in ("evaluation", "metrics", "score_distribution"):
        value = payload.get(container_key)
        if isinstance(value, dict):
            evaluation_data[container_key] = value

    return evaluation_data


def _load_model_metadata(model: TrainedModel) -> dict:
    metadata: dict = {}

    if model.config_json:
        try:
            parsed = json.loads(model.config_json)
            if isinstance(parsed, dict):
                metadata.update(parsed)
        except json.JSONDecodeError:
            pass

    if model.file_path:
        storage = get_storage()
        config_identifier = build_companion_config_identifier(model.file_path)
        try:
            if storage.exists(config_identifier):
                parsed = json.loads(storage.download(config_identifier).decode("utf-8"))
                if isinstance(parsed, dict):
                    metadata.update(parsed)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError) as exc:
            logger.warning("Failed to load model metadata for %s: %s", model.file_path, exc)

    return metadata


async def _resolve_download_user(
    request: Request,
    token: str | None,
    db: AsyncSession,
    *,
    model_id: int,
) -> User:
    """Resolve the authenticated user for download/config endpoints.

    Browser downloads via <a href> cannot set Authorization headers, so the
    frontend uses a short-lived one-time download token in the query string.
    """
    user_id: int | None = None

    if token:
        user_id = decode_download_token(
            token,
            expected_action="model_download",
            expected_resource_id=model_id,
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
    return user


@router.get("", response_model=list[TrainedModelResponse])
async def list_models(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TrainedModelResponse]:
    """List all trained models for the current user."""
    result = await db.execute(
        select(TrainedModel)
        .where(TrainedModel.user_id == current_user.id)
        .order_by(TrainedModel.created_at.desc())
    )
    models = result.scalars().all()

    return [
        TrainedModelResponse(
            id=m.id,
            wake_word=m.wake_word,
            d_prime=m.d_prime,
            created_at=m.created_at,
            size_bytes=m.size_bytes,
        )
        for m in models
    ]


@router.get("/{model_id}/download")
async def download_model(
    model_id: int,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
    token: str | None = Query(default=None),
) -> Response:
    """Download a trained ONNX model file.

    Accepts a one-time query token for browser downloads or a standard
    ``Authorization: Bearer <token>`` header for API clients.
    """
    current_user = await _resolve_download_user(request, token, db, model_id=model_id)

    result = await db.execute(
        select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.id,
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    storage = get_storage()
    if not storage.exists(model.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model file missing from storage",
        )

    if isinstance(storage, LocalStorageBackend):
        filename = PurePosixPath(model.file_path).name or f"{model.wake_word}.onnx"
        return Response(
            content=storage.download(model.file_path),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return RedirectResponse(url=storage.presigned_url(model.file_path), status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/{model_id}/config", response_model=ModelConfigResponse)
async def get_model_config(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelConfigResponse:
    """Get training config and metrics for a trained model.

    Requires the standard ``Authorization: Bearer <token>`` header.
    """
    result = await db.execute(
        select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.id,
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    training_config = _load_model_metadata(model)

    # Try to read the companion .config.json file for additional metrics
    far_per_hour = None
    frr = None
    if training_config:
        far_per_hour = _extract_float(training_config, ("far_per_hour",))
        frr = _extract_float(training_config, ("frr",))

    return ModelConfigResponse(
        d_prime=model.d_prime,
        far_per_hour=far_per_hour,
        frr=frr,
        training_config=training_config,
    )


@router.get("/{model_id}/performance", response_model=ModelPerformanceResponse)
async def get_model_performance(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ModelPerformanceResponse:
    """Get performance details and any stored evaluation data for a model."""
    result = await db.execute(
        select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.id,
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    metadata = _load_model_metadata(model)
    positive_scores = _extract_scores(metadata, _POSITIVE_SCORE_KEYS)
    negative_scores = _extract_scores(metadata, _NEGATIVE_SCORE_KEYS)
    cohen_d = model.d_prime
    if cohen_d is None:
        cohen_d = _extract_float(metadata, _COHEN_D_KEYS)
    threshold = _extract_float(metadata, _THRESHOLD_KEYS)

    return ModelPerformanceResponse(
        model_name=model.wake_word,
        cohen_d=cohen_d,
        threshold=threshold,
        file_size=model.size_bytes,
        created_at=model.created_at,
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        evaluation_available=bool(positive_scores or negative_scores),
        evaluation_data=_collect_evaluation_data(metadata),
    )


@router.delete(
    "/{model_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_model(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Response:
    """Delete a trained model. Only the owner may delete their models."""
    result = await db.execute(
        select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.id,
        )
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    storage = get_storage()
    storage.delete(model.file_path)
    storage.delete(build_companion_config_identifier(model.file_path))

    await db.delete(model)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
