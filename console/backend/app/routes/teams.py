"""Team management routes."""

import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth import (
    create_team_invite_token,
    decode_team_invite_token,
    get_verified_user,
    make_get_team_member,
)
from app.config import settings
from app.database import get_db
from app.email_service import get_email_service
from app.models import Team, TeamMember, TrainedModel, User
from app.schemas import (
    MessageResponse,
    TeamCreateRequest,
    TeamInviteRequest,
    TeamListItemResponse,
    TeamMemberResponse,
    TeamResponse,
    TrainedModelResponse,
)

logger = logging.getLogger("violawake.teams")
router = APIRouter(prefix="/api/teams", tags=["teams"])


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _get_team_or_404(team_id: int, db: AsyncSession) -> Team:
    result = await db.execute(
        select(Team).options(selectinload(Team.members).selectinload(TeamMember.user))
        .where(Team.id == team_id)
    )
    team = result.scalar_one_or_none()
    if team is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Team not found")
    return team


def _member_response(m: TeamMember) -> TeamMemberResponse:
    return TeamMemberResponse(
        user_id=m.user_id,
        email=m.user.email,
        name=m.user.name,
        role=m.role,
        invited_at=m.invited_at,
        joined_at=m.joined_at,
    )


def _team_response(team: Team) -> TeamResponse:
    return TeamResponse(
        id=team.id,
        name=team.name,
        created_at=team.created_at,
        owner_id=team.owner_id,
        members=[_member_response(m) for m in team.members if m.joined_at is not None],
    )


def _test_invite_message(invite_token: str) -> str:
    """Return the invite response message, exposing the token only in tests."""
    if settings.env == "test":
        return f"Invitation sent. Test invite token: {invite_token}"
    return "Invitation sent"


async def _accept_invite(
    invite: dict,
    current_user: User,
    db: AsyncSession,
) -> TeamResponse:
    """Attach the current user to the invited team and return the team."""
    if invite["email"].casefold() != current_user.email.casefold():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This invite was issued for a different email address",
        )

    team_id = int(invite["team_id"])
    await _get_team_or_404(team_id, db)

    existing_result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == current_user.id,
        )
    )
    existing = existing_result.scalar_one_or_none()
    if existing is not None:
        if existing.joined_at is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Already a team member")
        existing.joined_at = datetime.now(timezone.utc)
        existing.role = invite["role"]
    else:
        member = TeamMember(
            team_id=team_id,
            user_id=current_user.id,
            role=invite["role"],
            joined_at=datetime.now(timezone.utc),
        )
        db.add(member)

    await db.flush()
    team = await _get_team_or_404(team_id, db)
    logger.info("User %s joined team %s with role %s", current_user.id, team_id, invite["role"])
    return _team_response(team)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    body: TeamCreateRequest,
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeamResponse:
    """Create a new team owned by the current user."""
    team = Team(name=body.name, owner_id=current_user.id)
    db.add(team)
    await db.flush()

    # Add the creator as the owner member
    owner_member = TeamMember(
        team_id=team.id,
        user_id=current_user.id,
        role="owner",
        joined_at=team.created_at,
    )
    db.add(owner_member)
    await db.flush()

    # Reload with members + users
    team = await _get_team_or_404(team.id, db)
    logger.info("Created team %s for user %s", team.id, current_user.id)
    return _team_response(team)


@router.get("", response_model=list[TeamListItemResponse])
async def list_teams(
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TeamListItemResponse]:
    """List all teams the current user belongs to (as owner or joined member)."""
    result = await db.execute(
        select(Team)
        .options(selectinload(Team.members))
        .join(TeamMember, TeamMember.team_id == Team.id)
        .where(
            TeamMember.user_id == current_user.id,
            TeamMember.joined_at.is_not(None),
        )
        .order_by(Team.created_at.desc())
    )
    teams = result.scalars().unique().all()
    return [
        TeamListItemResponse(
            id=t.id,
            name=t.name,
            created_at=t.created_at,
            owner_id=t.owner_id,
            member_count=sum(1 for m in t.members if m.joined_at is not None),
        )
        for t in teams
    ]


@router.post("/accept", response_model=TeamResponse)
async def accept_team_invite(
    token: str,
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeamResponse:
    """Accept a team invitation using a signed invite token."""
    invite = decode_team_invite_token(token)
    return await _accept_invite(invite, current_user, db)


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    team_id: int,
    _member: Annotated[TeamMember, Depends(make_get_team_member())],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeamResponse:
    """Get team details including all joined members."""
    team = await _get_team_or_404(team_id, db)
    return _team_response(team)


@router.post("/{team_id}/invite", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def invite_member(
    team_id: int,
    body: TeamInviteRequest,
    current_user: Annotated[User, Depends(get_verified_user)],
    _member: Annotated[TeamMember, Depends(make_get_team_member(["owner", "admin"]))],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Invite a user to the team by email."""
    team = await _get_team_or_404(team_id, db)

    # Prevent inviting someone who is already a joined member
    invited_user_result = await db.execute(select(User).where(User.email == body.email))
    invited_user = invited_user_result.scalar_one_or_none()

    if invited_user is not None:
        existing = await db.execute(
            select(TeamMember).where(
                TeamMember.team_id == team_id,
                TeamMember.user_id == invited_user.id,
                TeamMember.joined_at.is_not(None),
            )
        )
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User is already a team member",
            )

    # Admins cannot invite owners
    if body.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot invite a user as owner",
        )

    invite_token = create_team_invite_token(
        inviter_id=current_user.id,
        team_id=team_id,
        email=str(body.email),
        role=body.role.value,
    )

    email_svc = get_email_service()
    accept_url = email_svc._console_url("/teams/accept", token=invite_token)
    try:
        await email_svc.send_team_invite(
            to_email=str(body.email),
            team_name=team.name,
            inviter_name=current_user.name,
            accept_url=accept_url,
        )
    except Exception:
        logger.exception("Team invite email attempt failed for %s on team %s", body.email, team_id)

    logger.info("Invited %s to team %s by user %s", body.email, team_id, current_user.id)
    return MessageResponse(message=_test_invite_message(invite_token))


@router.post("/{team_id}/join", response_model=TeamResponse)
async def join_team(
    team_id: int,
    token: str,
    current_user: Annotated[User, Depends(get_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeamResponse:
    """Accept a team invitation using a signed invite token."""
    invite = decode_team_invite_token(token)

    if invite["team_id"] != team_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Token does not match team")
    return await _accept_invite(invite, current_user, db)


@router.delete("/{team_id}/members/{user_id}", response_model=MessageResponse)
async def remove_member(
    team_id: int,
    user_id: int,
    current_user: Annotated[User, Depends(get_verified_user)],
    _member: Annotated[TeamMember, Depends(make_get_team_member(["owner", "admin"]))],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Remove a member from the team. Admins cannot remove owners or other admins."""
    team = await _get_team_or_404(team_id, db)

    if user_id == team.owner_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot remove the team owner")

    target_result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
        )
    )
    target = target_result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    # Admins cannot remove other admins
    if _member.role == "admin" and target.role in ("owner", "admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role to remove this member")

    await db.delete(target)
    logger.info("User %s removed member %s from team %s", current_user.id, user_id, team_id)
    return MessageResponse(message="Member removed")


@router.post("/{team_id}/leave", response_model=MessageResponse)
async def leave_team(
    team_id: int,
    current_user: Annotated[User, Depends(get_verified_user)],
    _member: Annotated[TeamMember, Depends(make_get_team_member())],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Leave a team. Owners must delete the team instead."""
    team = await _get_team_or_404(team_id, db)

    if team.owner_id == current_user.id or _member.role == "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Team owners must delete the team instead of leaving it",
        )

    await db.delete(_member)
    logger.info("User %s left team %s", current_user.id, team_id)
    return MessageResponse(message="Left team")


@router.patch("/{team_id}/members/{user_id}", response_model=TeamMemberResponse)
async def change_member_role(
    team_id: int,
    user_id: int,
    role: str,
    current_user: Annotated[User, Depends(get_verified_user)],
    _member: Annotated[TeamMember, Depends(make_get_team_member(["owner"]))],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TeamMemberResponse:
    """Change a member's role. Only owners may do this. Cannot change another owner's role."""
    if role not in ("admin", "member"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role must be 'admin' or 'member'")

    team = await _get_team_or_404(team_id, db)

    if user_id == team.owner_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot change the owner's role")

    target_result = await db.execute(
        select(TeamMember)
        .options(selectinload(TeamMember.user))
        .where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == user_id,
            TeamMember.joined_at.is_not(None),
        )
    )
    target = target_result.scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Member not found")

    target.role = role
    await db.flush()
    logger.info("User %s changed role of %s to %s in team %s", current_user.id, user_id, role, team_id)
    return _member_response(target)


@router.delete("/{team_id}", response_model=MessageResponse)
async def delete_team(
    team_id: int,
    current_user: Annotated[User, Depends(get_verified_user)],
    _member: Annotated[TeamMember, Depends(make_get_team_member(["owner"]))],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> MessageResponse:
    """Delete a team. Only the owner may do this."""
    team = await _get_team_or_404(team_id, db)

    if team.owner_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the team owner can delete the team")

    await db.delete(team)
    logger.info("User %s deleted team %s", current_user.id, team_id)
    return MessageResponse(message="Team deleted")


@router.post("/{team_id}/models/{model_id}/share", response_model=TrainedModelResponse)
async def share_model(
    team_id: int,
    model_id: int,
    current_user: Annotated[User, Depends(get_verified_user)],
    _member: Annotated[TeamMember, Depends(make_get_team_member(["owner", "admin"]))],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> TrainedModelResponse:
    """Share one of the current user's models with the team."""
    # Verify team exists
    await _get_team_or_404(team_id, db)

    # Model must belong to the requesting user
    model_result = await db.execute(
        select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.user_id == current_user.id,
        )
    )
    model = model_result.scalar_one_or_none()
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")

    model.team_id = team_id
    await db.flush()
    logger.info("User %s shared model %s with team %s", current_user.id, model_id, team_id)
    return TrainedModelResponse(
        id=model.id,
        wake_word=model.wake_word,
        d_prime=model.d_prime,
        created_at=model.created_at,
        size_bytes=model.size_bytes,
    )


@router.get("/{team_id}/models", response_model=list[TrainedModelResponse])
async def list_team_models(
    team_id: int,
    _member: Annotated[TeamMember, Depends(make_get_team_member())],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[TrainedModelResponse]:
    """List all models shared with the team."""
    await _get_team_or_404(team_id, db)

    result = await db.execute(
        select(TrainedModel)
        .where(TrainedModel.team_id == team_id)
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
