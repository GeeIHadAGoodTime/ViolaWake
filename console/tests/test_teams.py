"""Unit tests for team management routes."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

backend_dir = str(Path(__file__).resolve().parents[1] / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.auth import _create_action_token, create_team_invite_token, decode_team_invite_token
from app.email_service import EmailService
from app.routes import teams as teams_routes
from app.schemas import (
    TeamCreateRequest,
    TeamInviteRequest,
    TeamMemberRole,
)


# ── Shared test infrastructure ────────────────────────────────────────────────
# We use SimpleNamespace throughout so SQLAlchemy mapper is never invoked.
# The route functions only access attributes, not ORM internals.


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _make_user(user_id: int, email: str, name: str = "Test User") -> SimpleNamespace:
    return SimpleNamespace(
        id=user_id,
        email=email,
        name=name,
        email_verified=True,
        password_hash="hashed",
        created_at=_utcnow(),
    )


def _make_team(team_id: int, name: str, owner: SimpleNamespace) -> SimpleNamespace:
    t = SimpleNamespace(
        id=team_id,
        name=name,
        owner_id=owner.id,
        owner=owner,
        created_at=_utcnow(),
        members=[],
    )
    return t


def _make_member(
    team: SimpleNamespace,
    user: SimpleNamespace,
    role: str,
    joined: bool = True,
) -> SimpleNamespace:
    m = SimpleNamespace(
        id=(team.id * 100) + user.id,
        team_id=team.id,
        user_id=user.id,
        user=user,
        team=team,
        role=role,
        invited_at=_utcnow(),
        joined_at=_utcnow() if joined else None,
    )
    team.members.append(m)
    return m


class _ScalarResult:
    """Minimal scalar result mock."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value

    def scalars(self) -> _ScalarResult:
        return self

    def unique(self) -> _ScalarResult:
        return self

    def all(self) -> list:
        if isinstance(self._value, (list, tuple)):
            return list(self._value)
        return [self._value] if self._value is not None else []


class FakeSession:
    """Fake async DB session for unit tests."""

    def __init__(self) -> None:
        self._added: list[Any] = []
        self._deleted: list[Any] = []
        self._execute_handler: Any = None  # callable(statement) -> _ScalarResult

    async def execute(self, statement: Any) -> _ScalarResult:
        if self._execute_handler:
            result = self._execute_handler(statement)
            # Support both sync returns and coroutines
            if hasattr(result, "__await__"):
                return await result
            return result
        return _ScalarResult(None)

    def add(self, obj: Any) -> None:
        self._added.append(obj)

    async def delete(self, obj: Any) -> None:
        self._deleted.append(obj)

    async def flush(self) -> None:
        for obj in self._added:
            if not getattr(obj, "id", None):
                obj.id = id(obj) % 10000
        self._added.clear()

    async def rollback(self) -> None:
        pass

    async def commit(self) -> None:
        pass


# ── Token helpers tests ───────────────────────────────────────────────────────


def test_team_invite_token_roundtrip() -> None:
    token = create_team_invite_token(
        inviter_id=1, team_id=5, email="alice@example.com", role="member"
    )
    decoded = decode_team_invite_token(token)
    assert decoded["inviter_id"] == 1
    assert decoded["team_id"] == 5
    assert decoded["email"] == "alice@example.com"
    assert decoded["role"] == "member"


def test_team_invite_token_wrong_purpose_rejected() -> None:
    from fastapi import HTTPException
    from datetime import timedelta

    token = _create_action_token(1, "verify_email", timedelta(hours=1))
    with pytest.raises(HTTPException) as exc_info:
        decode_team_invite_token(token)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_team_invite_email_uses_accept_url_and_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    def fake_send(params: dict[str, Any]) -> None:
        sent.update(params)

    monkeypatch.setattr("app.email_service.resend.Emails.send", fake_send)

    svc = EmailService(api_key="test-key", console_base_url="https://console.test")
    result = await svc.send_team_invite(
        to_email="bob@example.com",
        team_name="Alpha Team",
        inviter_name="Owner User",
        accept_url="https://console.test/teams/accept?token=abc123",
    )

    assert result is True
    assert sent["to"] == ["bob@example.com"]
    assert sent["subject"] == "You've been invited to join Alpha Team on ViolaWake"
    assert "Owner User invited you to join" in sent["html"]
    assert "https://console.test/teams/accept?token=abc123" in sent["html"]


# ── create_team ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_team_returns_team_response() -> None:
    owner = _make_user(1, "owner@example.com")
    db = FakeSession()

    call_count = {"n": 0}

    async def execute_handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        # First call: reload team with members
        if call_count["n"] == 1:
            team = _make_team(42, "My Team", owner)
            _make_member(team, owner, "owner")
            return _ScalarResult(team)
        return _ScalarResult(None)

    db._execute_handler = execute_handler

    resp = await teams_routes.create_team(
        body=TeamCreateRequest(name="My Team"),
        current_user=owner,
        db=db,
    )
    assert resp.name == "My Team"
    assert resp.owner_id == owner.id
    assert len(resp.members) == 1
    assert resp.members[0].role == "owner"


# ── list_teams ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_teams_returns_only_joined() -> None:
    owner = _make_user(1, "owner@example.com")
    db = FakeSession()

    team = _make_team(10, "Alpha Team", owner)
    _make_member(team, owner, "owner")

    db._execute_handler = lambda _stmt: _ScalarResult([team])

    resp = await teams_routes.list_teams(current_user=owner, db=db)
    assert len(resp) == 1
    assert resp[0].name == "Alpha Team"
    assert resp[0].member_count == 1


# ── invite_member ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invite_member_sends_email_without_token_in_production_response(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = _make_user(1, "owner@example.com")
    invitee_email = "bob@example.com"
    db = FakeSession()

    team = _make_team(10, "Alpha Team", owner)
    owner_member = _make_member(team, owner, "owner")
    sent_invites: list[dict[str, str]] = []

    class FakeEmailService:
        def _console_url(self, path: str, **query: str) -> str:
            token = query["token"]
            return f"https://console.test{path}?token={token}"

        async def send_team_invite(
            self,
            to_email: str,
            team_name: str,
            inviter_name: str,
            accept_url: str,
        ) -> bool:
            sent_invites.append(
                {
                    "to_email": to_email,
                    "team_name": team_name,
                    "inviter_name": inviter_name,
                    "accept_url": accept_url,
                }
            )
            return True

    monkeypatch.setattr(teams_routes.settings, "env", "production")
    monkeypatch.setattr(teams_routes, "get_email_service", lambda: FakeEmailService())

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # reload team with members
            return _ScalarResult(team)
        if call_count["n"] == 2:
            # invitee user lookup — not found
            return _ScalarResult(None)
        return _ScalarResult(None)

    db._execute_handler = handler

    resp = await teams_routes.invite_member(
        team_id=10,
        request=None,
        body=TeamInviteRequest(email=invitee_email, role=TeamMemberRole.member),
        current_user=owner,
        _member=owner_member,
        db=db,
    )
    assert resp.message == "Invitation sent"
    assert "token" not in resp.message.lower()
    assert len(sent_invites) == 1
    assert sent_invites[0]["to_email"] == invitee_email
    assert sent_invites[0]["team_name"] == "Alpha Team"
    assert sent_invites[0]["inviter_name"] == owner.name
    assert sent_invites[0]["accept_url"].startswith("https://console.test/teams/accept?token=")

    token = parse_qs(urlparse(sent_invites[0]["accept_url"]).query)["token"][0]
    decoded = decode_team_invite_token(token)
    assert decoded["email"] == invitee_email
    assert decoded["team_id"] == 10


@pytest.mark.asyncio
async def test_invite_member_can_include_token_when_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    owner = _make_user(1, "owner@example.com")
    db = FakeSession()

    team = _make_team(10, "Alpha Team", owner)
    owner_member = _make_member(team, owner, "owner")

    class FakeEmailService:
        def _console_url(self, path: str, **query: str) -> str:
            return f"https://console.test{path}?token={query['token']}"

        async def send_team_invite(
            self,
            to_email: str,
            team_name: str,
            inviter_name: str,
            accept_url: str,
        ) -> bool:
            return True

    monkeypatch.setattr(teams_routes.settings, "env", "test")
    monkeypatch.setattr(teams_routes, "get_email_service", lambda: FakeEmailService())

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        return _ScalarResult(None)

    db._execute_handler = handler

    resp = await teams_routes.invite_member(
        team_id=10,
        request=None,
        body=TeamInviteRequest(email="bob@example.com", role=TeamMemberRole.member),
        current_user=owner,
        _member=owner_member,
        db=db,
    )
    assert "Test invite token:" in resp.message
    token = resp.message.split("Test invite token: ")[1].strip()
    assert decode_team_invite_token(token)["team_id"] == 10


@pytest.mark.asyncio
async def test_invite_owner_role_forbidden() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)  # team reload
        return _ScalarResult(None)  # user not found

    db._execute_handler = handler

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.invite_member(
            team_id=10,
            request=None,
            body=TeamInviteRequest(email="x@example.com", role=TeamMemberRole.owner),
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_invite_already_joined_member_raises_conflict() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    bob_member = _make_member(team, bob, "member")

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        if call_count["n"] == 2:
            return _ScalarResult(bob)  # user found
        if call_count["n"] == 3:
            return _ScalarResult(bob_member)  # already a member
        return _ScalarResult(None)

    db._execute_handler = handler

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.invite_member(
            team_id=10,
            request=None,
            body=TeamInviteRequest(email="bob@example.com", role=TeamMemberRole.member),
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 409


# ── join_team ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_join_team_success() -> None:
    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")

    token = create_team_invite_token(
        inviter_id=owner.id, team_id=10, email=bob.email, role="member"
    )

    team = _make_team(10, "Alpha", owner)
    _make_member(team, owner, "owner")

    call_count = {"n": 0}

    class JoinSession(FakeSession):
        async def execute(self, stmt: Any) -> _ScalarResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _ScalarResult(team)  # team exists check
            if call_count["n"] == 2:
                return _ScalarResult(None)  # no existing membership
            if call_count["n"] == 3:
                # reload team after join — add bob's member record
                bob_member = _make_member(team, bob, "member")
                return _ScalarResult(team)
            return _ScalarResult(None)

    db = JoinSession()
    resp = await teams_routes.join_team(
        team_id=10,
        token=token,
        current_user=bob,
        db=db,
    )
    assert resp.id == 10


@pytest.mark.asyncio
async def test_accept_team_invite_success() -> None:
    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")

    token = create_team_invite_token(
        inviter_id=owner.id, team_id=10, email=bob.email, role="member"
    )

    team = _make_team(10, "Alpha", owner)
    _make_member(team, owner, "owner")

    call_count = {"n": 0}

    class AcceptSession(FakeSession):
        async def execute(self, stmt: Any) -> _ScalarResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _ScalarResult(team)
            if call_count["n"] == 2:
                return _ScalarResult(None)
            if call_count["n"] == 3:
                _make_member(team, bob, "member")
                return _ScalarResult(team)
            return _ScalarResult(None)

    resp = await teams_routes.accept_team_invite(
        token=token,
        current_user=bob,
        db=AcceptSession(),
    )
    assert resp.id == 10
    assert any(m.user_id == bob.id for m in resp.members)


@pytest.mark.asyncio
async def test_join_team_wrong_email_rejected() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    carol = _make_user(3, "carol@example.com")

    token = create_team_invite_token(
        inviter_id=owner.id, team_id=10, email="bob@example.com", role="member"
    )

    db = FakeSession()
    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.join_team(
            team_id=10,
            token=token,
            current_user=carol,
            db=db,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_join_team_wrong_team_id_rejected() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")

    token = create_team_invite_token(
        inviter_id=owner.id, team_id=99, email=bob.email, role="member"
    )

    db = FakeSession()
    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.join_team(
            team_id=10,
            token=token,
            current_user=bob,
            db=db,
        )
    assert exc_info.value.status_code == 400


# ── remove_member ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_remove_member_by_owner_succeeds() -> None:
    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    bob_member = _make_member(team, bob, "member")

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        if call_count["n"] == 2:
            return _ScalarResult(bob_member)
        return _ScalarResult(None)

    db._execute_handler = handler

    resp = await teams_routes.remove_member(
        team_id=10,
        user_id=bob.id,
        current_user=owner,
        _member=owner_member,
        db=db,
    )
    assert bob_member in db._deleted
    assert resp.message == "Member removed"


@pytest.mark.asyncio
async def test_remove_owner_is_forbidden() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    db._execute_handler = lambda _: _ScalarResult(team)

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.remove_member(
            team_id=10,
            user_id=owner.id,
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_admin_cannot_remove_another_admin() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    admin1 = _make_user(2, "admin1@example.com")
    admin2 = _make_user(3, "admin2@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    _make_member(team, owner, "owner")
    admin1_member = _make_member(team, admin1, "admin")
    admin2_member = _make_member(team, admin2, "admin")

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        if call_count["n"] == 2:
            return _ScalarResult(admin2_member)
        return _ScalarResult(None)

    db._execute_handler = handler

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.remove_member(
            team_id=10,
            user_id=admin2.id,
            current_user=admin1,
            _member=admin1_member,
            db=db,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_member_can_leave_team() -> None:
    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    _make_member(team, owner, "owner")
    bob_member = _make_member(team, bob, "member")
    db._execute_handler = lambda _: _ScalarResult(team)

    resp = await teams_routes.leave_team(
        team_id=10,
        current_user=bob,
        _member=bob_member,
        db=db,
    )
    assert resp.message == "Left team"
    assert bob_member in db._deleted


@pytest.mark.asyncio
async def test_owner_cannot_leave_team() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    db._execute_handler = lambda _: _ScalarResult(team)

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.leave_team(
            team_id=10,
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 403


# ── change_member_role ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_owner_can_promote_member_to_admin() -> None:
    owner = _make_user(1, "owner@example.com")
    bob = _make_user(2, "bob@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    bob_member = _make_member(team, bob, "member")

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        if call_count["n"] == 2:
            return _ScalarResult(bob_member)
        return _ScalarResult(None)

    db._execute_handler = handler

    resp = await teams_routes.change_member_role(
        team_id=10,
        user_id=bob.id,
        role="admin",
        current_user=owner,
        _member=owner_member,
        db=db,
    )
    assert resp.role == "admin"
    assert bob_member.role == "admin"


@pytest.mark.asyncio
async def test_change_role_to_invalid_value_rejected() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    db._execute_handler = lambda _: _ScalarResult(team)

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.change_member_role(
            team_id=10,
            user_id=2,
            role="superuser",
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_change_owner_role_forbidden() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    db._execute_handler = lambda _: _ScalarResult(team)

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.change_member_role(
            team_id=10,
            user_id=owner.id,
            role="admin",
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 403


# ── delete_team ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_owner_can_delete_team() -> None:
    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")
    db._execute_handler = lambda _: _ScalarResult(team)

    resp = await teams_routes.delete_team(
        team_id=10,
        current_user=owner,
        _member=owner_member,
        db=db,
    )
    assert resp.message == "Team deleted"
    assert team in db._deleted


# ── share_model / list_team_models ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_share_model_assigns_team_id() -> None:
    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")

    from datetime import datetime, timezone
    model = SimpleNamespace(
        id=42,
        user_id=owner.id,
        team_id=None,
        wake_word="hey nova",
        d_prime=2.5,
        size_bytes=1024,
        created_at=datetime.now(timezone.utc),
    )

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        if call_count["n"] == 2:
            return _ScalarResult(model)
        return _ScalarResult(None)

    db._execute_handler = handler

    resp = await teams_routes.share_model(
        team_id=10,
        model_id=42,
        current_user=owner,
        _member=owner_member,
        db=db,
    )
    assert model.team_id == 10
    assert resp.id == 42


@pytest.mark.asyncio
async def test_share_model_not_owner_of_model_raises_404() -> None:
    from fastapi import HTTPException

    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        return _ScalarResult(None)  # model not found for this user

    db._execute_handler = handler

    with pytest.raises(HTTPException) as exc_info:
        await teams_routes.share_model(
            team_id=10,
            model_id=99,
            current_user=owner,
            _member=owner_member,
            db=db,
        )
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_list_team_models_returns_team_models() -> None:
    owner = _make_user(1, "owner@example.com")
    db = FakeSession()
    team = _make_team(10, "T", owner)
    owner_member = _make_member(team, owner, "owner")

    from datetime import datetime, timezone
    model = SimpleNamespace(
        id=7,
        user_id=owner.id,
        team_id=10,
        wake_word="viola",
        d_prime=3.1,
        size_bytes=512,
        created_at=datetime.now(timezone.utc),
    )

    call_count = {"n": 0}

    async def handler(stmt: Any) -> _ScalarResult:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _ScalarResult(team)
        if call_count["n"] == 2:
            return _ScalarResult([model])
        return _ScalarResult([])

    db._execute_handler = handler

    resp = await teams_routes.list_team_models(
        team_id=10,
        _member=owner_member,
        db=db,
    )
    assert len(resp) == 1
    assert resp[0].wake_word == "viola"
