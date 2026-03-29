"""Add teams and team_members tables, team_id FK on recordings and trained_models.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-03-28
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- teams ---
    op.create_table(
        "teams",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("owner_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
    )

    # --- team_members ---
    op.create_table(
        "team_members",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=False, index=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
        sa.Column("role", sa.String(20), nullable=False, server_default=sa.text("'member'")),
        sa.Column("invited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("joined_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("team_id", "user_id", name="uq_team_member"),
    )

    # --- Add nullable team_id FK to recordings ---
    op.add_column(
        "recordings",
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=True, index=True),
    )

    # --- Add nullable team_id FK to trained_models ---
    op.add_column(
        "trained_models",
        sa.Column("team_id", sa.Integer, sa.ForeignKey("teams.id"), nullable=True, index=True),
    )


def downgrade() -> None:
    op.drop_column("trained_models", "team_id")
    op.drop_column("recordings", "team_id")
    op.drop_table("team_members")
    op.drop_table("teams")
