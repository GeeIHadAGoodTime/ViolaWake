"""Add failed_login_count and locked_until to users, deleted_at to recordings.

These columns were previously applied at runtime by _ensure_schema_updates()
in database.py but were missing from the Alembic migration chain, causing
Alembic-only deployments (e.g. PostgreSQL on Railway) to lack them.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-04-05
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Account-lockout columns on users
    op.add_column(
        "users",
        sa.Column("failed_login_count", sa.Integer, nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "users",
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
    )

    # Soft-delete support on recordings
    op.add_column(
        "recordings",
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("recordings", "deleted_at")
    op.drop_column("users", "locked_until")
    op.drop_column("users", "failed_login_count")
