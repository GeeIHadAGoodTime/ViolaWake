"""Add GDPR soft deletion fields and inbound email dedupe table.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-08
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_column_if_missing(table_name: str, column: sa.Column) -> None:
    bind = op.get_bind()
    columns = {item["name"] for item in sa.inspect(bind).get_columns(table_name)}
    if column.name not in columns:
        op.add_column(table_name, column)


def upgrade() -> None:
    _add_column_if_missing("users", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    _add_column_if_missing(
        "users",
        sa.Column("scheduled_hard_delete_at", sa.DateTime(timezone=True), nullable=True),
    )
    _add_column_if_missing("trained_models", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    _add_column_if_missing("training_jobs", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())
    if "inbound_email_auto_replies" not in table_names:
        op.create_table(
            "inbound_email_auto_replies",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("sender_email", sa.String(255), nullable=False, index=True),
            sa.Column("ticket_reference", sa.String(64), nullable=False, unique=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    table_names = set(sa.inspect(bind).get_table_names())
    if "inbound_email_auto_replies" in table_names:
        op.drop_table("inbound_email_auto_replies")
    op.drop_column("training_jobs", "deleted_at")
    op.drop_column("trained_models", "deleted_at")
    op.drop_column("users", "scheduled_hard_delete_at")
    op.drop_column("users", "deleted_at")
