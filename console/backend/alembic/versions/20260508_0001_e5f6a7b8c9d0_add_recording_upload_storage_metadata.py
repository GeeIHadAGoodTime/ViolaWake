"""Add recording upload storage metadata.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-08
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

from app.config import settings

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("recordings")}

    if "display_name" not in columns:
        op.add_column("recordings", sa.Column("display_name", sa.String(255), nullable=True))
    if "size_bytes" not in columns:
        op.add_column("recordings", sa.Column("size_bytes", sa.BigInteger(), nullable=True))
    if "uploaded_at" not in columns:
        if bind.dialect.name == "sqlite":
            op.add_column("recordings", sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=True))
        else:
            op.add_column(
                "recordings",
                sa.Column(
                    "uploaded_at",
                    sa.DateTime(timezone=True),
                    nullable=True,
                    server_default=sa.text("CURRENT_TIMESTAMP"),
                ),
            )

    bind.execute(text("UPDATE recordings SET uploaded_at = created_at WHERE uploaded_at IS NULL"))
    _backfill_size_bytes(bind)


def downgrade() -> None:
    bind = op.get_bind()
    columns = {column["name"] for column in sa.inspect(bind).get_columns("recordings")}
    if "uploaded_at" in columns:
        op.drop_column("recordings", "uploaded_at")
    if "size_bytes" in columns:
        op.drop_column("recordings", "size_bytes")
    if "display_name" in columns:
        op.drop_column("recordings", "display_name")


def _recording_path(identifier: str) -> Path:
    path = Path(identifier)
    if path.is_absolute():
        return path

    normalized = identifier.replace("\\", "/").strip().strip("/")
    parts = normalized.split("/")
    if parts and parts[0] == "recordings":
        return settings.upload_dir.joinpath(*parts[1:])
    return settings.upload_dir / normalized


def _backfill_size_bytes(bind) -> None:
    rows = bind.execute(
        text("SELECT id, file_path FROM recordings WHERE size_bytes IS NULL")
    ).fetchall()
    for recording_id, file_path in rows:
        try:
            size_bytes = _recording_path(str(file_path)).stat().st_size
        except OSError:
            continue
        bind.execute(
            text("UPDATE recordings SET size_bytes = :size_bytes WHERE id = :recording_id"),
            {"size_bytes": size_bytes, "recording_id": recording_id},
        )
