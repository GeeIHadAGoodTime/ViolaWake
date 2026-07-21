"""Retire the dead Postgres training_jobs table.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-07-21

The ``training_jobs`` table was never written at runtime (Postgres
``pg_stat_user_tables`` reported ``n_tup_ins=0`` on prod despite job ids up to
87+ existing). Training-job state is persisted exclusively by the async job
queue in its own SQLite store (``app.job_queue.JobQueue`` ->
``settings.data_dir / "job_queue.db"``, table ``jobs``), which is the source of
truth for create/list/get/cancel, user-facing history, GDPR export
(``queue_jobs``), and account-deletion purge (``JobQueue.delete_jobs_for_user``).

The Postgres table, its ``deleted_at`` soft-delete column, and the ORM
references that read/soft-delete/hard-delete it were all no-ops. This migration
drops the empty table (zero data loss — it has never held a row). See issue
#1438. The ``VIOLAWAKE_POST_TRAINING_RETENTION_HOURS`` setting is unrelated (it
governs *recording* retention) and is intentionally left untouched.

The drop is guarded on table existence so it is safe to run against any
environment (including ones already lacking the table), and the downgrade
recreates the table in its post-GDPR-migration shape (with ``deleted_at``) for
reversibility.
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if "training_jobs" in set(sa.inspect(bind).get_table_names()):
        op.drop_table("training_jobs")


def downgrade() -> None:
    bind = op.get_bind()
    if "training_jobs" not in set(sa.inspect(bind).get_table_names()):
        op.create_table(
            "training_jobs",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False, index=True),
            sa.Column("wake_word", sa.String(100), nullable=False),
            sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'queued'")),
            sa.Column("progress", sa.Float, nullable=False, server_default=sa.text("0.0")),
            sa.Column("epochs", sa.Integer, nullable=False, server_default=sa.text("50")),
            sa.Column("d_prime", sa.Float, nullable=True),
            sa.Column("model_id", sa.Integer, sa.ForeignKey("trained_models.id"), nullable=True),
            sa.Column("error", sa.Text, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )
