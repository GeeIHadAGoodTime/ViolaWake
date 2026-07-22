"""SQLAlchemy async database setup."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

# Railway deployments should set VIOLAWAKE_DB_URL to a full
# postgresql+asyncpg://... URL so the backend uses persistent PostgreSQL
# instead of the default filesystem-backed SQLite database.
DATABASE_URL = (
    settings.db_url.strip()
    if settings.db_url and settings.db_url.strip()
    else f"sqlite+aiosqlite:///{settings.db_path}"
)

engine = create_async_engine(DATABASE_URL, echo=False)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def _ensure_schema_updates(connection: Connection) -> None:
    """Apply lightweight schema updates for local-first deployments."""
    inspector = inspect(connection)
    table_names = set(inspector.get_table_names())
    if "users" not in table_names:
        return

    user_columns = {column["name"] for column in inspector.get_columns("users")}
    if "email_verified" not in user_columns:
        connection.execute(
            text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN NOT NULL DEFAULT FALSE")
        )
    if "failed_login_count" not in user_columns:
        try:
            connection.execute(
                text(
                    "ALTER TABLE users ADD COLUMN IF NOT EXISTS "
                    "failed_login_count INTEGER DEFAULT 0 NOT NULL"
                )
            )
        except Exception:
            connection.execute(
                text("ALTER TABLE users ADD COLUMN failed_login_count INTEGER DEFAULT 0 NOT NULL")
            )
    if "locked_until" not in user_columns:
        try:
            connection.execute(
                text("ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP")
            )
        except Exception:
            connection.execute(text("ALTER TABLE users ADD COLUMN locked_until TIMESTAMP"))
    if "deleted_at" not in user_columns:
        connection.execute(text("ALTER TABLE users ADD COLUMN deleted_at TIMESTAMP"))
    if "scheduled_hard_delete_at" not in user_columns:
        connection.execute(text("ALTER TABLE users ADD COLUMN scheduled_hard_delete_at TIMESTAMP"))

    # Team FK columns on recordings and trained_models (nullable, so no default needed)
    if "recordings" in table_names:
        recording_columns = {col["name"] for col in inspector.get_columns("recordings")}
        if "team_id" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN team_id INTEGER REFERENCES teams(id)"))

    if "trained_models" in table_names:
        model_columns = {col["name"] for col in inspector.get_columns("trained_models")}
        if "team_id" not in model_columns:
            connection.execute(text("ALTER TABLE trained_models ADD COLUMN team_id INTEGER REFERENCES teams(id)"))
        if "deleted_at" not in model_columns:
            connection.execute(text("ALTER TABLE trained_models ADD COLUMN deleted_at TIMESTAMP"))

    # (The Postgres ``training_jobs`` table was never written at runtime and is
    # retired by migration 20260721_0001 — issue #1438. No reconcile needed.)

    # Soft-delete support: recordings are marked deleted_at after training completes
    if "recordings" in table_names:
        recording_columns = {col["name"] for col in inspector.get_columns("recordings")}
        if "deleted_at" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN deleted_at TIMESTAMP"))
        if "display_name" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN display_name VARCHAR(255)"))
        if "size_bytes" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN size_bytes BIGINT"))
        if "uploaded_at" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN uploaded_at TIMESTAMP"))

        connection.execute(
            text("UPDATE recordings SET uploaded_at = created_at WHERE uploaded_at IS NULL")
        )
        _backfill_recording_size_bytes(connection)


def _local_recording_path(identifier: str) -> Path:
    path = Path(identifier)
    if path.is_absolute():
        return path

    normalized = identifier.replace("\\", "/").strip().strip("/")
    parts = normalized.split("/")
    if parts and parts[0] == "recordings":
        return settings.upload_dir.joinpath(*parts[1:])
    return settings.upload_dir / normalized


def _backfill_recording_size_bytes(connection: Connection) -> None:
    rows = connection.execute(
        text("SELECT id, file_path FROM recordings WHERE size_bytes IS NULL")
    ).fetchall()
    for recording_id, file_path in rows:
        try:
            size_bytes = _local_recording_path(str(file_path)).stat().st_size
        except OSError:
            continue
        connection.execute(
            text("UPDATE recordings SET size_bytes = :size_bytes WHERE id = :recording_id"),
            {"size_bytes": size_bytes, "recording_id": recording_id},
        )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """Create all tables (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_schema_updates)
