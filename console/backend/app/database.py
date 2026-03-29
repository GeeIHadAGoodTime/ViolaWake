"""SQLAlchemy async database setup."""

from __future__ import annotations

from collections.abc import AsyncGenerator

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

    # Team FK columns on recordings and trained_models (nullable, so no default needed)
    if "recordings" in table_names:
        recording_columns = {col["name"] for col in inspector.get_columns("recordings")}
        if "team_id" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN team_id INTEGER REFERENCES teams(id)"))

    if "trained_models" in table_names:
        model_columns = {col["name"] for col in inspector.get_columns("trained_models")}
        if "team_id" not in model_columns:
            connection.execute(text("ALTER TABLE trained_models ADD COLUMN team_id INTEGER REFERENCES teams(id)"))

    # Soft-delete support: recordings are marked deleted_at after training completes
    if "recordings" in table_names:
        recording_columns = {col["name"] for col in inspector.get_columns("recordings")}
        if "deleted_at" not in recording_columns:
            connection.execute(text("ALTER TABLE recordings ADD COLUMN deleted_at TIMESTAMP"))


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
