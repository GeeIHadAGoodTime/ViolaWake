"""Alembic environment configuration for ViolaWake Console.

Reads the database URL from app.config.settings so that the same
VIOLAWAKE_DB_URL / VIOLAWAKE_DB_PATH environment variables are used
for both the running application and schema migrations.

Supports both SQLite (dev) and PostgreSQL (production) via async engines.
"""
from __future__ import annotations

import asyncio
import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.config import settings
from app.database import Base

# Import all models so that Base.metadata is fully populated.
import app.models  # noqa: F401

# Alembic Config object — gives access to alembic.ini values.
config = context.config

# Set up Python logging from alembic.ini unless we are being called
# programmatically (e.g., from pytest) with an existing configuration.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# Target metadata for autogenerate support.
target_metadata = Base.metadata

# Resolve the database URL from application settings, not alembic.ini.
database_url = settings.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Configures the context with just a URL and not an engine.  Calls to
    context.execute() emit the given SQL string to the script output.
    """
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given synchronous connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # render_as_batch=True is required for SQLite, which does not support
        # ALTER TABLE ... DROP COLUMN natively.  It is harmless on PostgreSQL.
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Create an async engine and run migrations."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = database_url
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode with an async engine."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
