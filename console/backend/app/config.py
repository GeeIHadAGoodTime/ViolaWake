"""Application configuration for ViolaWake Console backend."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _generate_dev_secret_key() -> str:
    """Generate a development-only JWT key."""
    return secrets.token_urlsafe(32)


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Environment
    env: str = "development"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    db_path: Path = Path(__file__).resolve().parent.parent / "data" / "violawake.db"
    upload_dir: Path = Path(__file__).resolve().parent.parent / "data" / "recordings"
    models_dir: Path = Path(__file__).resolve().parent.parent / "data" / "models"

    # Object storage
    r2_endpoint: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = "violawake"

    # Database
    db_url: str = ""  # Optional full SQLAlchemy async URL, e.g. Railway PostgreSQL

    # Auth
    secret_key: str = ""
    algorithm: str = "HS256"
    access_token_expire_hours: int = 24

    # CORS
    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Training
    training_timeout: int = 1800  # seconds (30 minutes)
    max_concurrent_jobs: int = 2
    negatives_corpus_dir: str = ""  # Path to curated negative audio corpus (paid tier)

    # Stripe billing
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_developer: str = ""  # Stripe Price ID for $29/mo Developer tier
    stripe_price_business: str = ""  # Stripe Price ID for $99/mo Business tier
    sentry_dsn: str = ""

    # Console URLs (for Stripe checkout redirect)
    console_base_url: str = "http://localhost:5173"

    # Email
    resend_api_key: str = ""

    model_config = SettingsConfigDict(
        env_prefix="VIOLAWAKE_",
        env_file=".env",
        extra="ignore",
    )

    @field_validator("env", mode="before")
    @classmethod
    def normalize_env(cls, value: Any) -> str:
        """Accept mixed-case env names while keeping comparisons consistent."""
        if value is None:
            return "development"
        return str(value).strip().lower() or "development"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: Any) -> Any:
        """Accept comma-separated CORS origins from env vars."""
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("["):
                return value
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def validate_production_settings(self) -> Self:
        """Resolve development defaults and enforce production requirements."""
        if not self.secret_key:
            if self.is_production:
                raise ValueError(
                    "VIOLAWAKE_SECRET_KEY must be set when VIOLAWAKE_ENV=production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )
            self.secret_key = _generate_dev_secret_key()
        return self

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def database_url(self) -> str:
        """Return the configured database URL, defaulting to local SQLite."""
        if self.db_url and self.db_url.strip():
            return self.db_url.strip()
        return f"sqlite+aiosqlite:///{self.db_path}"

    @property
    def database_log_target(self) -> str:
        """Return a safe database identifier for logs without leaking credentials."""
        if self.db_url and self.db_url.strip():
            return "VIOLAWAKE_DB_URL"
        return str(self.db_path)

    @property
    def effective_cors_origins(self) -> list[str]:
        """Return CORS origins with production domains appended when in production."""
        origins = list(self.cors_origins)
        if self.is_production:
            prod_origins = [
                "https://console.violawake.com",
                "https://violawake.com",
            ]
            for origin in prod_origins:
                if origin not in origins:
                    origins.append(origin)
        return origins

    @property
    def billing_enabled(self) -> bool:
        """Billing features require a configured Stripe secret key."""
        return bool(self.stripe_secret_key)


settings = Settings()

# Ensure runtime directories exist for uploads, models, and the default SQLite path.
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
