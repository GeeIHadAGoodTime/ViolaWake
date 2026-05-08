"""Application configuration for ViolaWake Console backend."""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

DEFAULT_CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


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
    tmp_dir: Path = Path(__file__).resolve().parent.parent / "data" / "tmp"
    upload_volume_path: Path = Path("/app/data")
    upload_global_max_used_bytes: int = 50 * 1024 * 1024 * 1024
    upload_global_min_free_bytes: int = 5 * 1024 * 1024 * 1024
    use_decoder_sidecar: bool = False
    decoder_sidecar_url: str = "http://decoder:8001/decode"

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
    access_token_expire_hours: int = 2
    trusted_proxy_count: int = 0

    # CORS
    cors_origins: Annotated[list[str], NoDecode] = DEFAULT_CORS_ORIGINS.copy()

    # Training
    training_timeout: int = 1800  # seconds (30 minutes)
    max_concurrent_jobs: int = 2
    negatives_corpus_dir: str = ""  # Path to curated negative audio corpus (paid tier)

    # Retention cleanup (0 = disabled)
    recording_retention_days: int = 90  # Days to keep recordings; 0 disables automatic cleanup
    model_retention_days: int = 365  # Days to keep trained models; 0 disables automatic cleanup
    post_training_retention_hours: int = 24  # Hours to keep recordings after training completes; 0 disables

    # Admin
    admin_token: str = ""  # When set, enables POST /api/admin/cleanup (protect with a strong secret)

    # Stripe billing
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_developer: str = ""  # Stripe Price ID for $29/mo Developer tier
    stripe_price_business: str = ""  # Stripe Price ID for $99/mo Business tier
    sentry_dsn: str = ""

    # Free trial
    trial_days: int = 14  # 0 to disable free trial for paid tiers

    # Console URLs (for Stripe checkout redirect)
    console_base_url: str = "http://localhost:5173"

    # Email
    resend_api_key: str = ""
    email_inbound_webhook_secret: str = ""
    support_autoreply_window_hours: int = 24

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
        _MIN_SECRET_KEY_LENGTH = 32
        _INSECURE_PLACEHOLDERS = {"changeme", "secret", "password", "test", "dev"}

        key = self.secret_key.strip()
        key_is_empty = not key
        key_is_placeholder = key.lower() in _INSECURE_PLACEHOLDERS

        if key_is_empty or key_is_placeholder:
            if self.is_production:
                raise ValueError(
                    "VIOLAWAKE_SECRET_KEY must be set to a unique, random value "
                    "when VIOLAWAKE_ENV=production. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )
            import logging
            _logger = logging.getLogger("violawake.config")
            self.secret_key = _generate_dev_secret_key()
            _logger.warning(
                "VIOLAWAKE_SECRET_KEY was empty or insecure — generated a random "
                "development key. DO NOT use this in production."
            )
        elif len(key) < _MIN_SECRET_KEY_LENGTH:
            if self.is_production:
                raise ValueError(
                    f"VIOLAWAKE_SECRET_KEY is too short ({len(key)} chars). "
                    f"Production requires at least {_MIN_SECRET_KEY_LENGTH} characters. "
                    "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
                )
            import logging
            _logger = logging.getLogger("violawake.config")
            _logger.warning(
                "VIOLAWAKE_SECRET_KEY is only %d characters (minimum %d recommended). "
                "Short keys are brute-forceable. This is acceptable for development only.",
                len(key),
                _MIN_SECRET_KEY_LENGTH,
            )
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
        """Return CORS origins.

        If ``cors_origins`` was explicitly set via env var, use exactly those.
        Otherwise fall back to sensible defaults based on the environment.
        """
        if self.cors_origins != DEFAULT_CORS_ORIGINS:
            return list(self.cors_origins)
        if self.is_production:
            return [
                "https://console.violawake.com",
                "https://violawake.com",
            ]
        return list(DEFAULT_CORS_ORIGINS)

    @property
    def billing_enabled(self) -> bool:
        """Billing features require a configured Stripe secret key."""
        return bool(self.stripe_secret_key)


settings = Settings()

# Ensure runtime directories exist for uploads, models, and the default SQLite path.
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.models_dir.mkdir(parents=True, exist_ok=True)
settings.tmp_dir.mkdir(parents=True, exist_ok=True)
