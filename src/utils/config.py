"""
Charter-AI — Application Configuration.

Loads settings from environment variables / .env file using Pydantic BaseSettings.
All configuration is centralized here to avoid scattered os.getenv() calls.
"""

from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root = charter-ai/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from .env / environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- Database ----------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "charter_ai"
    postgres_user: str = "charter_ai"
    postgres_password: str = "changeme_in_production"
    database_url: str | None = None  # Override full URL if needed

    @property
    def db_url_async(self) -> str:
        """Construct async database URL."""
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def db_url_sync(self) -> str:
        """Construct sync database URL (for Alembic, ingestion scripts)."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ---------- API ----------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    sih_demo_mode: bool = True

    @field_validator("api_cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    # ---------- Paths ----------
    raw_data_dir: str = str(_PROJECT_ROOT / "data" / "raw")
    processed_data_dir: str = str(_PROJECT_ROOT / "data" / "processed")
    trained_models_dir: str = str(_PROJECT_ROOT / "trained_models")

    # ---------- ML ----------
    model_version: str = "v0.1.0"
    forecast_default_horizon_days: int = 30

    # ---------- Logging ----------
    log_level: str = "INFO"
    log_format: str = "json"


def get_settings() -> Settings:
    """Factory that returns a cached Settings instance."""
    return Settings()
