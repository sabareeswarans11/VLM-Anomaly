"""Runtime configuration loaded from environment variables.

Implemented in task 02. This module currently exposes the Settings shape
so other packages can import it without import-time side effects.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level settings read from environment + .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VLM_ANOMALY_",
        extra="ignore",
    )

    # API keys (no prefix — these mirror common conventions)
    together_api_key: str | None = Field(default=None, alias="TOGETHER_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    # MLflow
    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")

    # Paths (these DO take the VLM_ANOMALY_ prefix)
    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    models_dir: Path = Path("./models")

    # Budgets
    default_budget_usd: float = 5.0


def get_settings() -> Settings:
    """Return a fresh Settings instance."""
    return Settings()
