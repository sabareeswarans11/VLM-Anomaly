"""Runtime configuration loaded from environment variables + .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Top-level settings resolved from environment variables and ``.env``.

    Env-var lookup rules:
    - API keys use their conventional names (``TOGETHER_API_KEY``, etc.)
      via ``alias``; the ``VLM_ANOMALY_`` prefix is NOT applied to them.
    - Path / budget fields use the ``VLM_ANOMALY_`` prefix
      (e.g. ``VLM_ANOMALY_DATA_DIR``).

    All paths are normalised to absolute on construction so callers never
    have to worry about relative-vs-absolute.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="VLM_ANOMALY_",
        extra="ignore",
        populate_by_name=True,
        env_ignore_empty=True,  # treat VAR= in .env as unset, falling back to default
    )

    # ── API keys ─────────────────────────────────────────────────────────────
    # alias bypasses the VLM_ANOMALY_ prefix for these well-known names.
    together_api_key: str | None = Field(default=None, alias="TOGETHER_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    openrouter_api_key: str | None = Field(default=None, alias="OPEN_ROUTER")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")

    # ── Paths (take the VLM_ANOMALY_ prefix) ─────────────────────────────────
    data_dir: Path = Path("./data")
    results_dir: Path = Path("./results")
    models_dir: Path = Path("./models")

    # ── Budget ────────────────────────────────────────────────────────────────
    default_budget_usd: float = Field(default=5.0, gt=0.0)

    # ── Logging ───────────────────────────────────────────────────────────────
    # Set LOG_FORMAT=json in production to switch structlog to JSON renderer.
    log_format: str = Field(default="console", alias="LOG_FORMAT")

    # ── Path normalisation ────────────────────────────────────────────────────

    @field_validator("data_dir", "results_dir", "models_dir", mode="before")
    @classmethod
    def _resolve_path(cls, v: str | Path) -> Path:
        return Path(v).resolve()


def get_settings() -> Settings:
    """Return a populated :class:`Settings` instance.

    Reads ``.env`` if present; all fields have sensible defaults so this
    works on a fresh clone with no environment configuration.
    """
    return Settings()
