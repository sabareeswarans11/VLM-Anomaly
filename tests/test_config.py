"""Unit tests for vlm_anomaly.config and vlm_anomaly.logging."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from vlm_anomaly.config import Settings, get_settings

# ---------------------------------------------------------------------------
# Settings — defaults
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    def test_get_settings_works_without_env(self, tmp_path: Path) -> None:
        """Must not raise even on a fresh clone with no .env file."""
        with patch.dict(os.environ, {}, clear=False):
            s = Settings(_env_file=str(tmp_path / "nonexistent.env"))
        assert s.default_budget_usd == 5.0

    def test_api_keys_default_to_none(self, tmp_path: Path) -> None:
        # Clear API-key env vars AND redirect env_file so a real .env on disk
        # doesn't leak keys into the test.
        clean = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        with patch.dict(os.environ, clean, clear=True):
            s = Settings(_env_file=str(tmp_path / "empty.env"))
        assert s.together_api_key is None
        assert s.gemini_api_key is None
        assert s.anthropic_api_key is None
        assert s.groq_api_key is None

    def test_paths_are_absolute(self) -> None:
        s = get_settings()
        assert s.data_dir.is_absolute()
        assert s.results_dir.is_absolute()
        assert s.models_dir.is_absolute()

    def test_default_paths_resolve(self) -> None:
        s = get_settings()
        # Resolved path must end with the expected directory names
        assert s.data_dir.name == "data"
        assert s.results_dir.name == "results"
        assert s.models_dir.name == "models"


# ---------------------------------------------------------------------------
# Settings — reading from environment
# ---------------------------------------------------------------------------


class TestSettingsFromEnv:
    def test_together_api_key_read_from_env(self) -> None:
        with patch.dict(os.environ, {"TOGETHER_API_KEY": "sk-test-together"}):
            s = Settings()
        assert s.together_api_key == "sk-test-together"

    def test_anthropic_api_key_read_from_env(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            s = Settings()
        assert s.anthropic_api_key == "sk-ant-test"

    def test_custom_budget_from_env(self) -> None:
        with patch.dict(os.environ, {"VLM_ANOMALY_DEFAULT_BUDGET_USD": "10.0"}):
            s = Settings()
        assert s.default_budget_usd == pytest.approx(10.0)

    def test_custom_data_dir_from_env(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"VLM_ANOMALY_DATA_DIR": str(tmp_path / "mydata")}):
            s = Settings()
        assert s.data_dir == (tmp_path / "mydata").resolve()
        assert s.data_dir.is_absolute()

    def test_custom_results_dir_from_env(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"VLM_ANOMALY_RESULTS_DIR": str(tmp_path / "out")}):
            s = Settings()
        assert s.results_dir.is_absolute()
        assert s.results_dir.name == "out"

    def test_mlflow_tracking_uri_from_env(self) -> None:
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://localhost:5000"}):
            s = Settings()
        assert s.mlflow_tracking_uri == "http://localhost:5000"

    def test_log_format_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            s = Settings()
        assert s.log_format == "console"

    def test_log_format_json(self) -> None:
        with patch.dict(os.environ, {"LOG_FORMAT": "json"}):
            s = Settings()
        assert s.log_format == "json"


# ---------------------------------------------------------------------------
# Logging module
# ---------------------------------------------------------------------------


class TestLogging:
    def test_get_logger_returns_bound_logger(self) -> None:

        from vlm_anomaly.logging import get_logger

        log = get_logger("test.module")
        assert log is not None

    def test_configure_logging_console(self) -> None:
        from vlm_anomaly.logging import configure_logging

        configure_logging(json_logs=False, log_level="WARNING")

    def test_configure_logging_json(self) -> None:
        from vlm_anomaly.logging import configure_logging

        configure_logging(json_logs=True, log_level="ERROR")

    def test_logger_emits_without_error(self, capsys: pytest.CaptureFixture) -> None:
        from vlm_anomaly.logging import configure_logging, get_logger

        configure_logging(json_logs=False, log_level="DEBUG")
        log = get_logger("test")
        log.debug("hello", key="value")
