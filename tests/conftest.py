"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_anomaly.backends.mock import MockVLMBackend


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to tests/fixtures/."""
    return FIXTURES_DIR


@pytest.fixture
def mock_backend() -> MockVLMBackend:
    """Deterministic mock backend instance."""
    return MockVLMBackend()


@pytest.fixture
def sample_normal_image(fixtures_dir: Path) -> Path:
    """Path to the bundled normal sample image."""
    return fixtures_dir / "sample_normal.png"


@pytest.fixture
def sample_anomaly_image(fixtures_dir: Path) -> Path:
    """Path to the bundled anomaly sample image."""
    return fixtures_dir / "sample_anomaly.png"
