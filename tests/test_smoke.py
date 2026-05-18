"""Sanity checks that pass on a fresh clone with zero API keys and zero data."""

from __future__ import annotations

from pathlib import Path

from vlm_anomaly import __version__
from vlm_anomaly.backends.mock import MockVLMBackend
from vlm_anomaly.schemas import AnomalyPrediction, EvalResult, ExperimentConfig


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_mock_backend_is_deterministic(mock_backend: MockVLMBackend) -> None:
    image = Path("nonexistent/path/a.png")
    p1 = mock_backend.predict(image, "describe defects")
    p2 = mock_backend.predict(image, "describe defects")
    assert p1 == p2


def test_mock_backend_returns_valid_prediction(mock_backend: MockVLMBackend) -> None:
    pred = mock_backend.predict(Path("foo.png"), "hi")
    assert isinstance(pred, AnomalyPrediction)
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.cost_usd == 0.0


def test_eval_result_round_trips() -> None:
    result = EvalResult(
        model_id="mock-v0",
        backend="mock",
        dataset="mvtec",
        category="bottle",
        n_images=10,
        auroc=0.5,
        f1=0.5,
    )
    assert result.model_dump()["category"] == "bottle"


def test_experiment_config_defaults() -> None:
    cfg = ExperimentConfig(backend="mock", dataset="mvtec")
    assert cfg.seed == 42
    assert cfg.prompt == "generic.simple"
