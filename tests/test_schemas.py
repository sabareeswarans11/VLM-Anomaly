"""Unit tests for vlm_anomaly.schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_anomaly.schemas import (
    AnomalyPrediction,
    EvalResult,
    ExperimentConfig,
    PerImageResult,
    Region,
)

# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------


class TestRegion:
    def test_valid_bbox(self) -> None:
        r = Region(bbox=(0.0, 0.0, 100.0, 200.0), label="crack")
        assert r.bbox == (0.0, 0.0, 100.0, 200.0)

    def test_degenerate_bbox_allowed(self) -> None:
        # x1==x2, y1==y2 is a point — valid edge case
        Region(bbox=(50.0, 50.0, 50.0, 50.0), label="point")

    def test_inverted_bbox_rejected(self) -> None:
        with pytest.raises(ValueError, match="x2/y2 must be"):
            Region(bbox=(100.0, 0.0, 50.0, 200.0), label="bad")

    def test_json_round_trip(self) -> None:
        r = Region(bbox=(10.5, 20.5, 30.5, 40.5), label="scratch")
        reloaded = Region.model_validate_json(r.model_dump_json())
        assert reloaded == r


# ---------------------------------------------------------------------------
# AnomalyPrediction
# ---------------------------------------------------------------------------


class TestAnomalyPrediction:
    def test_minimal_construction(self) -> None:
        p = AnomalyPrediction(is_anomalous=True, confidence=0.9)
        assert p.is_anomalous is True
        assert p.parse_error is False
        assert p.regions == []

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValueError):
            AnomalyPrediction(is_anomalous=True, confidence=1.1)
        with pytest.raises(ValueError):
            AnomalyPrediction(is_anomalous=False, confidence=-0.1)

    def test_latency_and_cost_non_negative(self) -> None:
        with pytest.raises(ValueError):
            AnomalyPrediction(is_anomalous=False, confidence=0.0, latency_ms=-1.0)
        with pytest.raises(ValueError):
            AnomalyPrediction(is_anomalous=False, confidence=0.0, cost_usd=-0.01)

    def test_image_path_serialises_to_string(self) -> None:
        p = AnomalyPrediction(
            image_path=Path("/data/mvtec/bottle/test/broken_large/001.png"),
            is_anomalous=True,
            confidence=0.87,
        )
        payload = json.loads(p.model_dump_json())
        assert isinstance(payload["image_path"], str)
        assert "bottle" in payload["image_path"]

    def test_json_round_trip(self) -> None:
        p = AnomalyPrediction(
            image_path=Path("/tmp/img.png"),
            is_anomalous=True,
            confidence=0.75,
            description="crack on surface",
            defect_type="crack",
            regions=[Region(bbox=(10.0, 20.0, 100.0, 200.0), label="crack")],
            raw_response='{"is_anomalous": true}',
            latency_ms=1200.0,
            cost_usd=0.0042,
            tokens_in=512,
            tokens_out=84,
        )
        reloaded = AnomalyPrediction.model_validate_json(p.model_dump_json())
        assert reloaded.confidence == p.confidence
        assert reloaded.regions[0].label == "crack"
        assert reloaded.image_path == p.image_path

    def test_none_image_path_is_allowed(self) -> None:
        p = AnomalyPrediction(is_anomalous=False, confidence=0.1)
        assert p.image_path is None
        payload = json.loads(p.model_dump_json())
        assert payload["image_path"] is None


# ---------------------------------------------------------------------------
# PerImageResult
# ---------------------------------------------------------------------------


class TestPerImageResult:
    def _make(self, label: int = 1, is_anomalous: bool = True) -> PerImageResult:
        return PerImageResult(
            experiment_id="exp-001",
            model_id="mock-v0",
            backend="mock",
            dataset="mvtec",
            category="bottle",
            sample_label=label,
            prediction=AnomalyPrediction(is_anomalous=is_anomalous, confidence=0.9),
        )

    def test_correct_property_true(self) -> None:
        assert self._make(label=1, is_anomalous=True).correct is True
        assert self._make(label=0, is_anomalous=False).correct is True

    def test_correct_property_false(self) -> None:
        assert self._make(label=1, is_anomalous=False).correct is False
        assert self._make(label=0, is_anomalous=True).correct is False

    def test_sample_label_bounds(self) -> None:
        with pytest.raises(ValueError):
            PerImageResult(
                experiment_id="x",
                model_id="m",
                backend="b",
                dataset="d",
                category="c",
                sample_label=2,
                prediction=AnomalyPrediction(is_anomalous=True, confidence=0.5),
            )

    def test_timestamp_defaults_to_utc(self) -> None:
        r = self._make()
        assert r.timestamp.tzinfo is not None

    def test_json_round_trip(self) -> None:
        r = self._make()
        reloaded = PerImageResult.model_validate_json(r.model_dump_json())
        assert reloaded.experiment_id == r.experiment_id
        assert reloaded.prediction.confidence == r.prediction.confidence


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class TestEvalResult:
    def test_minimal_construction(self) -> None:
        r = EvalResult(model_id="padim", backend="anomalib", dataset="mvtec", category="bottle")
        assert r.split == "test"
        assert r.auroc is None

    def test_metric_bounds(self) -> None:
        with pytest.raises(ValueError):
            EvalResult(model_id="x", backend="y", dataset="z", category="w", auroc=1.5)

    def test_json_round_trip(self) -> None:
        r = EvalResult(
            model_id="patchcore",
            backend="anomalib",
            dataset="mvtec",
            category="bottle",
            n_images=83,
            auroc=0.95,
            f1=0.88,
        )
        reloaded = EvalResult.model_validate_json(r.model_dump_json())
        assert reloaded.auroc == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_defaults(self) -> None:
        cfg = ExperimentConfig(backend="mock", dataset="mvtec")
        assert cfg.categories is None
        assert cfg.seed == 42
        assert cfg.prompt == "generic.simple"
        assert cfg.budget_usd is None
        assert cfg.limit is None

    def test_explicit_categories(self) -> None:
        cfg = ExperimentConfig(backend="mock", dataset="mvtec", categories=["bottle", "cable"])
        assert cfg.categories == ["bottle", "cable"]

    def test_invalid_limit(self) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(backend="mock", dataset="mvtec", limit=0)

    def test_invalid_budget(self) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(backend="mock", dataset="mvtec", budget_usd=0.0)

    def test_json_round_trip(self) -> None:
        cfg = ExperimentConfig(
            backend="gemini",
            dataset="visa",
            categories=["candle"],
            prompt="manufacturing.detailed",
            limit=50,
            budget_usd=2.5,
        )
        reloaded = ExperimentConfig.model_validate_json(cfg.model_dump_json())
        assert reloaded.backend == "gemini"
        assert reloaded.categories == ["candle"]
        assert reloaded.budget_usd == pytest.approx(2.5)
