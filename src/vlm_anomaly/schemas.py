"""Shared Pydantic schemas for the entire toolkit.

Every module imports from here. Keep this file free of heavyweight deps —
only pydantic and stdlib.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer


class _BaseSchema(BaseModel):
    """Common config inherited by all schemas.

    ``json_encoders`` is not needed in v2 — Path serialises to string
    automatically in JSON mode.  We set ``populate_by_name=True`` so
    callers can use either the field name or an alias.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        use_enum_values=True,
    )


# ---------------------------------------------------------------------------
# Region
# ---------------------------------------------------------------------------

class Region(_BaseSchema):
    """A bounding-box region flagged by a VLM.

    Coords are floats because VLMs return normalised or float pixel values.
    Order: x1, y1, x2, y2 (top-left, bottom-right).
    """

    bbox: tuple[float, float, float, float]
    label: str

    @field_validator("bbox")
    @classmethod
    def bbox_must_be_valid(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = v
        if x2 < x1 or y2 < y1:
            raise ValueError(f"bbox x2/y2 must be >= x1/y1, got {v}")
        return v


# ---------------------------------------------------------------------------
# AnomalyPrediction  — one model call on one image
# ---------------------------------------------------------------------------

class AnomalyPrediction(_BaseSchema):
    """One model's prediction on one image. Backend-agnostic.

    ``image_path`` lets the evaluator correlate predictions back to
    ground-truth samples without a parallel tracking structure.

    ``confidence`` is the anomaly probability score used for AUROC.
    ``is_anomalous`` is the thresholded binary decision (threshold=0.5
    by default; the evaluator may sweep thresholds for F1 optimisation).
    """

    image_path: Path | None = None
    is_anomalous: bool
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""
    defect_type: str | None = None
    regions: list[Region] = Field(default_factory=list)
    raw_response: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    parse_error: bool = False


# ---------------------------------------------------------------------------
# PerImageResult  — written to results/*.json after each prediction
# ---------------------------------------------------------------------------

class PerImageResult(_BaseSchema):
    """One row persisted to disk after each inference call.

    These are the raw records that DuckDB aggregates into leaderboard rows.
    Keeping predictions and ground-truth together in one record means the
    aggregator never needs to re-join.
    """

    experiment_id: str
    model_id: str
    backend: str
    dataset: str
    category: str
    sample_label: int = Field(ge=0, le=1)        # 0 = normal, 1 = anomaly
    prediction: AnomalyPrediction
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def correct(self) -> bool:
        """True when the binary prediction matches the ground-truth label."""
        return bool(self.prediction.is_anomalous) == bool(self.sample_label)


# ---------------------------------------------------------------------------
# EvalResult  — aggregated metrics for one (model, dataset, category) cell
# ---------------------------------------------------------------------------

class EvalResult(_BaseSchema):
    """Aggregated metrics for one (model, dataset, category) cell.

    Produced by the aggregator from a collection of :class:`PerImageResult`
    records.  ``split`` is always ``"test"`` for this toolkit — VLMs are
    zero-shot and classical methods train internally.
    """

    model_id: str
    backend: str
    dataset: str
    category: str
    split: str = "test"
    n_images: int = Field(default=0, ge=0)
    auroc: float | None = Field(default=None, ge=0.0, le=1.0)
    f1: float | None = Field(default=None, ge=0.0, le=1.0)
    precision: float | None = Field(default=None, ge=0.0, le=1.0)
    recall: float | None = Field(default=None, ge=0.0, le=1.0)
    pro_score: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_latency_ms: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)


# ---------------------------------------------------------------------------
# ExperimentConfig  — inputs for a single evaluation run
# ---------------------------------------------------------------------------

class ExperimentConfig(_BaseSchema):
    """Inputs for a single evaluation run.

    ``categories=None`` means "run all categories in the dataset".
    An explicit list restricts the run to those categories only.
    ``budget_usd=None`` falls back to ``Settings.default_budget_usd``.
    """

    backend: str
    dataset: str
    categories: list[str] | None = None
    prompt: str = "generic.simple"
    limit: int | None = Field(default=None, ge=1)
    budget_usd: float | None = Field(default=None, gt=0.0)
    seed: int = 42
