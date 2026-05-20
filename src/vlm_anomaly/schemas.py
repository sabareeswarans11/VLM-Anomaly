"""Shared Pydantic schemas.

The full set of fields is fleshed out in task 02. The shapes here are the
minimum needed so downstream modules can typecheck against them.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class Region(BaseModel):
    """A bounding-box region flagged by a VLM.

    bbox coords are floats because VLMs return normalised or float pixel values.
    """

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    label: str


class AnomalyPrediction(BaseModel):
    """One model's prediction on one image. Backend-agnostic.

    ``image_path`` is carried here so the evaluator can correlate predictions
    back to ground-truth samples without maintaining a parallel list.
    """

    image_path: Path | None = None
    is_anomalous: bool
    confidence: float = Field(ge=0.0, le=1.0)
    description: str = ""
    defect_type: str | None = None
    regions: list[Region] = Field(default_factory=list)
    raw_response: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    parse_error: bool = False


class EvalResult(BaseModel):
    """Aggregated metrics for one (model, dataset, category) cell.

    ``split`` reflects the dataset split evaluated (always "test" for VLMs since
    they are zero-shot; "test" for classical methods too — training is internal).
    """

    model_id: str
    backend: str
    dataset: str
    category: str
    split: str = "test"  # open string; "test" is the only meaningful value for this toolkit
    n_images: int = 0
    auroc: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    pro_score: float | None = None
    mean_latency_ms: float = 0.0
    total_cost_usd: float = 0.0


class ExperimentConfig(BaseModel):
    """Inputs for a single evaluation run.

    ``categories=None`` means "run all categories in the dataset".
    An explicit list restricts the run to those categories only.
    """

    backend: str
    dataset: str
    categories: list[str] | None = None  # None = all categories
    prompt: str = "generic.simple"
    limit: int | None = None
    budget_usd: float | None = None
    seed: int = 42
