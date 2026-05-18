"""Shared Pydantic schemas.

The full set of fields is fleshed out in task 02. The shapes here are the
minimum needed so downstream modules can typecheck against them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Region(BaseModel):
    """A bounding-box region flagged by a VLM."""

    bbox: tuple[int, int, int, int]
    label: str


class AnomalyPrediction(BaseModel):
    """One model's prediction on one image. Backend-agnostic."""

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
    """Aggregated metrics for one (model, dataset, category) cell."""

    model_id: str
    backend: str
    dataset: str
    category: str
    split: Literal["train", "test"] = "test"
    n_images: int = 0
    auroc: float | None = None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None
    pro_score: float | None = None
    mean_latency_ms: float = 0.0
    total_cost_usd: float = 0.0


class ExperimentConfig(BaseModel):
    """Inputs for a single evaluation run."""

    backend: str
    dataset: str
    categories: list[str] = Field(default_factory=list)
    prompt: str = "generic.simple"
    limit: int | None = None
    budget_usd: float | None = None
    seed: int = 42
