"""Abstract base class for all VLM backends (cloud + on-device)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from vlm_anomaly.schemas import AnomalyPrediction


class VLMBackend(ABC):
    """Every cloud or edge VLM backend implements this contract.

    A backend takes an image path and a rendered prompt string and returns
    a fully populated :class:`AnomalyPrediction`. Latency, cost, and token
    counts must be filled in by the backend itself.
    """

    name: str

    @abstractmethod
    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        """Run inference on ``image`` with ``prompt`` and return a prediction."""
