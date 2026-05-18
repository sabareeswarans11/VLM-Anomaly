"""Deterministic mock VLM backend for tests and offline demos."""

from __future__ import annotations

import hashlib
from pathlib import Path

from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.schemas import AnomalyPrediction


class MockVLMBackend(VLMBackend):
    """Returns predictions seeded from a hash of the image path.

    This means the same image always gets the same prediction, which is what
    we need for deterministic unit tests without touching any network.
    """

    name = "mock"

    def __init__(self, anomaly_threshold: float = 0.5) -> None:
        self._threshold = anomaly_threshold

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        """Return a deterministic prediction derived from the image path."""
        digest = hashlib.sha256(str(image).encode()).digest()
        confidence = digest[0] / 255.0
        return AnomalyPrediction(
            is_anomalous=confidence >= self._threshold,
            confidence=confidence,
            description=f"mock prediction for {image.name}",
            defect_type="mock" if confidence >= self._threshold else None,
            raw_response="{}",
            latency_ms=1.0,
            cost_usd=0.0,
            tokens_in=len(prompt),
            tokens_out=0,
        )
