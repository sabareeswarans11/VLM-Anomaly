"""Abstract base class for all VLM backends (cloud + on-device)."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from vlm_anomaly.schemas import AnomalyPrediction

# Default retry policy for transient HTTP errors (429 rate-limit, 5xx).
_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)


class VLMBackend(ABC):
    """Every cloud or edge VLM backend implements this contract.

    Cloud backends use ``httpx.AsyncClient`` internally and call
    ``asyncio.run(_async_predict(...))`` from the synchronous ``predict``
    method, keeping the public interface simple for the evaluator.

    The backend is responsible for:
    - Base64-encoding the image.
    - Constructing the provider-specific request.
    - Measuring wall-clock latency.
    - Computing cost from token counts.
    - Returning a fully populated :class:`~vlm_anomaly.schemas.AnomalyPrediction`.
    """

    name: str

    @abstractmethod
    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        """Run inference on ``image`` with ``prompt`` and return a prediction.

        Args:
            image: Path to a readable image file.
            prompt: Rendered prompt string from the prompt library.

        Returns:
            A fully populated :class:`~vlm_anomaly.schemas.AnomalyPrediction`.
        """

    # ------------------------------------------------------------------
    # Shared async HTTP helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run(coro: Any) -> Any:
        """Run ``coro`` in a new event loop (safe outside Jupyter)."""
        return asyncio.run(coro)

    @staticmethod
    def _make_client(timeout: float = 60.0) -> httpx.AsyncClient:
        """Return a configured ``httpx.AsyncClient``."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            follow_redirects=True,
        )
