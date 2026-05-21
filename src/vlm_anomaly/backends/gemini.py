"""Google Gemini backend — Gemini 3 Flash and Gemini 3 Pro.

Uses the Gemini REST API directly (no Python SDK) via httpx so we stay
consistent with the rest of the codebase's HTTP layer.
"""

from __future__ import annotations

import time
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.config import get_settings
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import AnomalyPrediction
from vlm_anomaly.utils.image_utils import image_to_base64
from vlm_anomaly.utils.json_parsing import parse_anomaly_prediction_dict

log = get_logger(__name__)

_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Approximate per-token prices (USD) for Gemini 3 Flash and Pro (2026).
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3-flash": (0.00000010, 0.00000040),  # $0.10 / $0.40 per 1M
    "gemini-3-pro": (0.00000125, 0.00000500),  # $1.25 / $5.00 per 1M
    "gemini-2.5-flash-preview-05-20": (0.00000010, 0.00000040),
    "gemini-2.5-pro-preview-05-06": (0.00000125, 0.00000500),
}


class GeminiBackend(VLMBackend):
    """Google Gemini multimodal backend.

    Args:
        model: Gemini model name (e.g. ``"gemini-3-flash"``).
        api_key: Google AI API key.  Defaults to ``GEMINI_API_KEY`` env var.
        max_tokens: Maximum tokens to generate.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-3-flash",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_key = api_key or get_settings().gemini_api_key
        self.max_tokens = max_tokens
        price_in, price_out = _PRICES.get(model, (0.000001, 0.000004))
        self._price_in = price_in
        self._price_out = price_out

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        return self._run(self._async_predict(image, prompt))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    async def _async_predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set.")

        b64 = image_to_base64(image)
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {"maxOutputTokens": self.max_tokens},
        }

        url = f"{_API_BASE}/{self.model}:generateContent?key={self.api_key}"

        t0 = time.perf_counter()
        async with self._make_client() as client:
            resp = await client.post(url, json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        body = resp.json()

        raw = body["candidates"][0]["content"]["parts"][0]["text"]
        usage = body.get("usageMetadata", {})
        tokens_in = usage.get("promptTokenCount", 0)
        tokens_out = usage.get("candidatesTokenCount", 0)
        cost = tokens_in * self._price_in + tokens_out * self._price_out

        data, parse_error = parse_anomaly_prediction_dict(raw)
        log.debug(
            "gemini.predict",
            model=self.model,
            latency_ms=round(latency_ms),
            parse_error=parse_error,
        )

        return AnomalyPrediction(
            image_path=image,
            is_anomalous=data.get("is_anomalous", False),
            confidence=data.get("confidence", 0.0),
            description=data.get("description", ""),
            defect_type=data.get("defect_type"),
            regions=data.get("regions", []),
            raw_response=raw,
            latency_ms=latency_ms,
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            parse_error=parse_error,
        )
