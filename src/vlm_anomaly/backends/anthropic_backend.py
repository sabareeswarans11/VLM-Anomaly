"""Anthropic Claude backend — Claude Opus 4.6 / 4.7 and Sonnet 4.x.

Uses the Anthropic Messages API directly via httpx.
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

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

# Per-token prices (USD) — Anthropic list pricing (2026).
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (0.000015, 0.000075),
    "claude-opus-4-7": (0.000015, 0.000075),
    "claude-sonnet-4-6": (0.000003, 0.000015),
    "claude-haiku-4-5-20251001": (0.0000008, 0.000004),
}


class AnthropicBackend(VLMBackend):
    """Anthropic Claude multimodal backend.

    Args:
        model: Anthropic model ID (e.g. ``"claude-opus-4-7"``).
        api_key: Anthropic API key.  Defaults to ``ANTHROPIC_API_KEY`` env var.
        max_tokens: Maximum tokens to generate.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-4-7",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_key = api_key or get_settings().anthropic_api_key
        self.max_tokens = max_tokens
        price_in, price_out = _PRICES.get(model, (0.000015, 0.000075))
        self._price_in = price_in
        self._price_out = price_out

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        return self._run(self._async_predict(image, prompt))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    async def _async_predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")

        b64 = image_to_base64(image)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        t0 = time.perf_counter()
        async with self._make_client() as client:
            resp = await client.post(
                _API_URL,
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        body = resp.json()

        raw = body["content"][0]["text"]
        usage = body.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        cost = tokens_in * self._price_in + tokens_out * self._price_out

        data, parse_error = parse_anomaly_prediction_dict(raw)
        log.debug(
            "anthropic.predict",
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
