"""Together.ai backend — Qwen3-VL and other multimodal models.

Together.ai exposes an OpenAI-compatible chat-completions endpoint, so the
request format mirrors the OpenAI vision API.
"""

from __future__ import annotations

import time
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential

from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.config import get_settings
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import AnomalyPrediction
from vlm_anomaly.utils.image_utils import image_to_data_url
from vlm_anomaly.utils.json_parsing import parse_anomaly_prediction_dict

log = get_logger(__name__)

_API_URL = "https://api.together.xyz/v1/chat/completions"

# Approximate per-token prices (USD) for Qwen3-VL on Together.ai (2026).
_PRICE_IN_PER_TOKEN = 0.00000018   # $0.18 / 1M tokens
_PRICE_OUT_PER_TOKEN = 0.00000018


class TogetherBackend(VLMBackend):
    """Together.ai multimodal backend.

    Args:
        model: Together.ai model identifier.
        api_key: Together API key.  Defaults to ``TOGETHER_API_KEY`` env var.
        max_tokens: Maximum tokens to generate.
    """

    name = "together"

    def __init__(
        self,
        model: str = "Qwen/Qwen2-VL-72B-Instruct",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_key = api_key or get_settings().together_api_key
        self.max_tokens = max_tokens

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        return self._run(self._async_predict(image, prompt))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30), reraise=True)
    async def _async_predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is not set.")

        data_url = image_to_data_url(image)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
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
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        resp.raise_for_status()
        body = resp.json()

        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        cost = tokens_in * _PRICE_IN_PER_TOKEN + tokens_out * _PRICE_OUT_PER_TOKEN

        data, parse_error = parse_anomaly_prediction_dict(raw)
        log.debug("together.predict", model=self.model, latency_ms=round(latency_ms), parse_error=parse_error)

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
