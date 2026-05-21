"""Groq free-tier backend — Llama-4-Scout with vision.

Groq exposes an OpenAI-compatible chat-completions endpoint, so the
request format is identical to the Together.ai backend.  The free tier
has rate limits; this backend sets conservative defaults to avoid 429s.
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

_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqBackend(VLMBackend):
    """Groq free-tier multimodal backend (Llama-4-Scout).

    Args:
        model: Groq model name.
        api_key: Groq API key.  Defaults to ``GROQ_API_KEY`` env var.
        max_tokens: Maximum tokens to generate.
    """

    name = "groq"

    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_key = api_key or get_settings().groq_api_key
        self.max_tokens = max_tokens

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        return self._run(self._async_predict(image, prompt))

    # Groq free tier: longer back-off on 429s
    @retry(
        stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True
    )
    async def _async_predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set.")

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
        async with self._make_client(timeout=90.0) as client:
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

        data, parse_error = parse_anomaly_prediction_dict(raw)
        log.debug(
            "groq.predict", model=self.model, latency_ms=round(latency_ms), parse_error=parse_error
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
            cost_usd=0.0,  # free tier
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            parse_error=parse_error,
        )
