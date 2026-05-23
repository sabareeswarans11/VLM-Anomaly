"""OpenRouter backend — Qwen3-VL-32B and other multimodal models.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint.
Paid-tier limits are generous; we use a 2 s proactive floor to stay polite.

Pricing for Qwen3-VL-32B (discounted):
  Input:  $0.104 / 1M tokens
  Output: $0.416 / 1M tokens
"""

from __future__ import annotations

import asyncio
import threading
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

_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_PRICES: dict[str, tuple[float, float]] = {
    "qwen/qwen3-vl-32b-instruct": (0.000000104, 0.000000416),
    "qwen/qwen2.5-vl-72b-instruct": (0.000000400, 0.000000400),
    "qwen/qwen2.5-vl-7b-instruct": (0.000000100, 0.000000100),
}

_MIN_INTERVAL_S: float = 2.0

_rate_lock = threading.Lock()
_rate_state: dict[str, float] = {"last_call_ts": 0.0}


def _wait_for_rate_limit() -> None:
    with _rate_lock:
        now = time.monotonic()
        gap = _MIN_INTERVAL_S - (now - _rate_state["last_call_ts"])
        if gap > 0:
            time.sleep(gap)
        _rate_state["last_call_ts"] = time.monotonic()


class OpenRouterBackend(VLMBackend):
    """OpenRouter multimodal backend (Qwen3-VL and others).

    Args:
        model: OpenRouter model identifier (e.g. ``"qwen/qwen3-vl-32b-instruct"``).
        api_key: OpenRouter API key.  Defaults to ``OPEN_ROUTER`` env var.
        max_tokens: Maximum tokens to generate.
    """

    name = "openrouter"

    def __init__(
        self,
        model: str = "qwen/qwen3-vl-32b-instruct",
        api_key: str | None = None,
        max_tokens: int = 512,
    ) -> None:
        self.model = model
        self.api_key = api_key or get_settings().openrouter_api_key
        self.max_tokens = max_tokens
        price_in, price_out = _PRICES.get(model, (0.000001, 0.000004))
        self._price_in = price_in
        self._price_out = price_out

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        if not self.api_key:
            raise ValueError("OPEN_ROUTER API key is not set.")
        _wait_for_rate_limit()
        return self._run(self._async_predict(image, prompt))

    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    async def _async_predict(self, image: Path, prompt: str) -> AnomalyPrediction:

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
        async with self._make_client(timeout=120.0) as client:
            resp = await client.post(
                _API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/sabareeswarans11/VLM-Anomaly",
                    "X-Title": "VLM-Anomaly",
                },
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", 30))
            retry_after = min(retry_after, 60.0)
            log.warning("openrouter.rate_limited", retry_after_s=retry_after, model=self.model)
            await asyncio.sleep(retry_after)
            resp.raise_for_status()

        resp.raise_for_status()
        body = resp.json()

        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        cost = tokens_in * self._price_in + tokens_out * self._price_out

        data, parse_error = parse_anomaly_prediction_dict(raw)
        log.debug(
            "openrouter.predict",
            model=self.model,
            latency_ms=round(latency_ms),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(cost, 6),
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
