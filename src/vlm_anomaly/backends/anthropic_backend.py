"""Anthropic Claude backend — Claude Opus 4.7 / Sonnet 4.6 / Haiku 4.5.

Supports both single-image and batched (up to 10 images per call) inference.
Batching is strongly recommended to maximise throughput within the TPM limits:

  Free tier Opus:  500K input TPM  → ~36 batch-of-10 calls/min
  Free tier Sonnet: 30K input TPM  → ~2  batch-of-10 calls/min

Cost (Opus 4.7): $15/1M input · $75/1M output
Cost (Sonnet 4.6): $3/1M input · $15/1M output
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
from vlm_anomaly.utils.image_utils import image_to_base64
from vlm_anomaly.utils.json_parsing import parse_anomaly_prediction_dict, parse_batch_response

log = get_logger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (0.000015, 0.000075),
    "claude-opus-4-7": (0.000015, 0.000075),
    "claude-sonnet-4-6": (0.000003, 0.000015),
    "claude-haiku-4-5-20251001": (0.0000008, 0.000004),
}

BATCH_SIZE = 10

# 2 s proactive floor — conservative for free-tier TPM budgets.
_MIN_INTERVAL_S: float = 2.0
_rate_lock = threading.Lock()
_rate_state: dict[str, float] = {"last_call_ts": 0.0}

_BATCH_SYSTEM_PROMPT = (
    "You are an industrial quality inspector. "
    "Analyze each numbered image for defects or anomalies. "
    "Reply ONLY with a valid JSON array — no markdown, no explanation."
)


def _wait_for_rate_limit() -> None:
    with _rate_lock:
        now = time.monotonic()
        gap = _MIN_INTERVAL_S - (now - _rate_state["last_call_ts"])
        if gap > 0:
            time.sleep(gap)
        _rate_state["last_call_ts"] = time.monotonic()


class AnthropicBackend(VLMBackend):
    """Anthropic Claude multimodal backend.

    Args:
        model: Anthropic model ID (e.g. ``"claude-opus-4-7"``).
        api_key: Anthropic API key.  Defaults to ``ANTHROPIC_API_KEY`` env var.
        max_tokens: Maximum tokens for single-image calls.
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

    # ------------------------------------------------------------------
    # Single-image interface (VLMBackend contract)
    # ------------------------------------------------------------------

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        _wait_for_rate_limit()
        return self._run(self._async_predict(image, prompt))

    @retry(
        stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True
    )
    async def _async_predict(self, image: Path, prompt: str) -> AnomalyPrediction:

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
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }

        t0 = time.perf_counter()
        async with self._make_client(timeout=120.0) as client:
            resp = await client.post(_API_URL, json=payload, headers=self._headers())
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 429:
            retry_after = min(float(resp.headers.get("retry-after", 30)), 60.0)
            log.warning("anthropic.rate_limited", retry_after_s=retry_after, model=self.model)
            await asyncio.sleep(retry_after)
            resp.raise_for_status()

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

    # ------------------------------------------------------------------
    # Batched interface — up to BATCH_SIZE images per API call
    # ------------------------------------------------------------------

    def predict_batch(self, images: list[Path], prompt: str) -> list[AnomalyPrediction]:
        """Classify up to ``BATCH_SIZE`` images in a single API call.

        Args:
            images: List of image paths (max ``BATCH_SIZE``).
            prompt: Prompt context (used to build the batch instruction).

        Returns:
            One :class:`AnomalyPrediction` per input image, in order.
        """
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        _wait_for_rate_limit()
        return self._run(self._async_predict_batch(images, prompt))

    @retry(
        stop=stop_after_attempt(6), wait=wait_exponential(multiplier=2, min=5, max=60), reraise=True
    )
    async def _async_predict_batch(
        self, images: list[Path], prompt: str
    ) -> list[AnomalyPrediction]:

        n = len(images)
        t0 = time.perf_counter()

        # Build interleaved image + index content
        content: list[dict] = []
        for i, img_path in enumerate(images):
            b64 = image_to_base64(img_path)
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                }
            )
            content.append({"type": "text", "text": f"[{i + 1}]"})

        content.append(
            {
                "type": "text",
                "text": (
                    f"For each of the {n} numbered images above, output a JSON array of exactly {n} objects:\n"
                    f'[{{"id":1,"is_anomalous":bool,"confidence":float,"defect_type":str_or_null,"description":str}}, ...]\n'
                    f"JSON array only."
                ),
            }
        )

        payload = {
            "model": self.model,
            "max_tokens": max(400, n * 80),
            "system": _BATCH_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
        }

        async with self._make_client(timeout=180.0) as client:
            resp = await client.post(_API_URL, json=payload, headers=self._headers())
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 429:
            retry_after = min(float(resp.headers.get("retry-after", 30)), 60.0)
            log.warning("anthropic.rate_limited", retry_after_s=retry_after, model=self.model)
            await asyncio.sleep(retry_after)
            resp.raise_for_status()

        resp.raise_for_status()
        body = resp.json()

        raw = body["content"][0]["text"]
        usage = body.get("usage", {})
        tokens_in = usage.get("input_tokens", 0)
        tokens_out = usage.get("output_tokens", 0)
        total_cost = tokens_in * self._price_in + tokens_out * self._price_out
        cost_per_img = total_cost / n

        parsed = parse_batch_response(raw, n)
        log.debug(
            "anthropic.predict_batch",
            model=self.model,
            batch_size=n,
            latency_ms=round(latency_ms),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=round(total_cost, 6),
            parse_errors=sum(1 for _, err in parsed if err),
        )

        results = []
        for img_path, (data, parse_error) in zip(images, parsed):
            results.append(
                AnomalyPrediction(
                    image_path=img_path,
                    is_anomalous=data.get("is_anomalous", False),
                    confidence=data.get("confidence", 0.0),
                    description=data.get("description", ""),
                    defect_type=data.get("defect_type"),
                    regions=data.get("regions", []),
                    raw_response=raw,
                    latency_ms=latency_ms / n,
                    cost_usd=cost_per_img,
                    tokens_in=tokens_in // n,
                    tokens_out=tokens_out // n,
                    parse_error=parse_error,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
