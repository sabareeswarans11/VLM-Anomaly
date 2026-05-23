"""Groq free-tier backend — Llama-4-Scout with vision.

Groq exposes an OpenAI-compatible chat-completions endpoint, so the
request format is identical to the Together.ai backend.  The free tier
has rate limits; this backend sets conservative defaults to avoid 429s.

Free-tier limits (meta-llama/llama-4-scout-17b-16e-instruct):
  - 1,000 req / day  (RPD)
  - 30,000 tokens / min  (TPM)
  Each vision request ≈ 2,500 tokens, so safe throughput is ~12 req/min.
  We enforce a 6-second floor between calls (≈10 req/min) and respect
  the Retry-After header on 429s so no images are ever dropped.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.config import get_settings
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import AnomalyPrediction
from vlm_anomaly.utils.image_utils import image_to_data_url
from vlm_anomaly.utils.json_parsing import parse_anomaly_prediction_dict

log = get_logger(__name__)

_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Proactive rate-limit floor: 6 s between calls ≈ 10 req/min, well under
# the 30k-TPM ceiling at ~2.5k tokens per vision request.
_MIN_INTERVAL_S: float = 5.5

# Module-level state shared across all GroqBackend instances in a process.
_rate_lock = threading.Lock()
_last_call_ts: float = 0.0


def _wait_for_rate_limit() -> None:
    """Block (synchronously) until the minimum inter-request interval has passed."""
    global _last_call_ts
    with _rate_lock:
        now = time.monotonic()
        gap = _MIN_INTERVAL_S - (now - _last_call_ts)
        if gap > 0:
            time.sleep(gap)
        _last_call_ts = time.monotonic()


class GroqBackend(VLMBackend):
    """Groq free-tier multimodal backend (Llama-4-Scout).

    Args:
        model: Groq model name.
        api_key: Groq API key.  Defaults to ``GROQ_API_KEY`` env var.
        max_tokens: Maximum tokens to generate.
        min_interval_s: Minimum seconds between API calls.  Override to
            tune throughput vs. rate-limit risk.
    """

    name = "groq"

    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key: str | None = None,
        max_tokens: int = 512,
        min_interval_s: float = _MIN_INTERVAL_S,
    ) -> None:
        self.model = model
        self.api_key = api_key or get_settings().groq_api_key
        self.max_tokens = max_tokens
        self.min_interval_s = min_interval_s

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        _wait_for_rate_limit()
        return self._run(self._async_predict(image, prompt))

    # Retry up to 8 times on transient errors; 429s are handled inline below.
    @retry(
        stop=stop_after_attempt(8),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        reraise=True,
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
        async with self._make_client(timeout=120.0) as client:
            resp = await client.post(
                _API_URL,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code == 429:
            retry_after = _parse_retry_after(resp)
            log.warning(
                "groq.rate_limited",
                retry_after_s=retry_after,
                remaining_requests=resp.headers.get("x-ratelimit-remaining-requests"),
                remaining_tokens=resp.headers.get("x-ratelimit-remaining-tokens"),
                reset_tokens=resp.headers.get("x-ratelimit-reset-tokens"),
            )
            await asyncio.sleep(retry_after)
            resp.raise_for_status()  # trigger tenacity retry

        resp.raise_for_status()
        body = resp.json()

        raw = body["choices"][0]["message"]["content"]
        usage = body.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)

        log.debug(
            "groq.predict",
            model=self.model,
            latency_ms=round(latency_ms),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            remaining_requests=resp.headers.get("x-ratelimit-remaining-requests"),
            remaining_tokens=resp.headers.get("x-ratelimit-remaining-tokens"),
        )

        data, parse_error = parse_anomaly_prediction_dict(raw)

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


def _parse_retry_after(resp: httpx.Response) -> float:
    """Return seconds to sleep from Retry-After or a safe default."""
    header = resp.headers.get("retry-after") or resp.headers.get("x-ratelimit-reset-tokens")
    if header:
        # Numeric seconds
        try:
            return float(header)
        except ValueError:
            pass
        # Groq uses strings like "4.85s" or "2m30s"
        return _parse_duration_str(header)
    return 60.0  # conservative fallback


def _parse_duration_str(s: str) -> float:
    """Parse Groq duration strings like '4.85s', '1m30s', '2h10m5s' into seconds."""
    s = s.strip()
    total = 0.0
    import re
    for value, unit in re.findall(r"([\d.]+)([hms])", s):
        v = float(value)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        else:
            total += v
    return total if total > 0 else 60.0
