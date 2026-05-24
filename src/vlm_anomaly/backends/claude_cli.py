"""Claude CLI backend — routes inference through the local `claude` CLI.

Uses ``--print`` (non-interactive / print mode) with the built-in Read tool
so the CLI reads the image file directly.  Authentication is handled via the
OAuth session stored in the macOS Keychain (set up with ``claude auth login``),
which is the subscription path — NOT an ANTHROPIC_API_KEY with per-token billing.
Cost is tracked via the ``total_cost_usd`` field the CLI reports in its JSON envelope.

Prerequisites:
  - ``claude`` CLI installed (``npm install -g @anthropic-ai/claude-code``).
  - Authenticated via ``claude auth login`` (OAuth, not API key).
  - The subprocess must inherit the full process environment so macOS Keychain
    is reachable.  Jupyter notebooks launched from a normal terminal session
    satisfy this automatically.

Usage::

    from vlm_anomaly.backends.claude_cli import ClaudeCliBackend
    backend = ClaudeCliBackend(model="claude-sonnet-4-6")
    result = backend.predict(Path("image.jpg"), prompt)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import AnomalyPrediction
from vlm_anomaly.utils.json_parsing import parse_anomaly_prediction_dict

log = get_logger(__name__)

# Timeout per subprocess call (seconds). Vision calls can be slow on first run.
_SUBPROCESS_TIMEOUT = 120

_SYSTEM_PROMPT = (
    "You are an industrial quality-inspection AI. "
    "When given an image file path, read the image using your Read tool, inspect it, "
    "then output ONLY a single valid JSON object — no markdown, no explanation, no extra text."
)

_INSTRUCTION = "Read the image file at this path: {image_path}\n\n{prompt}"


class ClaudeCliBackend(VLMBackend):
    """VLM backend that calls the local ``claude`` CLI via OAuth subscription.

    Authentication uses the macOS Keychain OAuth session (``claude auth login``),
    NOT an ANTHROPIC_API_KEY.  The subprocess inherits the full environment so
    Keychain is reachable from Jupyter or script contexts launched from a normal
    terminal.  ``ANTHROPIC_API_KEY`` is explicitly stripped from the env to
    prevent accidental API-key billing fallback.

    Args:
        model: Claude model alias or full ID passed to ``--model``.
        timeout: Subprocess timeout in seconds per image.
    """

    name = "claude_cli"

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        timeout: int = _SUBPROCESS_TIMEOUT,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self._cli = shutil.which("claude")
        if not self._cli:
            raise RuntimeError(
                "'claude' CLI not found on PATH. Install it with: npm install -g @anthropic-ai/claude-code"
            )

    def predict(self, image: Path, prompt: str) -> AnomalyPrediction:
        """Run the CLI for one image and parse the response.

        Args:
            image: Absolute path to a readable image file.
            prompt: Rendered prompt string from the prompt library.

        Returns:
            Fully populated :class:`~vlm_anomaly.schemas.AnomalyPrediction`.
        """
        image = Path(image).resolve()
        full_prompt = _INSTRUCTION.format(image_path=image, prompt=prompt)

        cmd = [
            self._cli,
            "--print",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--allowedTools",
            "Read",
            "--no-session-persistence",
            "--system-prompt",
            _SYSTEM_PROMPT,
            full_prompt,
        ]

        # Inherit the full environment so the CLI can reach the macOS Keychain
        # OAuth session (stored there when `claude auth login` was run interactively).
        # Overriding ANTHROPIC_API_KEY here would force API-key billing instead
        # of the subscription OAuth session — we deliberately do NOT set it so
        # the CLI falls back to OAuth.
        subprocess_env = os.environ.copy()
        subprocess_env.pop("ANTHROPIC_API_KEY", None)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=subprocess_env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log.warning("claude_cli.timeout", image=str(image), timeout=self.timeout)
            return _error_prediction(image, "subprocess timeout", 0.0, self.timeout * 1000)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Parse the CLI's JSON envelope first — even non-zero exits produce JSON
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError:
            log.warning(
                "claude_cli.bad_envelope",
                returncode=proc.returncode,
                stdout=proc.stdout[:200],
                stderr=proc.stderr[:200],
            )
            return _error_prediction(image, "bad CLI JSON envelope", 0.0, latency_ms)

        if envelope.get("is_error"):
            subtype = envelope.get("subtype", "unknown")
            result_text = envelope.get("result", "")
            log.warning(
                "claude_cli.error_result",
                subtype=subtype,
                result=result_text[:120],
                image=str(image),
                total_cost_usd=envelope.get("total_cost_usd", 0.0),
            )
            return _error_prediction(
                image,
                f"CLI error [{subtype}]: {result_text[:80]}",
                float(envelope.get("total_cost_usd", 0.0)),
                latency_ms,
            )

        raw: str = envelope.get("result", "")
        cost_usd: float = float(envelope.get("total_cost_usd", 0.0))

        data, parse_error = parse_anomaly_prediction_dict(raw)
        log.debug(
            "claude_cli.predict",
            model=self.model,
            latency_ms=round(latency_ms),
            cost_usd=round(cost_usd, 6),
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
            cost_usd=cost_usd,
            tokens_in=0,
            tokens_out=0,
            parse_error=parse_error,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_prediction(
    image: Path, reason: str, cost_usd: float, latency_ms: float
) -> AnomalyPrediction:
    return AnomalyPrediction(
        image_path=image,
        is_anomalous=False,
        confidence=0.0,
        description=reason,
        defect_type=None,
        regions=[],
        raw_response=reason,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        tokens_in=0,
        tokens_out=0,
        parse_error=True,
    )
