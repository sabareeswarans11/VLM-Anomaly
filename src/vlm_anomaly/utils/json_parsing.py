"""Robust extraction of structured JSON from free-form VLM responses.

VLMs frequently wrap JSON in markdown fences, add prose before/after, or
return subtly malformed JSON.  This module implements the five-step fallback
chain described in CLAUDE.md §7.3:

1. ``json.loads`` on the full response.
2. Extract the first balanced ``{…}`` block.
3. Strip code fences (` ```json … ``` ` or ` ``` … ``` `).
4. Regex extraction of the keys we care about.
5. Give up — return ``parse_error=True`` with the raw response preserved.
"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> tuple[dict[str, Any], bool]:
    """Extract the first JSON object from ``text``.

    Args:
        text: Raw string returned by a VLM.

    Returns:
        A tuple ``(data, parse_error)`` where ``data`` is the parsed dict
        (possibly empty) and ``parse_error`` is ``True`` when all fallbacks
        failed.
    """
    text = text.strip()

    # Step 1 — direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, False
    except json.JSONDecodeError:
        pass

    # Step 2 — extract first balanced { … } block
    block = _extract_balanced_braces(text)
    if block:
        try:
            result = json.loads(block)
            if isinstance(result, dict):
                return result, False
        except json.JSONDecodeError:
            pass

    # Step 3 — strip markdown code fences then retry
    stripped = _strip_code_fences(text)
    if stripped != text:
        try:
            result = json.loads(stripped)
            if isinstance(result, dict):
                return result, False
        except json.JSONDecodeError:
            pass
        # Also try balanced braces on the stripped version
        block2 = _extract_balanced_braces(stripped)
        if block2:
            try:
                result = json.loads(block2)
                if isinstance(result, dict):
                    return result, False
            except json.JSONDecodeError:
                pass

    # Step 4 — regex key extraction for the keys we care about
    extracted = _regex_extract(text)
    if extracted:
        return extracted, False

    # Step 5 — give up
    return {}, True


def parse_anomaly_prediction_dict(text: str) -> tuple[dict[str, Any], bool]:
    """Extract and normalise a VLM response into an ``AnomalyPrediction`` dict.

    Handles common VLM quirks: ``"yes"``/``"no"`` for booleans, string
    confidence values, missing optional keys.

    Args:
        text: Raw VLM response string.

    Returns:
        ``(data_dict, parse_error)`` ready to be passed to
        ``AnomalyPrediction(**data_dict)``.
    """
    data, parse_error = extract_json(text)
    if parse_error:
        return {}, True

    normalised: dict[str, Any] = {}

    # is_anomalous — coerce strings and integers
    raw_flag = data.get("is_anomalous")
    if isinstance(raw_flag, bool):
        normalised["is_anomalous"] = raw_flag
    elif isinstance(raw_flag, str):
        normalised["is_anomalous"] = raw_flag.lower() in ("true", "yes", "1")
    elif isinstance(raw_flag, int):
        normalised["is_anomalous"] = bool(raw_flag)
    else:
        return {}, True  # required field missing

    # confidence
    raw_conf = data.get("confidence")
    try:
        conf = (
            float(raw_conf)
            if raw_conf is not None
            else (1.0 if normalised["is_anomalous"] else 0.0)
        )
        normalised["confidence"] = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        normalised["confidence"] = 1.0 if normalised["is_anomalous"] else 0.0

    # optional string fields
    for key in ("description", "defect_type"):
        if key in data and isinstance(data[key], str):
            normalised[key] = data[key]

    # regions
    raw_regions = data.get("regions", [])
    if isinstance(raw_regions, list):
        normalised["regions"] = raw_regions

    return normalised, False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_balanced_braces(text: str) -> str:
    """Return the substring spanning the first balanced ``{…}`` block."""
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences."""
    # ```json\n...\n``` or ```\n...\n```
    pattern = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text


_BOOL_RE = re.compile(r'"is_anomalous"\s*:\s*(true|false|"true"|"false"|"yes"|"no"|1|0)', re.I)
_CONF_RE = re.compile(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)')
_DESC_RE = re.compile(r'"description"\s*:\s*"([^"]*)"')
_DEFT_RE = re.compile(r'"defect_type"\s*:\s*"([^"]*)"')


def _regex_extract(text: str) -> dict[str, Any]:
    """Best-effort extraction of known keys via regex when JSON parsing fails."""
    result: dict[str, Any] = {}

    m = _BOOL_RE.search(text)
    if not m:
        return {}
    raw = m.group(1).lower().strip('"')
    result["is_anomalous"] = raw in ("true", "yes", "1")

    m2 = _CONF_RE.search(text)
    if m2:
        result["confidence"] = max(0.0, min(1.0, float(m2.group(1))))
    else:
        result["confidence"] = 1.0 if result["is_anomalous"] else 0.0

    m3 = _DESC_RE.search(text)
    if m3:
        result["description"] = m3.group(1)

    m4 = _DEFT_RE.search(text)
    if m4:
        result["defect_type"] = m4.group(1)

    return result
