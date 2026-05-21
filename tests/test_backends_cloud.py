"""Cloud backend contract tests.

Unit tests run with no API keys and no network — they verify that every
backend honours the VLMBackend contract and that shared utilities work
correctly.

Live integration tests are gated behind ``@pytest.mark.integration`` and
require the corresponding API key to be set.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vlm_anomaly.backends.anthropic_backend import AnthropicBackend
from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.backends.gemini import GeminiBackend
from vlm_anomaly.backends.groq import GroqBackend
from vlm_anomaly.backends.mock import MockVLMBackend
from vlm_anomaly.backends.together import TogetherBackend
from vlm_anomaly.schemas import AnomalyPrediction
from vlm_anomaly.utils.cost_tracker import BudgetExceeded, CostTracker
from vlm_anomaly.utils.image_utils import image_to_base64, image_to_data_url, load_and_resize
from vlm_anomaly.utils.json_parsing import extract_json, parse_anomaly_prediction_dict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_NORMAL = Path(__file__).parent / "fixtures" / "sample_normal.png"
SAMPLE_ANOMALY = Path(__file__).parent / "fixtures" / "sample_anomaly.png"
PROMPT = "Is there a defect? Reply with JSON."

_GOOD_RAW = '{"is_anomalous": false, "confidence": 0.1, "description": "looks fine"}'
_BAD_RAW = (
    '{"is_anomalous": true, "confidence": 0.9, "description": "crack", "defect_type": "crack"}'
)


def _fake_response(raw: str, tokens_in: int = 600, tokens_out: int = 80) -> MagicMock:
    """Build a fake httpx Response with a Together/Groq-style body."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "choices": [{"message": {"content": raw}}],
        "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out},
    }
    return mock


def _fake_gemini_response(raw: str, tokens_in: int = 600, tokens_out: int = 80) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": raw}]}}],
        "usageMetadata": {"promptTokenCount": tokens_in, "candidatesTokenCount": tokens_out},
    }
    return mock


def _fake_anthropic_response(raw: str, tokens_in: int = 600, tokens_out: int = 80) -> MagicMock:
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {
        "content": [{"text": raw}],
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
    }
    return mock


# ---------------------------------------------------------------------------
# image_utils
# ---------------------------------------------------------------------------


class TestImageUtils:
    def test_load_and_resize_returns_rgb(self) -> None:
        img = load_and_resize(SAMPLE_NORMAL)
        assert img.mode == "RGB"

    def test_load_and_resize_respects_max_side(self) -> None:
        img = load_and_resize(SAMPLE_ANOMALY, max_side=4)
        assert max(img.size) <= 4

    def test_image_to_base64_is_string(self) -> None:
        b64 = image_to_base64(SAMPLE_NORMAL)
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_image_to_data_url_prefix(self) -> None:
        url = image_to_data_url(SAMPLE_NORMAL)
        assert url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# json_parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_direct_json(self) -> None:
        data, err = extract_json(_GOOD_RAW)
        assert err is False
        assert data["is_anomalous"] is False

    def test_json_wrapped_in_prose(self) -> None:
        text = f"Sure, here is my answer:\n{_BAD_RAW}\nLet me know if you need more."
        data, err = extract_json(text)
        assert err is False
        assert data["is_anomalous"] is True

    def test_json_in_code_fence(self) -> None:
        text = f"```json\n{_GOOD_RAW}\n```"
        data, err = extract_json(text)
        assert err is False
        assert data["confidence"] == pytest.approx(0.1)

    def test_json_in_plain_fence(self) -> None:
        text = f"```\n{_BAD_RAW}\n```"
        data, err = extract_json(text)
        assert err is False
        assert data["is_anomalous"] is True

    def test_fallback_regex(self) -> None:
        messy = 'The answer is: "is_anomalous": true, "confidence": 0.85'
        data, err = extract_json(messy)
        assert err is False
        assert data["is_anomalous"] is True
        assert data["confidence"] == pytest.approx(0.85)

    def test_all_fallbacks_fail(self) -> None:
        _, err = extract_json("I cannot determine the answer.")
        assert err is True

    def test_parse_anomaly_dict_coerces_string_bool(self) -> None:
        raw = '{"is_anomalous": "yes", "confidence": 0.9}'
        data, err = parse_anomaly_prediction_dict(raw)
        assert err is False
        assert data["is_anomalous"] is True

    def test_parse_anomaly_dict_coerces_int_bool(self) -> None:
        raw = '{"is_anomalous": 1, "confidence": 0.7}'
        data, err = parse_anomaly_prediction_dict(raw)
        assert err is False
        assert data["is_anomalous"] is True

    def test_parse_anomaly_dict_clamps_confidence(self) -> None:
        raw = '{"is_anomalous": false, "confidence": 1.5}'
        data, err = parse_anomaly_prediction_dict(raw)
        assert err is False
        assert data["confidence"] <= 1.0

    def test_fence_with_invalid_json_falls_to_regex(self) -> None:
        # Code fence strips OK but content is still broken JSON → regex fallback
        text = '```json\n{"is_anomalous": true BROKEN\n```'
        data, err = extract_json(text)
        # regex finds "is_anomalous": true
        assert err is False
        assert data["is_anomalous"] is True

    def test_balanced_braces_invalid_json_falls_to_step3(self) -> None:
        # Has braces but content is invalid JSON with no fence → regex
        text = 'Answer: { "is_anomalous": true INVALID } extra'
        data, err = extract_json(text)
        assert err is False
        assert data["is_anomalous"] is True

    def test_regex_extracts_description(self) -> None:
        text = '"is_anomalous": false, "confidence": 0.2, "description": "all good"'
        data, err = extract_json(text)
        assert err is False
        assert data.get("description") == "all good"

    def test_regex_extracts_defect_type(self) -> None:
        text = '"is_anomalous": true, "confidence": 0.9, "defect_type": "crack"'
        data, err = extract_json(text)
        assert err is False
        assert data.get("defect_type") == "crack"

    def test_parse_anomaly_dict_bad_confidence_type(self) -> None:
        # confidence is not a number → falls back to default
        raw = '{"is_anomalous": true, "confidence": "high"}'
        data, err = parse_anomaly_prediction_dict(raw)
        assert err is False
        assert data["confidence"] == 1.0  # anomalous default

    def test_parse_anomaly_dict_no_confidence_key(self) -> None:
        raw = '{"is_anomalous": false}'
        data, err = parse_anomaly_prediction_dict(raw)
        assert err is False
        assert data["confidence"] == 0.0  # normal default

    def test_parse_anomaly_dict_with_regions(self) -> None:
        raw = '{"is_anomalous": true, "confidence": 0.8, "regions": [{"bbox": [0,0,10,10], "label": "crack"}]}'
        data, err = parse_anomaly_prediction_dict(raw)
        assert err is False
        assert len(data["regions"]) == 1

    def test_parse_anomaly_dict_missing_required_field(self) -> None:
        raw = '{"confidence": 0.8}'
        _, err = parse_anomaly_prediction_dict(raw)
        assert err is True


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_record_accumulates(self) -> None:
        t = CostTracker()
        t.record(0.01)
        t.record(0.02)
        assert t.total_usd == pytest.approx(0.03)
        assert t.calls == 2

    def test_check_budget_passes(self) -> None:
        t = CostTracker(budget_usd=1.0)
        t.record(0.5)
        t.check_budget(0.4)  # 0.5 + 0.4 = 0.9 < 1.0 — should not raise

    def test_check_budget_raises(self) -> None:
        t = CostTracker(budget_usd=1.0)
        t.record(0.9)
        with pytest.raises(BudgetExceeded):
            t.check_budget(0.2)  # 0.9 + 0.2 = 1.1 > 1.0

    def test_no_budget_never_raises(self) -> None:
        t = CostTracker(budget_usd=None)
        t.record(9999.0)
        t.check_budget(9999.0)  # no limit

    def test_reset(self) -> None:
        t = CostTracker(budget_usd=5.0)
        t.record(2.0)
        t.reset()
        assert t.total_usd == 0.0
        assert t.calls == 0

    def test_remaining_usd(self) -> None:
        t = CostTracker(budget_usd=1.0)
        t.record(0.3)
        assert t.remaining_usd == pytest.approx(0.7)

    def test_summary_keys(self) -> None:
        t = CostTracker(budget_usd=5.0, name="exp-1")
        t.record(0.1)
        s = t.summary()
        assert s["name"] == "exp-1"
        assert s["calls"] == 1
        assert "total_usd" in s


# ---------------------------------------------------------------------------
# MockVLMBackend contract
# ---------------------------------------------------------------------------


class TestMockBackendContract:
    def test_implements_vlm_backend(self) -> None:
        assert isinstance(MockVLMBackend(), VLMBackend)

    def test_returns_anomaly_prediction(self) -> None:
        p = MockVLMBackend().predict(SAMPLE_NORMAL, PROMPT)
        assert isinstance(p, AnomalyPrediction)

    def test_image_path_set(self) -> None:
        p = MockVLMBackend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.image_path == SAMPLE_NORMAL

    def test_deterministic(self) -> None:
        b = MockVLMBackend()
        assert b.predict(SAMPLE_NORMAL, PROMPT) == b.predict(SAMPLE_NORMAL, PROMPT)

    def test_confidence_in_range(self) -> None:
        p = MockVLMBackend().predict(SAMPLE_NORMAL, PROMPT)
        assert 0.0 <= p.confidence <= 1.0

    def test_cost_is_zero(self) -> None:
        p = MockVLMBackend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Together backend contract (mocked HTTP)
# ---------------------------------------------------------------------------


class TestTogetherBackendContract:
    def _backend(self) -> TogetherBackend:
        return TogetherBackend(api_key="sk-fake")

    def _mock_post(self, raw: str):
        resp = _fake_response(raw)
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=False)
        cm.post = AsyncMock(return_value=resp)
        return patch("vlm_anomaly.backends.together.TogetherBackend._make_client", return_value=cm)

    def test_returns_anomaly_prediction(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert isinstance(p, AnomalyPrediction)

    def test_is_anomalous_false(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.is_anomalous is False

    def test_is_anomalous_true(self) -> None:
        with self._mock_post(_BAD_RAW):
            p = self._backend().predict(SAMPLE_ANOMALY, PROMPT)
        assert p.is_anomalous is True

    def test_image_path_set(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.image_path == SAMPLE_NORMAL

    def test_cost_positive(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.cost_usd > 0.0

    def test_tokens_recorded(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.tokens_in == 600
        assert p.tokens_out == 80

    def test_no_api_key_raises(self) -> None:
        b = TogetherBackend(api_key=None)
        b.api_key = None
        with self._mock_post(_GOOD_RAW):
            with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
                b.predict(SAMPLE_NORMAL, PROMPT)


# ---------------------------------------------------------------------------
# Gemini backend contract (mocked HTTP)
# ---------------------------------------------------------------------------


class TestGeminiBackendContract:
    def _backend(self) -> GeminiBackend:
        return GeminiBackend(api_key="fake-key")

    def _mock_post(self, raw: str):
        resp = _fake_gemini_response(raw)
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=False)
        cm.post = AsyncMock(return_value=resp)
        return patch("vlm_anomaly.backends.gemini.GeminiBackend._make_client", return_value=cm)

    def test_returns_anomaly_prediction(self) -> None:
        with self._mock_post(_BAD_RAW):
            p = self._backend().predict(SAMPLE_ANOMALY, PROMPT)
        assert isinstance(p, AnomalyPrediction)

    def test_is_anomalous_true(self) -> None:
        with self._mock_post(_BAD_RAW):
            p = self._backend().predict(SAMPLE_ANOMALY, PROMPT)
        assert p.is_anomalous is True

    def test_cost_computed(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.cost_usd > 0.0

    def test_no_api_key_raises(self) -> None:
        b = GeminiBackend(api_key=None)
        b.api_key = None
        with self._mock_post(_GOOD_RAW), pytest.raises(ValueError, match="GEMINI_API_KEY"):
            b.predict(SAMPLE_NORMAL, PROMPT)


# ---------------------------------------------------------------------------
# Anthropic backend contract (mocked HTTP)
# ---------------------------------------------------------------------------


class TestAnthropicBackendContract:
    def _backend(self) -> AnthropicBackend:
        return AnthropicBackend(api_key="sk-ant-fake")

    def _mock_post(self, raw: str):
        resp = _fake_anthropic_response(raw)
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=False)
        cm.post = AsyncMock(return_value=resp)
        return patch(
            "vlm_anomaly.backends.anthropic_backend.AnthropicBackend._make_client", return_value=cm
        )

    def test_returns_anomaly_prediction(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert isinstance(p, AnomalyPrediction)

    def test_confidence_set(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert p.confidence == pytest.approx(0.1)

    def test_cost_computed(self) -> None:
        with self._mock_post(_BAD_RAW):
            p = self._backend().predict(SAMPLE_ANOMALY, PROMPT)
        assert p.cost_usd > 0.0

    def test_no_api_key_raises(self) -> None:
        b = AnthropicBackend(api_key=None)
        b.api_key = None
        with self._mock_post(_GOOD_RAW):
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                b.predict(SAMPLE_NORMAL, PROMPT)


# ---------------------------------------------------------------------------
# Groq backend contract (mocked HTTP)
# ---------------------------------------------------------------------------


class TestGroqBackendContract:
    def _backend(self) -> GroqBackend:
        return GroqBackend(api_key="gsk_fake")

    def _mock_post(self, raw: str):
        resp = _fake_response(raw)
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=cm)
        cm.__aexit__ = AsyncMock(return_value=False)
        cm.post = AsyncMock(return_value=resp)
        return patch("vlm_anomaly.backends.groq.GroqBackend._make_client", return_value=cm)

    def test_returns_anomaly_prediction(self) -> None:
        with self._mock_post(_GOOD_RAW):
            p = self._backend().predict(SAMPLE_NORMAL, PROMPT)
        assert isinstance(p, AnomalyPrediction)

    def test_cost_is_zero(self) -> None:
        with self._mock_post(_BAD_RAW):
            p = self._backend().predict(SAMPLE_ANOMALY, PROMPT)
        assert p.cost_usd == 0.0  # free tier

    def test_no_api_key_raises(self) -> None:
        b = GroqBackend(api_key=None)
        b.api_key = None
        with self._mock_post(_GOOD_RAW), pytest.raises(ValueError, match="GROQ_API_KEY"):
            b.predict(SAMPLE_NORMAL, PROMPT)


# ---------------------------------------------------------------------------
# Integration tests (require real API keys — skipped in CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_together_live(sample_normal_image: Path) -> None:
    from vlm_anomaly.config import get_settings

    key = get_settings().together_api_key
    if not key:
        pytest.skip("TOGETHER_API_KEY not set")
    b = TogetherBackend(api_key=key)
    p = b.predict(sample_normal_image, "Is there a defect? Reply with JSON.")
    assert isinstance(p, AnomalyPrediction)
    assert p.latency_ms > 0


@pytest.mark.integration
def test_gemini_live(sample_normal_image: Path) -> None:
    from vlm_anomaly.config import get_settings

    key = get_settings().gemini_api_key
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    b = GeminiBackend(api_key=key)
    p = b.predict(sample_normal_image, "Is there a defect? Reply with JSON.")
    assert isinstance(p, AnomalyPrediction)


@pytest.mark.integration
def test_anthropic_live(sample_normal_image: Path) -> None:
    from vlm_anomaly.config import get_settings

    key = get_settings().anthropic_api_key
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    b = AnthropicBackend(api_key=key)
    p = b.predict(sample_normal_image, "Is there a defect? Reply with JSON.")
    assert isinstance(p, AnomalyPrediction)


@pytest.mark.integration
def test_groq_live(sample_normal_image: Path) -> None:
    from vlm_anomaly.config import get_settings

    key = get_settings().groq_api_key
    if not key:
        pytest.skip("GROQ_API_KEY not set")
    b = GroqBackend(api_key=key)
    p = b.predict(sample_normal_image, "Is there a defect? Reply with JSON.")
    assert isinstance(p, AnomalyPrediction)
