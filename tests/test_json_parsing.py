"""JSON parsing tests — basic extract_json and batch response parsing."""

from __future__ import annotations

import json

from vlm_anomaly.utils.json_parsing import extract_json, parse_batch_response


def test_extracts_balanced_json_from_messy_response() -> None:
    text = 'Here is my answer: {"is_anomalous": true, "confidence": 0.9} — done.'
    data, err = extract_json(text)
    assert err is False
    assert data["is_anomalous"] is True


def test_marks_parse_error_when_all_fallbacks_fail() -> None:
    _, err = extract_json("The image looks fine to me.")
    assert err is True


# ---------------------------------------------------------------------------
# parse_batch_response
# ---------------------------------------------------------------------------

_ITEM_NORMAL = {"is_anomalous": False, "confidence": 0.05, "description": "ok"}
_ITEM_DEFECT = {"is_anomalous": True, "confidence": 0.92, "defect_type": "crack"}


class TestParseBatchResponse:
    def _make_array(self, items: list[dict]) -> str:
        return json.dumps(items)

    def test_parses_clean_array(self) -> None:
        text = self._make_array([_ITEM_NORMAL, _ITEM_DEFECT])
        results = parse_batch_response(text, n=2)
        assert len(results) == 2
        assert results[0][0]["is_anomalous"] is False
        assert results[1][0]["is_anomalous"] is True

    def test_parses_array_in_code_fence(self) -> None:
        text = "```json\n" + self._make_array([_ITEM_NORMAL]) + "\n```"
        results = parse_batch_response(text, n=1)
        assert len(results) == 1
        assert results[0][1] is False  # no parse error

    def test_pads_when_fewer_items_than_n(self) -> None:
        text = self._make_array([_ITEM_NORMAL])
        results = parse_batch_response(text, n=3)
        assert len(results) == 3
        # Padded entries have parse_error=True
        assert results[1][1] is True
        assert results[2][1] is True

    def test_truncates_when_more_items_than_n(self) -> None:
        text = self._make_array([_ITEM_NORMAL, _ITEM_DEFECT, _ITEM_NORMAL])
        results = parse_batch_response(text, n=2)
        assert len(results) == 2

    def test_total_parse_failure_returns_n_errors(self) -> None:
        results = parse_batch_response("I cannot answer this.", n=4)
        assert len(results) == 4
        assert all(err for _, err in results)

    def test_non_dict_item_is_error(self) -> None:
        text = json.dumps(["not a dict", _ITEM_NORMAL])
        results = parse_batch_response(text, n=2)
        assert results[0][1] is True  # string item → error
        assert results[1][1] is False  # valid dict → no error

    def test_parses_array_embedded_in_prose(self) -> None:
        text = "Here are my results:\n" + self._make_array([_ITEM_DEFECT]) + "\nDone."
        results = parse_batch_response(text, n=1)
        assert results[0][0]["is_anomalous"] is True

    def test_confidence_clamped_to_range(self) -> None:
        item = {"is_anomalous": True, "confidence": 2.5}
        text = json.dumps([item])
        results = parse_batch_response(text, n=1)
        assert 0.0 <= results[0][0]["confidence"] <= 1.0
