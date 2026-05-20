"""JSON parsing tests — covered fully in test_backends_cloud.py::TestJsonParsing.

These stubs exist so the test file is present; the real tests live alongside
the backend contract tests where they get richer fixture context.
"""

from __future__ import annotations

from vlm_anomaly.utils.json_parsing import extract_json, parse_anomaly_prediction_dict


def test_extracts_balanced_json_from_messy_response() -> None:
    text = 'Here is my answer: {"is_anomalous": true, "confidence": 0.9} — done.'
    data, err = extract_json(text)
    assert err is False
    assert data["is_anomalous"] is True


def test_marks_parse_error_when_all_fallbacks_fail() -> None:
    _, err = extract_json("The image looks fine to me.")
    assert err is True
