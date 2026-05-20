# Task 04 — Cloud VLM Backends

Implement the four hosted backends behind the `VLMBackend` contract.

## Deliverables

- [x] `backends/base.py` — `VLMBackend` ABC finalized with `_run` + `_make_client` helpers.
- [x] `backends/together.py` — Qwen2-VL-72B via Together.ai (OpenAI-compat endpoint).
- [x] `backends/gemini.py` — Gemini 3 Flash / Pro via REST API.
- [x] `backends/anthropic_backend.py` — Claude Opus 4.7 via Messages API.
- [x] `backends/groq.py` — Llama-4-Scout free tier via Groq OpenAI-compat endpoint.
- [x] `utils/image_utils.py` — resize (LANCZOS, max 1024px), JPEG encode, base64 / data URL.
- [x] `utils/json_parsing.py` — 5-step fallback chain (direct → balanced braces →
      fence strip → regex → parse_error).
- [x] `utils/cost_tracker.py` — thread-safe CostTracker with BudgetExceeded.
- [x] All backends use `httpx.AsyncClient` with `tenacity` retries.
- [x] `tests/test_backends_cloud.py` — 47 unit tests (mocked HTTP) + 4 live
      integration stubs gated behind `@pytest.mark.integration`.

## Done when

Each backend returns a valid `AnomalyPrediction` for the sample fixture image
when its API key is set, and the contract tests pass with no keys.
