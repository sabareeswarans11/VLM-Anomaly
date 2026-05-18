# Task 04 — Cloud VLM Backends

Implement the four hosted backends behind the `VLMBackend` contract.

## Deliverables

- [ ] `backends/base.py` — `VLMBackend` ABC finalized (already stubbed).
- [ ] `backends/together.py` — Qwen3-VL / DeepSeek-VL2 via Together.ai.
- [ ] `backends/gemini.py` — Gemini 3 Flash / Pro.
- [ ] `backends/anthropic_backend.py` — Claude Opus 4.6+.
- [ ] `backends/groq.py` — Llama-4-Scout free tier.
- [ ] `utils/image_utils.py` — resize / crop / base64 helpers.
- [ ] All backends use `httpx.AsyncClient` with retries via `tenacity`.
- [ ] Per-call cost tracked via `utils/cost_tracker.py`.
- [ ] `tests/test_backends_cloud.py` — contract tests against the mock; live
      calls gated behind `@pytest.mark.integration`.

## Done when

Each backend returns a valid `AnomalyPrediction` for the sample fixture image
when its API key is set, and the contract tests pass with no keys.
