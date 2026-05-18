# Task 06 — VLM Evaluator

Single orchestrator that wires backend + prompt + parser + metrics for both
cloud and edge runs.

## Deliverables

- [ ] `evaluators/prompt_library.py` — loads `prompts/*.yaml`, renders by
      `name.variant` key, exposes the byte-identical text used by the iOS app.
- [ ] `utils/json_parsing.py` — full fallback chain per CLAUDE.md §7.3.
- [ ] `utils/cost_tracker.py` — per-experiment running total + budget abort.
- [ ] `evaluators/vlm_evaluator.py` — iterates `AnomalyDataset` samples,
      calls backend, parses, scores, flushes JSON after every image.
- [ ] `tests/test_vlm_evaluator.py` + `tests/test_json_parsing.py` green.

## Done when

`python scripts/run_vlm_eval.py --backend mock --dataset mvtec --category
bottle --limit 5 --budget 0.01` writes a results JSON to `results/`.
