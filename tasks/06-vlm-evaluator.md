# Task 06 — VLM Evaluator

Single orchestrator that wires backend + prompt + parser + metrics for both
cloud and edge runs.

## Deliverables

- [x] `evaluators/prompt_library.py` — loads `prompts/*.yaml`, renders by
      `name.variant` key, validates names/variants, used by iOS app too.
- [x] `utils/json_parsing.py` — full 5-step fallback chain (done in task 04).
- [x] `utils/cost_tracker.py` — thread-safe CostTracker + BudgetExceeded (done in task 04).
- [x] `evaluators/base.py` — `compute_eval_result()` with AUROC, F1, precision, recall.
- [x] `evaluators/vlm_evaluator.py` — iterates samples, checks budget, calls backend,
      flushes `.jsonl` after every image, returns `EvalResult` per category.
- [x] `scripts/run_vlm_eval.py` — fully wired CLI with backend/dataset/prompt/limit/budget.
- [x] `tests/test_vlm_evaluator.py` — 21 tests (prompt library + evaluator + budget). Green.
- [x] `tests/test_json_parsing.py` — promoted to real tests (done in task 04).

## Done when

`python scripts/run_vlm_eval.py --backend mock --dataset mvtec --category
bottle --limit 5 --budget 0.01` writes a results JSON to `results/`.
