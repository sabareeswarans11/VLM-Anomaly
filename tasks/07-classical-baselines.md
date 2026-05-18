# Task 07 — Classical Baselines

Wrap Anomalib's PaDiM, PatchCore, EfficientAD, STFPM so they produce the
same `EvalResult` shape as the VLM evaluator.

## Deliverables

- [ ] `evaluators/classical_evaluator.py` — thin Anomalib wrapper, CPU-only
      by default, accepts model name + category, returns `EvalResult`.
- [ ] `scripts/run_classical_eval.py` filled in.
- [ ] `tests/test_classical_evaluator.py` — marked `@pytest.mark.slow`,
      runs PaDiM on a tiny fixture and asserts AUROC > 0.5.

## Done when

`python scripts/run_classical_eval.py --model padim --dataset mvtec
--category bottle` writes a results JSON in 2–10 min on CPU.
