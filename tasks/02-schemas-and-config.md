# Task 02 — Schemas + Config

Lock in the shapes the rest of the toolkit depends on.

## Deliverables

- [ ] `src/vlm_anomaly/schemas.py` finalized: `AnomalyPrediction`, `EvalResult`,
      `ExperimentConfig`, `Region`. JSON-serializable round-trip tested.
- [ ] `src/vlm_anomaly/config.py` finalized: `Settings` reading env + `.env`,
      `get_settings()` helper, path normalization to absolute.
- [ ] `structlog` configured (console renderer in dev, JSON in prod) and exposed
      via `vlm_anomaly.logging.get_logger`.
- [ ] Unit tests in `tests/test_schemas.py` and `tests/test_config.py`.

## Done when

`pytest tests/test_schemas.py tests/test_config.py -v` is green and `from
vlm_anomaly.config import get_settings; get_settings()` works without `.env`.
