# Task 08 — Aggregator + Statistical Analysis

Roll up the per-image JSON results into a leaderboard, and add the stats
needed to make honest claims in the README.

## Deliverables

- [ ] `analysis/aggregator.py` — DuckDB over `results/*.json`, group-by
      `(model, dataset, category)`, sorted leaderboard view.
- [ ] `analysis/statistical_tests.py` — McNemar test for paired model
      comparisons, bootstrap CIs for AUROC and F1.
- [ ] `tests/test_aggregator.py` — uses `tests/fixtures/sample_results.json`.

## Done when

`python -c "from vlm_anomaly.analysis.aggregator import leaderboard;
print(leaderboard('results/'))"` prints a ranked DataFrame.
