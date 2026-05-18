# Task 09 — Visualization + Report Generator

Produce the plots and the auto-generated markdown report.

## Deliverables

- [ ] `visualization/plots.py` — AUROC bar chart, cost-vs-accuracy scatter,
      per-category heatmap. Save as both PNG and SVG.
- [ ] `visualization/interactive.py` — JSON payload for the optional React
      + Recharts explorer.
- [ ] `analysis/report_generator.py` — emits `REPORT.md` with the
      leaderboard table, all plots inlined, and the narrative blurb from
      CLAUDE.md §13.2.

## Done when

`python -c "from vlm_anomaly.analysis.report_generator import generate;
generate('results/', 'REPORT.md')"` produces a complete markdown report
and PNGs land in `results/plots/`.
