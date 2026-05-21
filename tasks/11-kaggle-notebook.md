# Task 11 — Kaggle Notebook for the Full MVTec Sweep

Run the expensive 15-category × N-models sweep on a free Kaggle P100.

## Deliverables

- [x] `notebooks/02_mvtec_full_eval.ipynb` — installs `vlm-anomaly` from
      a Kaggle dataset, loads MVTec from a Kaggle dataset, reads API keys
      from Kaggle Secrets, runs the sweep, writes JSON to notebook output.
- [x] Notebook is idempotent — re-running picks up where it stopped.
- [x] Download instructions for pulling JSON back into `results/` locally.

## Done when

A reviewer can fork the notebook on Kaggle, hit "Run all", and a few hours
later have the full leaderboard data committed back to the repo.
