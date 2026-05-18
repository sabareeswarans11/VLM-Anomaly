# Task 03 — Dataset Loaders

Standardize MVTec AD and VisA behind a single `AnomalyDataset` interface.

## Deliverables

- [ ] `datasets/base.py` — `AnomalyDataset` ABC + `AnomalySample` dataclass.
- [ ] `datasets/mvtec.py` — auto-download with SHA-256 checksum, parse
      train/test splits and mask paths, expose all 15 categories.
- [ ] `datasets/visa.py` — same interface, 12 categories.
- [ ] `datasets/infra.py` — placeholder for a custom subset; documents the
      target schema even if empty.
- [ ] `scripts/download_mvtec.sh` and `scripts/download_visa.sh` filled in.
- [ ] `tests/test_datasets.py` — round-trip tests using a tiny synthetic
      fixture tree (no real download in CI).

## Done when

`python -c "from vlm_anomaly.datasets.mvtec import MVTec; MVTec().categories()"`
returns the 15 MVTec categories without touching the network.
