# Task 03 — Dataset Loaders

Standardize MVTec AD and VisA behind a single `AnomalyDataset` interface.

## Deliverables

- [x] `datasets/base.py` — `AnomalyDataset` ABC + `AnomalySample` dataclass.
- [x] `datasets/mvtec.py` — auto-download with SHA-256 checksum, parse
      train/test splits and mask paths, expose all 15 categories.
- [x] `datasets/visa.py` — same interface, 12 categories; auto-detects
      MVTec-style and raw VisA layouts.
- [x] `datasets/infra.py` — custom infrastructure subset; discovers
      categories dynamically from root_dir.
- [x] `scripts/download_mvtec.sh` and `scripts/download_visa.sh` filled in.
- [x] `tests/test_datasets.py` — 27 tests using synthetic fixture trees,
      covering MVTec, VisA (both layouts), and InfraAD.

## Done when

`python -c "from vlm_anomaly.datasets.mvtec import MVTec; MVTec().categories()"`
returns the 15 MVTec categories without touching the network.
