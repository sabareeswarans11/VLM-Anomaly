# Task 01 — Project Scaffold

Set up the repository so subsequent tasks have a stable foundation.

## Deliverables

- [x] `pyproject.toml` with base + extras (`classical`, `edge`, `edge-mlx`, `dev`).
- [x] `.python-version` = 3.11.
- [x] `.gitignore` covering Python, venv, datasets, models, mlruns, iOS, IDE, AI-assistant artifacts.
- [x] MIT `LICENSE`.
- [x] `README.md` (story-first).
- [x] `.env.template` mirroring CLAUDE.md §12.
- [x] `ruff` + `pytest` config in `pyproject.toml`.
- [x] `.pre-commit-config.yaml` with ruff hooks.
- [x] `src/vlm_anomaly/` package layout with every subpackage's `__init__.py`.
- [x] `tests/` with `conftest.py`, fixtures, and placeholder test modules.
- [x] `prompts/*.yaml` initial library.
- [x] `scripts/` CLI stubs.
- [x] `results/.gitkeep`.

## Done when

`ruff check . && ruff format --check . && pytest -m "not slow and not integration"` is green
on a fresh clone with no API keys.
