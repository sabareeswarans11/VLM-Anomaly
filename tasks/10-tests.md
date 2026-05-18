# Task 10 — Full Unit Suite Green Without API Keys

Bring every test module to a passing state. No real network. No real data.

## Deliverables

- [ ] `pytest -m "not slow and not integration"` is green on a fresh clone.
- [ ] Coverage ≥ 80% on `src/vlm_anomaly/` minus the `__init__.py` files
      and `cli.py`.
- [ ] CI workflow under `.github/workflows/ci.yml` runs ruff + pytest on
      push and PR (no key-protected jobs).

## Done when

A teammate can `git clone && uv pip install -e ".[dev]"` then `pytest -m
"not slow and not integration"` with no extra setup.
