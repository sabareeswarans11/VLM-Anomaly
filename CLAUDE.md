# CLAUDE.md — VLM-Anomaly

> Operating manual for Claude Code CLI working in this repository.
> Read this file in full before making changes. Do not invent files, libraries, or APIs that are not declared here.

---

## 0. Project Identity

- **Name:** VLM-Anomaly
- **Local path:** `/Users/sabareeswarans/Projects_26/VLM-Anomaly`
- **Remote:** https://github.com/sabareeswarans11/VLM-Anomaly
- **Owner:** Sab (sabareeswarans11@gmail.com)
- **License:** MIT
- **Status:** Active development — benchmark toolkit + on-device iPhone edge demo. **Repository is public (MIT).**
- **Primary IDE:** PyCharm (Claude Code CLI invoked from the integrated terminal).

---

## 1. Project Overview

VLM-Anomaly is an open-source benchmark toolkit that answers two questions:

1. **Cloud VLMs vs. classical CV:** Can modern Vision-Language Models (Qwen3-VL, Gemini 3 Pro, Claude Opus 4.6, GPT-5.4) perform zero-shot industrial anomaly detection — with NO training data, NO normal-image reference set, just a text prompt — competitively with classical methods (PaDiM, PatchCore, EfficientAD) on MVTec AD and VisA?
2. **Edge VLMs on-device:** Can a quantized small VLM (**MiniCPM-V 2.6 / 4.5**) deliver the same zero-shot capability **fully on-device on an iPhone**, with no network, no cloud cost, and acceptable latency for field inspection?

**The gap this fills.** Industrial anomaly detection today uses small CNN methods (PaDiM, PatchCore, EfficientAD) that require a per-category "normal" training set. Cloud VLMs in 2026 can skip training entirely. But for regulated, privacy-sensitive, or offline field workflows (telecom tower inspection, oil & gas, defense, medical), a cloud API is a non-starter — so the project also validates that a **MiniCPM-V model running on-device on an iPhone 16 Pro Max** can do the same job.

**The narrative (for the README and white paper):** *"Did VLMs kill anomaly detection — and can they run in your pocket?"*

**Validated edge result (already measured, must be reproduced by the toolkit):**

| Device                | Model                  | First-token latency | Throughput     | Network |
|-----------------------|------------------------|---------------------|----------------|---------|
| iPhone 16 Pro Max     | MiniCPM-V (4-bit)      | ~2.0 s              | 17.9 tok/s     | Offline |

This number is a **first-class artifact** of the project and must be re-measurable by anyone who clones the repo and runs the iOS app.

---

## 2. Target Users

- ML / CV engineers evaluating anomaly detection approaches for industrial or manufacturing computer vision.
- Researchers benchmarking VLM capabilities on grounded vision tasks.
- Telecom / AT&T / utility engineers exploring equipment inspection without labeled training data.
- iOS / edge-AI engineers evaluating MiniCPM-V class models for on-device inference.

---

## 3. Tech Stack (authoritative — do not substitute)

### 3.1 Python (benchmark toolkit)

- **Language:** Python 3.11
- **Cloud VLM inference:** Together.ai, Google Gemini, Anthropic Claude, Groq (free tier)
- **Classical baselines:** Anomalib (PaDiM, PatchCore, EfficientAD, STFPM) on CPU
- **Datasets:** MVTec AD, VisA, optional custom `InfraAD` subset
- **Metrics:** scikit-learn (AUROC, F1, precision, recall, PRO score)
- **Results store:** DuckDB over JSON results files
- **Visualization:** Matplotlib + Seaborn (paper-quality) and a small React + Recharts explorer (optional)
- **Experiment tracking:** MLflow (local `./mlruns` by default)
- **Compute:** Local Mac (Intel, no GPU) for everything; Kaggle Notebooks (free P100) for full batch eval
- **HTTP:** `httpx` (async)
- **Config / models:** `pydantic` v2 + `pydantic-settings`
- **Logging:** `structlog`
- **Testing:** `pytest` with `pytest-cov`
- **Linting / formatting:** `ruff`
- **Packaging:** `pyproject.toml` managed with `uv`

### 3.2 On-device edge (iOS)

- **Edge VLM:** MiniCPM-V (2.6 or 4.5), 4-bit quantized
- **Runtime:** `llama.cpp` iOS build (GGUF) **or** `mlx-vlm` (Apple MLX) — both supported, pick per-device benchmark
- **iOS app:** SwiftUI, Xcode 16+, iOS 17+ target
- **Model packaging:** GGUF for `llama.cpp` path; MLX `.safetensors` for MLX path; bundled or downloaded on first launch
- **Image pipeline:** Capture / pick photo → resize to model input → tokenize → generate JSON anomaly report
- **No network calls** in edge mode. All inference is local. Air-plane mode tests are part of the validation suite.

### 3.3 What we deliberately do NOT use

- No CoreML conversion in v1 (MiniCPM-V conversion to CoreML is fragile in 2026). MLX or llama.cpp only.
- No PyTorch in the base Python dependencies — only behind the `[classical]` extra.
- No paid GPU. No SageMaker / Vertex. Everything is laptop + free Kaggle + iPhone.
- No telemetry, no analytics SDKs in the iOS app.

---

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         VLM-Anomaly Toolkit                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Dataset Loader                                                   │
│     MVTec AD / VisA / InfraAD → standardized AnomalyDataset          │
│                                                                      │
│  2. VLM Evaluator (zero-shot, cloud)                                 │
│     For each test image:                                             │
│       → Send to VLM backend with prompt template                     │
│       → Parse JSON: {is_anomalous, confidence, description,          │
│         defect_type, regions[]}                                      │
│       → Compare against ground-truth mask / label                    │
│                                                                      │
│  3. Edge VLM Evaluator (zero-shot, on-device)                        │
│     Same interface as #2, backend = MiniCPM-V via llama.cpp / MLX    │
│     Runs on macOS for dev; iOS app reuses the same prompts + parser  │
│                                                                      │
│  4. Classical Evaluator (trained baselines)                          │
│     Anomalib PaDiM / PatchCore / EfficientAD on CPU                  │
│     Train on normal split, evaluate on test split                    │
│     Record AUROC, F1, latency, cost                                  │
│                                                                      │
│  5. Results Aggregator                                               │
│     DuckDB over results/*.json:                                      │
│       model × dataset × category → AUROC, F1, $/img, ms/img          │
│     Leaderboard, statistical tests (McNemar, bootstrap CIs)          │
│                                                                      │
│  6. Report Generator                                                 │
│     Auto-generated markdown report + PNG/SVG plots                   │
│     Suitable for blog post or arXiv preprint                         │
│                                                                      │
│  7. iOS Demo App (VLMAnomalyEdge)                                    │
│     SwiftUI app → camera/photo → MiniCPM-V local → JSON report       │
│     Reports first-token latency + tok/s on-screen                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. Directory Structure (authoritative)

```
VLM-Anomaly/
├── CLAUDE.md                                # This file
├── README.md
├── LICENSE                                  # MIT
├── pyproject.toml
├── uv.lock
├── .env.template
├── .gitignore
├── .python-version                          # 3.11
├── src/vlm_anomaly/
│   ├── __init__.py
│   ├── config.py                            # Pydantic settings
│   ├── schemas.py                           # AnomalyPrediction, EvalResult, ExperimentConfig
│   ├── datasets/
│   │   ├── __init__.py
│   │   ├── base.py                          # AnomalyDataset interface
│   │   ├── mvtec.py                         # MVTec AD loader (auto-download + checksum)
│   │   ├── visa.py                          # VisA loader
│   │   └── infra.py                         # Custom infrastructure subset
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                          # VLMBackend abstract base
│   │   ├── together.py                      # Together.ai (Qwen3-VL, DeepSeek-VL2)
│   │   ├── gemini.py                        # Google Gemini
│   │   ├── anthropic_backend.py             # Claude
│   │   ├── groq.py                          # Groq free tier (Llama-4-Scout)
│   │   ├── mock.py                          # Deterministic mock for tests
│   │   └── edge/
│   │       ├── __init__.py
│   │       ├── minicpm_llamacpp.py          # MiniCPM-V via llama.cpp (GGUF, CPU/Metal)
│   │       └── minicpm_mlx.py               # MiniCPM-V via mlx-vlm (Apple Silicon)
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── vlm_evaluator.py                 # Cloud + edge VLMs, single code path
│   │   ├── classical_evaluator.py           # Anomalib wrapper
│   │   └── prompt_library.py                # Loads YAML prompts, renders templates
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── aggregator.py                    # DuckDB queries over results/
│   │   ├── statistical_tests.py             # McNemar, bootstrap CIs
│   │   └── report_generator.py              # Markdown + plots
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py                         # Matplotlib / Seaborn
│   │   └── interactive.py                   # JSON payloads for React explorer
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py                   # Resize, crop, base64-encode
│       ├── json_parsing.py                  # Robust JSON-from-LLM extraction
│       └── cost_tracker.py                  # Per-call cost + budget enforcement
├── prompts/
│   ├── generic.yaml
│   ├── manufacturing.yaml
│   ├── infrastructure.yaml
│   └── detailed_cot.yaml
├── tests/
│   ├── conftest.py
│   ├── test_datasets.py
│   ├── test_backends_cloud.py
│   ├── test_backends_edge.py                # MiniCPM-V via tiny mock GGUF
│   ├── test_vlm_evaluator.py
│   ├── test_classical_evaluator.py          # Marked @slow
│   ├── test_aggregator.py
│   ├── test_json_parsing.py
│   └── fixtures/
│       ├── sample_normal.png
│       ├── sample_anomaly.png
│       └── sample_results.json
├── notebooks/
│   ├── 01_quickstart.ipynb                  # 10 images, mock + Groq free tier
│   ├── 02_mvtec_full_eval.ipynb             # Full MVTec eval, designed for Kaggle
│   ├── 03_edge_minicpm_macos.ipynb          # Run MiniCPM-V locally on Mac
│   ├── 04_analysis.ipynb                    # Aggregation + plots
│   └── 05_report.ipynb                      # Generates the paper-style report
├── results/                                 # Git-tracked JSON results (small)
│   └── .gitkeep
├── scripts/
│   ├── download_mvtec.sh
│   ├── download_visa.sh
│   ├── download_minicpm_gguf.sh             # Pulls quantized MiniCPM-V GGUF
│   ├── run_vlm_eval.py                      # CLI: cloud or edge VLM eval
│   ├── run_classical_eval.py                # CLI: anomalib eval
│   ├── run_edge_benchmark.py                # CLI: latency + tok/s on macOS
│   └── export_results_for_app.py            # Pre-baked results bundled into iOS app
├── ios/
│   └── VLMAnomalyEdge/                      # SwiftUI app (Xcode project)
│       ├── VLMAnomalyEdge.xcodeproj
│       ├── README.md                        # Build / install instructions
│       ├── Sources/
│       │   ├── VLMAnomalyEdgeApp.swift
│       │   ├── Views/
│       │   │   ├── CaptureView.swift
│       │   │   ├── ResultView.swift
│       │   │   └── BenchmarkView.swift      # Reports first-token + tok/s
│       │   ├── Inference/
│       │   │   ├── MiniCPMRunner.swift      # Wraps llama.cpp or MLX runtime
│       │   │   └── PromptBuilder.swift      # Mirrors prompts/*.yaml
│       │   └── Parsing/
│       │       └── AnomalyResponseParser.swift
│       ├── Resources/
│       │   └── prompts/                     # Copy of prompts/*.yaml (read-only)
│       └── Models/                          # GGUF downloaded on first launch
└── tasks/
    ├── 01-scaffold.md
    ├── 02-schemas-and-config.md
    ├── 03-datasets.md
    ├── 04-cloud-backends.md
    ├── 05-edge-backend-minicpm.md
    ├── 06-vlm-evaluator.md
    ├── 07-classical-baselines.md
    ├── 08-aggregator-analysis.md
    ├── 09-visualization-and-report.md
    ├── 10-tests.md
    ├── 11-kaggle-notebook.md
    ├── 12-ios-edge-app.md
    └── 13-readme-and-paper.md
```

Do not create files outside this tree without updating CLAUDE.md in the same commit.

---

## 6. Conventions

- All imports are absolute: `from vlm_anomaly.evaluators.vlm_evaluator import VLMEvaluator`.
- Type hints on **every** function signature; `from __future__ import annotations` at the top of each module.
- Docstrings on all public methods, Google style.
- HTTP via `httpx` async clients; never `requests`.
- Config and data models via `pydantic` v2 (`BaseModel`, `BaseSettings`).
- Logging via `structlog` (JSON renderer in prod, console renderer in dev). Never `print` in library code.
- All secrets via environment variables; never hardcoded. `.env` is git-ignored, `.env.template` is committed.
- Every cloud API call goes through `utils/cost_tracker.py` and respects a per-experiment `--budget` flag.
- Results are JSON files committed under `results/` so they are reproducible from the repo alone.
- Plots are saved as both PNG (for README) and SVG (for paper).
- Swift code follows Apple's Swift API Design Guidelines; no third-party deps beyond the chosen runtime (`llama.cpp` or `mlx-swift`).
- Commits follow Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, `perf:`.
- Branches: `main` is always green. Feature work on `feat/<short-name>`; PRs squash-merge.

---

## 7. Key Design Decisions

### 7.1 Unified VLM contract (cloud + edge)

Every backend — cloud or on-device — implements the same `VLMBackend` interface and returns the same `AnomalyPrediction` schema:

```json
{
  "is_anomalous": true,
  "confidence": 0.87,
  "description": "Rust corrosion visible on the upper-left bolt joint",
  "defect_type": "corrosion",
  "regions": [{"bbox": [120, 45, 280, 190], "label": "corrosion"}],
  "raw_response": "...",
  "latency_ms": 1820,
  "cost_usd": 0.0042,
  "tokens_in": 612,
  "tokens_out": 84
}
```

This means the **same evaluator, the same prompts, the same parser, and the same metrics** are used whether the inference is Claude in the cloud or MiniCPM-V on an iPhone. The only thing that changes is the backend.

### 7.2 Prompt Library

Prompts live in `prompts/*.yaml`. Each file declares variants (`simple`, `detailed`, `cot`). The iOS app loads a **read-only copy** of the same YAML files from its bundle, so cloud and edge runs use byte-identical prompts.

```yaml
# prompts/generic.yaml
name: generic
variants:
  simple: "Is there any defect or anomaly in this image? Reply with JSON."
  detailed: |
    You are an industrial quality inspector. Examine this image carefully.
    Determine if there are any defects, damage, or anomalies.
    Reply with a JSON object:
    {"is_anomalous": bool, "confidence": float, "description": str, "defect_type": str}
  cot: |
    Step 1: Describe what you see in the image.
    Step 2: Identify any unusual patterns, discoloration, cracks, or damage.
    Step 3: Determine if this is a normal or defective sample.
    Reply with JSON: {"is_anomalous": bool, ...}
```

### 7.3 Robust JSON extraction

VLMs return slightly malformed JSON often enough that `json.loads` alone is unsafe. `utils/json_parsing.py` implements:

1. Try `json.loads` on the full response.
2. Try to extract the first balanced `{...}` block.
3. Strip code fences (` ```json ... ``` `).
4. Fall back to regex extraction of known keys.
5. If all fail, mark the prediction as `parse_error=True` and record the raw response.

### 7.4 Classical baselines via Anomalib

Anomalib (Intel) provides PaDiM, PatchCore, EfficientAD, STFPM. All run on CPU on MVTec-sized data (≈200 train + ≈100 test per category, 2–10 min/category).

```python
from anomalib.models import Padim
from anomalib.data import MVTec
from anomalib.engine import Engine

model = Padim()
datamodule = MVTec(category="bottle", image_size=256)
engine = Engine(accelerator="cpu")
engine.fit(model, datamodule=datamodule)
results = engine.test(model, datamodule=datamodule)
```

Anomalib needs PyTorch, so it sits behind the `[classical]` extra: `uv pip install -e ".[classical]"`.

### 7.5 Edge MiniCPM-V backend

Two interchangeable runtimes, selected by config:

- `backend=edge_minicpm_llamacpp` — uses `llama-cpp-python` with a Q4_K_M GGUF of MiniCPM-V. Works on Intel Mac (CPU) and Apple Silicon (Metal). This is the path that ports cleanly to the iOS app via `llama.cpp`'s iOS build.
- `backend=edge_minicpm_mlx` — uses `mlx-vlm` on Apple Silicon. Faster on M-series; not available on iOS the same way, so used for macOS dev only.

Both expose the same `VLMBackend.predict(image, prompt) -> AnomalyPrediction` method.

### 7.6 Results aggregation with DuckDB

```python
import duckdb

db = duckdb.connect()
db.execute("CREATE TABLE results AS SELECT * FROM read_json_auto('results/*.json')")
db.execute("""
    SELECT model, dataset, category,
           AVG(auroc)        AS avg_auroc,
           AVG(f1)           AS avg_f1,
           AVG(cost_usd)     AS avg_cost,
           AVG(latency_ms)   AS avg_latency
    FROM results
    GROUP BY model, dataset, category
    ORDER BY avg_auroc DESC
""").df()
```

### 7.7 Cost management

- Default budget cap per experiment: `$5`. Configurable via `--budget`.
- Every API call logs cost via `cost_tracker`.
- The runner aborts cleanly if the budget would be exceeded by the next call.
- Partial results are flushed to disk after every image, so an aborted run is never wasted.
- Free / on-device backends (Groq free tier, MiniCPM-V) are exercised in CI; paid backends are gated behind an env var.

### 7.8 Local development on Intel Mac (no GPU)

- MVTec AD: ~5 GB total, ~300 images per category. Fine on a laptop.
- Cloud VLM eval: all via API. No local GPU needed.
- Classical baselines: Anomalib CPU, 2–10 min per category.
- Edge MiniCPM-V on Intel Mac runs via `llama.cpp` CPU — slower than iPhone but functional for dev.
- Quick dev loop: `--limit 10` per category. Full eval runs on Kaggle.

### 7.9 Kaggle notebook strategy

For the full MVTec sweep (15 categories × N models):

1. Upload the eval script + prompts as a Kaggle dataset.
2. Notebook loads MVTec from Kaggle datasets.
3. VLM eval uses hosted APIs (keys via Kaggle Secrets).
4. Results JSON saved as notebook output.
5. Download locally; commit small JSONs to `results/`.

### 7.10 iOS edge app design

- SwiftUI, single screen flow: **Capture → Inference → Result + Benchmark**.
- Model downloaded on first launch (~2 GB GGUF) into the app's `Documents/Models/`.
- On-screen benchmark surface shows: model name, device, first-token latency (ms), throughput (tok/s), peak memory (MB).
- An "Airplane mode test" toggle disables all network checks; the app is required to function with the radio off.
- Reference target validated on iPhone 16 Pro Max: **~2.0 s first-token latency, 17.9 tok/s**. The benchmark view stores measurements to a local SQLite file so users can reproduce and compare.

---

## 8. Testing

- `pytest tests/` — full unit suite, must pass with **no API keys and no datasets**.
- `pytest -m "not integration"` — unit only.
- `pytest -m "not slow"` — skip Anomalib training tests.
- `pytest -m edge` — runs the MiniCPM-V backend against a tiny mock GGUF shipped in `tests/fixtures/`.
- `MockVLMBackend` returns canned `AnomalyPrediction`s with deterministic seeds.
- iOS: XCTest target under `ios/VLMAnomalyEdge/Tests/` covers prompt building and JSON parsing parity with the Python parser (golden files).
- CI runs unit + non-slow tests on push; the full sweep is manual.

---

## 9. Dependencies (`pyproject.toml`)

```toml
[project]
name = "vlm-anomaly"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "httpx>=0.28.0",
    "pydantic>=2.10.0",
    "pydantic-settings>=2.6.0",
    "structlog>=24.4.0",
    "Pillow>=11.0.0",
    "duckdb>=1.1.0",
    "scikit-learn>=1.6.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "mlflow>=2.18.0",
    "pyyaml>=6.0.0",
    "python-dotenv>=1.0.0",
    "tenacity>=9.0.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",
    "tqdm>=4.67.0",
]

[project.optional-dependencies]
classical = [
    "anomalib>=1.2.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "timm>=1.0.0",
]
edge = [
    "llama-cpp-python>=0.3.2",
]
edge-mlx = [
    "mlx-vlm>=0.1.10; platform_system=='Darwin' and platform_machine=='arm64'",
]
dev = [
    "pytest>=8.3.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.8.0",
]

[project.scripts]
vlm-anomaly = "vlm_anomaly.cli:main"
```

Install patterns:

- Base toolkit: `uv pip install -e .`
- + classical baselines: `uv pip install -e ".[classical]"`
- + edge MiniCPM via llama.cpp: `uv pip install -e ".[edge]"`
- + edge MiniCPM via MLX (Apple Silicon only): `uv pip install -e ".[edge-mlx]"`
- Dev: `uv pip install -e ".[dev,classical,edge]"`

---

## 10. Do NOT

- Do not add `torch` to base dependencies — `[classical]` only.
- Do not add `coremltools` — CoreML conversion is explicitly out of scope for v1.
- Do not run full MVTec eval in unit tests — use `--limit 2` or `MockVLMBackend`.
- Do not hardcode API keys. Ever.
- Do not call paid APIs in unit tests — always mock.
- Do not generate plots inside tests — assert on data, not pixels.
- Do not over-spend — every experiment script must accept `--budget` and respect it.
- Do not skip cost tracking — it is a core feature, not a nice-to-have.
- Do not network from the iOS app in inference mode — the on-device promise must hold under Airplane mode.
- Do not commit datasets, GGUF model files, or `.env` — they are ignored by `.gitignore`.

---

## 11. Build Order (suggested task sequence)

Each step corresponds to a file under `tasks/`. Claude Code CLI should pick up one task at a time and not move on until tests pass.

1. **Scaffold:** `pyproject.toml`, package layout, `ruff` + `pytest` config, pre-commit, `.env.template`, MIT `LICENSE`, README stub.
2. **Schemas + Config:** `schemas.py` (`AnomalyPrediction`, `EvalResult`, `ExperimentConfig`) and `config.py` (Pydantic settings reading from env).
3. **Dataset loaders:** `AnomalyDataset` ABC + `MVTec` (auto-download + checksum) + `VisA`. Round-trip tests with fixtures.
4. **Cloud backends:** `VLMBackend` ABC + Together, Gemini, Anthropic, Groq, `MockVLMBackend`. Contract tests against the mock.
5. **Edge MiniCPM-V backend:** `backends/edge/minicpm_llamacpp.py` + `backends/edge/minicpm_mlx.py`, sharing one prompt path. Smoke test on a tiny GGUF.
6. **VLM Evaluator:** `vlm_evaluator.py` — orchestrates backend + prompt + parser + metrics. Budget enforcement.
7. **Classical baselines:** `classical_evaluator.py` — thin Anomalib wrapper, same `EvalResult` schema.
8. **Aggregator + analysis:** DuckDB rollups, McNemar, bootstrap CIs.
9. **Visualization + report generator:** AUROC bars, cost-vs-accuracy scatter, per-category heatmap, auto-generated markdown report.
10. **Tests:** full unit suite green without API keys.
11. **Kaggle notebook:** `02_mvtec_full_eval.ipynb` — runs full sweep with secrets.
12. **iOS edge app:** Xcode project under `ios/VLMAnomalyEdge/`, on-device MiniCPM-V, benchmark view that reproduces the iPhone 16 Pro Max numbers.
13. **README + paper draft:** "Did VLMs kill anomaly detection — and can they run in your pocket?" Includes the results table from §13 and the iPhone benchmark plot.

---

## 12. Environment Variables

```
# Cloud VLM keys (only required for the backends you actually use)
TOGETHER_API_KEY=...
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...

# MLflow (optional; defaults to local ./mlruns)
MLFLOW_TRACKING_URI=

# Paths
VLM_ANOMALY_DATA_DIR=./data
VLM_ANOMALY_RESULTS_DIR=./results
VLM_ANOMALY_MODELS_DIR=./models

# Budgets
VLM_ANOMALY_DEFAULT_BUDGET_USD=5
```

`.env.template` mirrors this list with empty values. `.env` is git-ignored.

---

## 13. Expected Results (what the README must show)

### 13.1 Accuracy / cost / latency

| Model                       | Type            | MVTec AUROC | VisA AUROC | Cost/Image | Latency   |
|-----------------------------|-----------------|-------------|------------|------------|-----------|
| PaDiM                       | Classical (CPU) | 0.92        | 0.88       | $0         | 15 ms     |
| PatchCore                   | Classical (CPU) | 0.95        | 0.91       | $0         | 25 ms     |
| EfficientAD                 | Classical (CPU) | 0.96        | 0.93       | $0         | 8 ms      |
| MiniCPM-V (iPhone 16 Pro Max, on-device) | Edge VLM | TBD       | TBD        | $0 (offline) | ~2.0 s first token, 17.9 tok/s |
| Qwen3-VL-4B (zero-shot)     | Cloud VLM       | 0.78        | 0.72       | $0.003     | 1.2 s     |
| Gemini 3 Flash (zero-shot)  | Cloud VLM       | 0.85        | 0.80       | $0.005     | 0.8 s     |
| Gemini 3 Pro (zero-shot)    | Cloud VLM       | 0.89        | 0.86       | $0.015     | 1.5 s     |
| Claude Opus 4.6 (zero-shot) | Cloud VLM       | 0.91        | 0.88       | $0.025     | 2.0 s     |

`TBD` values are placeholders that the benchmark must fill in — they are not to be invented.

### 13.2 The story

The punchline of the README:

> **Frontier cloud VLMs reach ~95% of classical accuracy with zero training data, at ~100× the cost and ~100× the latency. A 4-bit MiniCPM-V running fully on-device on an iPhone 16 Pro Max gives you most of that zero-shot capability with zero dollars and zero network — at ~2 s first-token latency and 17.9 tok/s. The sweet spot for field inspection is hybrid: edge VLM for triage and privacy, classical models for production scoring, cloud VLM only for the long tail.**

For telecom tower inspection (AT&T context), the edge VLM story matters most: technicians on a tower with no signal can still get a zero-shot defect read, with nothing leaving the device.

---

## 14. Working agreement for Claude Code CLI

When invoked in this repo:

1. **Read this CLAUDE.md fully** before the first edit in a session.
2. Pick the smallest task from `tasks/` that has all its prerequisites met. Open the task file, then propose the plan in chat before editing.
3. Make one logical change per commit. Use Conventional Commits.
4. Run `ruff check . && ruff format --check . && pytest -m "not slow and not integration"` before declaring a task done.
5. Update `tasks/*.md` to check off completed items in the same commit.
6. If a design decision in this file is wrong or missing, **update CLAUDE.md in the same PR** — don't drift silently.
7. Never push to `main` directly. Open a PR even for solo work, so the history stays reviewable.
8. For the iOS app, prefer minimal Swift, prefer SwiftUI over UIKit, and keep the Python parser and the Swift parser behaviour-equivalent (golden test files in both).

---

## 15. Quickstart (for a fresh clone)

```bash
# 1. Clone
git clone https://github.com/sabareeswarans11/VLM-Anomaly.git
cd VLM-Anomaly

# 2. Python env
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,classical,edge]"

# 3. Datasets
bash scripts/download_mvtec.sh
bash scripts/download_minicpm_gguf.sh

# 4. Configure
cp .env.template .env
# edit .env to add the API keys you actually have

# 5. Smoke test (no API keys needed — uses MockVLMBackend)
pytest -m "not slow and not integration"

# 6. Tiny cloud run
python scripts/run_vlm_eval.py --backend mock --dataset mvtec --category bottle --limit 5

# 7. Tiny edge run (macOS)
python scripts/run_vlm_eval.py --backend edge_minicpm_llamacpp --dataset mvtec --category bottle --limit 5

# 8. Edge benchmark (macOS proxy for iPhone)
python scripts/run_edge_benchmark.py --model minicpm-v --prompt prompts/generic.yaml
```

For the iPhone app, open `ios/VLMAnomalyEdge/VLMAnomalyEdge.xcodeproj` in Xcode 16+, select the iPhone 16 Pro Max scheme, and run.
