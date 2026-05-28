# VLM-Anomaly

A benchmark toolkit evaluating **zero-shot Vision-Language Model anomaly detection** against classical computer vision baselines on MVTec AD, with a validated on-device edge deployment via MiniCPM-V on iPhone.

[![CI](https://github.com/sabareeswarans11/VLM-Anomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/sabareeswarans11/VLM-Anomaly/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Industrial anomaly detection has traditionally depended on classical CNN methods (PaDiM, PatchCore, EfficientAD) that require a per-category set of normal training images. This project benchmarks whether modern Vision-Language Models can replace that training requirement entirely — using only a text prompt at inference time — and whether the same capability can run **fully offline on a smartphone**.

Two questions drive the work:

1. **Cloud VLMs vs. classical CV** — Do frontier VLMs (Claude Opus 4.7, Gemini 2.5 Flash, Qwen3-VL-32B) match classical baselines on MVTec AD under zero-shot conditions (no training data, no reference images)?
2. **On-device edge inference** — Can a 4-bit quantized MiniCPM-V running locally on an iPhone 16 Pro Max deliver competitive zero-shot detection with no network dependency?

---

## Benchmark Results (MVTec AD — 15 categories, 1,725 images)

| Model | Type | Mean AUROC | Cost / Image | Notes |
|---|---|---|---|---|
| PatchCore | Classical | **0.9147** | $0 | Anomalib, Kaggle P100 |
| PaDiM | Classical | **0.9098** | $0 | Anomalib, CPU |
| Claude Opus 4.7 (few-shot ens.) | Cloud VLM | 0.7709 | ~$0.10 | 4-prompt ensemble, 2/15 cats |
| Claude Opus 4.7 (zero-shot) | Cloud VLM | 0.7519 | ~$0.008 | All 15 categories |
| MiniCPM-V 4.6 Q4_K_M (on-device) | Edge VLM | 0.6523 | $0 | iPhone / Intel CPU, offline |
| Gemini 2.5 Flash (zero-shot) | Cloud VLM | 0.6104 | ~$0.001 | All 15 categories |
| Qwen3-VL-32B (zero-shot) | Cloud VLM | 0.1782 | ~$0.0002 | Via OpenRouter |

### Actual provider spend (full 15-category sweep, 1,725 images)

| Model | Spend | Cost / Image |
|---|---|---|
| PaDiM / PatchCore | $0.00 | $0 |
| MiniCPM-V (on-device) | $0.00 | $0 |
| Qwen3-VL-32B | $0.28 | ~$0.0002 |
| Gemini 2.5 Flash | $2.37 | ~$0.0014 |
| Claude Opus 4.7 | $13.36 | ~$0.0077 |

---

## Edge Validation — iPhone 16 Pro Max

| Device | Model | First-token latency | Throughput | Network |
|---|---|---|---|---|
| iPhone 16 Pro Max | MiniCPM-V 4.6 (4-bit Q4_K_M) | ~2.0 s | 17.9 tok/s | Offline (airplane mode) |

The on-device result is reproduced by the iOS app's benchmark view and logged to local SQLite, so any user can verify the numbers on their own device.

For field inspection workflows (telecom tower inspection, oil & gas, defense, medical imaging) where cloud APIs are unavailable or prohibited, a technician with no signal can still obtain a zero-shot defect assessment — nothing leaves the device.

---

## Architecture

```
Dataset loader  (MVTec AD / VisA / InfraAD)
       │
       ▼
VLMEvaluator
       │
       ├──► Cloud backend  (Anthropic / Gemini / OpenRouter)
       │         │
       └──► Edge backend   (MiniCPM-V via llama.cpp or mlx-vlm)
                 │
                 ▼
         AnomalyPrediction  { is_anomalous, confidence, description,
                               defect_type, regions, latency_ms, cost_usd }
                 │
                 ▼
  results/{id}_{dataset}_{category}.jsonl   (one record per image, flushed live)
                 │
                 ▼
  aggregator.leaderboard("results/")        (DuckDB, AUROC / F1 recomputed)
                 │
                 ▼
  report_generator.generate(...)            (REPORT.md + SVG/PNG plots)
```

Every backend — cloud API or on-device — implements the same `VLMBackend` interface and returns the same schema. The same prompt library, JSON parser, and metrics pipeline apply to both, so cloud and edge runs are directly comparable.

---

## Repository Layout

```
src/vlm_anomaly/
├── backends/
│   ├── anthropic_backend.py      Claude (zero-shot + few-shot ensemble)
│   ├── gemini.py                 Gemini 2.5 Flash / Pro
│   ├── openrouter.py             Qwen3-VL-32B via OpenRouter
│   ├── mock.py                   Deterministic mock (CI / unit tests)
│   └── edge/
│       ├── minicpm_llamacpp.py   MiniCPM-V via llama.cpp (CPU / Metal)
│       └── minicpm_mlx.py        MiniCPM-V via mlx-vlm (Apple Silicon)
├── datasets/                     MVTec AD, VisA, InfraAD loaders
├── evaluators/
│   ├── vlm_evaluator.py          Cloud + edge orchestration, budget enforcement
│   ├── classical_evaluator.py    Anomalib wrapper (PaDiM, PatchCore, EfficientAD)
│   ├── few_shot_evaluator.py     Few-shot ensemble (ref images + 4-prompt voting)
│   └── prompt_library.py         YAML prompt loader
├── analysis/
│   ├── aggregator.py             DuckDB leaderboard, AUROC/F1 recomputation
│   ├── statistical_tests.py      McNemar test, bootstrap confidence intervals
│   └── report_generator.py       Auto-generates REPORT.md + plots
└── visualization/
    ├── plots.py                  AUROC bar, cost-vs-accuracy scatter, heatmap
    └── interactive.py            JSON payloads for React explorer

prompts/                          YAML prompt library (mirrored in iOS app bundle)
├── generic.yaml
├── manufacturing.yaml
├── infrastructure.yaml
├── detailed_cot.yaml
└── claude_opus_enhanced.yaml     Category-specific few-shot prompts

scripts/
├── run_vlm_eval.py               Cloud or edge VLM evaluation CLI
├── run_classical_eval.py         Anomalib classical baseline CLI
├── run_minicpm_local_sweep.py    MiniCPM-V full MVTec sweep (direct mtmd API)
└── run_edge_benchmark.py         Latency / throughput benchmark (macOS proxy)

notebooks/
├── 05_mvtec_patchcore_eval_kaggle.ipynb   PatchCore sweep on Kaggle P100
├── 10_mvtec_claude_opus_few_shot_ensemble.ipynb
└── ...                                    Quickstart, Gemini, Qwen, analysis

ios/VLMAnomalyEdge/               SwiftUI app — MiniCPM-V on-device
├── Sources/
│   ├── Inference/MiniCPMRunner.swift       llama.cpp wrapper
│   ├── Views/BenchmarkView.swift           Latency + tok/s display
│   └── Parsing/AnomalyResponseParser.swift JSON parser (mirrors Python)
└── Models/                                 GGUF downloaded on first launch

results/                          Committed JSON/JSONL results (reproducible)
```

---

## Quickstart

```bash
git clone https://github.com/sabareeswarans11/VLM-Anomaly.git
cd VLM-Anomaly

# Python 3.11 environment
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure API keys (only the ones you have)
cp .env.template .env

# Unit tests — no API keys, no dataset needed
pytest -m "not slow and not integration and not edge"

# Regenerate REPORT.md + plots from committed results
python -c "
from vlm_anomaly.analysis.report_generator import generate
generate('results/', 'REPORT.md')
"
```

**Cloud evaluation** (any single backend):
```bash
python scripts/run_vlm_eval.py \
  --backend anthropic --dataset mvtec --category bottle --limit 10
```

**Classical baselines** (requires MVTec data + PyTorch):
```bash
uv pip install -e ".[classical]"
bash scripts/download_mvtec.sh
python scripts/run_classical_eval.py --model padim --dataset mvtec --category bottle
```

**Edge sweep** (MiniCPM-V, CPU, no API key):
```bash
uv pip install -e ".[edge]"
bash scripts/download_minicpm_gguf.sh
python scripts/run_minicpm_local_sweep.py --threads 8 --max-tokens 150
```

**Full Kaggle sweep** (free P100 GPU):  
Fork [`notebooks/05_mvtec_patchcore_eval_kaggle.ipynb`](notebooks/05_mvtec_patchcore_eval_kaggle.ipynb), attach the MVTec dataset, add API keys as Kaggle Secrets, run all cells.

---

## iOS App

Open `ios/VLMAnomalyEdge/VLMAnomalyEdge.xcodeproj` in Xcode 16+. The app downloads the MiniCPM-V 4.6 Q4_K_M GGUF (~2 GB) on first launch into the app's `Documents/Models/` directory.

- **Capture tab** — camera or photo library → local inference → JSON anomaly report
- **Benchmark tab** — first-token latency, throughput (tok/s), peak memory, logged to local SQLite
- **Airplane mode** — all inference is local; the app works with the radio fully off

The Swift JSON parser (`AnomalyResponseParser.swift`) is behaviour-equivalent to the Python parser (`utils/json_parsing.py`), verified by shared golden test files.

---

## Install Options

| Command | Includes |
|---|---|
| `uv pip install -e ".[dev]"` | Base toolkit + pytest + ruff |
| `uv pip install -e ".[classical]"` | + Anomalib / PyTorch (PaDiM, PatchCore) |
| `uv pip install -e ".[edge]"` | + llama-cpp-python (MiniCPM-V CPU/Metal) |
| `uv pip install -e ".[edge-mlx]"` | + mlx-vlm (Apple Silicon only) |

---

## CI

- **Lint**: `ruff check` + `ruff format --check`
- **Tests**: `pytest -m "not slow and not integration and not edge"` — no API keys required
- **Coverage**: 89% (gate: 80%)

---

## License

MIT — see [`LICENSE`](LICENSE).
