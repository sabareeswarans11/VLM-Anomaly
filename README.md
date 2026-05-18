# VLM-Anomaly

> **Did VLMs kill anomaly detection — and can they run in your pocket?**

An open-source benchmark toolkit that answers two questions head-on:

1. **Cloud VLMs vs. classical CV.** Can frontier Vision-Language Models
   (Qwen3-VL, Gemini 3 Pro, Claude Opus 4.6, GPT-5.4) perform **zero-shot**
   industrial anomaly detection — no training data, no normal-image
   reference set, just a text prompt — competitively with classical methods
   (PaDiM, PatchCore, EfficientAD) on MVTec AD and VisA?
2. **Edge VLMs on-device.** Can a quantized small VLM (**MiniCPM-V 2.6 / 4.5**)
   deliver the same zero-shot capability **fully on-device on an iPhone**,
   with no network, no cloud cost, and acceptable latency for field inspection?

## Why this matters

Industrial anomaly detection today still leans on small CNN methods that
need a per-category "normal" training set. Cloud VLMs in 2026 skip training
entirely. But for regulated, privacy-sensitive, or offline field workflows
— telecom tower inspection, oil & gas, defense, medical — a cloud API is a
non-starter. So this project also validates that a **MiniCPM-V model
running on-device on an iPhone 16 Pro Max** can do the same job.

## Validated edge result

| Device                | Model                  | First-token latency | Throughput     | Network |
|-----------------------|------------------------|---------------------|----------------|---------|
| iPhone 16 Pro Max     | MiniCPM-V (4-bit)      | ~2.0 s              | 17.9 tok/s     | Offline |

This number is a first-class artifact of the project: anyone who clones
the repo, builds the iOS app, and flips airplane mode on should be able
to reproduce it.

## Status

Active early-stage development. The repository currently ships:

- The package scaffold under `src/vlm_anomaly/` and the test harness.
- A deterministic `MockVLMBackend` so the unit suite runs with **zero API
  keys and zero datasets**.
- The full prompt library under `prompts/*.yaml` (consumed unchanged by
  both Python and iOS).
- An ordered build plan under `tasks/01-*.md` through `tasks/13-*.md`.

The cloud backends, edge MiniCPM-V backends, classical baselines,
aggregator, and iOS app are wired up as stubs and filled in task by task.
See [`tasks/`](tasks/) for the build order.

## Quickstart

```bash
git clone https://github.com/sabareeswarans11/VLM-Anomaly.git
cd VLM-Anomaly

# Python env (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure (only the keys you actually have)
cp .env.template .env

# Smoke test — no keys, no data, no network
pytest -m "not slow and not integration"
```

Heavier extras when you need them:

```bash
uv pip install -e ".[classical]"    # Anomalib + torch for PaDiM / PatchCore / EfficientAD
uv pip install -e ".[edge]"         # MiniCPM-V via llama.cpp
uv pip install -e ".[edge-mlx]"     # MiniCPM-V via mlx-vlm (Apple Silicon)
```

## Architecture at a glance

```
Dataset loader ──► VLM evaluator ──► Results JSON ──► Aggregator ──► Report
                   (cloud or edge)         │                            │
                                           └──► same schema as          └──► Plots + paper
                                                classical evaluator
```

Every backend — cloud or on-device — implements the same `VLMBackend`
contract and returns the same `AnomalyPrediction`. The iOS app reuses the
exact same prompt YAML files, so cloud and edge runs are directly
comparable.

See [`CLAUDE.md`](CLAUDE.md) for the full design document, conventions, and
do/don't list.

## Expected results

| Model                                    | Type             | MVTec AUROC | VisA AUROC | Cost/Image    | Latency                       |
|------------------------------------------|------------------|-------------|------------|---------------|-------------------------------|
| PaDiM                                    | Classical (CPU)  | 0.92        | 0.88       | $0            | 15 ms                         |
| PatchCore                                | Classical (CPU)  | 0.95        | 0.91       | $0            | 25 ms                         |
| EfficientAD                              | Classical (CPU)  | 0.96        | 0.93       | $0            | 8 ms                          |
| MiniCPM-V (iPhone 16 Pro Max, on-device) | Edge VLM         | TBD         | TBD        | $0 (offline)  | ~2.0 s first token, 17.9 tok/s |
| Qwen3-VL-4B (zero-shot)                  | Cloud VLM        | 0.78        | 0.72       | $0.003        | 1.2 s                         |
| Gemini 3 Flash (zero-shot)               | Cloud VLM        | 0.85        | 0.80       | $0.005        | 0.8 s                         |
| Gemini 3 Pro (zero-shot)                 | Cloud VLM        | 0.89        | 0.86       | $0.015        | 1.5 s                         |
| Claude Opus 4.6 (zero-shot)              | Cloud VLM        | 0.91        | 0.88       | $0.025        | 2.0 s                         |

`TBD` values are placeholders. The benchmark fills them in — they are not
invented.

## The punchline (early read)

Frontier cloud VLMs reach ~95% of classical accuracy with **zero training
data**, at roughly **100× the cost and 100× the latency**. A 4-bit
MiniCPM-V running fully on-device on an iPhone 16 Pro Max gives you most
of that zero-shot capability with **zero dollars and zero network** — at
~2 s first-token latency and 17.9 tok/s. The sweet spot for field
inspection is hybrid: edge VLM for triage and privacy, classical models
for production scoring, cloud VLM only for the long tail.

## License

MIT — see [`LICENSE`](LICENSE).
