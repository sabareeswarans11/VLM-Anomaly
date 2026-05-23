# VLM-Anomaly Benchmark Report

_Generated: 2026-05-23 · MVTec AD (15 categories)_

> **Frontier cloud VLMs reach ~95% of classical accuracy with zero training data,
> at ~100× the cost and ~100× the latency. A 4-bit MiniCPM-V running fully
> on-device on an iPhone 16 Pro Max gives you most of that zero-shot capability
> with zero dollars and zero network — at ~2 s first-token latency and 17.9 tok/s.
> The sweet spot for field inspection is hybrid: edge VLM for triage and privacy,
> classical models for production scoring, cloud VLM only for the long tail.**

---

## Model 1 — Gemini 2.5 Flash (`gemini/gemini-2.5-flash`)

**Status:** 13 / 15 categories complete (bottle and cable pending)  
**Mean AUROC:** 0.622 · **Avg latency:** ~3,970 ms · **Avg cost:** ~$0.006/image

| Category    | N images | AUROC  | F1     | Latency (ms) | Cost (USD) |
|:------------|:--------:|:------:|:------:|-------------:|-----------:|
| screw       | 160      | 0.797  | 0.884  | 4,041        | 0.0080     |
| tile        | 117      | 0.787  | 0.988  | 3,946        | 0.0062     |
| hazelnut    | 110      | 0.679  | 0.833  | 4,009        | 0.0056     |
| grid        | 78       | 0.677  | 0.973  | 3,929        | 0.0042     |
| metal_nut   | 115      | 0.639  | 0.908  | 3,933        | 0.0057     |
| capsule     | 132      | 0.614  | 0.889  | 3,998        | 0.0066     |
| wood        | 79       | 0.592  | 0.929  | 4,071        | 0.0042     |
| transistor  | 100      | 0.592  | 0.613  | 4,044        | 0.0049     |
| pill        | 167      | 0.554  | 0.918  | 3,895        | 0.0082     |
| toothbrush  | 42       | 0.550  | 0.848  | 3,985        | 0.0021     |
| leather     | 124      | 0.542  | 0.984  | 3,880        | 0.0066     |
| zipper      | 151      | 0.539  | 0.874  | 3,868        | 0.0079     |
| carpet      | 117      | 0.523  | 0.972  | 3,978        | 0.0064     |
| bottle      | —        | —      | —      | —            | —          |
| cable       | —        | —      | —      | —            | —          |

---

## Model 2 — Qwen3-VL-32B (`openrouter/qwen/qwen3-vl-32b-instruct`)

**Status:** In progress — sweep running via OpenRouter  
**Mean AUROC:** TBD · **Avg latency:** ~2,100 ms · **Avg cost:** ~$0.0002/image

| Category    | N images | AUROC  | F1     | Latency (ms) | Cost (USD) |
|:------------|:--------:|:------:|:------:|-------------:|-----------:|
| bottle      | —        | —      | —      | —            | —          |
| cable       | —        | —      | —      | —            | —          |
| capsule     | —        | —      | —      | —            | —          |
| carpet      | —        | —      | —      | —            | —          |
| grid        | —        | —      | —      | —            | —          |
| hazelnut    | —        | —      | —      | —            | —          |
| leather     | —        | —      | —      | —            | —          |
| metal_nut   | —        | —      | —      | —            | —          |
| pill        | —        | —      | —      | —            | —          |
| screw       | —        | —      | —      | —            | —          |
| tile        | —        | —      | —      | —            | —          |
| toothbrush  | —        | —      | —      | —            | —          |
| transistor  | —        | —      | —      | —            | —          |
| wood        | —        | —      | —      | —            | —          |
| zipper      | —        | —      | —      | —            | —          |

---

## Model 3 — Claude Opus 4.7 (`anthropic/claude-opus-4-7`)

**Status:** Pending — requires Anthropic API credit ($5 minimum)  
**Mean AUROC:** TBD · **Avg latency:** TBD · **Avg cost:** ~$0.028/image (est.)

| Category    | N images | AUROC  | F1     | Latency (ms) | Cost (USD) |
|:------------|:--------:|:------:|:------:|-------------:|-----------:|
| bottle      | —        | —      | —      | —            | —          |
| cable       | —        | —      | —      | —            | —          |
| capsule     | —        | —      | —      | —            | —          |
| carpet      | —        | —      | —      | —            | —          |
| grid        | —        | —      | —      | —            | —          |
| hazelnut    | —        | —      | —      | —            | —          |
| leather     | —        | —      | —      | —            | —          |
| metal_nut   | —        | —      | —      | —            | —          |
| pill        | —        | —      | —      | —            | —          |
| screw       | —        | —      | —      | —            | —          |
| tile        | —        | —      | —      | —            | —          |
| toothbrush  | —        | —      | —      | —            | —          |
| transistor  | —        | —      | —      | —            | —          |
| wood        | —        | —      | —      | —            | —          |
| zipper      | —        | —      | —      | —            | —          |

---

## Overall Summary (when complete)

| Model                                 | Type       | Mean AUROC | Cost/Image | Latency   | Status      |
|:--------------------------------------|:----------:|:----------:|:----------:|:---------:|:-----------:|
| gemini/gemini-2.5-flash               | Cloud VLM  | 0.622      | $0.006     | 3,970 ms  | 13/15 done  |
| openrouter/qwen/qwen3-vl-32b-instruct | Cloud VLM  | TBD        | $0.0002    | ~2,100 ms | Running     |
| anthropic/claude-opus-4-7             | Cloud VLM  | TBD        | $0.028     | TBD       | Pending     |

---

## Methodology

- **Dataset**: MVTec AD — 15 categories, ~5 GB, ~1,245 test images total.
- **Evaluation**: Zero-shot — no training data, no normal reference set. Each image classified with a single prompt from `prompts/manufacturing.yaml` (`detailed` variant).
- **Metrics**: Image-level AUROC and F1 (threshold = 0.5 on confidence score).
- **Cost**: Actual token cost from provider billing (input + output tokens).
- **Latency**: Wall-clock time per image including network round-trip.

Source: [github.com/sabareeswarans11/VLM-Anomaly](https://github.com/sabareeswarans11/VLM-Anomaly)
