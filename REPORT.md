# VLM-Anomaly Benchmark Report

_MVTec AD — 15 categories, 1,725 test images — zero-shot evaluation_

> **Paper in preparation.** Full ROC curves, statistical tests (McNemar, bootstrap CIs),
> cost-accuracy scatter plots, and per-category heatmaps will appear in the accompanying paper
> targeting **REAI Conference 2026** at SMU University
> (Track 1: Resource-Efficient Foundation Models).

---

## Leaderboard

| Model | Type | Mean AUROC | Latency |
|:------|:-----|----------:|--------:|
| classical/patchcore | Classical (trained) | 0.9147 | — |
| classical/padim | Classical (trained) | 0.9098 | ~72 s/cat |
| anthropic/claude-opus-4-7 (few-shot ens.) | Cloud VLM | 0.7709 | ~1.5 s/img |
| anthropic/claude-opus-4-7 | Cloud VLM | 0.7519 | ~1.1 s/img |
| edge/minicpm-v-4.6-q4km | Edge VLM (on-device) | 0.6523 | ~20 s/img¹ |
| gemini/gemini-2.5-flash | Cloud VLM | 0.6104 | ~4.0 s/img |
| openrouter/qwen3-vl-32b | Cloud VLM | 0.1782 | ~2.1 s/img |

¹ Intel CPU (dev machine). iPhone 16 Pro Max: ~2.0 s first-token, 17.9 tok/s.

---

## Per-Category AUROC

| Category | PatchCore | PaDiM | Claude Opus 4.7 | MiniCPM-V (edge) | Gemini 2.5 Flash | Qwen3-VL-32B |
|:---------|----------:|------:|----------------:|-----------------:|-----------------:|-------------:|
| bottle | 0.9952 | 0.9984 | 0.9103 | 0.7619 | 0.5429 | 0.2087 |
| cable | 0.8703 | 0.8765 | 0.8850 | 0.5000 | 0.5300 | 0.3493 |
| capsule | 0.8823 | 0.8855 | 0.4378 | 0.5275 | 0.6137 | 0.1492 |
| carpet | 0.9675 | 0.9912 | 0.6878 | 0.5562 | 0.5229 | 0.0502 |
| grid | 0.6734 | 0.8864 | 0.7962 | 0.5877 | 0.6767 | 0.0547 |
| hazelnut | 1.0000 | 0.7368 | 0.8709 | 0.8500 | 0.6793 | 0.2989 |
| leather | 0.9997 | 0.9976 | 0.7069 | 0.7989 | 0.5416 | 0.0126 |
| metal_nut | 0.9888 | 0.9761 | 0.8262 | 0.6559 | 0.6393 | 0.0970 |
| pill | 0.8633 | 0.8898 | 0.6900 | 0.6148 | 0.5540 | 0.4816 |
| screw | 0.8446 | 0.8092 | 0.5388 | 0.5000 | 0.7969 | 0.1380 |
| tile | 0.9188 | 0.9181 | 0.9080 | 0.9345 | 0.7870 | 0.0231 |
| toothbrush | 0.8806 | 0.8694 | 0.8514 | 0.7431 | 0.5500 | 0.1528 |
| transistor | 0.9029 | 0.9688 | 0.6990 | 0.5625 | 0.5917 | 0.3388 |
| wood | 0.9860 | 0.9588 | 0.8904 | 0.6833 | 0.5917 | 0.0250 |
| zipper | 0.9464 | 0.8842 | 0.5804 | 0.5084 | 0.5390 | 0.2935 |
| **Mean** | **0.9147** | **0.9098** | **0.7519** | **0.6523** | **0.6104** | **0.1782** |

---

## Methodology

- **Dataset**: MVTec AD (15 object/texture categories, ~5 GB). Ground truth: `good/` folder → label 0; all other subdirectories → label 1.
- **Metric**: Image-level AUROC (`sklearn.metrics.roc_auc_score`). F1 at threshold 0.5 on `confidence` score.
- **Zero-shot VLMs**: One image + one text prompt per inference. No training data. No reference normal images (except few-shot ensemble variant).
- **Classical baselines**: PaDiM and PatchCore trained on each category's normal split via Anomalib. PatchCore evaluated on Kaggle P100 GPU; PaDiM on CPU.
- **Edge VLM**: MiniCPM-V 4.6 Q4_K_M GGUF via llama.cpp direct mtmd API. C_calibrated_scale prompt. Evaluated on Intel CPU; iPhone 16 Pro Max numbers from iOS app benchmark view.
- **Few-shot ensemble**: Claude Opus 4.7 with 2 normal reference images in context + 4-prompt voting. Consolidated AUROC across all 15 categories (2 categories replaced by ensemble results).

Source: [github.com/sabareeswarans11/VLM-Anomaly](https://github.com/sabareeswarans11/VLM-Anomaly)
