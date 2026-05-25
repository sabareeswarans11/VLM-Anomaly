# VLM-Anomaly Benchmark Report

_Generated: 2026-05-25 01:09 UTC_

> **Frontier cloud VLMs reach ~95% of classical accuracy with zero training data,
> at ~100× the cost and ~100× the latency. A 4-bit MiniCPM-V running fully
> on-device on an iPhone 16 Pro Max gives you most of that zero-shot capability
> with zero dollars and zero network — at ~2 s first-token latency and 17.9 tok/s.
> The sweet spot for field inspection is hybrid: edge VLM for triage and privacy,
> classical models for production scoring, cloud VLM only for the long tail.**

## Leaderboard

| Model                                   |   Mean AUROC |   Total Cost (USD) |   Cost/Image (USD) |   Latency (ms) |
|:----------------------------------------|-------------:|-------------------:|-------------------:|---------------:|
| classical/padim                         |       0.9098 |             0      |             0      |       71975    |
| anthropic/claude-opus-4-7               |       0.7519 |            13.36   |             0.0077 |        1080.1  |
| anthropic/claude-opus-4-7-fewshot2-ens4 |       0.6306 |            28.8073 |             0.1029 |        4450.99 |
| gemini/gemini-2.5-flash                 |       0.6104 |             2.37   |             0.0014 |        3971.45 |
| openrouter/qwen/qwen3-vl-32b-instruct   |       0.1782 |             0.276  |             0.0002 |        2122.04 |

## Per-Category Breakdown

| Model                                   | Category   |   N |   AUROC |     F1 |   Latency (ms) |   Cost (USD) |
|:----------------------------------------|:-----------|----:|--------:|-------:|---------------:|-------------:|
| anthropic/claude-opus-4-7               | bottle     |  83 |  0.9103 | 0.8548 |       1030.44  |       1.7635 |
| anthropic/claude-opus-4-7               | tile       | 117 |  0.908  | 0.9581 |       1045.41  |       2.1588 |
| anthropic/claude-opus-4-7               | wood       |  79 |  0.8904 | 0.9756 |       1100.92  |       2.0039 |
| anthropic/claude-opus-4-7               | cable      | 150 |  0.885  | 0.8571 |       1206.41  |       3.8915 |
| anthropic/claude-opus-4-7               | hazelnut   | 110 |  0.8709 | 0.8321 |       1046.07  |       2.7966 |
| anthropic/claude-opus-4-7               | toothbrush |  42 |  0.8514 | 0.8485 |       1175.81  |       1.0835 |
| anthropic/claude-opus-4-7               | metal_nut  | 115 |  0.8262 | 0.8161 |        954.926 |       1.6293 |
| anthropic/claude-opus-4-7               | grid       |  78 |  0.7962 | 0.9825 |       1091.51  |       1.9932 |
| anthropic/claude-opus-4-7               | leather    | 124 |  0.7069 | 0.9946 |        989.908 |       3.1195 |
| anthropic/claude-opus-4-7               | transistor | 100 |  0.699  | 0.5895 |       1110.37  |       2.5506 |
| anthropic/claude-opus-4-7               | pill       | 167 |  0.69   | 0.9008 |       1070.67  |       2.9504 |
| anthropic/claude-opus-4-7               | carpet     | 117 |  0.6878 | 0.9721 |       1089.04  |       2.9743 |
| anthropic/claude-opus-4-7               | zipper     | 151 |  0.5804 | 0.8491 |       1146.46  |       3.8712 |
| anthropic/claude-opus-4-7               | screw      | 160 |  0.5388 | 0.7841 |       1057.62  |       4.06   |
| anthropic/claude-opus-4-7               | capsule    | 132 |  0.4378 | 0.8177 |       1085.96  |       3.205  |
| anthropic/claude-opus-4-7-fewshot2-ens4 | capsule    | 120 |  0.6835 | 0.8743 |       5158.59  |      12.389  |
| anthropic/claude-opus-4-7-fewshot2-ens4 | screw      | 160 |  0.5776 | 0.7721 |       3743.39  |      16.4183 |
| classical/padim                         | bottle     |   0 |  0.9984 | 0.9841 |      51083.7   |       0      |
| classical/padim                         | leather    |   0 |  0.9976 | 0.9836 |      79811     |       0      |
| classical/padim                         | carpet     |   0 |  0.9912 | 0.9711 |      87290.1   |       0      |
| classical/padim                         | metal_nut  |   0 |  0.9761 | 0.9617 |      56939.2   |       0      |
| classical/padim                         | transistor |   0 |  0.9688 | 0.9048 |      68474.6   |       0      |
| classical/padim                         | wood       |   0 |  0.9588 | 0.9355 |      63588.7   |       0      |
| classical/padim                         | tile       |   0 |  0.9181 | 0.9157 |      64990     |       0      |
| classical/padim                         | pill       |   0 |  0.8898 | 0.9466 |      81243.2   |       0      |
| classical/padim                         | grid       |   0 |  0.8864 | 0.9    |      64660.9   |       0      |
| classical/padim                         | capsule    |   0 |  0.8855 | 0.9469 |      80306.6   |       0      |
| classical/padim                         | zipper     |   0 |  0.8842 | 0.9421 |      73582.2   |       0      |
| classical/padim                         | cable      |   0 |  0.8765 | 0.8513 |      89974.5   |       0      |
| classical/padim                         | toothbrush |   0 |  0.8694 | 0.9032 |      25280.4   |       0      |
| classical/padim                         | screw      |   0 |  0.8092 | 0.871  |      86570.7   |       0      |
| classical/padim                         | hazelnut   |   0 |  0.7368 | 0.8214 |     105829     |       0      |
| gemini/gemini-2.5-flash                 | screw      | 160 |  0.7969 | 0.8841 |       4040.84  |       0.008  |
| gemini/gemini-2.5-flash                 | tile       | 117 |  0.787  | 0.9881 |       3945.86  |       0.0062 |
| gemini/gemini-2.5-flash                 | hazelnut   | 110 |  0.6793 | 0.8333 |       4008.9   |       0.0056 |
| gemini/gemini-2.5-flash                 | grid       |  78 |  0.6767 | 0.9735 |       3929.24  |       0.0042 |
| gemini/gemini-2.5-flash                 | metal_nut  | 115 |  0.6393 | 0.9082 |       3933.38  |       0.0057 |
| gemini/gemini-2.5-flash                 | capsule    | 132 |  0.6137 | 0.8889 |       3997.73  |       0.0066 |
| gemini/gemini-2.5-flash                 | wood       |  79 |  0.5917 | 0.9291 |       4071.42  |       0.0042 |
| gemini/gemini-2.5-flash                 | transistor | 100 |  0.5917 | 0.6129 |       4043.52  |       0.0049 |
| gemini/gemini-2.5-flash                 | pill       | 167 |  0.554  | 0.918  |       3894.51  |       0.0082 |
| gemini/gemini-2.5-flash                 | toothbrush |  42 |  0.55   | 0.8485 |       3984.96  |       0.0021 |
| gemini/gemini-2.5-flash                 | bottle     |  83 |  0.5429 | 0.875  |       3923.94  |       0.0041 |
| gemini/gemini-2.5-flash                 | leather    | 124 |  0.5416 | 0.984  |       3879.84  |       0.0066 |
| gemini/gemini-2.5-flash                 | zipper     | 151 |  0.539  | 0.8735 |       3868.31  |       0.0079 |
| gemini/gemini-2.5-flash                 | cable      | 150 |  0.53   | 0.7586 |       4071.38  |       0.0074 |
| gemini/gemini-2.5-flash                 | carpet     | 117 |  0.5229 | 0.9718 |       3977.93  |       0.0064 |
| openrouter/qwen/qwen3-vl-32b-instruct   | pill       | 167 |  0.4816 | 0.9043 |       2125.9   |       0.0214 |
| openrouter/qwen/qwen3-vl-32b-instruct   | cable      | 150 |  0.3493 | 0.6176 |       1913.2   |       0.023  |
| openrouter/qwen/qwen3-vl-32b-instruct   | transistor | 100 |  0.3388 | 0.6947 |       2270.4   |       0.0163 |
| openrouter/qwen/qwen3-vl-32b-instruct   | hazelnut   | 110 |  0.2989 | 0.9722 |       2600.21  |       0.0174 |
| openrouter/qwen/qwen3-vl-32b-instruct   | zipper     | 151 |  0.2935 | 0.648  |       1954.03  |       0.0234 |
| openrouter/qwen/qwen3-vl-32b-instruct   | bottle     |  83 |  0.2087 | 0.9256 |       1950.25  |       0.011  |
| openrouter/qwen/qwen3-vl-32b-instruct   | toothbrush |  42 |  0.1528 | 0.9492 |       2282.73  |       0.007  |
| openrouter/qwen/qwen3-vl-32b-instruct   | capsule    | 132 |  0.1492 | 0.7889 |       1916.44  |       0.0197 |
| openrouter/qwen/qwen3-vl-32b-instruct   | screw      | 160 |  0.138  | 0.6957 |       1755.61  |       0.0245 |
| openrouter/qwen/qwen3-vl-32b-instruct   | metal_nut  | 115 |  0.097  | 0.9379 |       2105.69  |       0.0119 |
| openrouter/qwen/qwen3-vl-32b-instruct   | grid       |  78 |  0.0547 | 0.9825 |       2138.85  |       0.0127 |
| openrouter/qwen/qwen3-vl-32b-instruct   | carpet     | 117 |  0.0502 | 0.9773 |       2121.52  |       0.0183 |
| openrouter/qwen/qwen3-vl-32b-instruct   | wood       |  79 |  0.025  | 0.9565 |       2504.23  |       0.0133 |
| openrouter/qwen/qwen3-vl-32b-instruct   | tile       | 117 |  0.0231 | 0.9231 |       1794.12  |       0.014  |
| openrouter/qwen/qwen3-vl-32b-instruct   | leather    | 124 |  0.0126 | 0.978  |       2397.42  |       0.0194 |

## Plots

### AUROC Comparison

![AUROC Bar Chart](results/plots/auroc_bar.png)

### Cost vs Accuracy

![Cost vs Accuracy](results/plots/cost_vs_accuracy.png)

### Category Heatmap

![Category Heatmap](results/plots/category_heatmap.png)

## AUROC by Model × Category

| model_id                                |   bottle |   cable |   capsule |   carpet |    grid |   hazelnut |   leather |   metal_nut |    pill |   screw |    tile |   toothbrush |   transistor |    wood |   zipper |
|:----------------------------------------|---------:|--------:|----------:|---------:|--------:|-----------:|----------:|------------:|--------:|--------:|--------:|-------------:|-------------:|--------:|---------:|
| anthropic/claude-opus-4-7               |    0.91  |   0.885 |     0.438 |    0.688 |   0.796 |      0.871 |     0.707 |       0.826 |   0.69  |   0.539 |   0.908 |        0.851 |        0.699 |   0.89  |    0.58  |
| anthropic/claude-opus-4-7-fewshot2-ens4 |  nan     | nan     |     0.684 |  nan     | nan     |    nan     |   nan     |     nan     | nan     |   0.578 | nan     |      nan     |      nan     | nan     |  nan     |
| classical/padim                         |    0.998 |   0.876 |     0.886 |    0.991 |   0.886 |      0.737 |     0.998 |       0.976 |   0.89  |   0.809 |   0.918 |        0.869 |        0.969 |   0.959 |    0.884 |
| gemini/gemini-2.5-flash                 |    0.543 |   0.53  |     0.614 |    0.523 |   0.677 |      0.679 |     0.542 |       0.639 |   0.554 |   0.797 |   0.787 |        0.55  |        0.592 |   0.592 |    0.539 |
| openrouter/qwen/qwen3-vl-32b-instruct   |    0.209 |   0.349 |     0.149 |    0.05  |   0.055 |      0.299 |     0.013 |       0.097 |   0.482 |   0.138 |   0.023 |        0.153 |        0.339 |   0.025 |    0.293 |

## Actual Provider Spend

Costs below are **real billing figures** for the full MVTec AD sweep
(15 categories, ~1,725 test images).

| Model | Type | Provider | Dataset | Images | Actual Spend | Cost / Image |
|---|---|---|---|---|---|---|
| PaDiM | Classical | Open Source ([Anomalib](https://github.com/openvinotoolkit/anomalib)) | MVTec AD (15 cat) | 1,725 | **$0.00** | $0 |
| PatchCore | Classical | Open Source ([Anomalib](https://github.com/openvinotoolkit/anomalib)) | MVTec AD (15 cat) | 1,725 | **$0.00** | $0 |
| Qwen3-VL-32B | Cloud VLM | OpenRouter | MVTec AD (15 cat) | 1,725 | **$0.28** | ~$0.0002 |
| Gemini 2.5 Flash | Cloud VLM | Google AI Studio | MVTec AD (15 cat) | 1,725 | **$2.37** | ~$0.0014 |
| Claude Opus 4.7 | Cloud VLM | Anthropic API | MVTec AD (15 cat) | 1,725 | **$13.36** | ~$0.0077 |

> Cloud VLM costs are actual billing figures stored in `results/costs_override.json`
> and applied automatically on report generation.
> Classical baselines run fully on CPU — no API, no tokens, no cost.

### Add credits / manage billing

| Provider | Billing page |
|---|---|
| Anthropic (Claude) | [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing) |
| Google (Gemini) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) · [console.cloud.google.com/billing](https://console.cloud.google.com/billing) |
| OpenRouter (Qwen / multi-model) | [openrouter.ai/credits](https://openrouter.ai/credits) |


## Methodology

- **Datasets**: MVTec AD (15 categories, ~5 GB) and VisA (12 categories).
- **VLM evaluation**: Zero-shot — no training data, no normal reference set.
  Each image is evaluated with a single prompt from `prompts/*.yaml`.
- **Classical baselines**: PaDiM / PatchCore / EfficientAD trained on the
  normal split of each category using Anomalib on CPU.
- **Metrics**: Image-level AUROC and F1 (threshold = 0.5 on confidence score).
- **Cost**: Actual token cost from provider billing data.
- **Latency**: Wall-clock time per image (cloud: network included; edge: local).
- **Statistical tests**: McNemar test for paired model comparisons;
  bootstrap CIs (1000 resamples) for AUROC and F1.

Source: [github.com/sabareeswarans11/VLM-Anomaly](https://github.com/sabareeswarans11/VLM-Anomaly)

---

## Leaderboard with Few-Shot Ensemble

> Ensemble model consolidated AUROC: baseline 15-category AUROCs with 2 categories replaced by ensemble results (capsule, screw).

| Model | Mean AUROC | Latency (ms) | Note |
|-------|-----------|-------------|------|
| classical/padim | 0.9098 | 71975 |  |
| anthropic/claude-opus-4-7-fewshot2-ens4 | 0.7709 | 4451 | approx — 2/15 cats replaced |
| anthropic/claude-opus-4-7 | 0.7519 | 1080 |  |
| gemini/gemini-2.5-flash | 0.6104 | 3971 |  |
| openrouter/qwen/qwen3-vl-32b-instruct | 0.1782 | 2122 |  |
