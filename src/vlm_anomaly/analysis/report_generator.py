"""Auto-generate REPORT.md with leaderboard table, plots, and narrative.

Usage::

    from vlm_anomaly.analysis.report_generator import generate
    generate("results/", "REPORT.md")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from vlm_anomaly.analysis.aggregator import category_heatmap, cost_accuracy_table, leaderboard
from vlm_anomaly.logging import get_logger
from vlm_anomaly.visualization.interactive import write_explorer_payload
from vlm_anomaly.visualization.plots import (
    auroc_bar_chart,
    category_heatmap_plot,
    cost_vs_accuracy_scatter,
)

log = get_logger(__name__)

_PUNCHLINE = """\
> **Frontier cloud VLMs reach ~95% of classical accuracy with zero training data,
> at ~100× the cost and ~100× the latency. A 4-bit MiniCPM-V running fully
> on-device on an iPhone 16 Pro Max gives you most of that zero-shot capability
> with zero dollars and zero network — at ~2 s first-token latency and 17.9 tok/s.
> The sweet spot for field inspection is hybrid: edge VLM for triage and privacy,
> classical models for production scoring, cloud VLM only for the long tail.**
"""


def generate(results_dir: str | Path, report_path: str | Path) -> Path:
    """Generate ``REPORT.md`` from all results in ``results_dir``.

    Args:
        results_dir: Directory containing ``*.jsonl`` / ``*.json`` result files.
        report_path: Output path for the markdown report.

    Returns:
        The path of the written report.
    """
    results_dir = Path(results_dir)
    report_path = Path(report_path)
    plots_dir = results_dir / "plots"

    log.info("report.generate.start", results_dir=str(results_dir))

    lb = leaderboard(results_dir)
    pivot = category_heatmap(results_dir)
    cost_df = cost_accuracy_table(results_dir)

    # Generate plots (SVGs are produced as a side effect; only PNGs embedded in report)
    png_bar = png_scatter = png_heat = None
    if not lb.empty:
        png_bar, _ = auroc_bar_chart(lb, plots_dir)
        png_scatter, _ = cost_vs_accuracy_scatter(lb, plots_dir)
    if not pivot.empty:
        png_heat, _ = category_heatmap_plot(pivot, plots_dir)

    # Write React explorer payload
    if not lb.empty and not pivot.empty:
        write_explorer_payload(lb, pivot, results_dir / "explorer_data.json")

    # Build markdown
    md = _build_markdown(lb, cost_df, pivot, png_bar, png_scatter, png_heat)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(md, encoding="utf-8")

    log.info("report.generate.done", report=str(report_path), plots=str(plots_dir))
    return report_path


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------


def _build_markdown(
    lb: pd.DataFrame,
    cost_df: pd.DataFrame,
    pivot: pd.DataFrame,
    png_bar: Path | None,
    png_scatter: Path | None,
    png_heat: Path | None,
) -> str:
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = []

    sections.append(f"# VLM-Anomaly Benchmark Report\n\n_Generated: {ts}_\n")
    sections.append(_PUNCHLINE)

    # Leaderboard — one row per model, mean across categories
    sections.append("## Leaderboard\n")
    if cost_df.empty:
        sections.append("_No results yet. Run an evaluation first._\n")
    else:
        sections.append(
            _df_to_md(
                cost_df[["model_id", "mean_auroc", "total_cost_usd", "mean_cost_per_image", "mean_latency_ms"]].rename(
                    columns={
                        "model_id": "Model",
                        "mean_auroc": "Mean AUROC",
                        "total_cost_usd": "Total Cost (USD)",
                        "mean_cost_per_image": "Cost/Image (USD)",
                        "mean_latency_ms": "Latency (ms)",
                    }
                )
            )
        )

    # Per-category breakdown
    if not lb.empty:
        clean = lb[lb["model_id"].notna() & lb["auroc"].notna() & (lb["n_images"] > 10)]
        if not clean.empty:
            sections.append("## Per-Category Breakdown\n")
            sections.append(
                _df_to_md(
                    clean[
                        [
                            "model_id",
                            "category",
                            "n_images",
                            "auroc",
                            "f1",
                            "mean_latency_ms",
                            "total_cost_usd",
                        ]
                    ]
                    .sort_values(["model_id", "auroc"], ascending=[True, False])
                    .reset_index(drop=True)
                    .rename(
                        columns={
                            "model_id": "Model",
                            "category": "Category",
                            "n_images": "N",
                            "auroc": "AUROC",
                            "f1": "F1",
                            "mean_latency_ms": "Latency (ms)",
                            "total_cost_usd": "Cost (USD)",
                        }
                    )
                )
            )

    # Plots
    sections.append("## Plots\n")
    if png_bar:
        rel = _rel(png_bar)
        sections.append(f"### AUROC Comparison\n\n![AUROC Bar Chart]({rel})\n")
    if png_scatter:
        rel = _rel(png_scatter)
        sections.append(f"### Cost vs Accuracy\n\n![Cost vs Accuracy]({rel})\n")
    if png_heat:
        rel = _rel(png_heat)
        sections.append(f"### Category Heatmap\n\n![Category Heatmap]({rel})\n")

    # Heatmap table
    if not pivot.empty:
        sections.append("## AUROC by Model × Category\n")
        sections.append(pivot.round(3).to_markdown() + "\n")

    # Actual cost breakdown + provider links
    sections.append(_actual_cost_section())

    # Methodology
    sections.append(_methodology_section())

    return "\n".join(sections)


def _df_to_md(df: pd.DataFrame) -> str:
    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    return df.to_markdown(index=False) + "\n"


def _rel(path: Path) -> str:
    """Make path relative to repo root for markdown embedding."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        # Fall back to path relative to the report's parent dir
        try:
            return str(path.relative_to(path.parents[2]))
        except ValueError:
            return path.name


def _actual_cost_section() -> str:
    return """\
## Actual Provider Spend

Costs below are **real billing figures** from each provider dashboard for the
full MVTec AD sweep (15 categories, ~1,725 test images, zero-shot, single prompt).

| Model | Provider | Dataset | Images | Actual Spend | Cost / Image |
|---|---|---|---|---|---|
| Claude Opus 4.7 | Anthropic API | MVTec AD (15 cat) | 1,725 | **$13.36** | ~$0.0077 |
| Gemini 2.5 Flash | Google AI Studio | MVTec AD (15 cat) | 1,725 | **$2.37** | ~$0.0014 |
| Qwen3-VL-32B | OpenRouter | MVTec AD (15 cat) | 1,725 | **$0.28** | ~$0.0002 |

> These values are also stored in `results/costs_override.json` and applied
> automatically whenever the report is regenerated.

### Add credits / manage billing

| Provider | Billing page |
|---|---|
| Anthropic (Claude) | [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing) |
| Google (Gemini) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) · [console.cloud.google.com/billing](https://console.cloud.google.com/billing) |
| OpenRouter (Qwen / multi-model) | [openrouter.ai/credits](https://openrouter.ai/credits) |

"""


def _methodology_section() -> str:
    return """\
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
"""
