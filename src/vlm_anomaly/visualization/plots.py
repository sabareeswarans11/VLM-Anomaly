"""Paper-quality matplotlib/seaborn plots.

All plots are saved as both PNG (for README) and SVG (for paper).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)

_FIGSIZE_BAR = (10, 5)
_FIGSIZE_SCATTER = (8, 6)
_FIGSIZE_HEATMAP = (12, 5)
_DPI = 150


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> tuple[Path, Path]:
    """Save ``fig`` as ``{name}.png`` and ``{name}.svg`` under ``out_dir``.

    Args:
        fig: Matplotlib figure.
        out_dir: Output directory (created if missing).
        name: File stem (no extension).

    Returns:
        ``(png_path, svg_path)``
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, dpi=_DPI, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def auroc_bar_chart(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Horizontal bar chart of mean AUROC per model.

    Args:
        df: Leaderboard DataFrame from :func:`~vlm_anomaly.analysis.aggregator.leaderboard`.
        out_dir: Where to save the figure.

    Returns:
        ``(png_path, svg_path)``
    """
    agg = df.groupby("model_id")["auroc"].mean().dropna().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=_FIGSIZE_BAR)
    colors = [
        "#2196F3"
        if "anomalib" not in str(df[df["model_id"] == m]["backend"].iloc[0])
        else "#4CAF50"
        for m in agg.index
    ]
    bars = ax.barh(agg.index, agg.values, color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xlabel("Mean AUROC")
    ax.set_title("Model AUROC Comparison (MVTec AD)", fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(x=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Legend
    from matplotlib.patches import Patch

    legend = [
        Patch(color="#4CAF50", label="Classical (Anomalib)"),
        Patch(color="#2196F3", label="VLM (zero-shot)"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9)
    fig.tight_layout()
    return save_fig(fig, out_dir, "auroc_bar")


def cost_vs_accuracy_scatter(df: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Scatter plot of cost-per-image vs mean AUROC.

    Args:
        df: Leaderboard DataFrame.
        out_dir: Where to save the figure.

    Returns:
        ``(png_path, svg_path)``
    """
    agg = (
        df.groupby("model_id")
        .agg(
            auroc=("auroc", "mean"),
            cost_per_image=(
                "total_cost_usd",
                lambda x: (x / df.loc[x.index, "n_images"].replace(0, 1)).mean(),
            ),
            backend=("backend", "first"),
        )
        .dropna(subset=["auroc"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=_FIGSIZE_SCATTER)
    for _, row in agg.iterrows():
        color = "#4CAF50" if row["backend"] == "anomalib" else "#2196F3"
        ax.scatter(row["cost_per_image"], row["auroc"], color=color, s=100, zorder=3)
        ax.annotate(
            row["model_id"],
            (row["cost_per_image"], row["auroc"]),
            textcoords="offset points",
            xytext=(6, 3),
            fontsize=8,
        )

    ax.set_xlabel("Cost per Image (USD)")
    ax.set_ylabel("Mean AUROC")
    ax.set_title("Cost vs Accuracy Trade-off", fontweight="bold")
    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return save_fig(fig, out_dir, "cost_vs_accuracy")


def category_heatmap_plot(pivot: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    """Heatmap of AUROC for each model × category cell.

    Args:
        pivot: Pivot table from
            :func:`~vlm_anomaly.analysis.aggregator.category_heatmap`.
        out_dir: Where to save the figure.

    Returns:
        ``(png_path, svg_path)``
    """
    fig, ax = plt.subplots(figsize=_FIGSIZE_HEATMAP)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "AUROC"},
    )
    ax.set_title("AUROC by Model × Category", fontweight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Model")
    fig.tight_layout()
    return save_fig(fig, out_dir, "category_heatmap")
