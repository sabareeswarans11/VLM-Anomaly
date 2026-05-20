"""JSON payload builders for the optional React + Recharts explorer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def leaderboard_json(df: pd.DataFrame) -> list[dict]:
    """Convert a leaderboard DataFrame to a Recharts-friendly list of dicts.

    Args:
        df: Leaderboard DataFrame from the aggregator.

    Returns:
        List of records suitable for ``JSON.stringify`` and consumption by
        a Recharts ``BarChart`` or ``ScatterChart``.
    """
    records = df.to_dict(orient="records")
    # Round floats for compact JSON
    for rec in records:
        for key in ("auroc", "f1", "precision", "recall"):
            if rec.get(key) is not None:
                rec[key] = round(float(rec[key]), 4)
        for key in ("mean_latency_ms", "total_cost_usd"):
            if rec.get(key) is not None:
                rec[key] = round(float(rec[key]), 6)
    return records


def heatmap_json(pivot: pd.DataFrame) -> dict:
    """Convert the model × category AUROC pivot to a nested dict for Recharts.

    Args:
        pivot: Pivot table from the aggregator.

    Returns:
        ``{"models": [...], "categories": [...], "cells": [...]}``
    """
    models = list(pivot.index)
    categories = list(pivot.columns)
    cells = []
    for model in models:
        for cat in categories:
            val = pivot.loc[model, cat]
            if pd.notna(val):
                cells.append({"model": model, "category": cat, "auroc": round(float(val), 4)})
    return {"models": models, "categories": categories, "cells": cells}


def write_explorer_payload(df: pd.DataFrame, pivot: pd.DataFrame, out_path: Path) -> Path:
    """Write a single JSON file consumed by the React explorer.

    Args:
        df: Leaderboard DataFrame.
        pivot: Model × category AUROC pivot.
        out_path: Destination path (created with parents if needed).

    Returns:
        The path that was written.
    """
    payload = {
        "leaderboard": leaderboard_json(df),
        "heatmap": heatmap_json(pivot),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path
