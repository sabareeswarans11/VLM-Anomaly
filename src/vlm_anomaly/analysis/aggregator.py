"""DuckDB-based results aggregator.

Reads all ``*.jsonl`` (per-image VLM records) and ``*.json`` (pre-aggregated
classical baseline records) from a results directory, recomputes metrics, and
returns a ranked leaderboard DataFrame.

Usage::

    from vlm_anomaly.analysis.aggregator import leaderboard
    df = leaderboard("results/")
    print(df)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vlm_anomaly.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def leaderboard(results_dir: str | Path) -> pd.DataFrame:
    """Return a ranked leaderboard from all results in ``results_dir``.

    Reads both JSONL (per-image VLM records) and JSON (pre-aggregated
    classical baseline records).  Per-image records are re-aggregated so
    that all rows share the same schema.

    Args:
        results_dir: Directory containing ``*.jsonl`` and/or ``*.json`` files.

    Returns:
        DataFrame with columns: ``model_id``, ``backend``, ``dataset``,
        ``category``, ``n_images``, ``auroc``, ``f1``, ``mean_latency_ms``,
        ``total_cost_usd``.  Sorted by ``auroc`` descending.
    """
    results_dir = Path(results_dir)
    frames: list[pd.DataFrame] = []

    jsonl_files = sorted(results_dir.glob("*.jsonl"))
    json_files = sorted(results_dir.glob("*.json"))

    if jsonl_files:
        df = _aggregate_jsonl(jsonl_files)
        if not df.empty:
            frames.append(df)

    if json_files:
        df = _load_precomputed_json(json_files)
        if not df.empty:
            frames.append(df)

    if not frames:
        log.warning("aggregator.empty", results_dir=str(results_dir))
        return pd.DataFrame(columns=[
            "model_id", "backend", "dataset", "category",
            "n_images", "auroc", "f1", "mean_latency_ms", "total_cost_usd",
        ])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("auroc", ascending=False, na_position="last")
    return combined.reset_index(drop=True)


def category_heatmap(results_dir: str | Path) -> pd.DataFrame:
    """Return a model × category AUROC pivot table.

    Args:
        results_dir: Directory containing results files.

    Returns:
        DataFrame with models as rows and categories as columns.
    """
    lb = leaderboard(results_dir)
    if lb.empty:
        return pd.DataFrame()
    return lb.pivot_table(
        index="model_id", columns="category", values="auroc", aggfunc="mean"
    )


def cost_accuracy_table(results_dir: str | Path) -> pd.DataFrame:
    """Return per-model mean AUROC and mean cost per image.

    Args:
        results_dir: Directory containing results files.

    Returns:
        DataFrame with ``model_id``, ``mean_auroc``, ``mean_cost_per_image``,
        ``mean_latency_ms`` columns.
    """
    lb = leaderboard(results_dir)
    if lb.empty:
        return pd.DataFrame()

    agg = lb.groupby("model_id").agg(
        mean_auroc=("auroc", "mean"),
        mean_cost_per_image=("total_cost_usd", lambda x: (x / lb.loc[x.index, "n_images"].replace(0, 1)).mean()),
        mean_latency_ms=("mean_latency_ms", "mean"),
        backend=("backend", "first"),
    ).reset_index()
    return agg.sort_values("mean_auroc", ascending=False)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_jsonl(files: list[Path]) -> pd.DataFrame:
    """Re-aggregate per-image JSONL records using DuckDB."""
    if not files:
        return pd.DataFrame()

    # Build a glob pattern DuckDB can use
    parent = files[0].parent
    glob = str(parent / "*.jsonl")

    try:
        con = duckdb.connect()
        con.execute(f"""
            CREATE OR REPLACE VIEW raw AS
            SELECT * FROM read_ndjson_auto('{glob}', ignore_errors=true)
        """)

        # Flatten nested prediction struct
        df = con.execute("""
            SELECT
                model_id,
                backend,
                dataset,
                category,
                COUNT(*)                                    AS n_images,
                AVG(CAST(sample_label AS DOUBLE))           AS anomaly_rate,
                AVG(prediction.latency_ms)                  AS mean_latency_ms,
                SUM(prediction.cost_usd)                    AS total_cost_usd
            FROM raw
            GROUP BY model_id, backend, dataset, category
        """).df()
        con.close()
    except Exception as exc:
        log.warning("aggregator.jsonl.error", error=str(exc))
        return pd.DataFrame()

    # Recompute AUROC/F1 per group using scikit-learn
    rows = []
    for _, grp in df.iterrows():
        row = grp.to_dict()
        auroc, f1 = _recompute_metrics_for_group(parent, grp)
        row["auroc"] = auroc
        row["f1"] = f1
        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _recompute_metrics_for_group(results_dir: Path, grp: pd.Series) -> tuple[float | None, float | None]:
    """Read JSONL for one (model, dataset, category) cell and compute metrics."""
    import json
    from vlm_anomaly.evaluators.base import _safe_auroc, _safe_f1

    labels, scores, preds = [], [], []
    pattern = f"*_{grp['dataset']}_{grp['category']}.jsonl"
    for path in results_dir.glob(pattern):
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                if rec.get("model_id") != grp["model_id"]:
                    continue
                labels.append(rec["sample_label"])
                scores.append(rec["prediction"]["confidence"])
                preds.append(int(rec["prediction"]["is_anomalous"]))
            except (json.JSONDecodeError, KeyError):
                continue

    if not labels:
        return None, None

    auroc = _safe_auroc(labels, scores)
    f1, _, _ = _safe_f1(labels, preds)
    return auroc, f1


def _load_precomputed_json(files: list[Path]) -> pd.DataFrame:
    """Load pre-aggregated EvalResult JSON files (classical baselines)."""
    import json

    rows = []
    for path in files:
        if path.name == "sample_results.json":
            continue  # skip test fixture
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                rows.extend(data)
            elif isinstance(data, dict):
                rows.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Keep only columns we care about; tolerate missing columns
    want = ["model_id", "backend", "dataset", "category", "n_images",
            "auroc", "f1", "mean_latency_ms", "total_cost_usd"]
    for col in want:
        if col not in df.columns:
            df[col] = None
    return df[want]
