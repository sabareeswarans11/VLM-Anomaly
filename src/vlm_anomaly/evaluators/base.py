"""Common evaluator base — shared metric computation."""

from __future__ import annotations

from vlm_anomaly.schemas import EvalResult, PerImageResult


def compute_eval_result(
    rows: list[PerImageResult],
    model_id: str,
    backend: str,
    dataset: str,
    category: str,
) -> EvalResult:
    """Compute aggregate metrics from a list of per-image results.

    Args:
        rows: All :class:`~vlm_anomaly.schemas.PerImageResult` records for
            this ``(model, dataset, category)`` cell.
        model_id: Display name for the model.
        backend: Backend identifier string.
        dataset: Dataset name (``"mvtec"``, ``"visa"``, …).
        category: Category name within the dataset.

    Returns:
        An :class:`~vlm_anomaly.schemas.EvalResult` with AUROC, F1,
        precision, recall, mean latency, and total cost filled in.
        Returns ``None`` for metrics that cannot be computed (e.g. AUROC
        when all labels are the same class).
    """
    if not rows:
        return EvalResult(model_id=model_id, backend=backend, dataset=dataset, category=category)

    labels = [r.sample_label for r in rows]
    scores = [r.prediction.confidence for r in rows]
    preds = [int(r.prediction.is_anomalous) for r in rows]

    auroc = _safe_auroc(labels, scores)
    f1, precision, recall = _safe_f1(labels, preds)

    return EvalResult(
        model_id=model_id,
        backend=backend,
        dataset=dataset,
        category=category,
        n_images=len(rows),
        auroc=auroc,
        f1=f1,
        precision=precision,
        recall=recall,
        mean_latency_ms=sum(r.prediction.latency_ms for r in rows) / len(rows),
        total_cost_usd=sum(r.prediction.cost_usd for r in rows),
    )


def _safe_auroc(labels: list[int], scores: list[float]) -> float | None:
    try:
        from sklearn.metrics import roc_auc_score

        if len(set(labels)) < 2:
            return None  # only one class — AUROC undefined
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None


def _safe_f1(
    labels: list[int], preds: list[int]
) -> tuple[float | None, float | None, float | None]:
    try:
        from sklearn.metrics import f1_score, precision_score, recall_score

        kw = {"zero_division": 0}
        return (
            float(f1_score(labels, preds, **kw)),
            float(precision_score(labels, preds, **kw)),
            float(recall_score(labels, preds, **kw)),
        )
    except Exception:
        return None, None, None
