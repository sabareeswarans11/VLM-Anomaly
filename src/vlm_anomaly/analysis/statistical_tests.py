"""Statistical tests for comparing model pairs.

McNemar test for paired binary predictions and bootstrap confidence
intervals for AUROC and F1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class McNemarResult:
    """Result of a McNemar paired significance test."""

    model_a: str
    model_b: str
    statistic: float
    p_value: float
    significant: bool  # p < 0.05


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a scalar metric."""

    metric: str
    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95


def mcnemar_test(
    labels: list[int],
    preds_a: list[int],
    preds_b: list[int],
    model_a: str = "A",
    model_b: str = "B",
    alpha: float = 0.05,
) -> McNemarResult:
    """McNemar test comparing two models on the same test set.

    Tests whether model A and model B make significantly different errors
    on the same paired images.

    Args:
        labels: Ground-truth binary labels (0/1).
        preds_a: Binary predictions from model A.
        preds_b: Binary predictions from model B.
        model_a: Display name for model A.
        model_b: Display name for model B.
        alpha: Significance level.

    Returns:
        :class:`McNemarResult` with the test statistic and p-value.
    """
    from scipy.stats import chi2

    n = len(labels)
    if n != len(preds_a) or n != len(preds_b):
        raise ValueError("labels, preds_a, preds_b must all have the same length")

    # Contingency: cases where only one model is correct
    b = sum(1 for gt, a, pb in zip(labels, preds_a, preds_b) if (a == gt) and (pb != gt))
    c = sum(1 for gt, a, pb in zip(labels, preds_a, preds_b) if (a != gt) and (pb == gt))

    if b + c == 0:
        return McNemarResult(model_a, model_b, 0.0, 1.0, False)

    # Edwards-corrected McNemar statistic
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - chi2.cdf(statistic, df=1))
    return McNemarResult(model_a, model_b, statistic, p_value, p_value < alpha)


def bootstrap_auroc_ci(
    labels: list[int],
    scores: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap confidence interval for AUROC.

    Args:
        labels: Ground-truth binary labels (0/1).
        scores: Predicted anomaly probability scores.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Desired CI coverage (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        :class:`BootstrapCI` with estimate and CI bounds.
    """

    rng = random.Random(seed)
    n = len(labels)
    aurocs: list[float] = []

    base_auroc = _safe_roc_auc(labels, scores)

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        s_labels = [labels[i] for i in indices]
        s_scores = [scores[i] for i in indices]
        val = _safe_roc_auc(s_labels, s_scores)
        if val is not None:
            aurocs.append(val)

    if not aurocs:
        return BootstrapCI("auroc", base_auroc or 0.0, 0.0, 1.0, confidence)

    arr = np.array(aurocs)
    alpha = 1 - confidence
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return BootstrapCI("auroc", base_auroc or float(arr.mean()), lower, upper, confidence)


def bootstrap_f1_ci(
    labels: list[int],
    preds: list[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """Bootstrap confidence interval for F1.

    Args:
        labels: Ground-truth binary labels.
        preds: Binary model predictions.
        n_bootstrap: Number of resamples.
        confidence: CI coverage.
        seed: Random seed.

    Returns:
        :class:`BootstrapCI` with estimate and CI bounds.
    """
    from sklearn.metrics import f1_score

    rng = random.Random(seed)
    n = len(labels)
    f1s: list[float] = []
    base_f1 = float(f1_score(labels, preds, zero_division=0))

    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        s_labels = [labels[i] for i in indices]
        s_preds = [preds[i] for i in indices]
        try:
            f1s.append(float(f1_score(s_labels, s_preds, zero_division=0)))
        except Exception:
            continue

    if not f1s:
        return BootstrapCI("f1", base_f1, 0.0, 1.0, confidence)

    arr = np.array(f1s)
    alpha = 1 - confidence
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return BootstrapCI("f1", base_f1, lower, upper, confidence)


def _safe_roc_auc(labels: list[int], scores: list[float]) -> float | None:
    from sklearn.metrics import roc_auc_score

    try:
        if len(set(labels)) < 2:
            return None
        return float(roc_auc_score(labels, scores))
    except Exception:
        return None
