"""Aggregator and statistical tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_anomaly.analysis.aggregator import (
    category_heatmap,
    cost_accuracy_table,
    leaderboard,
)
from vlm_anomaly.analysis.statistical_tests import (
    BootstrapCI,
    McNemarResult,
    bootstrap_auroc_ci,
    bootstrap_f1_ci,
    mcnemar_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def results_dir_precomputed(tmp_path: Path) -> Path:
    """Pre-aggregated EvalResult JSON (classical baseline style)."""
    data = [
        {"model_id": "padim", "backend": "anomalib", "dataset": "mvtec",
         "category": "bottle", "n_images": 83, "auroc": 0.92, "f1": 0.88,
         "mean_latency_ms": 15.0, "total_cost_usd": 0.0},
        {"model_id": "patchcore", "backend": "anomalib", "dataset": "mvtec",
         "category": "bottle", "n_images": 83, "auroc": 0.95, "f1": 0.91,
         "mean_latency_ms": 25.0, "total_cost_usd": 0.0},
        {"model_id": "mock-vlm", "backend": "mock", "dataset": "mvtec",
         "category": "cable", "n_images": 58, "auroc": 0.78, "f1": 0.70,
         "mean_latency_ms": 1200.0, "total_cost_usd": 0.18},
    ]
    (tmp_path / "classical.json").write_text(json.dumps(data))
    return tmp_path


@pytest.fixture
def results_dir_jsonl(tmp_path: Path) -> Path:
    """JSONL per-image records for one VLM run."""
    from vlm_anomaly.schemas import AnomalyPrediction, PerImageResult

    rows = []
    for i in range(4):
        label = i % 2
        confidence = 0.9 if label == 1 else 0.1
        pred = AnomalyPrediction(
            is_anomalous=bool(label),
            confidence=confidence,
            latency_ms=1.0,
            cost_usd=0.002,
        )
        row = PerImageResult(
            experiment_id="test-exp",
            model_id="mock",
            backend="mock",
            dataset="mvtec",
            category="bottle",
            sample_label=label,
            prediction=pred,
        )
        rows.append(row)

    out = tmp_path / "test-exp_mvtec_bottle.jsonl"
    out.write_text("\n".join(r.model_dump_json() for r in rows))
    return tmp_path


# ---------------------------------------------------------------------------
# Aggregator — precomputed JSON
# ---------------------------------------------------------------------------

class TestLeaderboardPrecomputed:
    def test_returns_dataframe(self, results_dir_precomputed: Path) -> None:
        df = leaderboard(results_dir_precomputed)
        assert not df.empty

    def test_sorted_by_auroc_desc(self, results_dir_precomputed: Path) -> None:
        df = leaderboard(results_dir_precomputed)
        assert df["auroc"].is_monotonic_decreasing

    def test_expected_columns(self, results_dir_precomputed: Path) -> None:
        df = leaderboard(results_dir_precomputed)
        for col in ["model_id", "auroc", "f1", "dataset", "category"]:
            assert col in df.columns

    def test_patchcore_ranks_first(self, results_dir_precomputed: Path) -> None:
        df = leaderboard(results_dir_precomputed)
        assert df.iloc[0]["model_id"] == "patchcore"

    def test_empty_dir_returns_empty_df(self, tmp_path: Path) -> None:
        df = leaderboard(tmp_path)
        assert df.empty

    def test_category_heatmap(self, results_dir_precomputed: Path) -> None:
        hm = category_heatmap(results_dir_precomputed)
        assert not hm.empty
        assert "bottle" in hm.columns or "cable" in hm.columns

    def test_cost_accuracy_table(self, results_dir_precomputed: Path) -> None:
        df = cost_accuracy_table(results_dir_precomputed)
        assert "mean_auroc" in df.columns
        assert "mean_cost_per_image" in df.columns


# ---------------------------------------------------------------------------
# Aggregator — JSONL re-aggregation
# ---------------------------------------------------------------------------

class TestLeaderboardJsonl:
    def test_reads_jsonl(self, results_dir_jsonl: Path) -> None:
        df = leaderboard(results_dir_jsonl)
        assert not df.empty

    def test_auroc_computed(self, results_dir_jsonl: Path) -> None:
        df = leaderboard(results_dir_jsonl)
        row = df[df["model_id"] == "mock"].iloc[0]
        assert row["auroc"] is not None
        assert 0.0 <= row["auroc"] <= 1.0

    def test_n_images_correct(self, results_dir_jsonl: Path) -> None:
        df = leaderboard(results_dir_jsonl)
        row = df[df["model_id"] == "mock"].iloc[0]
        assert row["n_images"] == 4

    def test_sample_fixture_json_skipped(self, tmp_path: Path) -> None:
        # The fixture sample_results.json should be skipped (it has 'sample' in the name)
        (tmp_path / "sample_results.json").write_text(json.dumps([
            {"model_id": "x", "backend": "y", "dataset": "z", "category": "w",
             "n_images": 1, "auroc": 0.5, "f1": 0.5,
             "mean_latency_ms": 1.0, "total_cost_usd": 0.0}
        ]))
        df = leaderboard(tmp_path)
        assert df.empty  # sample_results.json is excluded


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestMcNemar:
    def _make_data(self):
        labels = [1, 1, 0, 0, 1, 0, 1, 0]
        # A is always right, B makes mistakes
        preds_a = labels[:]
        preds_b = [1 - x for x in labels]
        return labels, preds_a, preds_b

    def test_returns_mcnemar_result(self) -> None:
        labels, preds_a, preds_b = self._make_data()
        r = mcnemar_test(labels, preds_a, preds_b, "A", "B")
        assert isinstance(r, McNemarResult)

    def test_perfect_vs_inverse_is_significant(self) -> None:
        labels, preds_a, preds_b = self._make_data()
        r = mcnemar_test(labels, preds_a, preds_b)
        assert r.significant is True

    def test_identical_preds_not_significant(self) -> None:
        labels = [1, 0, 1, 0]
        preds = [1, 0, 1, 0]
        r = mcnemar_test(labels, preds, preds)
        assert r.p_value == pytest.approx(1.0)
        assert r.significant is False

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            mcnemar_test([1, 0], [1], [0, 1])


class TestBootstrapCI:
    def _data(self):
        labels = [1, 1, 0, 0] * 10
        scores = [0.9, 0.8, 0.2, 0.1] * 10
        preds  = [1, 1, 0, 0] * 10
        return labels, scores, preds

    def test_auroc_ci_returns_bootstrap_ci(self) -> None:
        labels, scores, _ = self._data()
        ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=100)
        assert isinstance(ci, BootstrapCI)
        assert ci.metric == "auroc"

    def test_auroc_ci_bounds_ordered(self) -> None:
        labels, scores, _ = self._data()
        ci = bootstrap_auroc_ci(labels, scores, n_bootstrap=100)
        assert ci.lower <= ci.estimate <= ci.upper

    def test_f1_ci_returns_bootstrap_ci(self) -> None:
        labels, _, preds = self._data()
        ci = bootstrap_f1_ci(labels, preds, n_bootstrap=100)
        assert isinstance(ci, BootstrapCI)
        assert ci.metric == "f1"

    def test_f1_ci_bounds_in_range(self) -> None:
        labels, _, preds = self._data()
        ci = bootstrap_f1_ci(labels, preds, n_bootstrap=100)
        assert 0.0 <= ci.lower <= ci.upper <= 1.0

    def test_reproducible_with_seed(self) -> None:
        labels, scores, _ = self._data()
        ci1 = bootstrap_auroc_ci(labels, scores, seed=42)
        ci2 = bootstrap_auroc_ci(labels, scores, seed=42)
        assert ci1.lower == ci2.lower
        assert ci1.upper == ci2.upper
