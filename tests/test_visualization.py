"""Tests for visualization.plots, visualization.interactive, and report_generator."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vlm_anomaly.analysis.report_generator import generate
from vlm_anomaly.visualization.interactive import (
    heatmap_json,
    leaderboard_json,
    write_explorer_payload,
)
from vlm_anomaly.visualization.plots import (
    auroc_bar_chart,
    category_heatmap_plot,
    cost_vs_accuracy_scatter,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_lb() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "padim",
                "backend": "anomalib",
                "dataset": "mvtec",
                "category": "bottle",
                "n_images": 83,
                "auroc": 0.92,
                "f1": 0.88,
                "mean_latency_ms": 15.0,
                "total_cost_usd": 0.0,
            },
            {
                "model_id": "patchcore",
                "backend": "anomalib",
                "dataset": "mvtec",
                "category": "bottle",
                "n_images": 83,
                "auroc": 0.95,
                "f1": 0.91,
                "mean_latency_ms": 25.0,
                "total_cost_usd": 0.0,
            },
            {
                "model_id": "mock-vlm",
                "backend": "mock",
                "dataset": "mvtec",
                "category": "cable",
                "n_images": 58,
                "auroc": 0.78,
                "f1": 0.70,
                "mean_latency_ms": 1200.0,
                "total_cost_usd": 0.18,
            },
            {
                "model_id": "mock-vlm",
                "backend": "mock",
                "dataset": "mvtec",
                "category": "bottle",
                "n_images": 83,
                "auroc": 0.75,
                "f1": 0.68,
                "mean_latency_ms": 1100.0,
                "total_cost_usd": 0.25,
            },
        ]
    )


@pytest.fixture
def sample_pivot(sample_lb: pd.DataFrame) -> pd.DataFrame:
    return sample_lb.pivot_table(
        index="model_id", columns="category", values="auroc", aggfunc="mean"
    )


# ---------------------------------------------------------------------------
# visualization.plots
# ---------------------------------------------------------------------------


class TestPlots:
    def test_auroc_bar_chart_creates_files(self, sample_lb: pd.DataFrame, tmp_path: Path) -> None:
        png, svg = auroc_bar_chart(sample_lb, tmp_path)
        assert png.exists()
        assert svg.exists()
        assert png.suffix == ".png"
        assert svg.suffix == ".svg"

    def test_auroc_bar_chart_file_nonempty(self, sample_lb: pd.DataFrame, tmp_path: Path) -> None:
        png, _ = auroc_bar_chart(sample_lb, tmp_path)
        assert png.stat().st_size > 1000

    def test_cost_scatter_creates_files(self, sample_lb: pd.DataFrame, tmp_path: Path) -> None:
        png, svg = cost_vs_accuracy_scatter(sample_lb, tmp_path)
        assert png.exists() and svg.exists()

    def test_category_heatmap_creates_files(
        self, sample_pivot: pd.DataFrame, tmp_path: Path
    ) -> None:
        png, svg = category_heatmap_plot(sample_pivot, tmp_path)
        assert png.exists() and svg.exists()

    def test_plots_dir_created_if_missing(self, sample_lb: pd.DataFrame, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "plots"
        auroc_bar_chart(sample_lb, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# visualization.interactive
# ---------------------------------------------------------------------------


class TestInteractive:
    def test_leaderboard_json_returns_list(self, sample_lb: pd.DataFrame) -> None:
        records = leaderboard_json(sample_lb)
        assert isinstance(records, list)
        assert len(records) == len(sample_lb)

    def test_leaderboard_json_floats_rounded(self, sample_lb: pd.DataFrame) -> None:
        records = leaderboard_json(sample_lb)
        for rec in records:
            if rec.get("auroc") is not None:
                assert len(str(rec["auroc"]).split(".")[-1]) <= 4

    def test_leaderboard_json_serializable(self, sample_lb: pd.DataFrame) -> None:
        records = leaderboard_json(sample_lb)
        json.dumps(records)  # must not raise

    def test_heatmap_json_structure(self, sample_pivot: pd.DataFrame) -> None:
        data = heatmap_json(sample_pivot)
        assert "models" in data
        assert "categories" in data
        assert "cells" in data

    def test_heatmap_json_cells_have_required_keys(self, sample_pivot: pd.DataFrame) -> None:
        data = heatmap_json(sample_pivot)
        for cell in data["cells"]:
            assert "model" in cell
            assert "category" in cell
            assert "auroc" in cell

    def test_write_explorer_payload(
        self, sample_lb: pd.DataFrame, sample_pivot: pd.DataFrame, tmp_path: Path
    ) -> None:
        out = tmp_path / "explorer_data.json"
        result = write_explorer_payload(sample_lb, sample_pivot, out)
        assert result == out
        assert out.exists()
        data = json.loads(out.read_text())
        assert "leaderboard" in data
        assert "heatmap" in data


# ---------------------------------------------------------------------------
# analysis.report_generator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    @pytest.fixture
    def results_dir(self, tmp_path: Path, sample_lb: pd.DataFrame) -> Path:
        """Write a minimal JSON results file so the generator has data."""
        records = sample_lb.to_dict(orient="records")
        (tmp_path / "classical.json").write_text(json.dumps(records))
        return tmp_path

    def test_generate_creates_report(self, results_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        p = generate(str(results_dir), str(out))
        assert p == out
        assert out.exists()

    def test_report_nonempty(self, results_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        generate(str(results_dir), str(out))
        assert out.stat().st_size > 500

    def test_report_contains_leaderboard_header(self, results_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        generate(str(results_dir), str(out))
        content = out.read_text()
        assert "Leaderboard" in content

    def test_report_contains_punchline(self, results_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        generate(str(results_dir), str(out))
        content = out.read_text()
        assert "zero training data" in content

    def test_plots_created(self, results_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        generate(str(results_dir), str(out))
        plots = list(results_dir.glob("plots/*.png"))
        assert len(plots) >= 2  # at least bar chart + scatter

    def test_generate_empty_results_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_results"
        empty.mkdir()
        out = tmp_path / "REPORT.md"
        generate(str(empty), str(out))
        assert out.exists()
        assert "No results yet" in out.read_text()

    def test_explorer_json_created(self, results_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "REPORT.md"
        generate(str(results_dir), str(out))
        explorer = results_dir / "explorer_data.json"
        assert explorer.exists()
