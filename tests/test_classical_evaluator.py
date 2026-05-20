"""Classical evaluator tests.

Unit tests check the evaluator wiring without Anomalib.
The @slow integration test requires the [classical] extra + MVTec data.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vlm_anomaly.evaluators.classical_evaluator import (
    ClassicalEvaluator,
    _parse_anomalib_results,
)
from vlm_anomaly.schemas import EvalResult


class TestClassicalEvaluatorUnit:
    def test_invalid_model_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            ClassicalEvaluator(model_name="invalid", dataset_name="mvtec", category="bottle")  # type: ignore[arg-type]

    def test_missing_anomalib_raises_import_error(self, tmp_path: Path) -> None:
        from vlm_anomaly.config import Settings
        settings = Settings(_env_file=str(tmp_path / "e.env"), results_dir=str(tmp_path))
        ev = ClassicalEvaluator("padim", "mvtec", "bottle", settings=settings)
        with patch.dict("sys.modules", {"anomalib": None}):
            with pytest.raises(ImportError, match="uv pip install"):
                ev.run()

    def test_parse_anomalib_results_image_auroc(self) -> None:
        metrics = [{"image_AUROC": 0.93, "image_F1Score": 0.88, "num_samples": 83}]
        r = _parse_anomalib_results(metrics, "padim", "mvtec", "bottle", 30000.0)
        assert isinstance(r, EvalResult)
        assert r.auroc == pytest.approx(0.93)
        assert r.f1 == pytest.approx(0.88)
        assert r.backend == "anomalib"

    def test_parse_anomalib_results_fallback_keys(self) -> None:
        metrics = [{"AUROC": 0.91, "F1Score": 0.85}]
        r = _parse_anomalib_results(metrics, "patchcore", "mvtec", "cable", 60000.0)
        assert r.auroc == pytest.approx(0.91)

    def test_parse_anomalib_results_empty(self) -> None:
        r = _parse_anomalib_results([], "padim", "mvtec", "bottle", 0.0)
        assert r.auroc is None
        assert r.n_images == 0

    def test_result_cost_always_zero(self) -> None:
        metrics = [{"image_AUROC": 0.9}]
        r = _parse_anomalib_results(metrics, "padim", "mvtec", "bottle", 1000.0)
        assert r.total_cost_usd == 0.0

    def test_flush_writes_json(self, tmp_path: Path) -> None:
        from vlm_anomaly.config import Settings
        import json
        settings = Settings(_env_file=str(tmp_path / "e.env"), results_dir=str(tmp_path))
        ev = ClassicalEvaluator("padim", "mvtec", "bottle", settings=settings)
        result = EvalResult(
            model_id="padim", backend="anomalib", dataset="mvtec",
            category="bottle", auroc=0.92, f1=0.88, n_images=83,
        )
        ev._flush_result(result)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data[0]["auroc"] == pytest.approx(0.92)


@pytest.mark.slow
def test_padim_trains_and_evaluates() -> None:
    """Full PaDiM run on MVTec bottle — requires [classical] + downloaded data."""
    try:
        import anomalib  # noqa: F401
    except ImportError:
        pytest.skip("anomalib not installed — run: uv pip install -e '.[classical]'")

    from vlm_anomaly.config import get_settings
    settings = get_settings()
    data_dir = settings.data_dir / "mvtec" / "bottle"
    if not data_dir.exists():
        pytest.skip("MVTec bottle not found — run: bash scripts/download_mvtec.sh")

    ev = ClassicalEvaluator("padim", "mvtec", "bottle", image_size=256)
    result = ev.run()
    assert result.auroc is not None
    assert result.auroc > 0.5, f"Expected AUROC > 0.5, got {result.auroc}"
