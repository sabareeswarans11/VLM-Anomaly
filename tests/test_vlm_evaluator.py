"""VLM Evaluator and Prompt Library tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_anomaly.backends.mock import MockVLMBackend
from vlm_anomaly.evaluators.prompt_library import PromptLibrary
from vlm_anomaly.evaluators.vlm_evaluator import VLMEvaluator
from vlm_anomaly.schemas import EvalResult, ExperimentConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _touch(path: Path, content: bytes = b"PNG") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


@pytest.fixture
def tiny_mvtec(tmp_path: Path) -> Path:
    """Two normal + two anomaly test images, one category."""
    root = tmp_path / "mvtec"
    cat = root / "bottle"
    for i in range(2):
        _touch(cat / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        _touch(cat / "test" / "broken_large" / f"{i:03d}.png")
        _touch(cat / "ground_truth" / "broken_large" / f"{i:03d}_mask.png")
    # Need train dir too for completeness
    _touch(cat / "train" / "good" / "000.png")
    return root


@pytest.fixture
def mock_backend() -> MockVLMBackend:
    return MockVLMBackend()


@pytest.fixture
def prompt_lib(tmp_path: Path) -> PromptLibrary:
    """Minimal prompt library with a single 'test' name."""
    p = tmp_path / "prompts" / "test.yaml"
    p.parent.mkdir(parents=True)
    p.write_text("name: test\nvariants:\n  simple: Is there a defect? Reply JSON.\n")
    return PromptLibrary(prompts_dir=tmp_path / "prompts")


@pytest.fixture
def mvtec_dataset(tiny_mvtec: Path):
    from vlm_anomaly.datasets.mvtec import MVTec

    return MVTec(root_dir=tiny_mvtec)


@pytest.fixture
def base_config() -> ExperimentConfig:
    return ExperimentConfig(
        backend="mock",
        dataset="mvtec",
        categories=["bottle"],
        prompt="test.simple",
        limit=None,
        budget_usd=None,
    )


# ---------------------------------------------------------------------------
# PromptLibrary
# ---------------------------------------------------------------------------


class TestPromptLibrary:
    def test_loads_real_prompts_dir(self) -> None:
        lib = PromptLibrary()
        assert "generic.simple" in lib.available_keys()
        assert "generic.detailed" in lib.available_keys()
        assert "manufacturing.simple" in lib.available_keys()

    def test_render_returns_string(self) -> None:
        lib = PromptLibrary()
        text = lib.render("generic.simple")
        assert isinstance(text, str)
        assert len(text) > 10

    def test_render_strips_trailing_whitespace(self) -> None:
        lib = PromptLibrary()
        text = lib.render("generic.simple")
        assert text == text.rstrip()

    def test_render_invalid_name_raises(self) -> None:
        lib = PromptLibrary()
        with pytest.raises(ValueError, match="not found"):
            lib.render("nonexistent.simple")

    def test_render_invalid_variant_raises(self) -> None:
        lib = PromptLibrary()
        with pytest.raises(ValueError, match="Variant"):
            lib.render("generic.nonexistent")

    def test_render_missing_dot_raises(self) -> None:
        lib = PromptLibrary()
        with pytest.raises(ValueError, match="<name>.<variant>"):
            lib.render("generic")

    def test_available_keys_sorted(self) -> None:
        lib = PromptLibrary()
        keys = lib.available_keys()
        assert keys == sorted(keys)

    def test_available_names(self) -> None:
        lib = PromptLibrary()
        names = lib.available_names()
        assert "generic" in names
        assert "manufacturing" in names

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PromptLibrary(prompts_dir=tmp_path / "nonexistent")

    def test_custom_dir(self, prompt_lib: PromptLibrary) -> None:
        assert "test.simple" in prompt_lib.available_keys()
        assert "Is there a defect" in prompt_lib.render("test.simple")

    def test_all_real_prompts_render(self) -> None:
        lib = PromptLibrary()
        for key in lib.available_keys():
            text = lib.render(key)
            assert isinstance(text, str) and len(text) > 0


# ---------------------------------------------------------------------------
# VLMEvaluator — basic runs
# ---------------------------------------------------------------------------


class TestVLMEvaluator:
    def _make_evaluator(
        self,
        backend,
        dataset,
        config,
        prompt_lib,
        results_dir: Path,
    ) -> VLMEvaluator:
        from vlm_anomaly.config import Settings

        # Use field names (not env-var names) as constructor kwargs.
        settings = Settings(
            _env_file=str(results_dir / "empty.env"),
            results_dir=str(results_dir),
        )
        return VLMEvaluator(
            backend=backend,
            dataset=dataset,
            config=config,
            settings=settings,
            prompt_library=prompt_lib,
        )

    def test_run_returns_eval_results(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        results = ev.run()
        assert len(results) == 1
        assert isinstance(results[0], EvalResult)

    def test_run_correct_category(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        results = ev.run()
        assert results[0].category == "bottle"

    def test_run_n_images_correct(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        # 2 normal + 2 anomaly = 4 images
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        results = ev.run()
        assert results[0].n_images == 4

    def test_run_with_limit(self, mock_backend, mvtec_dataset, prompt_lib, tmp_path) -> None:
        config = ExperimentConfig(
            backend="mock",
            dataset="mvtec",
            categories=["bottle"],
            prompt="test.simple",
            limit=2,
        )
        ev = self._make_evaluator(mock_backend, mvtec_dataset, config, prompt_lib, tmp_path)
        results = ev.run()
        assert results[0].n_images == 2

    def test_results_jsonl_written(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        ev.run()
        jsonl_files = list(tmp_path.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        lines = jsonl_files[0].read_text().strip().splitlines()
        assert len(lines) == 4  # one line per image

    def test_jsonl_lines_are_valid_json(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        import json

        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        ev.run()
        for line in list(tmp_path.glob("*.jsonl"))[0].read_text().splitlines():
            data = json.loads(line)
            assert "prediction" in data
            assert "sample_label" in data

    def test_cost_tracker_records(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        ev.run()
        # Mock backend cost is 0 — tracker still records calls
        assert ev.tracker.calls == 4

    def test_metrics_populated(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        results = ev.run()
        r = results[0]
        # With 2 normal + 2 anomaly labels, AUROC should be computable
        assert r.auroc is not None
        assert 0.0 <= r.auroc <= 1.0

    def test_f1_populated(
        self, mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path
    ) -> None:
        ev = self._make_evaluator(mock_backend, mvtec_dataset, base_config, prompt_lib, tmp_path)
        results = ev.run()
        r = results[0]
        assert r.f1 is not None


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestVLMEvaluatorBudget:
    def test_respects_budget_zero_stops_immediately(
        self, mock_backend, mvtec_dataset, prompt_lib, tmp_path
    ) -> None:
        from vlm_anomaly.config import Settings

        config = ExperimentConfig(
            backend="mock",
            dataset="mvtec",
            categories=["bottle"],
            prompt="test.simple",
            budget_usd=100.0,  # high enough for mock
        )
        settings = Settings(
            _env_file=str(tmp_path / "empty.env"),
            VLM_ANOMALY_RESULTS_DIR=str(tmp_path),
        )
        settings = Settings(
            _env_file=str(tmp_path / "empty.env"),
            results_dir=str(tmp_path),
        )
        ev = VLMEvaluator(
            backend=mock_backend,
            dataset=mvtec_dataset,
            config=config,
            settings=settings,
            prompt_library=prompt_lib,
        )
        # Push total past the budget so the very first check_budget fires.
        ev.tracker._total = 100.001
        results = ev.run()
        # Budget pre-check fires before any image → 0 images processed.
        assert results[0].n_images == 0
