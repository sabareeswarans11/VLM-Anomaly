"""VLM Evaluator — orchestrates backend + prompt + metrics for one experiment.

The evaluator:
1. Iterates :class:`~vlm_anomaly.datasets.base.AnomalySample` records from
   the dataset.
2. Renders a prompt via :class:`~vlm_anomaly.evaluators.prompt_library.PromptLibrary`.
3. Calls :meth:`~vlm_anomaly.backends.base.VLMBackend.predict`.
4. Records cost and checks the budget cap.
5. Flushes a :class:`~vlm_anomaly.schemas.PerImageResult` to disk after
   **every** image so partial runs are always recoverable.
6. Computes and returns an :class:`~vlm_anomaly.schemas.EvalResult` per category.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from vlm_anomaly.backends.base import VLMBackend
from vlm_anomaly.config import Settings, get_settings
from vlm_anomaly.datasets.base import AnomalyDataset
from vlm_anomaly.evaluators.base import compute_eval_result
from vlm_anomaly.evaluators.prompt_library import PromptLibrary
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import EvalResult, ExperimentConfig, PerImageResult
from vlm_anomaly.utils.cost_tracker import BudgetExceeded, CostTracker

log = get_logger(__name__)

# Conservative per-image cost estimate used for budget pre-check.
# Backends with known pricing use their own estimate; free backends use 0.
_FREE_BACKENDS = {"mock", "groq"}
_COST_ESTIMATE_USD = 0.05  # worst-case upper bound per image


class VLMEvaluator:
    """Run a zero-shot VLM anomaly evaluation.

    Args:
        backend: Any :class:`~vlm_anomaly.backends.base.VLMBackend` instance.
        dataset: Dataset implementing :class:`~vlm_anomaly.datasets.base.AnomalyDataset`.
        config: Experiment settings (categories, prompt key, limit, budget).
        settings: Application settings.  Defaults to :func:`~vlm_anomaly.config.get_settings`.
        prompt_library: Pre-built :class:`PromptLibrary`.  Built from the
            default prompts directory if omitted.
    """

    def __init__(
        self,
        backend: VLMBackend,
        dataset: AnomalyDataset,
        config: ExperimentConfig,
        settings: Settings | None = None,
        prompt_library: PromptLibrary | None = None,
    ) -> None:
        self.backend = backend
        self.dataset = dataset
        self.config = config
        self.settings = settings or get_settings()
        self.prompt_lib = prompt_library or PromptLibrary()

        budget = config.budget_usd or self.settings.default_budget_usd
        self.tracker = CostTracker(budget_usd=budget, name=config.backend)

        self.experiment_id = str(uuid.uuid4())[:8]
        self._results_dir = self.settings.results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[EvalResult]:
        """Execute the full evaluation and return per-category results.

        Iterates all configured categories (or all dataset categories when
        ``config.categories is None``), applies the per-category sample
        limit, and writes a ``results/{experiment_id}_{category}.jsonl``
        file incrementally.

        Returns:
            One :class:`~vlm_anomaly.schemas.EvalResult` per category.
        """
        categories = self.config.categories or self.dataset.categories()
        prompt_text = self.prompt_lib.render(self.config.prompt)

        log.info(
            "evaluator.run.start",
            experiment_id=self.experiment_id,
            backend=self.backend.name,
            dataset=self.dataset.name,
            categories=categories,
            prompt=self.config.prompt,
            limit=self.config.limit,
            budget_usd=self.tracker.budget_usd,
        )

        eval_results: list[EvalResult] = []

        for category in categories:
            result = self._run_category(category, prompt_text)
            eval_results.append(result)
            log.info(
                "evaluator.category.done",
                category=category,
                n_images=result.n_images,
                auroc=result.auroc,
                f1=result.f1,
                cost_usd=result.total_cost_usd,
            )

        log.info(
            "evaluator.run.done",
            experiment_id=self.experiment_id,
            total_cost_usd=self.tracker.total_usd,
            total_calls=self.tracker.calls,
        )
        return eval_results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_category(self, category: str, prompt_text: str) -> EvalResult:
        samples = self.dataset.samples(category, split="test")
        if self.config.limit:
            samples = samples[: self.config.limit]

        out_path = self._results_dir / f"{self.experiment_id}_{self.dataset.name}_{category}.jsonl"
        rows: list[PerImageResult] = []

        for sample in samples:
            # Budget pre-check
            estimate = 0.0 if self.backend.name in _FREE_BACKENDS else _COST_ESTIMATE_USD
            try:
                self.tracker.check_budget(estimate)
            except BudgetExceeded as exc:
                log.warning("evaluator.budget_exceeded", category=category, detail=str(exc))
                break

            try:
                prediction = self.backend.predict(sample.image_path, prompt_text)
            except Exception as exc:
                log.error("evaluator.predict.error", image=str(sample.image_path), error=str(exc))
                continue

            prediction = prediction.model_copy(update={"image_path": sample.image_path})
            self.tracker.record(prediction.cost_usd)

            row = PerImageResult(
                experiment_id=self.experiment_id,
                model_id=self.config.backend,
                backend=self.backend.name,
                dataset=self.dataset.name,
                category=category,
                sample_label=sample.label,
                prediction=prediction,
            )
            rows.append(row)
            _append_jsonl(out_path, row)

            log.debug(
                "evaluator.predict.ok",
                image=sample.image_path.name,
                is_anomalous=prediction.is_anomalous,
                confidence=round(prediction.confidence, 3),
                latency_ms=round(prediction.latency_ms),
                cost_usd=prediction.cost_usd,
            )

        return compute_eval_result(
            rows,
            model_id=self.config.backend,
            backend=self.backend.name,
            dataset=self.dataset.name,
            category=category,
        )


def _append_jsonl(path: Path, row: PerImageResult) -> None:
    """Append one JSON line to ``path`` (creates the file if needed)."""
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")
