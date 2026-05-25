"""Few-shot ensemble evaluator for Claude Opus (fewshot2-ens4).

Runs the 8 underperforming MVTec categories with:
  - 2 normal reference images injected into the (cached) system message
  - 4 category-specific ensemble prompts
  - Up to BATCH_SIZE test images per API call
  - Final anomaly score = mean confidence across 4 prompts

Supports mid-run resume: if a prior JSONL exists for a category, already-done
images are skipped and processing continues from where it left off.

Results use model_id ``"anthropic/claude-opus-4-7-fewshot2-ens4"`` so they
appear as a distinct row in the leaderboard alongside the vanilla baseline.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vlm_anomaly.backends.anthropic_backend import BATCH_SIZE, AnthropicBackend
from vlm_anomaly.config import Settings, get_settings
from vlm_anomaly.datasets.base import AnomalyDataset, AnomalySample
from vlm_anomaly.evaluators.base import compute_eval_result
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import AnomalyPrediction, EvalResult, PerImageResult
from vlm_anomaly.utils.cost_tracker import BudgetExceeded, CostTracker

log = get_logger(__name__)

MODEL_ID = "anthropic/claude-opus-4-7-fewshot2-ens4"
_DEFAULT_YAML = Path(__file__).parents[3] / "prompts" / "claude_opus_enhanced.yaml"


class FewShotEnsembleEvaluator:
    """Evaluate MVTec categories using few-shot normal references + prompt ensembles.

    Args:
        backend: An :class:`~vlm_anomaly.backends.anthropic_backend.AnthropicBackend`
            instance.  Must support ``predict_batch_with_refs()``.
        dataset: MVTec (or any) :class:`~vlm_anomaly.datasets.base.AnomalyDataset`.
        categories: List of category names to evaluate.  Must all appear in the
            enhanced YAML.
        enhanced_yaml: Path to ``prompts/claude_opus_enhanced.yaml``.
        budget_usd: Per-run budget cap in USD.  ``None`` = unlimited.
        limit: Max test images per category (``None`` = all).
        settings: Application settings.
    """

    def __init__(
        self,
        backend: AnthropicBackend,
        dataset: AnomalyDataset,
        categories: list[str],
        enhanced_yaml: Path | str = _DEFAULT_YAML,
        budget_usd: float | None = None,
        limit: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.backend = backend
        self.dataset = dataset
        self.categories = categories
        self.limit = limit
        self.settings = settings or get_settings()

        self.tracker = CostTracker(
            budget_usd=budget_usd, name="few_shot_ensemble"
        )  # None = unlimited
        self.experiment_id = str(uuid.uuid4())[:8]
        self._results_dir = self.settings.results_dir
        self._results_dir.mkdir(parents=True, exist_ok=True)

        with open(enhanced_yaml, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._config: dict[str, Any] = raw.get("categories", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[EvalResult]:
        """Evaluate all configured categories and return per-category results."""
        log.info(
            "few_shot_eval.run.start",
            experiment_id=self.experiment_id,
            model_id=MODEL_ID,
            categories=self.categories,
            budget_usd=self.tracker.budget_usd,
        )

        results: list[EvalResult] = []
        for category in self.categories:
            if category not in self._config:
                log.warning("few_shot_eval.category.not_in_yaml", category=category)
                continue
            result = self._run_category(category)
            results.append(result)

        log.info(
            "few_shot_eval.run.done",
            experiment_id=self.experiment_id,
            total_cost_usd=self.tracker.total_usd,
        )
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_category(self, category: str) -> EvalResult:
        cat_cfg = self._config[category]
        system_text: str = cat_cfg["system"].strip()
        prompts: list[dict] = cat_cfg["prompts"]
        ref_indices_spec: list = cat_cfg.get("ref_indices", [0, "mid"])

        # ── Reference images from train/good/ ────────────────────────────────
        ref_images = self._pick_ref_images(category, ref_indices_spec)
        if not ref_images:
            log.warning("few_shot_eval.no_refs_found", category=category)

        # ── All test samples ──────────────────────────────────────────────────
        samples = self.dataset.samples(category, split="test")
        if self.limit:
            samples = samples[: self.limit]

        # ── Resume: pick up where a prior partial run left off ────────────────
        prior_file, prior_rows = _find_prior_run(self._results_dir, category)
        done_paths: set[str] = {str(r.prediction.image_path) for r in prior_rows}
        rows: list[PerImageResult] = list(prior_rows)

        # Append to the existing file so the partial run stays in one place
        out_path = prior_file or (
            self._results_dir / f"{self.experiment_id}_mvtec_{category}_fewshot_ens.jsonl"
        )

        remaining = [s for s in samples if str(s.image_path) not in done_paths]
        total = len(samples)

        if prior_rows:
            print(
                f"  [{category}] resuming — {len(prior_rows)}/{total} done,"
                f" {len(remaining)} remaining"
            )
        else:
            print(f"  [{category}] starting — {total} images, batch_size={BATCH_SIZE}")

        # ── Process remaining in batches ──────────────────────────────────────
        for batch_start in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[batch_start : batch_start + BATCH_SIZE]

            try:
                self.tracker.check_budget(0.15 * len(prompts))
            except BudgetExceeded as exc:
                log.warning("few_shot_eval.budget_exceeded", category=category, detail=str(exc))
                break

            batch_rows = self._run_batch(batch, ref_images, system_text, prompts, category)
            rows.extend(batch_rows)
            for r in batch_rows:
                _append_jsonl(out_path, r)

            done_so_far = len(rows)
            print(
                f"    {category}  {done_so_far}/{total} images done"
                f"  (batch {batch_start // BATCH_SIZE + 1}/"
                f"{(len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE})"
            )

        result = compute_eval_result(
            rows,
            model_id=MODEL_ID,
            backend=self.backend.name,
            dataset=self.dataset.name,
            category=category,
        )

        auroc_str = f"{result.auroc:.4f}" if result.auroc is not None else "n/a"
        f1_str = f"{result.f1:.4f}" if result.f1 is not None else "n/a"
        print(
            f"\n  [{category}] DONE"
            f"  n={result.n_images}"
            f"  AUROC={auroc_str}"
            f"  F1={f1_str}"
            f"  cost=${result.total_cost_usd:.3f}\n"
        )
        log.info(
            "few_shot_eval.category.done",
            category=category,
            n_images=result.n_images,
            auroc=result.auroc,
            f1=result.f1,
            cost_usd=result.total_cost_usd,
        )
        return result

    def _run_batch(
        self,
        batch: list[AnomalySample],
        ref_images: list[Path],
        system_text: str,
        prompts: list[dict],
        category: str,
    ) -> list[PerImageResult]:
        """Run all 4 prompts for one batch and return ensemble predictions."""
        img_paths = [s.image_path for s in batch]
        n = len(img_paths)

        # Collect confidence scores per image across prompts: shape (n_prompts, n_images)
        all_confidences: list[list[float]] = []
        all_is_anomalous: list[list[bool]] = []
        all_descriptions: list[list[str]] = []
        all_defect_types: list[list[str | None]] = []
        total_cost = 0.0
        total_latency = 0.0
        last_raw_response = ""

        for prompt_cfg in prompts:
            prompt_text: str = prompt_cfg["text"].strip()
            try:
                preds: list[AnomalyPrediction] = self.backend.predict_batch_with_refs(
                    ref_images=ref_images,
                    images=img_paths,
                    system_text=system_text,
                    prompt=prompt_text,
                )
            except Exception as exc:
                log.error(
                    "few_shot_eval.batch_error",
                    category=category,
                    prompt_id=prompt_cfg.get("id"),
                    error=str(exc),
                )
                # Neutral fallback so the ensemble degrades gracefully
                preds = [
                    AnomalyPrediction(
                        image_path=p,
                        is_anomalous=False,
                        confidence=0.5,
                        parse_error=True,
                    )
                    for p in img_paths
                ]

            total_cost += sum(p.cost_usd for p in preds)
            total_latency += sum(p.latency_ms for p in preds)
            last_raw_response = preds[0].raw_response if preds else ""

            all_confidences.append([p.confidence for p in preds])
            all_is_anomalous.append([p.is_anomalous for p in preds])
            all_descriptions.append([p.description for p in preds])
            all_defect_types.append([p.defect_type for p in preds])

            self.tracker.record(sum(p.cost_usd for p in preds))

        # ── Ensemble: mean confidence, majority vote for is_anomalous ─────────
        conf_matrix = np.array(all_confidences)  # (n_prompts, n_images)
        ensemble_confidences = conf_matrix.mean(axis=0).tolist()

        anom_matrix = np.array(all_is_anomalous, dtype=int)
        ensemble_is_anomalous = (anom_matrix.sum(axis=0) > (len(prompts) / 2)).tolist()

        cost_per_img = total_cost / n
        latency_per_img = total_latency / n

        rows: list[PerImageResult] = []
        for i, sample in enumerate(batch):
            best_prompt_idx = int(conf_matrix[:, i].argmax())
            prediction = AnomalyPrediction(
                image_path=sample.image_path,
                is_anomalous=ensemble_is_anomalous[i],
                confidence=ensemble_confidences[i],
                description=all_descriptions[best_prompt_idx][i],
                defect_type=all_defect_types[best_prompt_idx][i],
                raw_response=last_raw_response,
                latency_ms=latency_per_img,
                cost_usd=cost_per_img,
            )
            rows.append(
                PerImageResult(
                    experiment_id=self.experiment_id,
                    model_id=MODEL_ID,
                    backend=self.backend.name,
                    dataset=self.dataset.name,
                    category=category,
                    sample_label=sample.label,
                    prediction=prediction,
                )
            )
            log.info(
                "few_shot_eval.image.done",
                image=sample.image_path.name,
                label=sample.label,
                confidence=round(ensemble_confidences[i], 3),
                is_anomalous=ensemble_is_anomalous[i],
            )

        return rows

    def _pick_ref_images(self, category: str, spec: list) -> list[Path]:
        """Resolve reference image indices into actual file paths."""
        try:
            train_good = sorted((self.dataset.root_dir / category / "train" / "good").glob("*.png"))
        except AttributeError:
            train_samples = self.dataset.samples(category, split="train")
            train_good = [s.image_path for s in train_samples]

        if not train_good:
            return []

        paths: list[Path] = []
        for entry in spec:
            idx = len(train_good) // 2 if entry == "mid" else int(entry)
            paths.append(train_good[min(idx, len(train_good) - 1)])
        return paths


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _find_prior_run(results_dir: Path, category: str) -> tuple[Path | None, list[PerImageResult]]:
    """Return (file_path, rows) for the most recent partial/complete run, or (None, [])."""
    for f in sorted(
        results_dir.glob(f"*_mvtec_{category}_fewshot_ens.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    ):
        if f.stat().st_size < 10:
            continue
        try:
            lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            if json.loads(lines[0]).get("model_id") != MODEL_ID:
                continue
            rows = [PerImageResult.model_validate_json(ln) for ln in lines]
            return f, rows
        except Exception:
            pass
    return None, []


def _append_jsonl(path: Path, row: PerImageResult) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(row.model_dump_json() + "\n")
