"""Anomalib wrapper for classical anomaly-detection baselines.

Wraps PaDiM, PatchCore, EfficientAD, and STFPM from the Anomalib library
behind the same ``EvalResult`` schema as the VLM evaluator so both appear
in one unified leaderboard.

Requires the ``[classical]`` extra::

    uv pip install -e ".[classical]"

All models run on CPU by default (``accelerator="cpu"``).
Training takes 2–10 minutes per category on a laptop.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Literal

from vlm_anomaly.config import Settings, get_settings
from vlm_anomaly.logging import get_logger
from vlm_anomaly.schemas import EvalResult

log = get_logger(__name__)

ModelName = Literal["padim", "patchcore", "efficientad", "stfpm"]

_MODEL_NAMES: set[str] = {"padim", "patchcore", "efficientad", "stfpm"}


def best_accelerator() -> str:
    """Return the best accelerator Lightning supports on this machine.

    Uses Lightning's own accelerator checks rather than PyTorch's — on Intel
    Macs with AMD GPUs, ``torch.backends.mps.is_available()`` returns True but
    Lightning's ``MPSAccelerator`` rejects it (Apple Silicon only).
    """
    try:
        from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator

        if CUDAAccelerator.is_available():
            return "gpu"
        if MPSAccelerator.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class ClassicalEvaluator:
    """Run a classical Anomalib baseline and return an ``EvalResult``.

    Args:
        model_name: One of ``padim``, ``patchcore``, ``efficientad``, ``stfpm``.
        dataset_name: ``"mvtec"`` or ``"visa"``.
        category: Dataset category (e.g. ``"bottle"``).
        image_size: Resize images to this square size before training/eval.
        settings: App settings; defaults to ``get_settings()``.
    """

    def __init__(
        self,
        model_name: ModelName,
        dataset_name: Literal["mvtec", "visa"],
        category: str,
        image_size: int = 256,
        accelerator: str = "cpu",
        settings: Settings | None = None,
    ) -> None:
        if model_name not in _MODEL_NAMES:
            raise ValueError(f"Unknown model {model_name!r}. Valid: {sorted(_MODEL_NAMES)}")
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.category = category
        self.image_size = image_size
        self.accelerator = accelerator
        self.settings = settings or get_settings()
        self.experiment_id = str(uuid.uuid4())[:8]

    def run(self) -> EvalResult:
        """Train and evaluate the model on the given category.

        Returns:
            ``EvalResult`` with AUROC, F1, precision, recall, and latency.

        Raises:
            ImportError: If ``anomalib`` is not installed.
            FileNotFoundError: If the dataset root does not exist.
        """
        _check_anomalib()

        from anomalib.engine import Engine

        data_root = self.settings.data_dir
        model = _build_model(self.model_name)
        datamodule = _build_datamodule(self.dataset_name, self.category, data_root, self.image_size)

        log.info(
            "classical.run.start",
            model=self.model_name,
            dataset=self.dataset_name,
            category=self.category,
            image_size=self.image_size,
            accelerator=self.accelerator,
        )

        # WideResNet50 (PatchCore backbone) has hundreds of nested modules.
        # Rich's tree renderer hits Python's default recursion limit (1000)
        # in two callbacks: RichModelSummary and RichProgressBar.
        # Disable both, and raise the limit as a safety net for any other
        # rich-based renderer that might walk the module tree.
        engine = Engine(
            accelerator=self.accelerator,
            devices=1,
            max_epochs=1,
            enable_model_summary=False,
            enable_progress_bar=False,
        )

        _saved_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(5000)
        try:
            t0 = time.perf_counter()
            engine.fit(model, datamodule=datamodule)
            results_list = engine.test(model, datamodule=datamodule, verbose=False)
        finally:
            sys.setrecursionlimit(_saved_limit)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        eval_result = _parse_anomalib_results(
            results_list,
            model_id=f"classical/{self.model_name}",
            dataset=self.dataset_name,
            category=self.category,
            elapsed_ms=elapsed_ms,
        )

        self._flush_result(eval_result)

        log.info(
            "classical.run.done",
            model=self.model_name,
            category=self.category,
            auroc=eval_result.auroc,
            f1=eval_result.f1,
            elapsed_s=round(elapsed_ms / 1000, 1),
        )
        return eval_result

    def _flush_result(self, result: EvalResult) -> None:
        out_dir = self.settings.results_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = (
            out_dir
            / f"{self.experiment_id}_{self.dataset_name}_{self.category}_{self.model_name}.json"
        )
        path.write_text(json.dumps([result.model_dump()], indent=2))
        log.info("classical.result.written", path=str(path))


# ---------------------------------------------------------------------------
# Anomalib helpers
# ---------------------------------------------------------------------------


def _check_anomalib() -> None:
    try:
        import anomalib  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Anomalib is required for classical baselines.\n"
            "Install it with: uv pip install -e '.[classical]'"
        ) from exc


def _build_model(name: str):
    if name == "padim":
        from anomalib.models import Padim

        return Padim()
    if name == "patchcore":
        from anomalib.models import Patchcore

        return Patchcore()
    if name == "efficientad":
        from anomalib.models import EfficientAd

        return EfficientAd()
    if name == "stfpm":
        from anomalib.models import Stfpm

        return Stfpm()
    raise ValueError(f"Unknown model: {name!r}")


def _build_datamodule(dataset_name: str, category: str, data_root: Path, image_size: int):
    # anomalib 2.x uses MVTecAD (not MVTec) and no image_size param.
    # Resize is handled via augmentations.
    import torchvision.transforms.v2 as T  # noqa: N812

    aug = T.Resize((image_size, image_size), antialias=True)

    if dataset_name == "mvtec":
        from anomalib.data import MVTecAD

        return MVTecAD(
            root=str(data_root / "mvtec"),
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
            augmentations=aug,
        )
    if dataset_name == "visa":
        from anomalib.data import VisaAD  # type: ignore[attr-defined]

        return VisaAD(
            root=str(data_root / "visa"),
            category=category,
            train_batch_size=32,
            eval_batch_size=32,
            num_workers=0,
            augmentations=aug,
        )
    raise ValueError(f"Unknown dataset: {dataset_name!r}")


def _parse_anomalib_results(
    results_list: list[dict],
    model_id: str,
    dataset: str,
    category: str,
    elapsed_ms: float,
) -> EvalResult:
    """Extract metrics from Anomalib's test() output dict."""
    metrics: dict = results_list[0] if results_list else {}

    def _get(key: str) -> float | None:
        # Anomalib prefixes metrics with "test/" or "image_" depending on version
        for candidate in [key, f"test/{key}", f"image_{key}", f"pixel_{key}"]:
            val = metrics.get(candidate)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return None

    return EvalResult(
        model_id=model_id,
        backend="anomalib",
        dataset=dataset,
        category=category,
        n_images=int(
            metrics.get("test/num_samples")
            or metrics.get("num_samples")
            or metrics.get("test/dataset_size")
            or 0
        ),
        auroc=_get("image_AUROC") or _get("AUROC") or _get("auroc"),
        f1=_get("image_F1Score") or _get("F1Score") or _get("f1"),
        precision=_get("image_Precision") or _get("Precision"),
        recall=_get("image_Recall") or _get("Recall"),
        mean_latency_ms=elapsed_ms,
        total_cost_usd=0.0,
    )
