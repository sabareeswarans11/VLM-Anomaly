"""InfraAD — custom infrastructure inspection subset.

This dataset covers telecom tower, antenna, cable tray, junction box,
and pipeline images for field-inspection use cases (AT&T / utility context).

Layout (same convention as MVTec-style VisA):

    {root_dir}/
    └── {category}/
        ├── train/
        │   └── good/
        ├── test/
        │   ├── good/
        │   └── {defect}/   # corrosion | loose_hardware | cracked_weld | …
        └── ground_truth/
            └── {defect}/

Categories are NOT hardcoded — they are discovered from ``root_dir`` at
runtime, making it easy to add new equipment types without code changes.

This loader is a placeholder for task 03.  Full implementation — including
a contribution guide for adding new infrastructure categories — is slated
for a later milestone once the base benchmark is complete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vlm_anomaly.config import get_settings
from vlm_anomaly.datasets.base import AnomalyDataset, AnomalySample
from vlm_anomaly.logging import get_logger

log = get_logger(__name__)


class InfraAD(AnomalyDataset):
    """Loader for the custom InfraAD infrastructure inspection subset.

    Args:
        root_dir: Path containing the per-category subdirectories.
            Defaults to ``{data_dir}/infra`` from
            :class:`~vlm_anomaly.config.Settings`.
    """

    name = "infra"

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is None:
            root_dir = get_settings().data_dir / "infra"
        self.root_dir = Path(root_dir).resolve()

    def categories(self) -> list[str]:
        """Return category names discovered from ``root_dir``.

        Returns an empty list if ``root_dir`` does not yet exist.
        """
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def samples(
        self, category: str, split: Literal["train", "test"] = "test"
    ) -> list[AnomalySample]:
        """Return samples for ``category``/``split`` using the MVTec-style layout.

        Raises:
            ValueError: If ``category`` is not found under ``root_dir``.
            FileNotFoundError: If ``root_dir`` does not exist.
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"InfraAD root not found: {self.root_dir}. "
                "Add your infrastructure images following the layout in this module's docstring."
            )

        cat_dir = self.root_dir / category
        if not cat_dir.exists():
            raise ValueError(
                f"Category {category!r} not found in {self.root_dir}. "
                f"Available: {self.categories()}"
            )

        return _load_mvtec_style(cat_dir, category, split)

    def download(self) -> None:
        """InfraAD is a private/custom dataset — no automated download."""
        log.warning(
            "infra.download.noop",
            message=(
                "InfraAD is a custom dataset. Populate root_dir manually "
                "following the layout documented in datasets/infra.py."
            ),
            root_dir=str(self.root_dir),
        )


def _load_mvtec_style(
    cat_dir: Path, category: str, split: Literal["train", "test"]
) -> list[AnomalySample]:
    samples: list[AnomalySample] = []

    if split == "train":
        good_dir = cat_dir / "train" / "good"
        if good_dir.exists():
            for p in sorted(good_dir.glob("*.png")):
                samples.append(AnomalySample(image_path=p, label=0, category=category))
        return samples

    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    if not test_dir.exists():
        return samples

    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        is_anomalous = sub.name != "good"
        for img in sorted(sub.glob("*.png")):
            mask_path: Path | None = None
            if is_anomalous:
                candidate = gt_dir / sub.name / f"{img.stem}_mask.png"
                mask_path = candidate if candidate.exists() else None
            samples.append(
                AnomalySample(
                    image_path=img,
                    label=1 if is_anomalous else 0,
                    category=category,
                    mask_path=mask_path,
                    defect_type=sub.name if is_anomalous else None,
                )
            )

    return samples
