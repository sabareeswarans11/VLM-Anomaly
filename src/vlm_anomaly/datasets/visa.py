"""VisA (Visual Anomaly) dataset loader.

Two on-disk layouts are supported and auto-detected:

**MVTec-style** (preprocessed, recommended):

    {root_dir}/
    └── {category}/
        ├── train/
        │   └── good/
        ├── test/
        │   ├── good/
        │   └── bad/
        └── ground_truth/
            └── bad/

**Raw VisA** (as released by the authors):

    {root_dir}/
    └── {category}/
        └── Data/
            ├── Images/
            │   ├── Normal/
            │   └── Anomaly/
            └── Masks/
                └── Anomaly/

``categories()`` returns the 12 known category names without requiring
``root_dir`` to exist on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from vlm_anomaly.config import get_settings
from vlm_anomaly.datasets.base import AnomalyDataset, AnomalySample
from vlm_anomaly.logging import get_logger

log = get_logger(__name__)

CATEGORIES: list[str] = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]


class VisA(AnomalyDataset):
    """Loader for the VisA (Visual Anomaly) dataset (12 categories).

    Args:
        root_dir: Path containing the per-category subdirectories.
            Defaults to ``{data_dir}/visa`` from
            :class:`~vlm_anomaly.config.Settings`.
    """

    name = "visa"

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is None:
            root_dir = get_settings().data_dir / "visa"
        self.root_dir = Path(root_dir).resolve()

    # ------------------------------------------------------------------
    # AnomalyDataset interface
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return the 12 VisA category names (no disk access required)."""
        return list(CATEGORIES)

    def samples(
        self, category: str, split: Literal["train", "test"] = "test"
    ) -> list[AnomalySample]:
        """Return all samples for ``category`` in the given ``split``.

        Args:
            category: Must be one of :data:`CATEGORIES`.
            split: ``"train"`` → normal images only.
                   ``"test"``  → normal + anomalous.

        Raises:
            ValueError: If ``category`` is not a known VisA category.
            FileNotFoundError: If ``root_dir/{category}`` does not exist.
        """
        if category not in CATEGORIES:
            raise ValueError(f"Unknown VisA category {category!r}. Valid: {CATEGORIES}")

        cat_dir = self.root_dir / category
        if not cat_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: {cat_dir}. "
                f"Run `bash scripts/download_visa.sh` first."
            )

        layout = _detect_layout(cat_dir)
        log.debug("visa.samples", category=category, layout=layout, split=split)

        if layout == "mvtec":
            return _mvtec_style_samples(cat_dir, category, split)
        return _raw_visa_samples(cat_dir, category, split)

    def download(self) -> None:
        """Download VisA into ``root_dir``.

        VisA requires accepting the dataset licence on the official page.
        This method logs instructions rather than silently fetching the data.
        """
        log.warning(
            "visa.download.manual",
            message=(
                "VisA requires accepting a licence. "
                "Download from https://github.com/amazon-science/spot-diff "
                "and extract into the directory shown below."
            ),
            target_dir=str(self.root_dir),
        )


# ------------------------------------------------------------------
# Layout detection
# ------------------------------------------------------------------


def _detect_layout(cat_dir: Path) -> Literal["mvtec", "raw"]:
    """Return ``'mvtec'`` if the MVTec-style layout is present, else ``'raw'``."""
    if (cat_dir / "train").exists() or (cat_dir / "test").exists():
        return "mvtec"
    return "raw"


# ------------------------------------------------------------------
# MVTec-style helpers
# ------------------------------------------------------------------


def _mvtec_style_samples(
    cat_dir: Path, category: str, split: Literal["train", "test"]
) -> list[AnomalySample]:
    samples: list[AnomalySample] = []

    if split == "train":
        good_dir = cat_dir / "train" / "good"
        return [
            AnomalySample(image_path=p, label=0, category=category)
            for p in sorted(good_dir.glob("*.png"))
        ]

    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"

    for sub in sorted(test_dir.iterdir()):
        if not sub.is_dir():
            continue
        is_anomalous = sub.name != "good"
        for img in sorted(_glob_images(sub)):
            mask_path: Path | None = None
            if is_anomalous:
                mask_path = gt_dir / sub.name / f"{img.stem}_mask.png"
                if not mask_path.exists():
                    mask_path = None
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


# ------------------------------------------------------------------
# Raw VisA helpers
# ------------------------------------------------------------------


def _raw_visa_samples(
    cat_dir: Path, category: str, split: Literal["train", "test"]
) -> list[AnomalySample]:
    data_dir = cat_dir / "Data"
    normal_dir = data_dir / "Images" / "Normal"
    anomaly_dir = data_dir / "Images" / "Anomaly"
    masks_dir = data_dir / "Masks" / "Anomaly"

    samples: list[AnomalySample] = []

    if split in ("train", "test"):
        # In the raw release there is no train/test split file included here;
        # we expose all normal images under "train" and the full set under "test".
        for img in sorted(_glob_images(normal_dir)):
            samples.append(AnomalySample(image_path=img, label=0, category=category))

    if split == "test":
        for img in sorted(_glob_images(anomaly_dir)):
            mask_path = masks_dir / f"{img.stem}.png"
            samples.append(
                AnomalySample(
                    image_path=img,
                    label=1,
                    category=category,
                    mask_path=mask_path if mask_path.exists() else None,
                    defect_type="anomaly",
                )
            )

    return samples


def _glob_images(directory: Path) -> list[Path]:
    """Return sorted PNG and JPG images in ``directory``."""
    imgs = (
        list(directory.glob("*.png"))
        + list(directory.glob("*.jpg"))
        + list(directory.glob("*.JPG"))
    )
    return sorted(imgs)
