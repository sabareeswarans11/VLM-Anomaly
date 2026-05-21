"""MVTec AD dataset loader.

Directory layout expected on disk (standard MVTec release):

    {root_dir}/
    └── {category}/
        ├── train/
        │   └── good/          # normal training images
        ├── test/
        │   ├── good/          # normal test images   (label=0)
        │   └── {defect}/      # anomalous test images (label=1, defect_type={defect})
        └── ground_truth/
            └── {defect}/      # binary PNG masks — same stem as test image + _mask suffix

``categories()`` returns the 15 known category names without requiring
``root_dir`` to exist on disk, so it works on a fresh clone.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Literal

from vlm_anomaly.config import get_settings
from vlm_anomaly.datasets.base import AnomalyDataset, AnomalySample
from vlm_anomaly.logging import get_logger

log = get_logger(__name__)

# Published SHA-256 of the official MVTec AD archive (mvtec_anomaly_detection.tar.xz)
_MVTEC_URL = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
_MVTEC_SHA256 = "cf4313b13603ab4c36b9b1ced58a1c1e43d0ff381c01f36e7f1aadf8e1b36f5d"

CATEGORIES: list[str] = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]


class MVTec(AnomalyDataset):
    """Loader for the MVTec Anomaly Detection dataset (15 categories).

    Args:
        root_dir: Path to the directory that contains the per-category
            subdirectories.  Defaults to ``{data_dir}/mvtec`` from
            :class:`~vlm_anomaly.config.Settings`.
    """

    name = "mvtec"

    def __init__(self, root_dir: Path | str | None = None) -> None:
        if root_dir is None:
            root_dir = get_settings().data_dir / "mvtec"
        self.root_dir = Path(root_dir).resolve()

    # ------------------------------------------------------------------
    # AnomalyDataset interface
    # ------------------------------------------------------------------

    def categories(self) -> list[str]:
        """Return the 15 MVTec category names (no disk access required)."""
        return list(CATEGORIES)

    def samples(
        self, category: str, split: Literal["train", "test"] = "test"
    ) -> list[AnomalySample]:
        """Return all samples for ``category`` in the given ``split``.

        Args:
            category: Must be one of :data:`CATEGORIES`.
            split: ``"train"`` → normal images only (for classical baselines).
                   ``"test"``  → normal + anomalous (for evaluation).

        Raises:
            ValueError: If ``category`` is not a known MVTec category.
            FileNotFoundError: If ``root_dir/{category}`` does not exist.
        """
        if category not in CATEGORIES:
            raise ValueError(f"Unknown MVTec category {category!r}. Valid: {CATEGORIES}")

        cat_dir = self.root_dir / category
        if not cat_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: {cat_dir}. "
                f"Run `bash scripts/download_mvtec.sh` first."
            )

        if split == "train":
            return self._train_samples(cat_dir, category)
        return self._test_samples(cat_dir, category)

    def download(self) -> None:
        """Download and verify the MVTec AD archive into ``root_dir``.

        Skips the download if all 15 category directories already exist.
        Verifies the SHA-256 of the downloaded archive before extraction.
        """
        if all((self.root_dir / c).exists() for c in CATEGORIES):
            log.info("mvtec.download.skip", reason="all categories already present")
            return

        self.root_dir.mkdir(parents=True, exist_ok=True)
        archive = self.root_dir.parent / "mvtec_anomaly_detection.tar.xz"

        if not archive.exists():
            log.info("mvtec.download.start", url=_MVTEC_URL, dest=str(archive))
            urllib.request.urlretrieve(_MVTEC_URL, archive)

        log.info("mvtec.download.verify", path=str(archive))
        _verify_sha256(archive, _MVTEC_SHA256)

        log.info("mvtec.download.extract", dest=str(self.root_dir))
        shutil.unpack_archive(str(archive), str(self.root_dir))
        log.info("mvtec.download.done")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_samples(self, cat_dir: Path, category: str) -> list[AnomalySample]:
        good_dir = cat_dir / "train" / "good"
        return [
            AnomalySample(
                image_path=p,
                label=0,
                category=category,
            )
            for p in sorted(good_dir.glob("*.png"))
        ]

    def _test_samples(self, cat_dir: Path, category: str) -> list[AnomalySample]:
        samples: list[AnomalySample] = []
        test_dir = cat_dir / "test"
        gt_dir = cat_dir / "ground_truth"

        for defect_dir in sorted(test_dir.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            is_anomalous = defect != "good"

            for img_path in sorted(defect_dir.glob("*.png")):
                mask_path: Path | None = None
                if is_anomalous:
                    # Mask filename: same stem + "_mask.png"
                    mask_path = gt_dir / defect / f"{img_path.stem}_mask.png"
                    if not mask_path.exists():
                        mask_path = None

                samples.append(
                    AnomalySample(
                        image_path=img_path,
                        label=1 if is_anomalous else 0,
                        category=category,
                        mask_path=mask_path,
                        defect_type=defect if is_anomalous else None,
                    )
                )

        return samples


def _verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, got {digest}")
