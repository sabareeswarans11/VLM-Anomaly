"""Abstract base class for anomaly-detection datasets."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AnomalySample:
    """A single image with its ground-truth label and optional mask."""

    image_path: Path
    label: Literal[0, 1]  # 0 = normal, 1 = anomaly
    category: str
    mask_path: Path | None = None
    defect_type: str | None = None


class AnomalyDataset(ABC):
    """Common interface that every dataset loader must implement."""

    name: str

    @abstractmethod
    def categories(self) -> list[str]:
        """All category names available in this dataset."""

    @abstractmethod
    def samples(
        self, category: str, split: Literal["train", "test"] = "test"
    ) -> list[AnomalySample]:
        """All samples for the given category and split."""

    @abstractmethod
    def download(self) -> None:
        """Download the dataset to the configured data directory if missing."""
