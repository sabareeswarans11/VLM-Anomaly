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
    """Common interface that every dataset loader must implement.

    Subclasses must accept ``root_dir`` in ``__init__`` and store it as
    ``self.root_dir``.  This is enforced by convention (not the ABC) so that
    callers can always predict where data lives without interrogating each
    concrete class.
    """

    name: str
    root_dir: Path  # set by every concrete subclass __init__

    @abstractmethod
    def categories(self) -> list[str]:
        """All category names available in this dataset."""

    @abstractmethod
    def samples(
        self, category: str, split: Literal["train", "test"] = "test"
    ) -> list[AnomalySample]:
        """All samples for the given category and split.

        Args:
            category: One of the strings returned by :meth:`categories`.
            split: ``"train"`` returns normal-only images used to train
                classical baselines; ``"test"`` returns the labelled
                evaluation set (both normal and anomalous).
        """

    @abstractmethod
    def download(self) -> None:
        """Download the dataset to ``self.root_dir`` if not already present."""
