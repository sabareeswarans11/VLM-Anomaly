"""Dataset loader tests — no network, no real data.

A synthetic fixture tree is built in a tmp_path so every test is hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_anomaly.datasets.base import AnomalySample
from vlm_anomaly.datasets.infra import InfraAD
from vlm_anomaly.datasets.mvtec import CATEGORIES as MVTEC_CATEGORIES
from vlm_anomaly.datasets.mvtec import MVTec
from vlm_anomaly.datasets.visa import CATEGORIES as VISA_CATEGORIES
from vlm_anomaly.datasets.visa import VisA

# ---------------------------------------------------------------------------
# Fixtures — synthetic on-disk trees
# ---------------------------------------------------------------------------


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PNG")  # minimal non-empty file
    return path


@pytest.fixture
def mvtec_root(tmp_path: Path) -> Path:
    """Tiny MVTec-style tree: bottle category, 2 train + 2 good test + 2 anomaly test."""
    root = tmp_path / "mvtec"
    cat = root / "bottle"

    # train/good
    for i in range(2):
        _touch(cat / "train" / "good" / f"{i:03d}.png")

    # test/good
    for i in range(2):
        _touch(cat / "test" / "good" / f"{i:03d}.png")

    # test/broken_large  + ground_truth
    for i in range(2):
        _touch(cat / "test" / "broken_large" / f"{i:03d}.png")
        _touch(cat / "ground_truth" / "broken_large" / f"{i:03d}_mask.png")

    return root


@pytest.fixture
def visa_root_mvtec_style(tmp_path: Path) -> Path:
    """Tiny VisA tree in MVTec-style layout: candle category."""
    root = tmp_path / "visa"
    cat = root / "candle"

    for i in range(2):
        _touch(cat / "train" / "good" / f"{i:03d}.png")
    for i in range(2):
        _touch(cat / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        _touch(cat / "test" / "bad" / f"{i:03d}.png")
        _touch(cat / "ground_truth" / "bad" / f"{i:03d}_mask.png")

    return root


@pytest.fixture
def visa_root_raw(tmp_path: Path) -> Path:
    """Tiny VisA tree in raw VisA layout: candle category."""
    root = tmp_path / "visa_raw"
    cat = root / "candle"

    for i in range(2):
        _touch(cat / "Data" / "Images" / "Normal" / f"normal_{i:03d}.png")
    for i in range(2):
        _touch(cat / "Data" / "Images" / "Anomaly" / f"anomaly_{i:03d}.png")
        _touch(cat / "Data" / "Masks" / "Anomaly" / f"anomaly_{i:03d}.png")

    return root


@pytest.fixture
def infra_root(tmp_path: Path) -> Path:
    """Tiny InfraAD tree: tower category."""
    root = tmp_path / "infra"
    cat = root / "tower"

    for i in range(2):
        _touch(cat / "train" / "good" / f"{i:03d}.png")
    for i in range(2):
        _touch(cat / "test" / "good" / f"{i:03d}.png")
    for i in range(2):
        _touch(cat / "test" / "corrosion" / f"{i:03d}.png")
        _touch(cat / "ground_truth" / "corrosion" / f"{i:03d}_mask.png")

    return root


# ---------------------------------------------------------------------------
# MVTec
# ---------------------------------------------------------------------------


class TestMVTec:
    def test_categories_requires_no_disk(self) -> None:
        ds = MVTec(root_dir="/nonexistent/path")
        cats = ds.categories()
        assert len(cats) == 15
        assert "bottle" in cats
        assert "zipper" in cats
        assert cats == sorted(cats)

    def test_categories_match_constant(self) -> None:
        assert MVTec().categories() == MVTEC_CATEGORIES

    def test_train_samples(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        samples = ds.samples("bottle", split="train")
        assert len(samples) == 2
        assert all(s.label == 0 for s in samples)
        assert all(s.mask_path is None for s in samples)
        assert all(s.category == "bottle" for s in samples)

    def test_test_samples_count(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        samples = ds.samples("bottle", split="test")
        # 2 good (label=0) + 2 broken_large (label=1)
        assert len(samples) == 4

    def test_test_samples_labels(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        samples = ds.samples("bottle", split="test")
        normals = [s for s in samples if s.label == 0]
        anomalies = [s for s in samples if s.label == 1]
        assert len(normals) == 2
        assert len(anomalies) == 2

    def test_anomaly_samples_have_defect_type(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        samples = ds.samples("bottle", split="test")
        for s in samples:
            if s.label == 1:
                assert s.defect_type == "broken_large"
            else:
                assert s.defect_type is None

    def test_mask_paths_resolve(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        anomalies = [s for s in ds.samples("bottle") if s.label == 1]
        assert all(s.mask_path is not None for s in anomalies)
        assert all(s.mask_path.exists() for s in anomalies)

    def test_normal_samples_have_no_mask(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        normals = [s for s in ds.samples("bottle") if s.label == 0]
        assert all(s.mask_path is None for s in normals)

    def test_samples_are_anomaly_sample_instances(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        for s in ds.samples("bottle"):
            assert isinstance(s, AnomalySample)

    def test_unknown_category_raises(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        with pytest.raises(ValueError, match="Unknown MVTec category"):
            ds.samples("nonexistent")

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        ds = MVTec(root_dir=tmp_path / "mvtec")
        with pytest.raises(FileNotFoundError):
            ds.samples("bottle")

    def test_root_dir_is_absolute(self, mvtec_root: Path) -> None:
        ds = MVTec(root_dir=mvtec_root)
        assert ds.root_dir.is_absolute()

    def test_download_skips_when_all_present(self, mvtec_root: Path) -> None:
        """download() should log-and-return when all 15 category dirs exist."""
        from unittest.mock import patch

        from vlm_anomaly.datasets import mvtec as mvtec_mod

        # Create stub dirs for all 15 categories
        for cat in mvtec_mod.CATEGORIES:
            (mvtec_root / cat).mkdir(exist_ok=True)
        ds = MVTec(root_dir=mvtec_root)
        # Should return without touching network
        with patch("urllib.request.urlretrieve") as mock_dl:
            ds.download()
            mock_dl.assert_not_called()

    def test_default_root_uses_settings(self) -> None:
        from unittest.mock import patch

        from vlm_anomaly.config import Settings

        with patch("vlm_anomaly.datasets.mvtec.get_settings") as mock_s:
            mock_s.return_value = Settings(
                _env_file="/nonexistent",
                data_dir="/tmp/mydata",
            )
            ds = MVTec()
        assert "mydata" in str(ds.root_dir)


# ---------------------------------------------------------------------------
# VisA — MVTec-style layout
# ---------------------------------------------------------------------------


class TestVisAMvtecStyle:
    def test_categories_requires_no_disk(self) -> None:
        ds = VisA(root_dir="/nonexistent")
        cats = ds.categories()
        assert len(cats) == 12
        assert "candle" in cats

    def test_categories_match_constant(self) -> None:
        assert VisA().categories() == VISA_CATEGORIES

    def test_train_samples(self, visa_root_mvtec_style: Path) -> None:
        ds = VisA(root_dir=visa_root_mvtec_style)
        samples = ds.samples("candle", split="train")
        assert len(samples) == 2
        assert all(s.label == 0 for s in samples)

    def test_test_samples_count(self, visa_root_mvtec_style: Path) -> None:
        ds = VisA(root_dir=visa_root_mvtec_style)
        samples = ds.samples("candle", split="test")
        assert len(samples) == 4  # 2 good + 2 bad

    def test_test_labels(self, visa_root_mvtec_style: Path) -> None:
        ds = VisA(root_dir=visa_root_mvtec_style)
        samples = ds.samples("candle", split="test")
        assert sum(s.label for s in samples) == 2

    def test_mask_paths_exist(self, visa_root_mvtec_style: Path) -> None:
        ds = VisA(root_dir=visa_root_mvtec_style)
        anomalies = [s for s in ds.samples("candle") if s.label == 1]
        assert all(s.mask_path is not None and s.mask_path.exists() for s in anomalies)

    def test_unknown_category_raises(self, visa_root_mvtec_style: Path) -> None:
        ds = VisA(root_dir=visa_root_mvtec_style)
        with pytest.raises(ValueError, match="Unknown VisA category"):
            ds.samples("nonexistent")


# ---------------------------------------------------------------------------
# VisA — raw layout
# ---------------------------------------------------------------------------


class TestVisARawLayout:
    def test_train_samples_from_raw(self, visa_root_raw: Path) -> None:
        ds = VisA(root_dir=visa_root_raw)
        samples = ds.samples("candle", split="train")
        assert len(samples) == 2
        assert all(s.label == 0 for s in samples)

    def test_test_samples_from_raw(self, visa_root_raw: Path) -> None:
        ds = VisA(root_dir=visa_root_raw)
        samples = ds.samples("candle", split="test")
        # 2 normal + 2 anomaly
        assert len(samples) == 4
        assert sum(s.label for s in samples) == 2

    def test_anomaly_masks_from_raw(self, visa_root_raw: Path) -> None:
        ds = VisA(root_dir=visa_root_raw)
        anomalies = [s for s in ds.samples("candle") if s.label == 1]
        assert all(s.mask_path is not None and s.mask_path.exists() for s in anomalies)


# ---------------------------------------------------------------------------
# InfraAD
# ---------------------------------------------------------------------------


class TestInfraAD:
    def test_categories_empty_when_root_missing(self, tmp_path: Path) -> None:
        ds = InfraAD(root_dir=tmp_path / "infra")
        assert ds.categories() == []

    def test_categories_discovered_from_disk(self, infra_root: Path) -> None:
        ds = InfraAD(root_dir=infra_root)
        assert ds.categories() == ["tower"]

    def test_train_samples(self, infra_root: Path) -> None:
        ds = InfraAD(root_dir=infra_root)
        samples = ds.samples("tower", split="train")
        assert len(samples) == 2
        assert all(s.label == 0 for s in samples)

    def test_test_samples(self, infra_root: Path) -> None:
        ds = InfraAD(root_dir=infra_root)
        samples = ds.samples("tower", split="test")
        assert len(samples) == 4
        anomalies = [s for s in samples if s.label == 1]
        assert all(s.defect_type == "corrosion" for s in anomalies)

    def test_missing_root_raises(self, tmp_path: Path) -> None:
        ds = InfraAD(root_dir=tmp_path / "infra")
        with pytest.raises(FileNotFoundError):
            ds.samples("tower")
