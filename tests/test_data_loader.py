import numpy as np
import pytest

from src import config
from src.data_loader import DatasetNotFoundError, build_metadata, load_processed_images, locate_dataset, read_image_rgb


def test_missing_dataset_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TRAIN_IMAGES_DIR", tmp_path / "nope")
    with pytest.raises(DatasetNotFoundError):
        locate_dataset()


def test_metadata_structure(patched_config):
    meta = build_metadata()
    assert {"path", "class_id", "group", "source", "roi_x1"} <= set(meta.columns)
    assert set(meta["source"]) == {"train_pool", "official_test"}
    assert meta["class_id"].nunique() == 5
    # every training frame of a track shares one group id
    assert meta[meta.source == "train_pool"].groupby("group").size().min() >= 1


def test_images_load_and_have_fixed_shape(patched_config):
    meta = build_metadata().head(12)
    images, ok = load_processed_images(meta, use_cache=False, n_jobs=1)
    assert ok.all()
    assert images.shape == (12, config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]) and images.dtype == np.uint8


def test_corrupt_file_detected(tmp_path):
    bad = tmp_path / "bad.ppm"
    bad.write_bytes(b"not an image")
    assert read_image_rgb(bad) is None
