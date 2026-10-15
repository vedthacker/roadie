import numpy as np

from src import config
from src.hog_features import extract_hog_batch, extract_hog_features, hog_feature_dim, hog_visualization


def _img(seed=0):
    return np.random.RandomState(seed).randint(0, 255, (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]), dtype=np.uint8)


def test_hog_dimension_matches_formula():
    assert extract_hog_features(_img()).shape == (hog_feature_dim(),)
    assert hog_feature_dim() == 1764  # 7*7 blocks * 4 cells * 9 bins for 32x32 / 4x4 cells


def test_hog_is_deterministic_and_finite():
    a, b = extract_hog_features(_img()), extract_hog_features(_img())
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_batch_matches_single():
    imgs = np.stack([_img(i) for i in range(4)])
    batch = extract_hog_batch(imgs, n_jobs=1)
    assert batch.shape == (4, hog_feature_dim()) and np.allclose(batch[2], extract_hog_features(imgs[2]))


def test_visualization_shape():
    feats, vis = hog_visualization(_img())
    assert vis.shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]) and len(feats) == hog_feature_dim()
