import numpy as np

from src import config
from src.preprocessing import crop_roi, fit_scaler, flatten_raw_pixels, preprocess_image


def test_preprocess_output_shape_and_dtype():
    img = np.random.RandomState(0).randint(0, 255, (57, 61, 3), dtype=np.uint8)
    out = preprocess_image(img, roi=(5, 5, 50, 50))
    assert out.shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]) and out.dtype == np.uint8


def test_roi_crop_and_degenerate_roi():
    img = np.zeros((40, 40, 3), np.uint8)
    assert crop_roi(img, (5, 5, 24, 24)).shape[:2] == (20, 20)
    assert crop_roi(img, (30, 30, 10, 10)).shape == img.shape  # degenerate -> unchanged


def test_grayscale_input_supported():
    assert preprocess_image(np.zeros((50, 50), np.uint8)).shape == (config.IMAGE_SIZE[1], config.IMAGE_SIZE[0])


def test_raw_pixels_scaled_to_unit_interval():
    X = flatten_raw_pixels(np.full((3, 32, 32), 255, np.uint8))
    assert X.shape == (3, 1024) and X.max() == 1.0


def test_scaler_is_fit_on_train_only():
    tr, te = np.random.RandomState(0).normal(5, 2, (100, 4)), np.random.RandomState(1).normal(50, 2, (20, 4))
    sc = fit_scaler(tr)
    assert np.allclose(sc.mean_, tr.mean(0))  # statistics come from train, not from test
    assert abs(sc.transform(te).mean()) > 5   # test is NOT re-centred
