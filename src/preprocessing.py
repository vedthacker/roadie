"""Image preprocessing. ONE implementation shared by training, evaluation and the app."""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from . import config


def crop_roi(img: np.ndarray, roi) -> np.ndarray:
    """Crop to the sign's region of interest (x1, y1, x2, y2) shipped with GTSRB (inclusive)."""
    if roi is None:
        return img
    x1, y1, x2, y2 = [int(v) for v in roi]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:  # degenerate ROI -> keep the whole image
        return img
    return img[y1:y2 + 1, x1:x2 + 1]


def to_grayscale(img_rgb: np.ndarray) -> np.ndarray:
    """RGB -> single channel (luminance). HOG captures shape from gradients, colour is dropped."""
    if img_rgb.ndim == 2:
        return img_rgb
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def resize_image(img: np.ndarray, size=config.IMAGE_SIZE) -> np.ndarray:
    """Resize to a fixed (width, height); INTER_AREA is the right choice for down-sampling."""
    return cv2.resize(img, tuple(size), interpolation=cv2.INTER_AREA)


def preprocess_image(img_rgb: np.ndarray, roi=None, size=config.IMAGE_SIZE) -> np.ndarray:
    """ROI crop (optional) -> grayscale -> resize. Returns uint8 array of shape (H, W)."""
    img = crop_roi(img_rgb, roi)
    gray = to_grayscale(img)
    return resize_image(gray, size).astype(np.uint8)


def flatten_raw_pixels(images: np.ndarray) -> np.ndarray:
    """Raw-pixel representation for the baseline: flatten and scale uint8 -> [0, 1] float32."""
    return images.reshape(len(images), -1).astype(np.float32) / config.RAW_PIXEL_SCALE


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit the scaler on TRAINING data only (leakage prevention)."""
    return StandardScaler().fit(X_train)
