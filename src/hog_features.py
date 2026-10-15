"""HOG feature extraction (skimage) with explicit parameters and on-disk caching."""
from __future__ import annotations

import hashlib
import json

import numpy as np
from joblib import Parallel, delayed
from skimage.feature import hog

from . import config


def extract_hog_features(gray_img: np.ndarray, params: dict | None = None) -> np.ndarray:
    """HOG descriptor of one preprocessed grayscale image (H, W) uint8."""
    p = params or config.HOG_PARAMS
    return hog(gray_img, feature_vector=True, **p).astype(np.float32)


def hog_visualization(gray_img: np.ndarray, params: dict | None = None):
    """Return (feature_vector, hog_image) for plotting."""
    p = params or config.HOG_PARAMS
    feats, vis = hog(gray_img, visualize=True, feature_vector=True, **p)
    return feats.astype(np.float32), vis


def hog_feature_dim(image_size=config.IMAGE_SIZE, params: dict | None = None) -> int:
    """Analytical HOG length (checked against the real output by the tests)."""
    p = params or config.HOG_PARAMS
    w, h = image_size
    cx, cy = p["pixels_per_cell"]
    bx, by = p["cells_per_block"]
    ncx, ncy = w // cx, h // cy
    return (ncx - bx + 1) * (ncy - by + 1) * bx * by * p["orientations"]


def extract_hog_batch(images: np.ndarray, params: dict | None = None, n_jobs: int = -1) -> np.ndarray:
    p = params or config.HOG_PARAMS
    feats = Parallel(n_jobs=n_jobs, batch_size=512)(delayed(extract_hog_features)(im, p) for im in images)
    return np.vstack(feats).astype(np.float32)


def hog_cache_key(images: np.ndarray, params: dict | None = None) -> str:
    p = params or config.HOG_PARAMS
    h = hashlib.md5(json.dumps(p, sort_keys=True, default=list).encode())
    h.update(str(images.shape).encode())
    h.update(np.ascontiguousarray(images[:: max(1, len(images) // 64)]).tobytes())
    return h.hexdigest()[:12]


def get_or_compute_hog(images: np.ndarray, params: dict | None = None, use_cache: bool = True) -> np.ndarray:
    """HOG for a whole image array; cached under data/processed/hog_<key>.npy."""
    config.ensure_dirs()
    cache = config.PROCESSED_DIR / f"hog_{hog_cache_key(images, params)}.npy"
    if use_cache and cache.exists():
        return np.load(cache)
    feats = extract_hog_batch(images, params)
    if use_cache:
        np.save(cache, feats)
    return feats
