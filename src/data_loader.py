"""Dataset discovery, metadata table, safe download and image loading for GTSRB."""
from __future__ import annotations

import hashlib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from . import config
from .preprocessing import preprocess_image

GT_COLUMNS = ["Filename", "Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId"]


class DatasetNotFoundError(FileNotFoundError):
    """Raised instead of silently assuming that the dataset exists."""


def locate_dataset(require_test: bool = False) -> dict:
    """Verify the GTSRB folder layout and return the important paths.

    Raises DatasetNotFoundError with instructions if the training images are missing.
    The labelled test set is optional (a grouped hold-out is used if it is absent).
    """
    msg = (
        f"GTSRB training images not found at {config.TRAIN_IMAGES_DIR}.\n"
        "Follow DATASET_DOWNLOAD.md (or run `python scripts/setup_project.py --download`)."
    )
    if not config.TRAIN_IMAGES_DIR.is_dir():
        raise DatasetNotFoundError(msg)
    class_dirs = sorted(d for d in config.TRAIN_IMAGES_DIR.iterdir() if d.is_dir() and d.name.isdigit())
    if not class_dirs:
        raise DatasetNotFoundError(msg + "\nNo numeric class folders (00000, 00001, ...) were found.")
    test_gt = None
    if config.RAW_DIR.is_dir():  # GT-final_test.csv sits at the root of its own zip, i.e. outside GTSRB/
        found = list(config.RAW_DIR.rglob(config.TEST_GT_CSV_NAME))
        test_gt = found[0] if found else None
    has_test = test_gt is not None and config.TEST_IMAGES_DIR.is_dir()
    if require_test and not has_test:
        raise DatasetNotFoundError(
            "Labelled official test set not found (need Final_Test/Images and GT-final_test.csv)."
        )
    return {"train_dir": config.TRAIN_IMAGES_DIR, "class_dirs": class_dirs,
            "test_dir": config.TEST_IMAGES_DIR if has_test else None, "test_gt": test_gt if has_test else None}


def _read_gt_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    missing = [c for c in GT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing expected columns {missing}")
    return df


def build_metadata(max_per_class: int | None = None, max_test: int | None = None,
                   seed: int = config.SEED) -> pd.DataFrame:
    """One row per image: path, label, track/group id, ROI, source ('train_pool'|'official_test').

    GTSRB training images come in *tracks* of 30 frames of the same physical sign; the track id
    is parsed from the file name <track>_<frame>.ppm and the group id is '<class>_<track>'.
    """
    info = locate_dataset()
    frames = []
    for class_dir in info["class_dirs"]:
        csvs = sorted(class_dir.glob("GT-*.csv"))
        if not csvs:
            raise DatasetNotFoundError(f"No annotation CSV in {class_dir}")
        df = _read_gt_csv(csvs[0])
        df["path"] = df["Filename"].map(lambda f, d=class_dir: str(d / f))
        df["folder_class"] = int(class_dir.name)
        frames.append(df)
    train = pd.concat(frames, ignore_index=True)
    train["track_id"] = train["Filename"].str.split("_").str[0].astype(int)
    train["group"] = train["ClassId"].astype(str) + "_" + train["track_id"].astype(str)
    train["source"] = "train_pool"
    if max_per_class:
        # sample whole tracks round-robin so the subset keeps the grouped structure
        rng = np.random.RandomState(seed)
        keep = []
        for _, cdf in train.groupby("ClassId"):
            groups = cdf["group"].unique()
            rng.shuffle(groups)
            taken = 0
            for g in groups:
                if taken >= max_per_class:
                    break
                idx = cdf.index[cdf["group"] == g]
                keep.extend(idx)
                taken += len(idx)
        train = train.loc[sorted(keep)]
    parts = [train]
    if info["test_gt"] is not None:
        test = _read_gt_csv(info["test_gt"])
        test["path"] = test["Filename"].map(lambda f: str(info["test_dir"] / f))
        test["folder_class"] = test["ClassId"]
        test["track_id"] = -1
        test["group"] = "test_" + test["Filename"]
        test["source"] = "official_test"
        if max_test and len(test) > max_test:
            test = test.sample(n=max_test, random_state=seed).sort_index()
        parts.append(test)
    meta = pd.concat(parts, ignore_index=True)
    meta = meta.rename(columns={"ClassId": "class_id", "Roi.X1": "roi_x1", "Roi.Y1": "roi_y1",
                                "Roi.X2": "roi_x2", "Roi.Y2": "roi_y2", "Width": "csv_width",
                                "Height": "csv_height", "Filename": "filename"})
    meta["class_id"] = meta["class_id"].astype(int)
    return meta.reset_index(drop=True)


def read_image_rgb(path) -> np.ndarray | None:
    """Return an RGB uint8 array, or None if the file is unreadable/corrupt."""
    try:
        with Image.open(path) as im:
            im.load()
            return np.asarray(im.convert("RGB"))
    except Exception:  # noqa: BLE001 - any decoder failure counts as corrupt
        return None


def md5_of_file(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_one(path, roi):
    img = read_image_rgb(path)
    if img is None:
        return None
    return preprocess_image(img, roi=roi)


def load_processed_images(meta: pd.DataFrame, use_cache: bool = True, n_jobs: int = -1):
    """Load + preprocess every image in `meta` (grayscale, ROI-cropped, resized, uint8).

    Returns (images[N,H,W] uint8, ok_mask[N] bool). Results are cached in data/processed/.
    """
    config.ensure_dirs()
    key_src = "|".join([str(config.IMAGE_SIZE), str(config.USE_ROI_CROP), str(len(meta)),
                        hashlib.md5("".join(meta["path"].astype(str)).encode()).hexdigest()])
    key = hashlib.md5(key_src.encode()).hexdigest()[:12]
    cache = config.PROCESSED_DIR / f"images_{key}.npz"
    if use_cache and cache.exists():
        with np.load(cache) as z:
            return z["images"], z["ok"]
    rois = [None] * len(meta)
    if config.USE_ROI_CROP:
        rois = list(meta[["roi_x1", "roi_y1", "roi_x2", "roi_y2"]].itertuples(index=False, name=None))
    results = Parallel(n_jobs=n_jobs, batch_size=256)(
        delayed(_load_one)(p, r) for p, r in zip(meta["path"], rois))
    w, h = config.IMAGE_SIZE
    ok = np.array([r is not None for r in results])
    images = np.zeros((len(meta), h, w), dtype=np.uint8)
    for i, r in enumerate(results):
        if r is not None:
            images[i] = r
    if use_cache:
        np.savez_compressed(cache, images=images, ok=ok)
    return images, ok


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_local_archives(dest: Path | None = None) -> list[str]:
    """Extract the official GTSRB zips found in data/raw/ (manual-download route) and write a manifest
    (file name, size, SHA-256, time) - this is the record of the exact version used."""
    import datetime as dt
    import json
    dest = Path(dest or config.RAW_DIR)
    manifest, done = [], []
    for name in config.DATASET_URLS:
        z = dest / name
        if not z.exists():
            continue
        with zipfile.ZipFile(z) as zf:
            root = dest.resolve()
            for member in zf.namelist():  # zip-slip guard
                if not (root / member).resolve().is_relative_to(root):
                    raise RuntimeError(f"Unsafe path in archive: {member}")
            zf.extractall(dest)
        manifest.append({"file": name, "bytes": z.stat().st_size, "sha256": _sha256(z), "source": config.DATASET_URLS[name],
                         "extracted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        done.append(name)
    if manifest:
        (dest / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    return done


def download_gtsrb(dest: Path | None = None) -> None:
    """Download the three official archives into data/raw/ and extract them. Explicit, never called implicitly.

    Raises RuntimeError with manual instructions if the network blocks the download.
    """
    dest = Path(dest or config.RAW_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    for name, url in config.DATASET_URLS.items():
        target = dest / name
        if not target.exists():
            print(f"Downloading {url}")
            try:
                urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed https URLs from config
            except Exception as exc:  # noqa: BLE001
                if target.exists():
                    target.unlink()
                raise RuntimeError(
                    f"Could not download {name} ({exc}). Download it manually from "
                    f"{config.DATASET_ARCHIVE_PAGE} and put it into {dest} (see DATASET_DOWNLOAD.md)."
                ) from exc
    print("Extracted:", extract_local_archives(dest))
