"""Dataset audit: corrupt files, label validity, duplicates, sizes, class balance, leakage risks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from . import config
from .data_loader import md5_of_file, read_image_rgb


def _probe(path: str):
    """(readable, width, height, md5, size_bytes) for one file."""
    if not Path(path).is_file():
        return False, 0, 0, None, 0
    img = read_image_rgb(path)
    if img is None:
        return False, 0, 0, None, int(Path(path).stat().st_size)
    h, w = img.shape[:2]
    return True, int(w), int(h), md5_of_file(path), int(Path(path).stat().st_size)


def probe_files(meta: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """Add columns readable, width, height, md5 (measured from the actual files)."""
    res = Parallel(n_jobs=n_jobs, batch_size=256)(delayed(_probe)(p) for p in meta["path"])
    out = meta.copy()
    out[["readable", "width", "height", "md5", "size_bytes"]] = pd.DataFrame(res, index=out.index)
    return out


def validate_dataset(meta: pd.DataFrame) -> pd.DataFrame:
    """Return `meta` with probe columns; raise ValueError on structural problems."""
    if meta.empty:
        raise ValueError("Empty metadata table")
    probed = probe_files(meta)
    return probed


def audit_dataset(probed: pd.DataFrame, save_path: Path | None = None) -> dict:
    """Compute dataset statistics for the reports. Everything here is measured, nothing assumed."""
    train = probed[probed["source"] == "train_pool"]
    test = probed[probed["source"] == "official_test"]
    ok = probed["readable"]
    counts = train["class_id"].value_counts().sort_index()

    label_problems = {
        "missing_label_rows": int(probed["class_id"].isna().sum()),
        "folder_vs_csv_label_mismatch": int((train["folder_class"] != train["class_id"]).sum()),
        "negative_class_ids": int((probed["class_id"] < 0).sum()),
    }
    train_ok = train[train["readable"]]
    dup_groups = train_ok[train_ok.duplicated("md5", keep=False)]
    # duplicates with *conflicting labels* are the dangerous ones
    conflicting = dup_groups.groupby("md5")["class_id"].nunique()
    test_ok = test[test["readable"]]
    cross = set(train_ok["md5"]) & set(test_ok["md5"]) if len(test_ok) else set()

    roi_bad = 0
    if {"roi_x1", "roi_x2"} <= set(probed.columns):
        roi_bad = int(((probed["roi_x2"] <= probed["roi_x1"]) | (probed["roi_y2"] <= probed["roi_y1"])
                       | (probed["roi_x2"] > probed["width"]) | (probed["roi_y2"] > probed["height"]))[ok].sum())
    sizes = probed[ok]
    audit = {
        "run_mode": config.RUN_MODE,
        "dataset": config.DATASET_NAME,
        "source_page": config.DATASET_ARCHIVE_PAGE,
        "citation": config.DATASET_CITATION,
        "n_files_total": int(len(probed)),
        "total_size_mb": round(float(probed["size_bytes"].sum()) / 1e6, 1),
        "n_train_pool_files": int(len(train)),
        "n_official_test_files": int(len(test)),
        "n_unreadable_or_missing": int((~ok).sum()),
        "unreadable_examples": probed.loc[~ok, "path"].head(10).tolist(),
        "file_formats": sorted({Path(p).suffix.lower() for p in probed["path"]}),
        "n_classes_train": int(counts.size),
        "n_classes_test": int(test["class_id"].nunique()) if len(test) else None,
        "class_ids_train": [int(c) for c in counts.index],
        "class_counts_train": {int(k): int(v) for k, v in counts.items()},
        "class_count_min": int(counts.min()), "class_count_max": int(counts.max()),
        "class_count_median": float(counts.median()),
        "imbalance_ratio_max_over_min": float(counts.max() / counts.min()),
        "n_tracks_train": int(train["group"].nunique()),
        "label_problems": label_problems,
        "duplicate_files_in_train": {
            "n_rows_in_duplicate_groups": int(len(dup_groups)),
            "n_duplicate_groups": int(dup_groups["md5"].nunique()),
            "groups_with_conflicting_labels": int((conflicting > 1).sum()),
            "method": "exact byte-level MD5 match (near-duplicates are handled by group-wise splitting)",
        },
        "train_test_identical_files": int(len(cross)),
        "roi_inconsistencies": roi_bad,
        "image_width": _describe(sizes["width"]), "image_height": _describe(sizes["height"]),
        "n_features_raw_pixels": int(config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1]),
        "samples_before_cleaning": int(len(probed)),
        "samples_after_cleaning": int(ok.sum()),
        "sanity_checks_vs_official_documentation": {
            "n_classes_matches_43": int(counts.size) == config.EXPECTED_N_CLASSES,
            "n_train_matches_39209": int(len(train)) == config.EXPECTED_N_TRAIN,
            "n_test_matches_12630": int(len(test)) == config.EXPECTED_N_TEST,
            "note": "False is expected when a development subset is used.",
        },
    }
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        Path(save_path).write_text(json.dumps(audit, indent=2))
    return audit


def _describe(s: pd.Series) -> dict:
    return {"min": int(s.min()), "p5": float(s.quantile(.05)), "median": float(s.median()),
            "mean": round(float(s.mean()), 2), "p95": float(s.quantile(.95)), "max": int(s.max())}


def clean_metadata(probed: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stage 3 cleaning. Only the TRAINING POOL is deduplicated; the official test set is never altered
    (apart from dropping files that cannot be decoded).

    Removed: (1) unreadable files, (2) training rows whose folder label disagrees with the official CSV,
    (3) training groups of byte-identical files with conflicting labels (ambiguous ground truth),
    (4) redundant exact duplicates with the same label (first copy kept).
    Near-duplicates (consecutive frames of one sign) are genuine samples and kept; they are handled by the
    group-aware split instead.
    """
    n0 = len(probed)
    df = probed.copy()
    log = {"n_before": int(n0)}
    unreadable = ~df["readable"]
    log["removed_unreadable"] = int(unreadable.sum())
    df = df[~unreadable]
    is_train = df["source"] == "train_pool"
    mismatch = is_train & (df["folder_class"] != df["class_id"])
    log["removed_label_mismatch_with_csv"] = int(mismatch.sum())
    df = df[~mismatch]
    is_train = df["source"] == "train_pool"
    tr = df[is_train]
    n_labels = tr.groupby("md5")["class_id"].transform("nunique")
    conflict = tr.index[n_labels > 1]
    log["removed_label_conflicting_duplicates"] = int(len(conflict))
    df = df.drop(index=conflict)
    tr = df[df["source"] == "train_pool"]
    redundant = tr.index[tr.duplicated("md5", keep="first")]
    log["removed_exact_duplicates_kept_first"] = int(len(redundant))
    df = df.drop(index=redundant)
    log["n_after"] = int(len(df))
    log["scope"] = ("full dataset" if config.RUN_MODE == "full_dataset" else "DEVELOPMENT SUBSET (not the full experiment)")
    return df.reset_index(drop=True), log
