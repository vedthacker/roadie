"""Leakage-aware train / validation / test split.

GTSRB training images are 30-frame *tracks* of the same physical sign, so a plain random split
would put near-identical frames in train and validation (optimistic scores). We therefore split
by group (class_track) with StratifiedGroupKFold.

Test set: the official labelled GTSRB test set if present (kept untouched until the final
evaluation); otherwise a grouped hold-out from the pool (clearly recorded in the split info).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config


def _grouped_holdout(indices, y, groups, fraction, seed):
    n_splits = max(2, int(round(1 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_rel, hold_rel = next(sgkf.split(np.zeros(len(indices)), y[indices], groups[indices]))
    return indices[train_rel], indices[hold_rel]


def split_dataset(meta: pd.DataFrame, seed: int = config.SEED) -> dict:
    """Return {'train','val','test'} index arrays (positions in `meta`) + info dict."""
    y = meta["class_id"].to_numpy()
    groups = meta["group"].to_numpy()
    pool = np.where(meta["source"].to_numpy() == "train_pool")[0]
    official_test = np.where(meta["source"].to_numpy() == "official_test")[0]
    if len(official_test):
        test_idx, test_source = official_test, "official_gtsrb_test_set"
    else:
        pool, test_idx = _grouped_holdout(pool, y, groups, config.HOLDOUT_TEST_FRACTION, seed)
        test_source = "grouped_holdout_from_training_pool (official labelled test set not available)"
    train_idx, val_idx = _grouped_holdout(pool, y, groups, config.VAL_FRACTION, seed)
    info = {"seed": seed, "test_source": test_source, "val_fraction": config.VAL_FRACTION,
            "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
            "split_unit": "group = class_track (all frames of one physical sign stay together)",
            "run_mode": config.RUN_MODE}
    return {"train": train_idx, "val": val_idx, "test": test_idx, "info": info}


def verify_no_leakage(meta: pd.DataFrame, splits: dict) -> dict:
    """Assert disjoint indices / groups (and file hashes if 'md5' column exists). Returns a report."""
    tr, va, te = splits["train"], splits["val"], splits["test"]
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te)), "index overlap"
    g = meta["group"]
    g_tr, g_va = set(g.iloc[tr]), set(g.iloc[va])
    assert not (g_tr & g_va), "track/group shared between train and validation"
    report = {"index_overlap": 0, "train_val_group_overlap": 0}
    if "official" not in splits["info"]["test_source"] and "holdout" in splits["info"]["test_source"]:
        assert not (g_tr & set(g.iloc[te])) and not (g_va & set(g.iloc[te])), "group shared with test"
        report["test_group_overlap"] = 0
    if "md5" in meta.columns:
        h = meta["md5"]
        report["train_test_identical_files"] = int(len(set(h.iloc[tr]) & set(h.iloc[te])))
        report["train_val_identical_files"] = int(len(set(h.iloc[tr]) & set(h.iloc[va])))
    return report


def class_distribution(meta: pd.DataFrame, idx) -> pd.Series:
    return meta.iloc[idx]["class_id"].value_counts().sort_index()
