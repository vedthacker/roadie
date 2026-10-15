"""GridSearchCV tuning. Leakage-safe: scaler lives INSIDE the CV pipeline, folds are grouped by
track, only TRAINING data is ever passed in (never validation/test)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config


def stratified_subsample(y, groups, max_samples, seed=config.SEED):
    """Group-aware stratified subsample of the TRAINING set used only to keep the grid search cheap."""
    n = len(y)
    if max_samples is None or n <= max_samples:
        return np.arange(n)
    n_splits = max(2, int(round(n / max_samples)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, idx = next(sgkf.split(np.zeros(n), y, groups))
    return np.sort(idx)


def tune_estimator(name, X_train, y_train, groups_train, class_weight=None, max_samples=config.TUNE_MAX_SAMPLES,
                   full_grid=False, n_jobs=1, seed=config.SEED):
    """Run GridSearchCV for 'svm' | 'knn' | 'tree'. Returns (search_result_dict, best_params_clean)."""
    if name == "svm":
        est = SVC(class_weight=class_weight, random_state=seed)
        grid = [dict(config.SVM_GRID), dict(config.SVM_LINEAR_GRID)] if full_grid else [dict(config.SVM_GRID)]
    elif name == "knn":
        est, grid = KNeighborsClassifier(metric="euclidean", n_jobs=1), [dict(config.KNN_GRID)]
    elif name == "tree":
        est, grid = DecisionTreeClassifier(class_weight=class_weight, random_state=seed), [dict(config.TREE_GRID)]
    else:
        raise ValueError(name)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", est)])  # scaler refit inside every fold
    idx = stratified_subsample(y_train, groups_train, max_samples, seed)
    Xs, ys, gs = X_train[idx], y_train[idx], groups_train[idx]
    cv = StratifiedGroupKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring=config.TUNING_SCORING, cv=cv, n_jobs=n_jobs, refit=False,
                          return_train_score=True, verbose=0)
    t0 = time.perf_counter()
    search.fit(Xs, ys, groups=gs)
    elapsed = time.perf_counter() - t0
    res = pd.DataFrame(search.cv_results_)
    res.columns = [c.replace('clf__', '') for c in res.columns]
    cols = [c for c in res.columns if c.startswith("param_") or c in (
        "mean_test_score", "std_test_score", "mean_train_score", "mean_fit_time", "rank_test_score")]
    best = {k.replace("clf__", ""): _py(v) for k, v in search.best_params_.items()}
    result = {
        "model": name, "scoring": config.TUNING_SCORING, "cv_folds": config.CV_FOLDS,
        "cv_type": "StratifiedGroupKFold (grouped by sign track)",
        "n_samples_used_in_search": int(len(idx)), "n_training_samples_available": int(len(y_train)),
        "subsampled": bool(len(idx) < len(y_train)), "search_space": _grid_repr(grid),
        "n_candidates": int(len(res)), "n_fits": int(len(res) * config.CV_FOLDS),
        "best_params": best, "best_cv_score": float(search.best_score_),
        "search_time_seconds": elapsed,
        "cv_results": res[cols].assign(**{c: res[c].astype(str) for c in cols if c.startswith("param_")})
        .to_dict(orient="records"),
    }
    return result, best


def _py(v):
    return v.item() if hasattr(v, "item") else v


def _grid_repr(grid):
    return [{k.replace("clf__", ""): [str(x) for x in v] for k, v in g.items()} for g in grid]
