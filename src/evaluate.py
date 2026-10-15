"""Metrics, comparison table helpers and the one-shot final test evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

from . import config

COMPARISON_COLUMNS = ["model", "feature_representation", "accuracy", "precision_macro", "recall_macro",
                      "f1_macro", "training_time_seconds", "inference_time_seconds", "notes",
                      "f1_weighted", "balanced_accuracy", "train_accuracy", "eval_split", "run_mode"]


class _Enc(json.JSONEncoder):
    def default(self, o):  # numpy-safe JSON
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, cls=_Enc))


def load_json(path):
    return json.loads(Path(path).read_text())


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p, "recall_macro": r,
            "f1_macro": f, "precision_weighted": pw, "recall_weighted": rw, "f1_weighted": fw,
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred)}


def evaluate_model(model, X, y, name: str, split: str, X_train=None, y_train=None) -> tuple[dict, np.ndarray]:
    """Predict, time it, compute metrics. Optionally also training accuracy (to show the gap)."""
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    infer = time.perf_counter() - t0
    out = {"model": name, "eval_split": split, "n_samples": int(len(y)), "run_mode": config.RUN_MODE,
           "inference_time_seconds": infer, "inference_ms_per_sample": 1000 * infer / max(1, len(y)),
           **compute_metrics(y, y_pred)}
    if X_train is not None:
        tm = compute_metrics(y_train, model.predict(X_train))
        out["train_metrics"] = tm
        out["train_accuracy"] = float(tm["accuracy"])
    return out, y_pred


def per_class_table(y_true, y_pred, labels=None) -> pd.DataFrame:
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else labels
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class_id": labels, "class_name": [config.class_name(c) for c in labels],
                         "precision": p, "recall": r, "f1": f, "support": s})


def confusion(y_true, y_pred, labels):
    return confusion_matrix(y_true, y_pred, labels=labels)


def top_confusions(cm: np.ndarray, labels, k: int = 10) -> pd.DataFrame:
    """Most frequent (true -> predicted) mistakes, off-diagonal only."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    rows = []
    for f in flat:
        i, j = divmod(int(f), off.shape[1])
        if off[i, j] > 0:
            rows.append({"true_class": int(labels[i]), "true_name": config.class_name(labels[i]),
                         "predicted_class": int(labels[j]), "predicted_name": config.class_name(labels[j]),
                         "count": int(off[i, j])})
    return pd.DataFrame(rows, columns=["true_class", "true_name", "predicted_class", "predicted_name", "count"])


def upsert_model_comparison(row: dict, path: Path | None = None) -> pd.DataFrame:
    """Insert/replace one model's row in results/model_comparison/model_comparison.csv."""
    path = Path(path or config.COMPARISON_DIR / "model_comparison.csv")
    df = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COMPARISON_COLUMNS)
    df = df[df["model"] != row["model"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    for c in COMPARISON_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COMPARISON_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def best_model(df: pd.DataFrame, metric: str = config.PRIMARY_METRIC, candidates=None) -> dict:
    """Automatically pick the winner by `metric` (never hard-coded)."""
    d = df if candidates is None else df[df["model"].isin(candidates)]
    d = d.dropna(subset=[metric])
    if d.empty:
        return {}
    r = d.sort_values(metric, ascending=False).iloc[0]
    return {"model": r["model"], "metric": metric, "value": float(r[metric])}


def comparison_row(metrics: dict, model: str, features: str, train_time: float, notes: str = "") -> dict:
    return {"model": model, "feature_representation": features, "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"], "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"], "training_time_seconds": train_time,
            "inference_time_seconds": metrics["inference_time_seconds"], "notes": notes,
            "f1_weighted": metrics["f1_weighted"], "balanced_accuracy": metrics["balanced_accuracy"],
            "train_accuracy": metrics.get("train_accuracy"), "eval_split": metrics["eval_split"],
            "run_mode": metrics["run_mode"]}
