"""Read-only view of everything the pipeline has produced, shaped for the report generator.

Missing files simply give None, and the report text then prints 'Pending execution'.
Nothing here computes a new result: it only re-arranges numbers already written by src/pipeline.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from src import config


def _json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def _csv(path, **kw):
    path = Path(path)
    return pd.read_csv(path, **kw) if path.exists() else None


_MET = ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "balanced_accuracy")


def _model(path):
    j = _json(path)
    if not j:
        return None
    return {**j, "train": j.get("train_metrics"), "val": {k: j[k] for k in _MET if k in j}}


class Results:
    def __init__(self):
        m = config.METRICS_DIR
        a = _json(m / "dataset_audit.json")
        if a:
            a = {**a, "n_files_indexed": a["n_files_total"], "n_readable": a["samples_after_cleaning"],
                 "n_classes": a["n_classes_train"], "n_tracks_total": a["n_tracks_train"],
                 "dataset_identifier": a["dataset"] + (" [SYNTHETIC STAND-IN]" if os.environ.get("ROADEYE_SYNTHETIC") else "")
                 + f" - run mode: {a['run_mode']}", "width": a["image_width"], "height": a["image_height"]}
        self.audit = a
        self.prep = _json(m / "data_preparation.json")
        self.pre = _json(m / "preprocessing_summary.json")
        self.bal = _json(m / "balancing.json")
        self.eda = _json(m / "eda_summary.json")
        files = {"baseline_svm": "baseline_svm.json", "knn": "knn_default.json", "decision_tree": "decision_tree_default.json",
                 "svm_default": "svm_default.json", "svm_tuned": "svm_tuned.json", "knn_tuned": "knn_tuned.json",
                 "decision_tree_tuned": "decision_tree_tuned.json"}
        self.models = {k: _model(m / v) for k, v in files.items()}
        gs = _json(m / "gridsearch_svm.json")
        self.params = None
        self.grid_svm = None
        if gs:
            res = pd.DataFrame(gs["cv_results"])
            top = res.sort_values("rank_test_score").iloc[0]
            self.params = {"best_params": gs["best_params"], "search_space": gs["search_space"],
                           "n_candidates": gs["n_candidates"], "cv_best_score_f1_macro": gs["best_cv_score"],
                           "cv_best_score_std": float(top["std_test_score"]),
                           "grid_search_wall_seconds": gs["search_time_seconds"], "subsampled": gs["subsampled"],
                           "n_samples_in_search": gs["n_samples_used_in_search"]}
            pcols = [c for c in res.columns if c.startswith("param_")]
            res["params"] = res[pcols].apply(lambda r: ", ".join(f"{c[6:]}={v}" for c, v in r.items() if str(v) != "nan"), axis=1)
            self.grid_svm = res.sort_values("rank_test_score")[["rank_test_score", "params", "mean_test_score", "std_test_score", "mean_train_score"]]
        self.meta = _json(config.MODELS_DIR / "svm" / config.METADATA_NAME)
        self.err = _json(m / "error_analysis.json")
        self.cmp_val = _csv(config.COMPARISON_DIR / "model_comparison.csv")
        self.cmp_test = _csv(config.COMPARISON_DIR / "model_comparison_test.csv")
        rq = _json(config.COMPARISON_DIR / "research_questions.json")
        self.rq = rq
        self.winner = None
        if rq:
            sc = {r.model: float(r.f1_macro) for r in self.cmp_val.itertuples()}
            self.winner = {"selection_metric": rq["primary_metric"], "selected_on": "validation split",
                           "hog_default_models_winner": (rq["RQ2_best_untuned_hog_classifier"] or {}).get("model"),
                           "overall_winner": (rq["overall_best_on_validation"] or {}).get("model"), "scores": sc}
        self.final = self._final()
        tc = _csv(m / "final_test_top_confusions.csv")
        if tc is not None:
            tc = tc.rename(columns={"true_class": "true_id", "predicted_class": "pred_id", "predicted_name": "pred_name"})
        self.top_conf = tc
        self.per_class = _csv(m / "final_test_per_class.csv")
        self.similar = _csv(m / "eda_similar_classes.csv")
        self.ref = _json(config.MODELS_DIR / "reference" / "reference_inspection.json")
        self.split_info = _json(m / "split_info.json")

    def _final(self):
        m = config.METRICS_DIR
        fin, allm = _json(m / "final_test_metrics.json"), _csv(m / "final_test_all_models.csv")
        if not fin or allm is None:
            return None
        tuned = self.models["svm_tuned"]
        row = lambda n: allm[allm.model == n].iloc[0] if (allm.model == n).any() else None  # noqa: E731
        out = {"test": {k: fin[k] for k in _MET if k in fin}, "train": tuned["train"], "validation": tuned["val"],
               "n_test_samples": fin["n_test"],
               "generalisation_gap_f1_macro_train_minus_test": (tuned["train"]["f1_macro"] - fin["f1_macro"]) if tuned.get("train") else None}
        raw, hog = row("Raw Pixel SVM"), row("HOG SVM")
        if raw is not None and hog is not None:
            out["research_question_1"] = {
                "raw_pixel_svm_test": {"f1_macro": float(raw.f1_macro), "accuracy": float(raw.accuracy)},
                "hog_svm_default_test": {"f1_macro": float(hog.f1_macro), "accuracy": float(hog.accuracy)},
                "delta_f1_macro": float(hog.f1_macro - raw.f1_macro), "delta_accuracy": float(hog.accuracy - raw.accuracy)}
        three = allm[allm.model.isin(["HOG KNN", "HOG Decision Tree", "HOG SVM"])]
        if len(three) == 3:
            out["research_question_2"] = {
                "test_scores": {r.model: float(r.f1_macro) for r in three.itertuples()},
                "winner_on_test": three.sort_values("f1_macro", ascending=False).iloc[0].model,
                "winner_on_validation": (self.rq["RQ2_best_untuned_hog_classifier"] or {}).get("model") if self.rq else None}
        return out

    @property
    def synthetic(self) -> bool:
        return "SYNTHETIC" in ((self.audit or {}).get("dataset_identifier", ""))

    @property
    def has_results(self) -> bool:
        return self.final is not None
