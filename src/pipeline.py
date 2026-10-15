"""Stage functions shared by scripts/run_pipeline.py and the notebooks (no duplicated logic).

Protocol (leakage prevention):
  1. test set (official GTSRB test set) is separated first and not touched until stage_final_eval
  2. TRAIN POOL -> grouped (by sign track) train / validation split
  3. scaler, class weights, models and the hyper-parameter search only ever see TRAIN
  4. validation is used for model comparison / selection
  5. frozen models are evaluated ONCE on the test set (guarded by a lock file)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr

from . import config
from . import visualization as viz
from .data_loader import build_metadata, load_processed_images
from .data_validation import audit_dataset, clean_metadata, validate_dataset
from .evaluate import (best_model, comparison_row, compute_metrics, confusion, evaluate_model, load_json,
                       per_class_table, save_json, top_confusions, upsert_model_comparison)
from .hog_features import get_or_compute_hog, hog_feature_dim
from .preprocessing import fit_scaler, flatten_raw_pixels
from .split_data import class_distribution, split_dataset, verify_no_leakage
from .train_decision_tree import train_decision_tree
from .train_knn import train_knn
from .train_svm import train_svm
from .tune_svm import tune_estimator

log = logging.getLogger("roadeye")
M = config.METRICS_DIR
HOG_LABEL = f"HOG ({hog_feature_dim()}-d)"
RAW_LABEL = f"Raw pixels ({config.IMAGE_SIZE[0] * config.IMAGE_SIZE[1]}-d)"


def setup_logging():
    config.ensure_dirs()
    if not log.handlers:
        log.setLevel(logging.INFO)
        for h in (logging.StreamHandler(), logging.FileHandler(config.LOGS_DIR / "pipeline.log")):
            h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
            log.addHandler(h)
    return log


# --------------------------------------------------------------------------------------
@dataclass
class Data:
    meta: pd.DataFrame
    images: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    splits: dict
    hog: np.ndarray = field(repr=False)
    raw: np.ndarray = field(repr=False)
    labels: np.ndarray = None

    def idx(self, name):
        return self.splits[name]

    def X(self, kind, name):
        return (self.hog if kind == "hog" else self.raw)[self.idx(name)]

    def Y(self, name):
        return self.y[self.idx(name)]


def _probed_path():
    key = f"{config.RUN_MODE}_{config.MAX_PER_CLASS}_{config.MAX_TEST}"
    return config.PROCESSED_DIR / f"metadata_probed_{key}.csv"


def _load_probed() -> pd.DataFrame:
    path = _probed_path()
    if path.exists():
        return pd.read_csv(path)
    probed = validate_dataset(build_metadata(config.MAX_PER_CLASS, config.MAX_TEST))
    probed.to_csv(path, index=False)
    return probed


# ---------------------------------------------------------------- stage 1: audit
def stage_audit() -> dict:
    """Probe every file (readable? size, md5), audit statistics, EDA-grade figures."""
    setup_logging()
    log.info("AUDIT: building metadata (run mode: %s)", config.RUN_MODE)
    meta = build_metadata(config.MAX_PER_CLASS, config.MAX_TEST)
    probed = validate_dataset(meta)
    probed.to_csv(_probed_path(), index=False)
    audit = audit_dataset(probed, M / "dataset_audit.json")
    viz.plot_class_distribution(audit["class_counts_train"])
    ok = probed[probed["readable"]]
    viz.plot_image_dimensions(ok["width"].to_numpy(), ok["height"].to_numpy())
    log.info("AUDIT done: %d files, %d unreadable, %d classes", audit["n_files_total"],
             audit["n_unreadable_or_missing"], audit["n_classes_train"])
    return audit


# ---------------------------------------------------------------- stage 2: prepare
def stage_prepare(save_figures: bool = True) -> Data:
    """Clean -> preprocess -> split -> HOG. Returns a Data object (caches make reruns cheap)."""
    setup_logging()
    config.ensure_dirs()
    probed = _load_probed()
    meta, clean_log = clean_metadata(probed)
    t0 = time.perf_counter()
    images, ok = load_processed_images(meta)
    t_pre = time.perf_counter() - t0
    if not ok.all():  # decodable by PIL during the audit but failed later (should not happen)
        log.warning("dropping %d unreadable images", (~ok).sum())
        meta, images = meta[ok].reset_index(drop=True), images[ok]
    y = meta["class_id"].to_numpy()
    groups = meta["group"].to_numpy()
    splits = split_dataset(meta)
    leak = verify_no_leakage(meta, splits)
    t0 = time.perf_counter()
    hog = get_or_compute_hog(images)
    t_hog = time.perf_counter() - t0
    raw = flatten_raw_pixels(images)
    labels = np.array(sorted(np.unique(y[np.concatenate([splits["train"], splits["val"]])])))
    data = Data(meta, images, y, groups, splits, hog, raw, labels)

    pool = np.concatenate([splits["train"], splits["val"]])
    n_total = len(meta)
    split_tbl = {s_: {"n_images": int(len(splits[s_])), "fraction": float(len(splits[s_]) / n_total),
                      "n_classes": int(len(np.unique(y[splits[s_]]))),
                      "n_groups_tracks": int(len(np.unique(groups[splits[s_]])))} for s_ in ("train", "val", "test")}
    info = {**splits["info"], "leakage_checks": leak, "n_features_raw": int(raw.shape[1]),
            "n_features_hog": int(hog.shape[1]), "hog_params": {k: list(v) if isinstance(v, tuple) else v
                                                                  for k, v in config.HOG_PARAMS.items()},
            "image_size": list(config.IMAGE_SIZE), "use_roi_crop": config.USE_ROI_CROP,
            "n_samples_after_cleaning": int(len(meta)),
            "class_counts": {s_: {int(k): int(v) for k, v in class_distribution(meta, splits[s_]).items()}
                             for s_ in ("train", "val", "test")}}
    save_json(info, M / "split_info.json")
    save_json({**clean_log, "split": split_tbl, "leakage_check": {"leakage_free": True, **leak}},
              M / "data_preparation.json")
    save_json({"n_samples_after_cleaning_and_preprocessing": int(len(meta)), "n_features_raw_pixels": int(raw.shape[1]),
               "n_features_hog": int(hog.shape[1]), "load_and_preprocess_seconds_this_call": t_pre,
               "hog_seconds_this_call": t_hog, "note": "seconds measure this call; cached runs are much faster"},
              M / "preprocessing_summary.json")
    order = np.concatenate([splits[s_] for s_ in ("train", "val", "test")])
    pd.DataFrame({"row": order, "split": sum(([s_] * len(splits[s_]) for s_ in ("train", "val", "test")), []),
                  "filename": meta["filename"].iloc[order].to_numpy(), "class_id": y[order]}
                 ).to_csv(config.PROCESSED_DIR / "splits.csv", index=False)
    if save_figures:
        viz.plot_sample_images(images[pool], y[pool])
        viz.plot_class_distribution({int(k): int(v) for k, v in class_distribution(meta, splits["train"]).items()},
                                    "class_distribution_train_split.png")
        tr = splits["train"]
        pick = [int(np.where(y[tr] == c)[0][0]) for c in labels[:: max(1, len(labels) // 6)][:6]]
        viz.plot_hog_visualization([images[tr][i] for i in pick], [f"class {y[tr][i]}" for i in pick])
        sub = np.random.RandomState(config.SEED).choice(len(tr), min(len(tr), 3000), replace=False)
        viz.plot_raw_vs_hog(raw[tr][sub], hog[tr][sub], y[tr][sub])
    log.info("PREPARE done: train=%d val=%d test=%d (%s)", len(splits["train"]), len(splits["val"]),
             len(splits["test"]), splits["info"]["test_source"])
    return data


def stage_balancing(data: Data) -> dict:
    """Measure imbalance, then decide class_weight BY EXPERIMENT on the validation split (train-only fitting)."""
    setup_logging()
    ytr, yva = data.Y("train"), data.Y("val")
    counts = pd.Series(ytr).value_counts().sort_index()
    ratio = float(counts.max() / counts.min())
    p = counts / counts.sum()
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p)))
    severity = ("balanced" if ratio <= 1.5 else "mild" if ratio <= 3 else "moderate" if ratio <= 10 else "severe")
    dist = {"imbalance_ratio_max_over_min": ratio, "min_class": int(counts.idxmin()), "min_count": int(counts.min()),
            "max_class": int(counts.idxmax()), "max_count": int(counts.max()), "normalised_entropy": entropy,
            "severity": severity}
    exp, selected = None, None
    if ratio > config.IMBALANCE_RATIO_THRESHOLD:
        _, Xtr, Xva = _scaled(data, "hog")
        exp = {}
        for label, cw in (("None", None), ("balanced", "balanced")):
            model, _ = train_svm(Xtr, ytr, class_weight=cw)
            ev, _ = evaluate_model(model, Xva, yva, f"svm class_weight={label}", "validation")
            exp[label] = {"val_accuracy": ev["accuracy"], "val_f1_macro": ev["f1_macro"],
                          "val_balanced_accuracy": ev["balanced_accuracy"]}
        selected = "balanced" if exp["balanced"]["val_f1_macro"] > exp["None"]["val_f1_macro"] else None
        decision = (f"Imbalance ratio {ratio:.2f} ({severity}). class_weight='balanced' vs None compared on the validation "
                    f"split by macro-F1 -> selected class_weight={selected}.")
    else:
        decision = f"Imbalance ratio {ratio:.2f} ({severity}): no balancing needed (class_weight=None)."
    out = {"train_distribution": dist, "experiment_on_validation": exp, "decision": decision,
           "selected_class_weight": selected, "run_mode": config.RUN_MODE,
           "alternatives_rejected": {
               "SMOTE": "interpolated HOG vectors are not valid sign images and risk leakage if applied before splitting",
               "random_oversampling": "duplicates near-identical frames, longer training, over-fitting risk",
               "undersampling": "discards scarce real data from already small classes"}}
    save_json(out, M / "balancing.json")
    log.info("BALANCING: %s", decision)
    return out


def class_weight_from_decision(data: Data | None = None):
    """Selected class_weight (None | 'balanced'); runs the balancing experiment first if it has not been run."""
    f = M / "balancing.json"
    if not f.exists():
        if data is None:
            return None
        stage_balancing(data)
    return load_json(f)["selected_class_weight"]


def stage_eda(data: Data) -> dict:
    """EDA on the TRAINING split only: brightness/contrast, class sizes, similar classes, anomalies."""
    setup_logging()
    tr = data.idx("train")
    imgs, y = data.images[tr], data.y[tr]
    flat = imgs.reshape(len(tr), -1).astype(np.float32)
    bright, contrast = flat.mean(axis=1), flat.std(axis=1)
    viz.plot_brightness_contrast(bright, contrast)
    viz.plot_class_mean_images(imgs, y)
    means = np.vstack([data.hog[tr][y == c].mean(axis=0) for c in data.labels])
    nm = means / np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
    sim = nm @ nm.T
    np.fill_diagonal(sim, -1)
    pairs = []
    for flat_i in np.argsort(sim, axis=None)[::-1]:
        i, j = divmod(int(flat_i), sim.shape[1])
        if i < j:
            pairs.append({"class_a": int(data.labels[i]), "name_a": config.class_name(data.labels[i]),
                          "class_b": int(data.labels[j]), "name_b": config.class_name(data.labels[j]),
                          "cosine_similarity": float(sim[i, j])})
        if len(pairs) == 10:
            break
    pd.DataFrame(pairs).to_csv(M / "eda_similar_classes.csv", index=False)
    viz.plot_similar_pairs(imgs, y, pairs[:6])
    mt = data.meta.iloc[tr]
    widths = {int(c): mt.loc[mt["class_id"] == c, "width"].to_numpy() for c in data.labels}
    viz.plot_size_by_class(widths)
    sizes = pd.Series({c: len(w) for c, w in widths.items()})
    med_w = pd.Series({c: float(np.median(w)) for c, w in widths.items()})
    rho = spearmanr(sizes.sort_index(), med_w.sort_index())[0] if len(sizes) > 2 else float("nan")
    q1, q3 = np.percentile(bright, [25, 75])
    iqr = q3 - q1
    out = {"run_mode": config.RUN_MODE, "n_train_images": int(len(tr)),
           "n_dark_train_images": int((bright < 40).sum()), "n_low_contrast_train_images": int((contrast < 15).sum()),
           "brightness": {"mean": float(bright.mean()), "std": float(bright.std()), "min": float(bright.min()),
                          "max": float(bright.max()),
                          "n_iqr_outliers": int(((bright < q1 - 1.5 * iqr) | (bright > q3 + 1.5 * iqr)).sum())},
           "contrast_mean": float(contrast.mean()),
           "spearman_class_size_vs_median_width": {"rho": float(rho)},
           "most_similar_class_pair": pairs[0] if pairs else None,
           "hog_feature_stats": {"mean": float(data.hog.mean()), "std": float(data.hog.std()),
                                 "share_zero": float((data.hog == 0).mean())}}
    save_json(out, M / "eda_summary.json")
    return out


# ---------------------------------------------------------------- helpers
def _scaled(data: Data, kind: str):
    scaler = fit_scaler(data.X(kind, "train"))  # TRAIN ONLY
    return scaler, scaler.transform(data.X(kind, "train")), scaler.transform(data.X(kind, "val"))


def _record(name, features, model, train_time, Xtr, ytr, Xva, yva, notes, model_dir=None, fname=None,
            metrics_file=None, cm_name=None):
    ev, pred = evaluate_model(model, Xva, yva, name, "validation", Xtr, ytr)
    row = comparison_row(ev, name, features, train_time, notes)
    upsert_model_comparison(row)
    if model_dir:
        joblib.dump(model, Path(model_dir) / fname)
    if metrics_file:
        save_json({**ev, "training_time_seconds": train_time, "features": features, "notes": notes,
                   "note_test": "Test-set results are produced once, by stage_final_eval."}, metrics_file)
    if cm_name:
        labels = np.unique(np.concatenate([yva, pred]))
        viz.plot_confusion_matrix(confusion(yva, pred, labels), labels, cm_name, f"{name} - validation set")
    return ev, pred


# ---------------------------------------------------------------- stage 3: baseline
def stage_baseline(data: Data) -> dict:
    setup_logging()
    cw = class_weight_from_decision(data)
    scaler, Xtr, Xva = _scaled(data, "raw")
    model, t = train_svm(Xtr, data.Y("train"), class_weight=cw)
    joblib.dump(scaler, config.MODELS_DIR / "baseline_svm" / "scaler.joblib")
    ev, _ = _record("Raw Pixel SVM", RAW_LABEL, model, t, Xtr, data.Y("train"), Xva, data.Y("val"),
                    "Baseline: untuned RBF SVM on flattened grayscale pixels", config.MODELS_DIR / "baseline_svm",
                    "baseline_svm.joblib", M / "baseline_svm.json", "baseline_confusion_matrix.png")
    log.info("BASELINE raw-pixel SVM val macro-F1 = %.4f", ev["f1_macro"])
    return ev


# ---------------------------------------------------------------- stage 4: compare
def stage_compare(data: Data) -> pd.DataFrame:
    """Same HOG features / same train-val split / same metrics for KNN, Decision Tree and SVM."""
    setup_logging()
    cw = class_weight_from_decision(data)
    scaler, Xtr, Xva = _scaled(data, "hog")
    ytr, yva = data.Y("train"), data.Y("val")
    specs = [("HOG KNN", train_knn, {}, "knn", "knn_default.joblib", "k=5, uniform, euclidean (no class weights)"),
             ("HOG Decision Tree", train_decision_tree, {"class_weight": cw}, "decision_tree",
              "decision_tree_default.joblib", f"default depth, class_weight={cw}"),
             ("HOG SVM", train_svm, {"class_weight": cw}, "svm", "hog_svm_default.joblib",
              f"RBF, C=1, gamma=scale, class_weight={cw}")]
    for name, fn, kw, folder, fname, notes in specs:
        model, t = fn(Xtr, ytr, **kw)
        ev, _ = _record(name, HOG_LABEL, model, t, Xtr, ytr, Xva, yva, notes, config.MODELS_DIR / folder, fname,
                        M / f"{folder}_default.json", f"{folder}_default_confusion_matrix.png")
        log.info("COMPARE %s val macro-F1 = %.4f", name, ev["f1_macro"])
    return finalize_comparison()


def finalize_comparison() -> pd.DataFrame:
    path = config.COMPARISON_DIR / "model_comparison.csv"
    df = pd.read_csv(path)
    order = ["Raw Pixel SVM", "HOG KNN", "HOG Decision Tree", "HOG SVM", "Tuned HOG KNN",
             "Tuned HOG Decision Tree", "Tuned HOG SVM"]
    df["o"] = df["model"].map({m: i for i, m in enumerate(order)})
    df = df.sort_values("o").drop(columns="o")
    df.to_csv(path, index=False)
    untuned_hog = ["HOG KNN", "HOG Decision Tree", "HOG SVM"]
    rq = {"primary_metric": config.PRIMARY_METRIC, "split": "validation", "run_mode": config.RUN_MODE,
          "RQ1_raw_vs_hog_svm": None, "RQ2_best_untuned_hog_classifier": best_model(df, candidates=untuned_hog),
          "overall_best_on_validation": best_model(df)}
    have = set(df["model"])
    if {"Raw Pixel SVM", "HOG SVM"} <= have:
        a = df[df.model == "Raw Pixel SVM"].iloc[0]
        b = df[df.model == "HOG SVM"].iloc[0]
        rq["RQ1_raw_vs_hog_svm"] = {"raw_f1_macro": float(a.f1_macro), "hog_f1_macro": float(b.f1_macro),
                                    "raw_accuracy": float(a.accuracy), "hog_accuracy": float(b.accuracy),
                                    "hog_improves_f1_macro": bool(b.f1_macro > a.f1_macro),
                                    "absolute_f1_macro_difference": float(b.f1_macro - a.f1_macro)}
    save_json(rq, config.COMPARISON_DIR / "research_questions.json")
    viz.plot_model_comparison(df, title="Validation-set comparison (macro-F1 and accuracy)")
    return df


# ---------------------------------------------------------------- stage 5: tuning
def stage_tune(data: Data, tune_others: bool = True, full_grid: bool = False, n_jobs: int = 1) -> dict:
    """GridSearchCV on TRAIN only (grouped CV), refit best on TRAIN, score on VALIDATION."""
    setup_logging()
    cw = class_weight_from_decision(data)
    scaler, Xtr, Xva = _scaled(data, "hog")
    ytr, yva, gtr = data.Y("train"), data.Y("val"), data.groups[data.idx("train")]
    raw_tr = data.X("hog", "train")  # unscaled: scaler is part of the CV pipeline
    out = {}
    plan = [("svm", "Tuned HOG SVM", train_svm, "svm", "tuned_svm_candidate.joblib", {"class_weight": cw})]
    if tune_others:
        plan += [("knn", "Tuned HOG KNN", train_knn, "knn", "knn_tuned.joblib", {}),
                 ("tree", "Tuned HOG Decision Tree", train_decision_tree, "decision_tree",
                  "decision_tree_tuned.joblib", {"class_weight": cw})]
    for key, name, fn, folder, fname, kw in plan:
        log.info("TUNE %s ...", key)
        res, best = tune_estimator(key, raw_tr, ytr, gtr, class_weight=cw if key != "knn" else None,
                                   full_grid=full_grid, n_jobs=n_jobs)
        model, t = fn(Xtr, ytr, **kw, **best)
        before = load_json(M / f"{folder}_default.json")
        ev, _ = _record(name, HOG_LABEL, model, t, Xtr, ytr, Xva, yva, f"GridSearchCV best: {best}",
                        config.MODELS_DIR / folder, fname, M / f"{folder}_tuned.json",
                        f"{folder}_tuned_confusion_matrix.png")
        res.update({"validation_f1_macro_before_tuning": before["f1_macro"],
                    "validation_f1_macro_after_tuning": ev["f1_macro"],
                    "validation_accuracy_before_tuning": before["accuracy"],
                    "validation_accuracy_after_tuning": ev["accuracy"], "run_mode": config.RUN_MODE})
        save_json(res, M / f"gridsearch_{key}.json")
        pd.DataFrame(res["cv_results"]).to_csv(M / f"gridsearch_{key}_cv_results.csv", index=False)
        out[key] = res
        log.info("TUNE %s best=%s cv=%.4f val-F1 %.4f -> %.4f", key, best, res["best_cv_score"],
                 before["f1_macro"], ev["f1_macro"])
        if key == "svm":
            viz.plot_tuning_heatmap(res["cv_results"])
            _save_final_model(data, scaler, model, best, res, ev, cw)
    finalize_comparison()
    return out


def _save_final_model(data, scaler, model, best, res, ev, cw):
    d = config.MODELS_DIR / "svm"
    joblib.dump(model, d / config.FINAL_MODEL_NAME)
    (d / "tuned_svm_candidate.joblib").unlink(missing_ok=True)
    joblib.dump(scaler, d / config.SCALER_NAME)
    classes = [int(c) for c in model.classes_]
    save_json(best, d / "best_parameters.json")
    save_json({"class_id_to_name": {str(c): config.class_name(c) for c in classes}}, d / config.LABEL_MAPPING_NAME)
    save_json({
        "project": "RoadEye", "model_type": "sklearn.svm.SVC (HOG features, StandardScaler)",
        "artifact_files": {"model": config.FINAL_MODEL_NAME, "scaler": config.SCALER_NAME,
                           "label_mapping": config.LABEL_MAPPING_NAME},
        "pipeline": ["ROI crop (training only; the app uses the whole image)", "grayscale", f"resize {config.IMAGE_SIZE}",
                     "HOG", "StandardScaler (fit on train)", "SVC"],
        "input_size": list(config.IMAGE_SIZE), "hog_config": {k: list(v) if isinstance(v, tuple) else v
                                                              for k, v in config.HOG_PARAMS.items()},
        "n_features": int(model.n_features_in_), "n_classes": len(classes), "classes": classes,
        "dataset": config.DATASET_NAME, "run_mode": config.RUN_MODE,
        "best_hyperparameters": best, "class_weight": cw,
        "cv_macro_f1_on_tuning_subsample": res["best_cv_score"],
        "validation_scores": {k: ev[k] for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro")},
        "random_seed": config.SEED, "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "numpy": np.__version__}}, d / config.METADATA_NAME)


# ---------------------------------------------------------------- stage 6: final test evaluation
def _error_analysis(data: Data, pred, yte, pc: pd.DataFrame) -> dict:
    """Systematic error patterns: speed-limit family, original image size, class size vs recall."""
    wrong = pred != yte
    mt = data.meta.iloc[data.idx("test")]
    min_side = np.minimum(mt["width"].to_numpy(), mt["height"].to_numpy())
    edges, names = [0, 30, 40, 50, 70, 10 ** 6], ["<30", "30-39", "40-49", "50-69", "70+"]
    by_size = []
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (min_side >= lo) & (min_side < hi)
        by_size.append({"original_min_side": nm, "n": int(m.sum()),
                        "error_rate": float(wrong[m].mean()) if m.sum() else None})
    speed = set(range(0, 9))
    sp = [(t in speed and p in speed) for t, p in zip(yte[wrong], pred[wrong])]
    ytr = data.Y("train")
    cnt = pd.Series(ytr).value_counts()
    recall = pc.set_index("class_id")["recall"]
    common = [c for c in recall.index if c in cnt.index and pc.set_index("class_id").loc[c, "support"] > 0]
    rho = spearmanr(cnt.loc[common], recall.loc[common])[0] if len(common) > 2 else float("nan")
    out = {"n_errors": int(wrong.sum()), "n_test": int(len(yte)), "error_rate": float(wrong.mean()),
           "share_of_errors_between_speed_limit_signs": float(np.mean(sp)) if sp else 0.0,
           "error_rate_by_original_image_size": by_size,
           "spearman_train_class_size_vs_recall": {"rho": float(rho)}}
    save_json(out, M / "error_analysis.json")
    return out


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def stage_final_eval(data: Data, force: bool = False) -> dict:
    """Evaluate the FROZEN final model (and, for reporting only, the other frozen models) on the test set.

    A lock file records the model hash; a second evaluation of the same model replays the saved
    predictions instead of touching the test set again. Re-evaluating a *changed* model needs force=True.
    """
    setup_logging()
    final_path = config.MODELS_DIR / "svm" / config.FINAL_MODEL_NAME
    if not final_path.exists():
        raise FileNotFoundError("Final model missing - run the tuning stage first.")
    lock = M / ".test_evaluated.lock"
    preds_path = M / "test_predictions.npz"
    h = _md5(final_path)
    yte = data.Y("test")
    labels = data.labels
    if lock.exists() and not force:
        if lock.read_text().split()[0] != h:
            raise RuntimeError("The final model changed after the test set was already used. Refusing to "
                               "re-evaluate (that would be test-set tuning). Pass force=True only to "
                               "deliberately overwrite, and document it.")
        log.info("Test set already evaluated for this model: replaying stored predictions.")
        z = np.load(preds_path, allow_pickle=False)
        pred = z["final_pred"]
        res = load_json(M / "final_test_metrics.json")
    else:
        log.info("FINAL EVALUATION on test set (single pass)")
        rows, preds = [], {}
        scaler_h = joblib.load(config.MODELS_DIR / "svm" / config.SCALER_NAME)
        scaler_r = joblib.load(config.MODELS_DIR / "baseline_svm" / "scaler.joblib")
        frozen = [("Raw Pixel SVM", "raw", scaler_r, config.MODELS_DIR / "baseline_svm" / "baseline_svm.joblib"),
                  ("HOG KNN", "hog", scaler_h, config.MODELS_DIR / "knn" / "knn_default.joblib"),
                  ("HOG Decision Tree", "hog", scaler_h, config.MODELS_DIR / "decision_tree" / "decision_tree_default.joblib"),
                  ("HOG SVM", "hog", scaler_h, config.MODELS_DIR / "svm" / "hog_svm_default.joblib"),
                  ("Tuned HOG KNN", "hog", scaler_h, config.MODELS_DIR / "knn" / "knn_tuned.joblib"),
                  ("Tuned HOG Decision Tree", "hog", scaler_h, config.MODELS_DIR / "decision_tree" / "decision_tree_tuned.joblib"),
                  ("Tuned HOG SVM", "hog", scaler_h, final_path)]
        for name, kind, sc, path in frozen:
            if not Path(path).exists():
                continue
            model = joblib.load(path)
            ev, p = evaluate_model(model, sc.transform(data.X(kind, "test")), yte, name, "test")
            rows.append(ev)
            preds[name] = p
        final = next(r for r in rows if r["model"] == "Tuned HOG SVM")
        pred = preds["Tuned HOG SVM"]
        pd.DataFrame(rows).to_csv(M / "final_test_all_models.csv", index=False)
        val_cmp = pd.read_csv(config.COMPARISON_DIR / "model_comparison.csv")[["model", "feature_representation", "training_time_seconds", "notes"]]
        pd.DataFrame(rows).merge(val_cmp, on="model", how="left").to_csv(
            config.COMPARISON_DIR / "model_comparison_test.csv", index=False)
        np.savez_compressed(preds_path, final_pred=pred, y_true=yte, test_rows=data.idx("test"))
        pc = per_class_table(yte, pred, labels)
        pc.to_csv(M / "final_test_per_class.csv", index=False)
        cm = confusion(yte, pred, labels)
        np.save(config.CONFUSION_DIR / "final_confusion_matrix.npy", cm)
        tc = top_confusions(cm, labels)
        tc.to_csv(M / "final_test_top_confusions.csv", index=False)
        _error_analysis(data, pred, yte, pc)
        val = load_json(M / "svm_tuned.json")
        trn = val.get("train_accuracy")
        res = {**final, "run_mode": config.RUN_MODE, "test_source": data.splits["info"]["test_source"],
               "n_test": int(len(yte)), "train_accuracy": trn, "validation_accuracy": val["accuracy"],
               "validation_f1_macro": val["f1_macro"], "test_accuracy": final["accuracy"],
               "generalisation_gap_val_minus_test_accuracy": val["accuracy"] - final["accuracy"],
               "worst_5_classes_by_f1": pc.sort_values("f1").head(5).to_dict(orient="records"),
               "top_confusions": tc.head(10).to_dict(orient="records"),
               "evaluated_model_md5": h, "forced_reevaluation": bool(force)}
        save_json(res, M / "final_test_metrics.json")
        lock.write_text(f"{h} {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}\n")
        with open(config.LOGS_DIR / "test_evaluation_log.txt", "a") as fh:
            fh.write(f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} evaluated model {h} force={force}\n")
    # figures (from stored predictions only)
    cm = confusion(yte, pred, labels)
    pc = per_class_table(yte, pred, labels)
    viz.plot_confusion_matrix(cm, labels, "final_confusion_matrix.png", "Final model (Tuned HOG SVM) - test set")
    viz.plot_per_class_f1(pc, "final_per_class_f1.png", "Per-class F1 on the test set")
    viz.plot_misclassified(data.images[data.idx("test")], yte, pred, "misclassified_examples.png")
    all_models = pd.read_csv(M / "final_test_all_models.csv")
    viz.plot_model_comparison(all_models, path="model_performance_comparison_test.png",
                              title="Final test-set results of all frozen models")
    return res
