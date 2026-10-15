"""Text of the 12 stage documents, the final report and helpers.

Rules
-----
* Every number comes from ``Results`` (files written by the pipeline). If a file does
  not exist the text shows ``Pending execution`` - nothing is typed in by hand.
* Statements that are *theory* (what we expect) are labelled as such and are separate
  from *measured* statements.

Block types:  ("h", level, text) ("p", text) ("b", [items]) ("t", header, rows, caption)
              ("img", filename, caption) ("note", text)
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.evaluate import load_json

PENDING = "Pending execution"


from scripts.results_loader import Results  # noqa: E402,F401


def f(x, nd=4):
    return PENDING if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x))


def pct(x, nd=2):
    return PENDING if x is None else f"{100 * x:.{nd}f}%"


def g(d, *path):
    """Safe nested get -> None if anything is missing."""
    for p in path:
        if d is None:
            return None
        try:
            d = d[p]
        except (KeyError, IndexError, TypeError):
            return None
    return d


def hog_desc() -> str:
    p = config.HOG_PARAMS
    return (f"{p['orientations']} orientations, {p['pixels_per_cell'][0]}x{p['pixels_per_cell'][1]} pixels per cell, "
            f"{p['cells_per_block'][0]}x{p['cells_per_block'][1]} cells per block, {p['block_norm']} block normalisation")


def metrics_rows(R, source="val"):
    keys = [("baseline_svm", "Raw Pixel SVM"), ("knn", "HOG KNN"), ("decision_tree", "HOG Decision Tree"),
            ("svm_default", "HOG SVM"), ("svm_tuned", "Tuned HOG SVM")]
    rows = []
    for key, name in keys:
        mm = g(R.models.get(key), source)
        rows.append([name] + ([f(mm["accuracy"]), f(mm["precision_macro"]), f(mm["recall_macro"]), f(mm["f1_macro"])]
                              if mm else [PENDING] * 4))
    return rows


def comparison_table(df, with_times=True):
    header = ["Model", "Features", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"] + (["Training time (s)", "Inference time (s)"] if with_times else [])
    if df is None or df.empty:
        return header, [[PENDING] * len(header)]
    rows = []
    for _, r in df.iterrows():
        row = [r["model"], r["feature_representation"], f(float(r["accuracy"])), f(float(r["precision_macro"])),
               f(float(r["recall_macro"])), f(float(r["f1_macro"]))]
        if with_times:
            row += [f"{r['training_time_seconds']:.2f}", f"{r['inference_time_seconds']:.3f}"]
        rows.append(row)
    return header, rows


# --------------------------------------------------------------------------- #
# Auto-generated analytical statements (only from measured numbers)
# --------------------------------------------------------------------------- #
def rq1_statement(R) -> str:
    q = g(R.final, "research_question_1")
    if not q:
        return f"Research Question 1 result: {PENDING}."
    d, a = q["delta_f1_macro"], q["delta_accuracy"]
    verdict = ("improved" if d > 0.005 else "did not meaningfully change" if abs(d) <= 0.005 else "reduced")
    return (f"On the untouched test split, HOG + SVM reached macro-F1 {q['hog_svm_default_test']['f1_macro']:.4f} "
            f"(accuracy {q['hog_svm_default_test']['accuracy']:.4f}) versus {q['raw_pixel_svm_test']['f1_macro']:.4f} "
            f"(accuracy {q['raw_pixel_svm_test']['accuracy']:.4f}) for raw pixels + SVM with identical SVM settings and split. "
            f"The difference is {d:+.4f} macro-F1 and {a:+.4f} accuracy, i.e. HOG {verdict} classification quality relative to raw pixels. "
            f"HOG also used {g(R.pre, 'n_features_hog')} features instead of {g(R.pre, 'n_features_raw_pixels')}.")


def rq2_statement(R) -> str:
    q = g(R.final, "research_question_2")
    if not q:
        return f"Research Question 2 result: {PENDING}."
    sc = ", ".join(f"{k} {v:.4f}" for k, v in q["test_scores"].items())
    txt = (f"Test macro-F1 with identical HOG features and default settings: {sc}. "
           f"Winner on the test split: {q['winner_on_test']}; winner on the validation split (the split used for selection): {q['winner_on_validation']}.")
    if q["winner_on_test"] != "HOG SVM":
        txt += (" SVM did NOT win this comparison; this is reported as measured. SVM remains the proposed model of the project "
                "because it is the focus of the study and is tuned in Stage 10.")
    return txt


def error_statement(R) -> str:
    e = R.err
    if not e:
        return f"Error-analysis findings: {PENDING}."
    parts = [f"{e['n_errors']} of {e['n_test']} test images were misclassified (error rate {e['error_rate']:.4f})."]
    if R.top_conf is not None and len(R.top_conf):
        t = R.top_conf.iloc[0]
        parts.append(f"The most frequent confusion is class {int(t.true_id)} ({t.true_name}) predicted as class {int(t.pred_id)} "
                     f"({t.pred_name}), {int(t['count'])} time(s).")
    parts.append(f"{100 * e['share_of_errors_between_speed_limit_signs']:.1f}% of the errors are between two speed-limit classes.")
    bs = [x for x in e["error_rate_by_original_image_size"] if x["error_rate"] is not None]
    if bs:
        parts.append("Error rate by original image size: " + "; ".join(f"{x['original_min_side']}: {x['error_rate']:.3f} (n={x['n']})" for x in bs) + ".")
    rho = e["spearman_train_class_size_vs_recall"]["rho"]
    parts.append(f"Spearman correlation between training-class size and per-class recall: {rho:.2f}.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Stage blocks
# --------------------------------------------------------------------------- #
def stage1(R):
    return [
        ("h", 2, "Project title"), ("p", "RoadEye - Traffic Sign Recognition Using HOG Features and SVM"),
        ("h", 2, "Background"),
        ("p", "Traffic signs encode the rules of the road in a compact visual language of shape, colour and pictogram. Driver-assistance "
              "systems (ADAS), road-asset inventories and navigation-map updates all need software that can recognise which sign is "
              "shown in an image. Modern solutions often use deep neural networks; this project deliberately studies the *classical "
              "machine-learning* route - hand-engineered features plus a conventional classifier - because it is transparent, cheap "
              "to train and lets us measure the contribution of each design decision."),
        ("h", 2, "Problem statement"),
        ("p", "Given a cropped colour photograph of a single traffic sign, predict which of the 43 GTSRB sign categories it belongs to, "
              "and determine experimentally (a) whether HOG feature engineering improves on raw pixels and (b) which syllabus classifier "
              "(KNN, Decision Tree, SVM) works best on the same HOG features."),
        ("h", 2, "Motivation"),
        ("b", ["Road-safety relevance: sign misreading is a known failure source for driver-assistance functions.",
               "Educational value: the project exercises the full ML pipeline - data audit, leakage-safe splitting, feature engineering, baseline, model comparison, tuning, evaluation and deployment demo.",
               "Low-resource: classical models train in minutes on a laptop CPU and are easy to explain and reproduce."]),
        ("h", 2, "Real-world use case"),
        ("p", "A sign-recognition module that receives an already-localised sign crop (from a detector or from a map-survey tool) and "
              "returns the sign class, e.g. to cross-check a speed-limit database or to flag a missing/damaged sign in a road-asset survey. "
              "A Streamlit demo shows this single-image classification."),
        ("h", 2, "Stakeholders"),
        ("b", ["Road users and drivers (safety benefit)", "Road authorities / asset managers (sign inventory and maintenance)",
               "Driver-assistance and map-data developers (recognition component)", "Course instructors and the student (assessment and learning)"]),
        ("h", 2, "Machine-learning formulation"),
        ("t", ["Item", "Definition"],
         [["Task", "Supervised multiclass image classification"],
          ["Input (x)", f"RGB image of one sign -> grayscale {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]} -> HOG feature vector"],
          ["Output (y)", "class id in {0, ..., 42} (sign category)"],
          ["Learning problem", "learn f(x) = y from labelled examples; generalise to unseen physical signs"],
          ["Primary metric", "macro-F1 (also accuracy, macro-precision, macro-recall, confusion matrix)"]], "ML formulation"),
        ("h", 2, "Constraints"),
        ("b", ["Syllabus: classical models only (KNN, Decision Tree, SVM); deep learning appears only as future scope.",
               "One-week scope and laptop-class compute (no GPU).", "No data leakage; test set used exactly once; all results measured by running the code."]),
        ("h", 2, "Scope and out-of-scope"),
        ("b", ["In scope: classification of cropped sign images; HOG vs raw pixels; KNN vs Decision Tree vs SVM; GridSearchCV; Streamlit demo.",
               "Out of scope: sign detection/localisation, video, real-time operation, CNN/YOLO models, deployment to vehicles or the cloud, safety certification."]),
        ("h", 2, "Expected impact"),
        ("p", "A reproducible, fully documented reference implementation showing how far classical features can go on a standard benchmark, "
              "with an honest account of its errors and limitations."),
        ("h", 2, "Research questions"),
        ("b", ["RQ1: Does HOG feature engineering improve traffic-sign classification compared with raw pixels (Raw Pixels + SVM vs HOG + SVM)?",
               "RQ2: Which of KNN, Decision Tree and SVM performs best on the same HOG features? (Determined experimentally - not assumed.)"]),
    ]


def stage2(R):
    a = R.audit
    rows = [["Source", "German Traffic Sign Recognition Benchmark (GTSRB), Institut fuer Neuroinformatik, Ruhr-Universitaet Bochum"],
            ["Reference", "J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks 32 (2012) 323-332."],
            ["Official page", "https://benchmark.ini.rub.de/"],
            ["Version used (this run)", g(a, "dataset_identifier") or PENDING],
            ["Image files indexed", f(g(a, "n_files_indexed"))], ["Readable images", f(g(a, "n_readable"))],
            ["Number of classes", f(g(a, "n_classes"))], ["Physical signs (tracks)", f(g(a, "n_tracks_total"))],
            ["File formats", f(g(a, "file_formats"))],
            ["Image width (min / median / max)", f"{g(a, 'width', 'min')} / {g(a, 'width', 'median')} / {g(a, 'width', 'max')}" if a else PENDING],
            ["Image height (min / median / max)", f"{g(a, 'height', 'min')} / {g(a, 'height', 'median')} / {g(a, 'height', 'max')}" if a else PENDING],
            ["Total size (MB)", f(g(a, "total_size_mb"), 1)],
            ["Target variable", "class_id (0-42) = folder name; cross-checked with official GT-xxxxx.csv"]]
    blocks = [("h", 2, "Dataset source and licence / usage terms"),
              ("p", "GTSRB is a public benchmark published for research on traffic-sign recognition. The licence / usage statement of the exact "
                    "copy used must be read from the page it was downloaded from (the official site or a mirror such as Kaggle, which states its "
                    "own licence). See DATASET_DOWNLOAD.md and LICENSE_NOTICE.md. The dataset is used here only for non-commercial coursework; "
                    "it is not redistributed inside the project ZIP."),
              ("h", 2, "Dataset summary (measured)"), ("t", ["Property", "Value"], rows, "Dataset summary - generated by Notebook 01")]
    blocks += [("h", 2, "Why GTSRB?"),
               ("b", ["Recognised benchmark with a published protocol, so results are meaningful and comparable.",
                      "43 classes with real-world variation (illumination, blur, scale, partial occlusion).",
                      "Large enough for classical ML and small enough (images of at most a few hundred pixels) for a laptop."]),
               ("h", 2, "Target definition and class distribution"),
               ("p", "The target is the sign category (43 classes: speed limits, prohibitory, warning, mandatory and end-of-restriction signs). "
                     "The distribution is shown below and analysed in Stage 5."),
               ("img", "class_distribution.png", "Images per class (measured)"),
               ("img", "sample_images.png", "One random image per class"),
               ("img", "image_size_distribution.png", "Original image sizes (resizing to one size is required)"),
               ("h", 2, "Dataset limitations"),
               ("b", ["All images are from German roads; other countries use different sign designs.",
                      "~30 frames per physical sign are highly correlated -> splitting by sign (group) is mandatory.",
                      "Images are already cropped to the sign (with a small border); there is no detection task.",
                      "Strong class imbalance is typical of GTSRB (see Stage 5 for the measured ratio).",
                      "Small, blurry or dark images carry little information at 32x32 pixels."])]
    return blocks


def stage3(R):
    p, a = R.prep, R.audit
    rows = [["Files indexed (before)", f(g(a, "n_files_indexed"))],
            ["Removed: unreadable / corrupt", f(g(p, "removed_unreadable"))],
            ["Removed: folder label disagrees with official CSV", f(g(p, "removed_label_mismatch_with_csv"))],
            ["Removed: byte-identical files with conflicting labels", f(g(p, "removed_label_conflicting_duplicates"))],
            ["Removed: redundant exact duplicates (first copy kept)", f(g(p, "removed_exact_duplicates_kept_first"))],
            ["Samples after cleaning", f(g(p, "n_after"))], ["Data scope", g(p, "scope") or PENDING]]
    sp = g(p, "split")
    srows = ([[k, f(v["n_images"]), pct(v["fraction"], 1), f(v["n_classes"]), f(v["n_groups_tracks"])] for k, v in sp.items()] if sp else [[PENDING] * 5])
    return [("h", 2, "Raw data structure"),
            ("p", "Images are organised as one folder per class (00000 ... 00042); in the official archive each folder also holds a semicolon-separated "
                  "GT-xxxxx.csv. File names encode the physical-sign track and frame (<track>_<frame>.ppm)."),
            ("h", 2, "Cleaning steps"),
            ("b", ["Corrupted files: every file is fully decoded; unreadable files are logged and removed.",
                   "Label validation: the class label is the folder name; the official CSV ClassId is an independent cross-check; disagreements are removed.",
                   "Duplicate checking: MD5 hash of the file bytes. Identical bytes with different labels are removed entirely; identical bytes with the same label keep one copy. "
                   "Near-duplicates (consecutive frames of one sign) are genuine samples and are kept - they are handled by the group-aware split.",
                   f"Image resizing and organisation: images are resized to {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]} in Stage 4 and cached in data/processed/ as arrays aligned to split.csv."]),
            ("h", 2, "Before / after sample counts (measured)"), ("t", ["Step", "Count"], rows, "Data cleaning log"),
            ("h", 2, "Train / validation / test split"),
            ("p", f"Test set: {g(R.split_info, 'test_source') or 'the official GTSRB test set (with its published labels GT-final_test.csv) when available; otherwise a grouped hold-out of the training pool, and the report states which one was used'}. "
                  f"It is separated first and not touched again until the final evaluation. The remaining training pool is split into training and validation (~{int((1 - config.VAL_FRACTION) * 100)}/{int(config.VAL_FRACTION * 100)}) "
                  f"with StratifiedGroupKFold (random seed {config.RANDOM_SEED}); the group is the physical sign (class + track). Splitting by sign prevents "
                  "near-identical frames from appearing in both training and validation, which would inflate every score. Because folds contain whole signs, the "
                  "fractions are approximate."),
            ("t", ["Subset", "Images", "Share", "Classes", "Physical signs"], srows, "Split sizes (measured)"),
            ("p", f"Leakage check (no shared file path, physical sign or identical image bytes between subsets): "
                  f"{g(p, 'leakage_check', 'leakage_free') if p else PENDING}.")]


def stage4(R):
    s = R.pre
    return [("h", 2, "Preprocessing pipeline"),
            ("t", ["Step", "What", "Why it is needed"],
             [["ROI crop", "crop to the sign's region of interest given in the official CSV (removes the ~10% border)", "keeps the sign, drops background"],
              ["Grayscale", "RGB -> 1 channel", "HOG relies on gradients; removes colour variation; fewer values"],
              ["Resize", f"{config.IMG_SIZE[0]}x{config.IMG_SIZE[1]} with INTER_AREA", "fixed-length feature vector; INTER_AREA is the correct filter for shrinking"],
              ["Pixel normalisation", "uint8 -> float32 in [0, 1]", "numerical range; no information is learned from data"],
              ["HOG extraction", hog_desc(), "edge-orientation descriptor robust to illumination"],
              ["Feature scaling", "StandardScaler fitted on TRAIN only", "SVM and KNN are scale sensitive; leakage-free"]], "Preprocessing steps"),
            ("h", 2, "Missing values, outliers, encoding"),
            ("b", ["Missing values: image data has no missing cells; unreadable files are the equivalent and are removed (Stage 3).",
                   "Outliers/anomalies: very small, dark or low-contrast images are quantified in EDA; they are real signs, so they are kept (removing them would bias the test set towards easy images).",
                   "Encoding: the target is already an integer class id (0-42); no categorical encoding of features is needed."]),
            ("h", 2, "Measured effect"),
            ("t", ["Item", "Value"],
             [["Samples after cleaning + preprocessing", f(g(s, "n_samples_after_cleaning_and_preprocessing"))],
              ["Raw-pixel features", f(g(s, "n_features_raw_pixels"))], ["HOG features", f(g(s, "n_features_hog"))]], "Preprocessing summary"),
            ("h", 2, "Data-leakage prevention"),
            ("b", ["Split first, fit afterwards: scaler, class-weight decision, models and GridSearchCV see the training split only.",
                   "Inside GridSearchCV the scaler is part of the pipeline, so it is re-fitted on each fold's training part.",
                   "Preprocessing steps that learn nothing from data (grayscale, resize, /255, HOG) are applied identically to all subsets.",
                   "No feature selection or balancing used validation/test information; the test split is evaluated once (audit log in results/logs/)."])]


def stage5(R):
    b = R.bal
    d = g(b, "train_distribution")
    rows = ([["Imbalance ratio (max/min class count)", f(d["imbalance_ratio_max_over_min"], 2)], ["Smallest class", f"class {d['min_class']} ({d['min_count']} images)"],
             ["Largest class", f"class {d['max_class']} ({d['max_count']} images)"], ["Normalised entropy (1 = balanced)", f(d["normalised_entropy"], 3)],
             ["Severity (rule: <=1.5 balanced, <=3 mild, <=10 moderate, else severe)", d["severity"]]] if d else [[PENDING, PENDING]])
    exp = g(b, "experiment_on_validation")
    erows = ([[k, f(v["val_accuracy"]), f(v["val_f1_macro"]), f(v["val_balanced_accuracy"])] for k, v in exp.items()] if exp else [[PENDING] * 4])
    return [("h", 2, "Class-distribution analysis (training split)"), ("t", ["Statistic", "Value"], rows, "Imbalance statistics (measured)"),
            ("img", "class_distribution.png", "Class distribution"),
            ("h", 2, "Strategy selection"),
            ("p", "Strategy is chosen by experiment, not by habit: the default HOG SVM is trained with class_weight=None and class_weight='balanced' on the training split "
                  "and the two are compared on the VALIDATION split by macro-F1."),
            ("t", ["class_weight", "Val accuracy", "Val macro-F1", "Val balanced accuracy"], erows, "Balancing experiment (validation split)"),
            ("p", f"Decision: {g(b, 'decision') or PENDING}"),
            ("p", f"Selected class_weight: {str(g(b, 'selected_class_weight')) if b else PENDING} (applies to SVM and Decision Tree; KNN has no class weights)."),
            ("h", 2, "Why alternatives were not used"),
            ("b", ["SMOTE / synthetic oversampling: interpolating HOG vectors does not create valid sign images and risks leakage if done before splitting; not needed when class weights suffice.",
                   "Random oversampling: duplicates training samples, increases training time and over-fitting risk.",
                   "Undersampling: discards scarce real data from already small classes."]),
            ("h", 2, "Confirmation of no leakage"),
            ("p", "Balancing decisions use the training split for fitting and the validation split for the comparison. Validation and test data are never resampled or re-weighted. "
                  "Macro-averaged metrics are used so that rare classes count as much as frequent ones.")]


def stage6(R):
    e = R.eda
    obs = []
    if e:
        obs += [f"Training images flagged dark (mean grey level < 40): {e['n_dark_train_images']}; low contrast (std < 15): {e['n_low_contrast_train_images']} of {e['n_train_images']}.",
                f"Spearman correlation between class size and median original width: {e['spearman_class_size_vs_median_width']['rho']:.2f}."]
        if e.get("most_similar_class_pair"):
            mp = e["most_similar_class_pair"]
            obs.append(f"Most similar pair by class-mean HOG cosine similarity: class {int(mp['class_a'])} ({mp['name_a']}) and class {int(mp['class_b'])} ({mp['name_b']}), cos = {mp['cosine_similarity']:.3f}.")
    else:
        obs.append(PENDING)
    sim = R.similar
    srows = ([[int(r.class_a), r["name_a"], int(r.class_b), r["name_b"], f"{r.cosine_similarity:.3f}"] for _, r in sim.head(8).iterrows()] if sim is not None else [[PENDING] * 5])
    return [("h", 2, "Questions asked"),
            ("b", ["Univariate: how many images per class, which sizes, how bright/contrasty are the images?",
                   "Bivariate: does image size relate to class? Which classes look alike?",
                   "Anomalies: dark, low-contrast, very small images.", "Which findings should change preprocessing or modelling?"]),
            ("img", "class_distribution.png", "Class distribution (target)"), ("img", "image_size_distribution.png", "Image size distribution"),
            ("img", "eda_brightness_contrast.png", "Brightness and contrast (training split)"),
            ("img", "eda_class_mean_images.png", "Class-mean images (training split)"),
            ("img", "eda_similar_classes.png", "Most similar class pairs (training split)"),
            ("t", ["Class A", "Name A", "Class B", "Name B", "Cosine similarity"], srows, "Most similar classes by mean-HOG cosine similarity"),
            ("img", "eda_size_by_class.png", "Original size per class"),
            ("h", 2, "Measured observations"), ("b", obs),
            ("h", 2, "Observations -> decisions"),
            ("t", ["Observation", "Decision"],
             [["Image sizes vary widely", "Resize to a single size"], ["Lighting varies; colour not essential", "Grayscale + HOG block normalisation"],
              ["Unequal class counts", "Validate class_weight on the validation split"], ["Frames per sign are near-identical", "Group-aware split and CV"],
              ["Some classes look alike", "Expect and analyse confusions; report macro-F1 and per-class results"]], "EDA-driven decisions")]


def stage7(R):
    s = R.pre
    return [("h", 2, "Representations compared"),
            ("t", ["Representation", "Definition", "Dimensions"],
             [["Raw pixels", f"flattened normalised {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]} grayscale image", f(g(s, "n_features_raw_pixels"))],
              ["HOG", hog_desc(), f(g(s, "n_features_hog"))]], "Feature representations"),
            ("h", 2, "How HOG works"),
            ("b", ["Compute image gradients (magnitude and direction) at every pixel.", "Split the image into cells; per cell build a histogram of gradient directions weighted by magnitude.",
                   "Group cells into overlapping blocks and normalise each block (L2-Hys) to reduce sensitivity to local contrast/brightness.", "Concatenate all block vectors into one feature vector."]),
            ("h", 2, "Why HOG suits traffic signs"),
            ("b", ["Signs are characterised by outlines (circle, triangle, octagon) and inner pictograms - both are edge structures that HOG encodes.",
                   "Block normalisation reduces the effect of illumination changes, a major variation source in GTSRB.",
                   "It is compact and fast, and needs no training (unlike learned CNN features) - appropriate for a classical-ML syllabus.",
                   "Limitation: HOG is not rotation or scale invariant and loses colour; at small image sizes fine digits are only a few pixels wide."]),
            ("h", 2, "HOG parameters"),
            ("t", ["Parameter", "Value", "Reason"],
             [["Image size", f"{config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}", "balance between detail and speed"], ["orientations", str(config.HOG_PARAMS["orientations"]), "standard Dalal-Triggs setting"],
              ["pixels_per_cell", str(config.HOG_PARAMS["pixels_per_cell"]), "cell large enough for stable histograms, small enough for pictograms"],
              ["cells_per_block", str(config.HOG_PARAMS["cells_per_block"]), "overlapping blocks for local normalisation"], ["block_norm", config.HOG_PARAMS["block_norm"], "robust normalisation"]], "HOG configuration"),
            ("img", "hog_visualization.png", "HOG visualisation for sample training images"), ("img", "raw_vs_hog_features.png", "Raw pixels vs HOG: dimensionality and PCA variance"),
            ("h", 2, "Feature scaling and selection"),
            ("p", "StandardScaler (zero mean, unit variance) is fitted on the training split only and applied to validation/test. No feature selection was applied: "
                  "HOG vectors are compact, selection on evaluation data risks leakage, and the RBF SVM copes with correlated features. HOG parameters were fixed a priori "
                  "(not tuned on the test split) and are stored in model_metadata.json so the app reproduces them exactly.")]


def stage8(R):
    m = R.models.get("baseline_svm")
    rows = ([[k, f(g(m, "train", k)), f(g(m, "val", k))] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")] if m else [[PENDING] * 3])
    return [("h", 2, "Baseline definition"),
            ("p", "Raw pixels + SVM: grayscale 32x32 image, flattened, standardised, default RBF SVM (C=1, gamma='scale'). It uses the same split, scaler type and class-weight decision as the proposed model."),
            ("h", 2, "Why a baseline is required"),
            ("p", "Without a reference point, a high score cannot be interpreted. The baseline shows what a model achieves *without* feature engineering, so the benefit of HOG can be measured instead of assumed."),
            ("h", 2, "Configuration and training procedure"),
            ("b", ["Features: raw normalised pixels (" + f(g(R.pre, "n_features_raw_pixels")) + " dimensions)", "Scaler fitted on training data only; SVC(kernel='rbf', C=1.0, gamma='scale')",
                   f"class_weight: {str(g(R.bal, 'selected_class_weight')) if R.bal else PENDING}", f"random_state: {config.RANDOM_SEED}", "Trained on the training split; evaluated on the validation split (test untouched)"]),
            ("h", 2, "Baseline metrics (measured)"),
            ("t", ["Metric", "Training", "Validation"], rows, "Baseline performance"),
            ("img", "baseline_confusion_matrix.png", "Baseline confusion matrix (validation, row-normalised)"),
            ("h", 2, "Limitations"),
            ("b", ["Pixel values are sensitive to brightness, contrast and small shifts.", "High dimensionality relative to information content.", "No explicit shape/edge description."])]


def stage9(R):
    h, r = comparison_table(R.cmp_val)
    ht, rt = comparison_table(R.cmp_test)
    w = R.winner or {}
    return [("h", 2, "Models and why they were selected"),
            ("t", ["Model", "Why", "Assumption / weakness"],
             [["KNN", "Simplest instance-based learner; strong baseline in feature space", "Local similarity is meaningful; slow prediction, scale sensitive, no model"],
              ["Decision Tree", "Interpretable, scale invariant", "Axis-aligned splits suffice; high variance / over-fits"],
              ["SVM (RBF)", "Maximum-margin classifier; effective on medium-size high-dimensional data; the proposed model", "Smooth non-linear boundary; scale sensitive; training cost grows with samples"]], "Model selection rationale"),
            ("h", 2, "Fairness of the comparison"),
            ("p", "Same cleaned data, same group-aware split, same HOG features, same StandardScaler (fit on train), same metrics; all models are scored on the validation split for selection "
                  "(table 1). After every model is frozen, the same models are evaluated once on the untouched test split (table 2)."),
            ("h", 2, "Table 1 - validation split (used for selection)"), ("t", h, r, "Model comparison - validation"),
            ("h", 2, "Table 2 - test split (evaluated once)"), ("t", ht, rt, "Model comparison - test"),
            ("img", "model_performance_comparison.png", "Validation comparison"), ("img", "model_performance_comparison_test.png", "Test comparison"),
            ("h", 2, "Winner (chosen by code, not by hand)"),
            ("p", f"Selection metric: {w.get('selection_metric', PENDING)} on the {w.get('selected_on', 'validation split')}. Best default HOG classifier: {w.get('hog_default_models_winner', PENDING)}. "
                  f"Best row overall (validation): {w.get('overall_winner', PENDING)}."),
            ("h", 2, "Research answers (measured)"), ("p", rq1_statement(R)), ("p", rq2_statement(R)),
            ("h", 2, "Why the winner performed better/worse - expected behaviour from theory"),
            ("b", ["Theory suggests the RBF SVM suits dense, high-dimensional HOG vectors because the margin maximisation and kernel give a smooth boundary that generalises from limited samples per class.",
                   "KNN depends on Euclidean distances in a 1764-dimensional space where distances become less discriminative; it also stores all training data and predicts slowly.",
                   "A single Decision Tree splits on single HOG bins and typically over-fits correlated high-dimensional features.",
                   "These are expectations; the measured tables above decide the actual ranking and any deviation must be discussed in the report."])]


def stage10(R):
    p = R.params
    g_df = R.grid_svm
    grows = ([[int(r.rank_test_score), r.params, f"{r.mean_test_score:.4f} +/- {r.std_test_score:.4f}", f"{r.mean_train_score:.4f}"] for _, r in g_df.head(8).iterrows()] if g_df is not None else [[PENDING] * 4])
    base, tuned = g(R.models.get("svm_default"), "val", "f1_macro"), g(R.models.get("svm_tuned"), "val", "f1_macro")
    return [("h", 2, "GridSearchCV set-up"),
            ("t", ["Item", "Value"],
             [["Search method", "GridSearchCV (exhaustive) over Pipeline(StandardScaler -> SVC)"], ["Data used", "training split only"],
              ["Cross-validation", f"{config.CV_FOLDS}-fold StratifiedGroupKFold (stratified by class, grouped by physical sign)"], ["Scoring metric", "macro-F1"],
              ["Search space", str(g(p, "search_space")) if p else PENDING], ["Candidates", f(g(p, "n_candidates"))]], "Tuning set-up"),
            ("p", "Why this design: the scaler sits inside the cross-validated pipeline (re-fitted per fold, no leakage); group-aware folds stop near-identical frames from straddling folds; "
                  "macro-F1 treats all classes equally; the grid is deliberately modest (kernel, C, gamma) to stay computationally reasonable."),
            ("h", 2, "Results (measured)"),
            ("t", ["Item", "Value"],
             [["Best parameters", str(g(p, "best_params")) if p else PENDING], ["Best CV macro-F1", f(g(p, "cv_best_score_f1_macro"))],
              ["CV std", f(g(p, "cv_best_score_std"))], ["Grid-search wall time (s)", f(g(p, "grid_search_wall_seconds"), 0)]], "Best configuration"),
            ("t", ["Rank", "Parameters", "CV macro-F1", "Train macro-F1"], grows, "Top SVM candidates"), ("img", "grid_search_svm.png", "SVM grid search results"),
            ("h", 2, "Before vs after tuning (validation split)"),
            ("t", ["Model", "Validation macro-F1"], [["HOG SVM (default)", f(base)], ["Tuned HOG SVM", f(tuned)], ["Change", f(None if base is None or tuned is None else tuned - base)]], "Effect of tuning"),
            ("p", "KNN and Decision Tree were tuned with small grids as well (results in results/metrics/grid_search_knn.csv and grid_search_decision_tree.csv), so the final comparison does not pit a tuned SVM against untuned rivals."),
            ("h", 2, "Why the final configuration was selected"),
            ("p", "The parameters with the highest mean CV macro-F1 on the training data were selected; the refit model was then scored once on the validation split. The test split played no role in tuning.")]


def stage11(R):
    fnl = R.final
    t = g(fnl, "test")
    rows = ([[k, f(g(fnl, "train", k)), f(g(fnl, "validation", k)), f(g(fnl, "test", k))] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted", "balanced_accuracy")] if t else [[PENDING] * 4])
    pc = R.per_class
    prow = ([[int(r.class_id), r.class_name, f"{r.precision:.3f}", f"{r.recall:.3f}", f"{r.f1:.3f}", int(r.support)] for _, r in pc.sort_values("f1").head(10).iterrows()] if pc is not None else [[PENDING] * 6])
    cf = R.top_conf
    crow = ([[f"{int(r.true_id)}: {r.true_name}", f"{int(r.pred_id)}: {r.pred_name}", int(r['count'])] for _, r in cf.head(8).iterrows()] if cf is not None and len(cf) else [[PENDING] * 3])
    return [("h", 2, "Validation strategy"),
            ("b", [f"Train / validation / test split (group-aware, stratified, seed {config.RANDOM_SEED}); test untouched until the end.",
                   f"{config.CV_FOLDS}-fold StratifiedGroupKFold cross-validation on the training split for tuning.", "Training and validation scores reported separately to expose over-fitting.",
                   "Final test evaluation performed once after model selection (audit log: results/logs/test_evaluation_log.txt)."]),
            ("h", 2, "Why macro-F1"),
            ("p", "GTSRB is imbalanced; accuracy can be high even if rare signs are ignored. Macro-F1 averages per-class F1 (harmonic mean of precision and recall) with equal weight, so it measures performance on every class."),
            ("h", 2, "Final metrics - Tuned HOG + SVM (measured)"), ("t", ["Metric", "Train", "Validation", "Test"], rows, "Final evaluation"),
            ("p", f"Generalisation gap (train - test macro-F1): {f(g(fnl, 'generalisation_gap_f1_macro_train_minus_test'))}."),
            ("img", "final_confusion_matrix.png", "Final confusion matrix (test, row-normalised)"), ("img", "final_per_class_f1.png", "Per-class F1 (test)"),
            ("h", 2, "Class-wise performance - ten weakest classes"), ("t", ["Class", "Name", "Precision", "Recall", "F1", "Support"], prow, "Weakest classes (test)"),
            ("h", 2, "Error analysis"), ("p", error_statement(R)), ("t", ["True class", "Predicted as", "Count"], crow, "Most frequent confusions (test)"),
            ("img", "misclassified_examples.png", "Random sample of misclassified test images"),
            ("h", 2, "Generalisation discussion"),
            ("p", "A small difference between validation and test scores indicates that model selection did not over-fit the validation split; a large train-test gap indicates over-fitting. "
                  "The split is by physical sign, so the test score estimates performance on *unseen signs from the same recording conditions*. It does not estimate performance on other countries, "
                  "other cameras, full street scenes or adversarial conditions.")]


def stage12(R):
    return [("h", 2, "Practical use cases"),
            ("b", ["Sign-classification module behind a detector in a driver-assistance prototype.", "Road-asset inventory: classify sign crops collected by survey vehicles.", "Education: transparent example of a complete classical-ML pipeline."]),
            ("h", 2, "Benefits"),
            ("b", ["Fast to train and run on a CPU; small model files.", "Interpretable pipeline - every step can be inspected and explained.", "Reproducible: fixed seeds, cached features, one-command pipeline, tests."]),
            ("h", 2, "Stakeholders"), ("p", "Road users, road authorities, ADAS/map developers, and students/instructors (see Stage 1)."),
            ("h", 2, "Risks"),
            ("b", ["Misclassification of safety-relevant signs (e.g. speed limits, stop/yield) could mislead a downstream system.", "Over-trust in the output: the app shows decision scores, not calibrated probabilities.",
                   "Distribution shift: other countries, weather, night, damaged or graffitied signs."]),
            ("h", 2, "Limitations"),
            ("b", ["Classifier only - requires an already-cropped sign; no detection.", "Trained and tested on German signs from one benchmark.", "HOG is not rotation/scale invariant and loses colour; low resolution (32x32) hides fine details such as digits.",
                   "No evaluation on real-time video, motion blur sequences or adversarial inputs.", "Test results estimate performance on unseen signs of the same benchmark only."]),
            ("h", 2, "Ethical considerations"),
            ("b", ["Safety: results must not be used to justify deployment in vehicles.", "Data: public benchmark, no personal data intended; some images may incidentally contain background scenes - used only for coursework under the dataset terms.",
                   "Honesty: all numbers are measured; external code/models are credited (EXTERNAL_RESOURCES.md).", "Generalisation claims are limited to what was tested."]),
            ("h", 2, "Why this system is NOT safety-certified"),
            ("p", "Safety-critical automotive software requires systematic hazard analysis, requirements traceability, validation on representative data from the intended operating domain, "
                  "and certification under standards such as ISO 26262 and ISO 21448 (SOTIF). None of this was performed. RoadEye is an educational prototype."),
            ("h", 2, "Future work"),
            ("b", ["Add sign detection (sliding window / region proposals) in front of the classifier.", "Higher resolution, colour features (HSV histograms), HOG + colour, PCA, probability calibration.",
                   "Deep learning (CNN) as a comparison - outside the current syllabus scope.", "Evaluate on other datasets/countries, night and weather conditions; calibration and rejection of unknown signs.", "Ensemble methods and error-driven augmentation for the weakest classes."])]


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8, 9: stage9, 10: stage10, 11: stage11, 12: stage12}

DOC_FILES = {1: ("01_problem_definition", "Stage 1 - Problem Definition & Use Case"), 2: ("02_data_acquisition_understanding", "Stage 2 - Data Acquisition & Understanding"),
             3: ("03_data_preparation", "Stage 3 - Data Preparation"), 4: ("04_data_preprocessing", "Stage 4 - Data Preprocessing"), 5: ("05_data_balancing", "Stage 5 - Data Balancing"),
             6: ("06_eda", "Stage 6 - Exploratory Data Analysis"), 7: ("07_feature_engineering", "Stage 7 - Feature Engineering / Selection"), 8: ("08_baseline_model", "Stage 8 - Baseline Model"),
             9: ("09_model_comparison", "Stage 9 - ML Modelling and Model Comparison"), 10: ("10_hyperparameter_tuning", "Stage 10 - Hyperparameter Tuning"),
             11: ("11_validation_evaluation", "Stage 11 - Validation & Evaluation"), 12: ("12_use_case_impact_limitations", "Stage 12 - Use Case, Impact & Limitations")}


# --------------------------------------------------------------------------- #
# Technical decisions (WHAT / WHY / HOW / WHAT CHANGED / HOW VALIDATED)
# --------------------------------------------------------------------------- #
def technical_decisions(R):
    q1 = g(R.final, "research_question_1")
    hog_val = (f"Validated by RQ1: macro-F1 {q1['raw_pixel_svm_test']['f1_macro']:.4f} (raw) -> {q1['hog_svm_default_test']['f1_macro']:.4f} (HOG) on test." if q1 else f"Validated by RQ1 ({PENDING}).")
    sel = str(g(R.bal, "selected_class_weight")) if R.bal else PENDING
    return [
        ("Why GTSRB?", "Use the German Traffic Sign Recognition Benchmark.", "Recognised benchmark, 43 classes, real variation, laptop-sized.", "Folder-per-class loader + audit (src/data_loader.py, src/data_validation.py).", "All results are comparable to the literature's benchmark setting.", "Audit of counts, sizes, labels, duplicates (Notebook 01)."),
        ("Why grayscale?", "Convert RGB to one channel.", "HOG uses gradients; colour adds lighting variation and 3x the data.", "cv2.cvtColor in preprocess_image().", "Smaller, more illumination-robust inputs.", "Consistent features between training and app (test_model_pipeline.py)."),
        ("Why 32x32 resizing?", "Resize every image to one size.", "Fixed-length vectors; sizes vary strongly in GTSRB.", "cv2.resize INTER_AREA.", "Uniform feature length.", "Image-size EDA (Notebook 02); HOG length test."),
        ("Why HOG?", "Use HOG features for the main model.", "Signs are edge/shape objects; HOG encodes edges and is illumination-robust.", f"skimage.feature.hog ({hog_desc()}).", "Features: raw pixels -> HOG.", hog_val),
        ("Why scaling?", "StandardScaler fitted on the training split.", "SVM and KNN are scale/distance sensitive.", "src.preprocessing.fit_scaler(); scaler inside the CV pipeline for GridSearchCV.", "Comparable feature ranges; leakage-free.", "Notebook 03 prints train vs val statistics proving the scaler was not fitted on val/test."),
        ("Why the selected split?", "Official test set held out; training pool split ~80/20 into train/validation, stratified and grouped by sign (seed 42).", "~30 frames per sign are near-identical; random splits leak.", "StratifiedGroupKFold in src/split_data.py.", "Test accuracy estimates performance on unseen signs.", "verify_no_leakage(): no shared index, no shared sign between train and validation, identical-file counts reported."),
        ("Why this balancing strategy?", f"Selected class_weight = {sel}.", "Imbalance measured; SMOTE on HOG is not meaningful; class weights alter only the loss.", "Validation-set comparison of class_weight None vs balanced (stage_balancing in src/pipeline.py).", "Weights applied to SVM/Decision Tree training.", f"Validation macro-F1 comparison ({PENDING if not R.bal else 'results/metrics/balancing.json'})."),
        ("Why SVM?", "SVM (RBF) is the main model.", "Maximum-margin classifier effective on high-dimensional features.", "sklearn.svm.SVC, tuned with GridSearchCV.", "Proposed model.", "Compared with KNN and Decision Tree on identical features (RQ2)."),
        ("Why KNN?", "Include K-Nearest Neighbours.", "Syllabus instance-based learner; simple reference.", "KNeighborsClassifier on scaled HOG.", "Comparison model.", "Same split/features/metrics as the others."),
        ("Why Decision Tree?", "Include a Decision Tree.", "Syllabus, interpretable, scale invariant.", "DecisionTreeClassifier (class-weighted if selected).", "Comparison model.", "Same split/features/metrics as the others."),
        ("Why GridSearchCV?", "Tune C, kernel and gamma (and small grids for KNN/Tree).", "Defaults are arbitrary; syllabus technique; modest grid is affordable.", "GridSearchCV over Pipeline with StratifiedGroupKFold, scoring macro-F1.", "Default SVM -> tuned SVM.", "CV on training data only; before/after on validation; test untouched."),
        ("Why macro-F1?", "Primary selection metric.", "Imbalanced multiclass: accuracy can hide poor rare-class performance.", "sklearn precision_recall_fscore_support(average='macro').", "Model selection and tuning use macro-F1.", "Accuracy, balanced accuracy, weighted-F1 and per-class tables reported alongside."),
        ("Why a baseline?", "Raw-pixel SVM baseline.", "A reference is needed to interpret gains from HOG.", "Same pipeline minus HOG.", "Quantifies the effect of feature engineering.", "RQ1 on the same split and SVM settings."),
    ]


def stage_blocks_all(R):
    return {k: fn(R) for k, fn in STAGES.items()}
