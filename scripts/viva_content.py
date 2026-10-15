"""Presentation outline and viva Q&A content (numbers pulled from Results when available)."""
from __future__ import annotations

from src import config
from src.hog_features import hog_feature_dim
from scripts.doc_content import PENDING, Results, f, g, hog_desc


def slides(R: Results):
    q1 = g(R.final, "research_question_1")
    t = g(R.final, "test")
    return [
        ("Title", ["RoadEye - Traffic Sign Recognition using HOG features and SVM", "Machine Learning Innovative Assignment - Nirma University", "[Student name] | [Roll no.] | [Date]"],
         "Introduce the project in one sentence: classical ML pipeline on GTSRB.", None),
        ("Problem", ["Input: cropped photo of one traffic sign; output: one of 43 classes", "Supervised multiclass image classification",
                     "RQ1: does HOG beat raw pixels?  RQ2: KNN vs Decision Tree vs SVM on the same HOG features?"], "State the two research questions clearly - the whole talk answers them.", None),
        ("Motivation", ["Sign recognition supports driver assistance and road-asset inventories", "Classical ML: cheap, transparent, reproducible - good for studying design decisions",
                        "Not safety-certified: an educational prototype"], "Mention scope: no detection, no deep learning.", None),
        ("Dataset", [f"GTSRB (Stallkamp et al., 2012); this run: {g(R.audit, 'dataset_identifier') or PENDING}", f"Images: {f(g(R.audit, 'n_readable'))}; classes: {f(g(R.audit, 'n_classes'))}; physical signs: {f(g(R.audit, 'n_tracks_total'))}",
                     "~30 frames per physical sign -> group-aware split"], "Show class_distribution.png.", "class_distribution.png"),
        ("Data preparation", [f"Cleaned: {f(g(R.prep, 'n_before'))} -> {f(g(R.prep, 'n_after'))} images (unreadable, label conflicts, exact duplicates removed)", "Official test set held out; train/validation split ~80/20 grouped by sign, seed 42",
                              f"Leakage check: {g(R.prep, 'leakage_check', 'leakage_free') if R.prep else PENDING}"], "Explain why a random split would leak.", None),
        ("EDA", ["Sizes vary -> resize; lighting varies -> grayscale + HOG normalisation", f"Imbalance ratio (train): {f(g(R.bal, 'train_distribution', 'imbalance_ratio_max_over_min'), 2)}", "Similar classes (e.g. speed limits) expected to be confused"],
         "Show eda_similar_classes.png.", "eda_similar_classes.png"),
        ("HOG feature engineering", [hog_desc(), f"{f(g(R.pre, 'n_features_hog'))} HOG features vs {f(g(R.pre, 'n_features_raw_pixels'))} raw pixels", "Scaler fitted on training data only"], "Explain cells, orientation histograms, block normalisation.", "hog_visualization.png"),
        ("Baseline", ["Raw pixels -> StandardScaler -> RBF SVM", f"Validation macro-F1: {f(g(R.models.get('baseline_svm'), 'val', 'f1_macro'))}", "Same split and SVM settings as the HOG model"], "Why a baseline matters.", "baseline_confusion_matrix.png"),
        ("Model comparison", [f"HOG + KNN / Decision Tree / SVM, validation macro-F1 scores: " + ("; ".join(f"{k} {v:.4f}" for k, v in (g(R.winner, 'scores') or {}).items() if k in ('HOG KNN', 'HOG Decision Tree', 'HOG SVM')) or PENDING),
                             f"Winner (automatic): {g(R.winner, 'hog_default_models_winner') or PENDING}"], "State honestly if SVM did not win.", "model_performance_comparison.png"),
        ("Hyperparameter tuning", ["GridSearchCV, 3-fold StratifiedGroupKFold on training data, scoring macro-F1", f"Best parameters: {g(R.params, 'best_params') or PENDING}", f"Best CV macro-F1: {f(g(R.params, 'cv_best_score_f1_macro'))}"], "Scaler is inside the CV pipeline.", "grid_search_svm.png"),
        ("Final results", [f"Test accuracy: {f(g(t, 'accuracy'))}; macro-F1: {f(g(t, 'f1_macro'))}", ("RQ1: HOG vs raw macro-F1 difference: " + (f"{q1['delta_f1_macro']:+.4f}" if q1 else PENDING)),
                           "Test set evaluated once, after model selection"], "Final confusion matrix.", "final_confusion_matrix.png"),
        ("Error analysis", [PENDING if not R.err else f"{R.err['n_errors']} errors / {R.err['n_test']} test images", "Look-alike classes, small/blurry images, rare classes", "Misclassified examples shown"], "Link errors back to the EDA similarity analysis.", "misclassified_examples.png"),
        ("Streamlit demo", ["Upload -> preview -> same preprocessing + HOG -> SVM prediction", "Shows decision scores, not probabilities", "Disclaimer: not safety-certified"], "Live demo: streamlit run app/app.py", None),
        ("Limitations", ["Classifier only (no detection); German signs only; 32x32 loses fine detail", "HOG not rotation/scale invariant; colour discarded", "Not evaluated on video, night or other countries"], "Be upfront.", None),
        ("Future scope & conclusion", ["Detection stage, colour+HOG, PCA, calibration; CNN as future comparison (outside syllabus)", "Answer RQ1 and RQ2 with the measured numbers", "Reproducible pipeline: one command"], "Close with the two answers.", None),
    ]


def viva(R: Results):
    p = config.HOG_PARAMS
    hogd = hog_desc()
    t = g(R.final, "test")
    final_f1 = f(g(t, "f1_macro"))
    return [
        ("HOG", [
            ("What is HOG?", "Histogram of Oriented Gradients: the image is divided into cells; each cell stores a histogram of gradient directions weighted by gradient magnitude; overlapping blocks of cells are normalised and concatenated into one feature vector."),
            ("Why HOG for traffic signs?", "Signs are defined by outlines and pictograms, i.e. edges. HOG encodes edge orientation and, through block normalisation, is robust to lighting changes."),
            ("What HOG parameters did you use?", f"{hogd} on {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]} grayscale images. They are defined once in src/config.py and saved in model_metadata.json."),
            ("How many HOG features do you get and how is that computed?", f"Blocks_x * blocks_y * cells_per_block_area * orientations = 7 * 7 * 4 * 9 = {hog_feature_dim()} for the configuration used (measured in this run: {f(g(R.pre, 'n_features_hog'))})."),
            ("What are HOG's weaknesses?", "Not rotation or scale invariant, discards colour, and at low resolution fine details such as digits are only a few pixels wide."),
        ]),
        ("SVM", [
            ("How does an SVM classify?", "It finds the hyperplane that maximises the margin between classes; with a kernel (RBF) the boundary becomes non-linear. Only support vectors define the boundary."),
            ("What do C and gamma control?", "C trades margin width against training errors (large C = fewer errors tolerated, risk of over-fitting). Gamma sets the RBF kernel width (large gamma = very local influence, risk of over-fitting)."),
            ("How does SVM handle 43 classes?", "scikit-learn's SVC trains one binary SVM per pair of classes (one-vs-one) and combines their votes."),
            ("Why must features be scaled for SVM?", "The RBF kernel uses Euclidean distance; features with large ranges would dominate. StandardScaler gives every feature zero mean and unit variance."),
            ("Why did you not show probabilities in the app?", "The SVC was trained without probability=True (Platt scaling). Its decision scores rank classes but are not calibrated probabilities, so showing a percentage would be misleading."),
        ]),
        ("KNN", [
            ("How does KNN work and what are its drawbacks here?", "It predicts the majority class among the k closest training points. It needs no training but stores all data, predicts slowly and relies on distances that become less meaningful in many dimensions."),
            ("Why is scaling important for KNN?", "Distances are computed across all features; unscaled features with larger ranges dominate the neighbour search."),
        ]),
        ("Decision Tree", [
            ("How does a Decision Tree work and why does it over-fit?", "It recursively splits on single-feature thresholds that best reduce impurity (Gini). Without depth limits it can fit the training data perfectly, so it memorises noise."),
            ("Does a Decision Tree need scaling?", "No - splits depend on feature order, not scale. We still used the same scaled input so the comparison is identical for all models."),
        ]),
        ("GridSearchCV & cross-validation", [
            ("What does GridSearchCV do?", "It trains and cross-validates every combination of the listed hyper-parameters and selects the one with the best mean validation score, then refits it on all training data."),
            ("What was your search space?", f"SVM: {g(R.params, 'search_space') or 'linear kernel with C in {0.1,1,10}; RBF kernel with C in {1,10,100} and gamma in {0.001, scale, 0.01}'}."),
            ("Why StratifiedGroupKFold?", "Stratified keeps class proportions in each fold; grouped keeps all frames of one physical sign in the same fold, otherwise near-duplicates leak between folds."),
            ("Why is the scaler inside the cross-validation pipeline?", "So it is fitted only on each fold's training part; fitting it on all data first would leak validation-fold statistics."),
            ("Did you tune on the test set?", "No. Tuning used training-only cross-validation; the test split was evaluated once after the models were frozen (audit log in results/logs)."),
        ]),
        ("Data leakage, split and imbalance", [
            ("What is data leakage and where could it occur here?", "Information from validation/test data influencing training: e.g. fitting the scaler on all data, balancing before splitting, tuning on test, or putting frames of one sign in both train and test. All were prevented."),
            ("Why split by physical sign?", "GTSRB has ~30 near-identical frames per sign; a random split would place siblings in train and test and inflate scores."),
            ("Is the dataset imbalanced? What did you do?", f"Measured imbalance ratio (train): {f(g(R.bal, 'train_distribution', 'imbalance_ratio_max_over_min'), 2)}. We compared class_weight=None vs 'balanced' on the validation set; selected: {str(g(R.bal, 'selected_class_weight')) if R.bal else PENDING}. No SMOTE: interpolated HOG vectors are not real sign images."),
            ("What is your random seed and why?", f"{config.RANDOM_SEED} for splits, models and CV so results are reproducible."),
        ]),
        ("Metrics and evaluation", [
            ("Why macro-F1 instead of accuracy?", "Macro-F1 averages per-class F1 equally, so poor performance on rare classes is visible; accuracy can look good while ignoring them."),
            ("Define precision, recall and F1.", "Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean of the two."),
            ("What does the confusion matrix show?", "Rows = true class, columns = predicted class; the diagonal is correct predictions, off-diagonal cells show which classes are confused."),
            ("What is your final test performance?", f"Tuned HOG + SVM: accuracy {f(g(t, 'accuracy'))}, macro-F1 {final_f1} (all values measured by the pipeline; see FINAL_REPORT)."),
            ("How do you know the model is not over-fitted?", "Training, validation and test scores are reported separately; the train-test gap and the validation-test gap are small if it generalises."),
        ]),
        ("GTSRB, baseline and project decisions", [
            ("What is GTSRB?", "The German Traffic Sign Recognition Benchmark: >39,000 training images of 43 sign classes photographed on German roads (Stallkamp et al., 2012)."),
            ("What is your baseline and why?", "Raw pixels + SVM with identical split/settings - it shows how much HOG adds."),
            ("Does HOG improve over raw pixels?", rq1_short(R)),
            ("Which classifier won among KNN, Decision Tree and SVM?", rq2_short(R)),
            ("What is feature engineering?", "Creating informative inputs from raw data - here, turning pixels into HOG edge-orientation features."),
            ("How is the model selected?", "By macro-F1 on the validation split / cross-validation on training data - never on the test set."),
        ]),
        ("Limitations and integrity", [
            ("What are the main limitations?", "Classification only (no detection), German signs only, 32x32 resolution, HOG not rotation/scale invariant, not evaluated on video or other countries, not safety-certified."),
            ("What would you do next?", "Add detection, colour features, PCA, calibration, and a CNN comparison (outside the syllabus)."),
            ("What external resources did you use?", "The GTSRB dataset and public GitHub repositories as reference only (see EXTERNAL_RESOURCES.md); the code, experiments and results are independently produced. Libraries: scikit-learn, scikit-image, OpenCV, NumPy, pandas, Streamlit."),
            ("Did you use AI assistance?", "State honestly according to your course policy: AI-assisted tools were used for coding/documentation support; you remain responsible for understanding and explaining every part."),
        ]),
    ]


def rq1_short(R):
    q = g(R.final, "research_question_1")
    if not q:
        return f"{PENDING} - run the pipeline; the answer is the macro-F1 difference between HOG+SVM and raw pixels+SVM on the test split."
    return f"Measured on the test split: macro-F1 {q['raw_pixel_svm_test']['f1_macro']:.4f} (raw) vs {q['hog_svm_default_test']['f1_macro']:.4f} (HOG), difference {q['delta_f1_macro']:+.4f}."


def rq2_short(R):
    q = g(R.final, "research_question_2")
    if not q:
        return f"{PENDING} - determined by the automatic comparison on identical HOG features."
    return f"Measured on the test split: {', '.join(f'{k} {v:.4f}' for k, v in q['test_scores'].items())} (macro-F1); winner on test: {q['winner_on_test']}."
