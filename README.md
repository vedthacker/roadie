# RoadEye

**Traffic Sign Recognition Using HOG Features and SVM** - Machine Learning Innovative Assignment
(Nirma University, B.Tech CSE, course 3CS101CC24).

> **Honest status:** all code, notebooks, tests, the app and the document generators are finished and were verified end to end on a small
> *synthetic* stand-in that mimics GTSRB's file format. The real GTSRB experiment has **not been run yet** (the dataset could not be
> downloaded in the environment where this package was built), so every measured number is *Pending execution*.
> Two commands (see "Running Training Pipeline") run the real experiment and fill in all documents. See `PROJECT_STATUS.md`.

## Overview
An end-to-end classical machine-learning system that takes a cropped traffic-sign image and predicts one of the 43 GTSRB classes.

```
Traffic sign image -> data preparation -> preprocessing (ROI crop, gray, 32x32) -> HOG -> StandardScaler
   -> KNN / Decision Tree / SVM -> comparison -> GridSearchCV -> final model -> test evaluation (once) -> Streamlit demo
```

## Problem Statement
Classify a cropped photograph of a German traffic sign into one of 43 categories using syllabus-relevant classical ML, and establish
experimentally **RQ1** whether HOG beats raw pixels and **RQ2** which classifier is best on HOG features.

## Objectives
1. A justified, reproducible, leakage-safe pipeline covering all 12 assignment stages.
2. **RQ1** - Raw pixels + SVM vs HOG + SVM. **RQ2** - HOG + KNN vs HOG + Decision Tree vs HOG + SVM (decided by data; SVM is not assumed to win).
3. Honest evaluation: macro-F1, confusion matrix, per-class results, error analysis, one-shot test evaluation.

## Dataset
German Traffic Sign Recognition Benchmark (GTSRB), official archives - **not included** (see `DATASET_DOWNLOAD.md`).
Cite: Stallkamp et al., *Neural Networks* 32 (2012) 323-332. The exact version/licence is recorded at download time (`LICENSE_NOTICE.md`).

## Architecture
`src/`: `config` (all parameters) - `data_loader` (locate, index, load, download) - `data_validation` (audit + cleaning) - `preprocessing` - `hog_features` (cached) -
`split_data` (sign-grouped) - `train_knn / train_decision_tree / train_svm` - `tune_svm` (GridSearchCV) - `evaluate` - `visualization` - `pipeline`
(stage functions shared by the notebooks and `scripts/run_pipeline.py`, so there is no duplicated logic).

## HOG Feature Engineering
ROI crop -> grayscale -> 32x32 -> HOG (9 orientations, 4x4-pixel cells, 2x2-cell blocks, L2-Hys, sqrt transform) = **1764 features**
(raw-pixel baseline: 1024). Settings live in `src/config.py` and are used identically for training, evaluation and the app.

## Machine Learning Models
KNN (k=5), Decision Tree, SVM (RBF, C=1) - identical HOG features, scaler and split. SVM is the proposed model; KNN and Tree are also tuned so the comparison is fair.

## Baseline
Raw grayscale pixels + StandardScaler + the *same default SVM*; only the features differ.

## Model Comparison
Same split, preprocessing, features and metrics; scored on the validation split (`results/model_comparison/model_comparison.csv`). Winner chosen by code, not by hand.

## Hyperparameter Tuning
`GridSearchCV` over `Pipeline(StandardScaler, classifier)`, 3-fold `StratifiedGroupKFold` (grouped by physical sign), scoring macro-F1, training data only
(on a stratified training subsample of at most 10,000 images for speed; the best setting is refit on the full training split).

## Evaluation
Track-grouped train/validation split; the official GTSRB test set is untouched until the end and evaluated **once** (lock file + model hash).
Metrics: accuracy, precision, recall, F1, macro-F1, balanced accuracy, confusion matrix, per-class report, error analysis.

## Results
<!-- RESULTS:START -->
_Run mode: full_dataset._

**Validation set (model selection):**

| Model | Features | Accuracy | Macro-F1 | Training time (s) |
|---|---|---|---|---|
| Raw Pixel SVM | Raw pixels (1024-d) | 0.7151 | 0.6550 | 108.8 |
| HOG KNN | HOG (1764-d) | 0.9033 | 0.8842 | 0.0 |
| HOG Decision Tree | HOG (1764-d) | 0.7791 | 0.7603 | 74.3 |
| HOG SVM | HOG (1764-d) | 0.9479 | 0.9361 | 135.0 |
| Tuned HOG KNN | HOG (1764-d) | 0.9047 | 0.8852 | 0.4 |
| Tuned HOG Decision Tree | HOG (1764-d) | 0.7700 | 0.7430 | 180.5 |
| Tuned HOG SVM | HOG (1764-d) | 0.9487 | 0.9356 | 141.2 |

* On the untouched test split, HOG + SVM reached macro-F1 0.9246 (accuracy 0.9391) versus 0.6664 (accuracy 0.7227) for raw pixels + SVM with identical SVM settings and split. The difference is +0.2582 macro-F1 and +0.2164 accuracy, i.e. HOG improved classification quality relative to raw pixels. HOG also used 1764 features instead of 1024.
* Test macro-F1 with identical HOG features and default settings: HOG KNN 0.8451, HOG Decision Tree 0.6981, HOG SVM 0.9246. Winner on the test split: HOG SVM; winner on the validation split (the split used for selection): HOG SVM.

**Final test set (evaluated once, tuned HOG + SVM):**

| Accuracy | Macro-precision | Macro-recall | Macro-F1 | Test images |
|---|---|---|---|---|
| 0.9408 | 0.9521 | 0.9069 | 0.9256 | 12,630 |
<!-- RESULTS:END -->

## Streamlit Demo
Upload a cropped sign image -> same preprocessing + HOG as training -> predicted class, top-3 SVM decision scores (margins, *not* probabilities) and a safety disclaimer.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt                          # exact tested versions: requirements-lock.txt
```

## Dataset Setup
```bash
python scripts/setup_project.py --download     # or put the 3 official zips in data/raw/ and run it without --download
```
Details, expected layout and licence note: `DATASET_DOWNLOAD.md`.

## Running Notebooks
Run `notebooks/01 ... 08` in order with Jupyter (they ship without outputs and stop with a clear message if the dataset is missing).
They call the same stage functions as the pipeline script. Regenerate them with `python scripts/build_notebooks.py`.

## Running Training Pipeline
```bash
python scripts/run_pipeline.py                  # audit -> prepare/EDA -> baseline -> compare -> tune -> final test evaluation -> reports
python scripts/run_pipeline.py --max-per-class 100 --max-test 2000   # quick DEVELOPMENT SUBSET (labelled as such in every output)
python scripts/generate_report_assets.py        # (re)build all .docx files, README results block and PROJECT_STATUS.md
python -m pytest tests                          # unit tests on synthetic fixtures
```
Outputs: `models/svm/` (final model), `results/metrics|figures|model_comparison/`, `docs/*.docx`.
Use a fresh `results/` for the final run: development-subset outputs are labelled but would otherwise be overwritten/mixed.

## Running Application
```bash
streamlit run app/app.py
```

## Project Structure
```
README.md  requirements*.txt  PROJECT_SCOPE.md  PROJECT_STATUS.md  EXTERNAL_RESOURCES.md  DATASET_DOWNLOAD.md  LICENSE_NOTICE.md
docs/  data/  notebooks/  src/  models/  results/  app/  tests/  scripts/
```

## Limitations
Classification only (no detection); German signs only; 32x32 resolution hides fine details; HOG is not rotation/scale invariant and drops colour;
not evaluated on video, night or other countries; **not safety-certified**.

## Future Scope
Add a detection stage; colour features; PCA; probability calibration; ensembles; a CNN comparison (outside the syllabus).

## External References
GTSRB and four public GitHub repositories used as reading material only - see `EXTERNAL_RESOURCES.md`. No external model was used.

## Academic Integrity
All experiments are produced by running this code; no numbers are typed in by hand. External resources are credited, AI assistance is declared
(`EXTERNAL_RESOURCES.md`), and the student must verify and be able to explain every part.
