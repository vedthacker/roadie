# Project scope - RoadEye

**Title:** RoadEye - Traffic Sign Recognition Using HOG Features and SVM
**Type:** supervised multiclass image classification - classical machine learning (Computer Vision + ITS)
**Course:** Machine Learning (3CS101CC24), Nirma University - Innovative Assignment

## Central story
GTSRB -> preprocessing -> raw-pixel baseline -> HOG feature engineering -> KNN / Decision Tree / SVM comparison ->
SVM GridSearchCV -> final evaluation (test set, once) -> error analysis -> Streamlit application.

## Research questions
1. **RQ1** - Does HOG improve classification over raw pixels? (Raw pixels + SVM vs HOG + SVM, same split/defaults.)
2. **RQ2** - Which syllabus classifier is best on identical HOG features? (KNN vs Decision Tree vs SVM; decided by data, SVM not assumed to win.)

## In scope
Dataset audit and cleaning - EDA - ROI crop, grayscale, 32x32, HOG - scaling (train-only) - raw-pixel baseline - KNN, Decision Tree, SVM -
class-balancing decision (measured) - GridSearchCV (grouped CV) - one-shot test evaluation - error analysis - Streamlit demo - tests - reports.

## Out of scope (not done on purpose)
CNN / YOLO / ResNet / EfficientNet (future scope only) - object detection in scenes - real-time video - autonomous driving - cloud or mobile
deployment - complex front end - safety certification.

## Syllabus alignment (course outline 3CS101CC24)
KNN (session 25) - Decision trees: ID3, C4.5, CART (26-27) - Support Vector Machines: hard/soft margin, kernels and the kernel trick (28-29) -
evaluation measures for classification (30) - feature scaling (15-16) - hyper-parameter tuning (14). Neural networks are in the syllabus too but are
deliberately **not** used here (the project is a classical-ML study; deep learning is mentioned only as future scope).

## Rules the project follows
No invented results - nothing is reported that was not produced by running the code - fixed seed (42) - no test-set tuning -
scaler/models/search fitted on training data only - reference repositories used for reading only (see `EXTERNAL_RESOURCES.md`) -
development-subset runs are always labelled "DEVELOPMENT SUBSET" in every metrics file, figure title and report (use a clean results/ folder for the final full run).

## Compute notes
`python scripts/run_pipeline.py --max-per-class 100 --max-test 2000` runs on a subset (labelled "development_subset") for quick checks.
The full run is the real experiment; the SVM grid search is the slow part (CPU only; allow from tens of minutes to a few hours depending on the machine - not measured here).
Feature extraction is cached in `data/processed/`.
