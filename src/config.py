"""Central configuration for RoadEye.

Every magic number used by the pipeline lives here, so notebooks, scripts, the
tests and the Streamlit app all share exactly the same preprocessing / HOG
settings (no hidden or duplicated preprocessing logic).
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (all relative to the project root -> no absolute paths anywhere)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ROADEYE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GTSRB_DIR = RAW_DIR / "GTSRB"
TRAIN_IMAGES_DIR = GTSRB_DIR / "Final_Training" / "Images"
TEST_IMAGES_DIR = GTSRB_DIR / "Final_Test" / "Images"
TEST_GT_CSV_NAME = "GT-final_test.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
COMPARISON_DIR = RESULTS_DIR / "model_comparison"
LOGS_DIR = RESULTS_DIR / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

# --------------------------------------------------------------------------
# Dataset identity (official GTSRB, hosted by the University of Copenhagen ERDA
# archive that the benchmark authors point to). Download is OPTIONAL and never
# silent: see data_loader.download_gtsrb().
# --------------------------------------------------------------------------
DATASET_NAME = "GTSRB - German Traffic Sign Recognition Benchmark (Final Training + Final Test + Test GT)"
DATASET_ARCHIVE_PAGE = (
    "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html"
)
_ERDA = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370"
DATASET_URLS = {
    "GTSRB_Final_Training_Images.zip": f"{_ERDA}/GTSRB_Final_Training_Images.zip",
    "GTSRB_Final_Test_Images.zip": f"{_ERDA}/GTSRB_Final_Test_Images.zip",
    "GTSRB_Final_Test_GT.zip": f"{_ERDA}/GTSRB_Final_Test_GT.zip",
}
DATASET_CITATION = (
    "J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. Man vs. computer: Benchmarking machine "
    "learning algorithms for traffic sign recognition. Neural Networks 32 (2012) 323-332."
)
# Figures quoted in the official documentation. They are ONLY used as a sanity
# check by the audit (warning if the downloaded copy differs); every number that
# appears in the reports is measured from the actual files.
EXPECTED_N_CLASSES = 43
EXPECTED_N_TRAIN = 39209
EXPECTED_N_TEST = 12630

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Run mode. A subset is NEVER silently presented as the full experiment: the
# mode is written into every metrics file / figure title / report.
#   ROADEYE_MAX_PER_CLASS=100  -> "development_subset" (training pool only)
#   ROADEYE_MAX_TEST=2000      -> caps the official test set size
# --------------------------------------------------------------------------
def _env_int(name: str):
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


MAX_PER_CLASS = _env_int("ROADEYE_MAX_PER_CLASS")
MAX_TEST = _env_int("ROADEYE_MAX_TEST")
RUN_MODE = "full_dataset" if (MAX_PER_CLASS is None and MAX_TEST is None) else "development_subset"

# --------------------------------------------------------------------------
# Preprocessing (single source of truth, also used by the Streamlit app)
# --------------------------------------------------------------------------
IMAGE_SIZE = (32, 32)  # (width, height) after ROI crop + resize
IMG_SIZE = IMAGE_SIZE  # alias used by the report generator
RANDOM_SEED = SEED
USE_ROI_CROP = True  # GTSRB ships an ROI per image (sign inside a ~10% border)
RAW_PIXEL_SCALE = 255.0  # raw baseline: uint8 -> [0, 1]

# --------------------------------------------------------------------------
# HOG configuration. 32x32 image, 4x4 cells -> 8x8 cells, 2x2 blocks with stride
# of one cell -> 7x7 blocks, 4 cells x 9 bins each -> 7*7*4*9 = 1764 features.
# --------------------------------------------------------------------------
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (4, 4),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True,  # gamma compression -> robust to illumination
}

# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
VAL_FRACTION = 0.20  # fraction of TRAIN POOL held out for validation (grouped by track)
HOLDOUT_TEST_FRACTION = 0.20  # only used if the official labelled test set is absent
CV_FOLDS = 3  # StratifiedGroupKFold inside GridSearchCV (training data only)

# --------------------------------------------------------------------------
# Models (untuned "reference" settings, identical data for every model)
# --------------------------------------------------------------------------
KNN_DEFAULT = {"n_neighbors": 5, "weights": "uniform", "metric": "euclidean"}
TREE_DEFAULT = {"random_state": SEED, "class_weight": None}
SVM_DEFAULT = {"kernel": "rbf", "C": 1.0, "gamma": "scale", "random_state": SEED}

# --------------------------------------------------------------------------
# Hyper-parameter search spaces (GridSearchCV)
# --------------------------------------------------------------------------
SVM_GRID = {
    "clf__kernel": ["rbf"],
    "clf__C": [1, 10, 100],
    "clf__gamma": ["scale", 0.001, 0.01],
}
SVM_LINEAR_GRID = {"clf__kernel": ["linear"], "clf__C": [0.01, 0.1, 1]}
KNN_GRID = {"clf__n_neighbors": [1, 3, 5, 7, 11], "clf__weights": ["uniform", "distance"]}
TREE_GRID = {
    "clf__max_depth": [10, 20, 40, None],
    "clf__min_samples_leaf": [1, 3, 5],
    "clf__criterion": ["gini", "entropy"],
}
TUNE_MAX_SAMPLES = 10000  # stratified, training-only subsample used INSIDE the grid search
TUNING_SCORING = "f1_macro"
PRIMARY_METRIC = "f1_macro"  # model selection metric (validation set)

# Imbalance rule: max/min class count above this -> use class_weight='balanced'
IMBALANCE_RATIO_THRESHOLD = 1.5  # <=1.5 treated as balanced; otherwise class_weight is validated on the validation split

# --------------------------------------------------------------------------
# Artifact names
# --------------------------------------------------------------------------
FINAL_MODEL_NAME = "best_svm_model.joblib"
SCALER_NAME = "scaler.joblib"
LABEL_MAPPING_NAME = "label_mapping.json"
METADATA_NAME = "model_metadata.json"

# --------------------------------------------------------------------------
# GTSRB class names (ClassId -> official sign description)
# --------------------------------------------------------------------------
CLASS_NAMES = {
    0: "Speed limit (20km/h)", 1: "Speed limit (30km/h)", 2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)", 4: "Speed limit (70km/h)", 5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)", 7: "Speed limit (100km/h)", 8: "Speed limit (120km/h)",
    9: "No passing", 10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection", 12: "Priority road", 13: "Yield",
    14: "Stop", 15: "No vehicles", 16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry", 18: "General caution", 19: "Dangerous curve to the left",
    20: "Dangerous curve to the right", 21: "Double curve", 22: "Bumpy road",
    23: "Slippery road", 24: "Road narrows on the right", 25: "Road work",
    26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Beware of ice/snow", 31: "Wild animals crossing",
    32: "End of all speed and passing limits", 33: "Turn right ahead",
    34: "Turn left ahead", 35: "Ahead only", 36: "Go straight or right",
    37: "Go straight or left", 38: "Keep right", 39: "Keep left",
    40: "Roundabout mandatory", 41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


def class_name(class_id: int) -> str:
    """Human readable name; falls back to 'Class N' for unknown ids."""
    return CLASS_NAMES.get(int(class_id), f"Class {int(class_id)}")


def ensure_dirs() -> None:
    """Create all output directories (idempotent)."""
    for path in (PROCESSED_DIR, METRICS_DIR, FIGURES_DIR, CONFUSION_DIR, COMPARISON_DIR, LOGS_DIR,
                 MODELS_DIR / "baseline_svm", MODELS_DIR / "knn", MODELS_DIR / "decision_tree",
                 MODELS_DIR / "svm", MODELS_DIR / "reference", DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
