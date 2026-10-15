"""Helpers for the Streamlit app. Uses src.preprocessing / src.hog_features, i.e. the exact
training-time pipeline (no duplicated preprocessing logic)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.hog_features import extract_hog_features  # noqa: E402
from src.preprocessing import preprocess_image  # noqa: E402

MODEL_DIR = config.MODELS_DIR / "svm"


def artifacts_available() -> bool:
    return all((MODEL_DIR / n).exists() for n in (config.FINAL_MODEL_NAME, config.SCALER_NAME, config.METADATA_NAME))


def load_artifacts(model_dir: Path = MODEL_DIR):
    model = joblib.load(model_dir / config.FINAL_MODEL_NAME)
    scaler = joblib.load(model_dir / config.SCALER_NAME)
    meta = json.loads((model_dir / config.METADATA_NAME).read_text())
    return model, scaler, meta


def predict_image(img_rgb: np.ndarray, model, scaler, top_k: int = 3):
    """img_rgb: uint8 RGB array of the (already cropped) sign. Returns top-k (class_id, name, score)."""
    gray = preprocess_image(img_rgb, roi=None)  # same function as training, whole image = ROI
    feats = extract_hog_features(gray)[None, :]
    scores = model.decision_function(scaler.transform(feats))[0]  # SVM margins, NOT probabilities
    order = np.argsort(scores)[::-1][:top_k]
    classes = model.classes_
    return gray, [(int(classes[i]), config.class_name(classes[i]), float(scores[i])) for i in order]
