"""RoadEye - Streamlit demo.  Run:  streamlit run app/app.py"""
import numpy as np
import streamlit as st
from PIL import Image

from utils import artifacts_available, load_artifacts, predict_image

st.set_page_config(page_title="RoadEye", page_icon="🚸", layout="centered")
st.title("RoadEye - Traffic Sign Recognition")
st.caption("HOG features + SVM, trained on GTSRB (classical machine learning, no deep learning)")

with st.expander("How it works", expanded=False):
    st.markdown(
        "1. Your image is converted to grayscale and resized to 32x32 (same function as in training).\n"
        "2. HOG (Histogram of Oriented Gradients) features are extracted with the training configuration.\n"
        "3. The features are standardised with the scaler fitted on the training set only.\n"
        "4. The tuned SVM predicts one of the GTSRB traffic-sign classes.")

if not artifacts_available():
    st.error("No trained model found in models/svm/. Run `python scripts/run_pipeline.py` first "
             "(see README.md). The app does not ship with fabricated predictions.")
    st.stop()


@st.cache_resource
def _load():
    return load_artifacts()


model, scaler, meta = _load()
st.sidebar.subheader("Model")
st.sidebar.write(f"Type: {meta['model_type']}")
st.sidebar.write(f"Classes: {meta['n_classes']}  |  Features: {meta['n_features']}")
st.sidebar.write(f"Hyper-parameters: {meta['best_hyperparameters']}")
st.sidebar.write(f"Validation macro-F1: {meta['validation_scores']['f1_macro']:.3f}")
st.sidebar.write(f"Run mode: {meta['run_mode']}")

up = st.file_uploader("Upload a photo of ONE traffic sign (tightly cropped works best)", type=["png", "jpg", "jpeg", "ppm", "bmp"])
if up is not None:
    img = np.asarray(Image.open(up).convert("RGB"))
    gray, top = predict_image(img, model, scaler)
    c1, c2 = st.columns(2)
    c1.image(img, caption="Uploaded image", use_container_width=True)
    c2.image(gray, caption="After preprocessing (grayscale, 32x32)", use_container_width=True, clamp=True)
    cid, name, score = top[0]
    st.success(f"Predicted class {cid}: {name}")
    st.write("Top-3 candidates (SVM decision-function scores; larger = more preferred. "
             "These are margins, **not probabilities**):")
    st.table({"ClassId": [t[0] for t in top], "Sign": [t[1] for t in top], "Decision score": [round(t[2], 3) for t in top]})

st.warning("Educational prototype. Trained only on German road signs, one sign per image, no detection step. "
           "NOT safety-certified and must never be used for real driving decisions.")
