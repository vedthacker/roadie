# RoadEye Streamlit app

```bash
python scripts/run_pipeline.py     # once: trains and saves models/svm/ (best_svm_model.joblib, scaler.joblib, ...)
streamlit run app/app.py
```

* `app.py` - interface (upload -> preview -> predict -> disclaimer). If no trained model exists it shows an explicit error and stops; it never fabricates predictions.
* `utils.py` - loads `models/svm/best_svm_model.joblib`, `scaler.joblib`, `model_metadata.json`; preprocessing and HOG are imported from `src/preprocessing.py` and `src/hog_features.py`, i.e. the exact training code and configuration.
* The app shows SVM **decision scores**, not probabilities: the SVC was trained without Platt scaling, so a percentage "confidence" would not be technically valid.
* Use cropped sign images (e.g. from the GTSRB test set). Full street scenes will not work - this is a classifier, not a detector.
