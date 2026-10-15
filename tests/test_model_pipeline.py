import joblib
import numpy as np

from src import config
from src import pipeline as P
from src.split_data import split_dataset, verify_no_leakage


def test_split_has_no_group_leakage(patched_config):
    from src.data_loader import build_metadata
    meta = build_metadata()
    sp = split_dataset(meta)
    report = verify_no_leakage(meta, sp)
    assert report["train_val_group_overlap"] == 0
    assert set(meta.iloc[sp["test"]]["source"]) == {"official_test"}


def test_pipeline_end_to_end_and_prediction(patched_config):
    P.stage_audit()
    data = P.stage_prepare(save_figures=False)
    P.stage_baseline(data)
    df = P.stage_compare(data)
    assert {"Raw Pixel SVM", "HOG KNN", "HOG Decision Tree", "HOG SVM"} <= set(df["model"])
    P.stage_tune(data, tune_others=False)
    res = P.stage_final_eval(data)
    assert 0.0 <= res["accuracy"] <= 1.0 and res["n_test"] == len(data.idx("test"))
    # model loading + prediction pipeline (same code path as the Streamlit app)
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT / "app"))
    import utils
    model, scaler, meta = utils.load_artifacts(config.MODELS_DIR / "svm")
    from PIL import Image
    path = data.meta["path"].iloc[data.idx("test")[0]]
    gray, top = utils.predict_image(np.asarray(Image.open(path).convert("RGB")), model, scaler)
    assert gray.shape == (32, 32) and len(top) == 3 and top[0][2] >= top[1][2]
    assert meta["n_features"] == 1764
    # test-set lock: second evaluation replays instead of re-scoring
    assert (config.METRICS_DIR / ".test_evaluated.lock").exists()
    again = P.stage_final_eval(data)
    assert again["accuracy"] == res["accuracy"]


def test_lock_blocks_changed_model(patched_config):
    import pytest
    P.stage_audit()
    data = P.stage_prepare(save_figures=False)
    P.stage_baseline(data); P.stage_compare(data); P.stage_tune(data, tune_others=False)
    P.stage_final_eval(data)
    path = config.MODELS_DIR / "svm" / config.FINAL_MODEL_NAME
    m = joblib.load(path)
    m.C = m.C * 2  # pretend somebody retrained after peeking at test
    joblib.dump(m, path)
    with pytest.raises(RuntimeError):
        P.stage_final_eval(data)
