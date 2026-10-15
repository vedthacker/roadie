import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture(scope="session")
def synthetic_root(tmp_path_factory):
    from synthetic import make_synthetic_gtsrb
    root = tmp_path_factory.mktemp("synthetic_data")
    make_synthetic_gtsrb(root, n_classes=5, tracks=6, frames=5, n_test=40)
    return root


@pytest.fixture()
def patched_config(synthetic_root, tmp_path, monkeypatch):
    """Point every path in src.config at the synthetic dataset and a temp output dir."""
    from src import config
    g = synthetic_root / "GTSRB"
    monkeypatch.setattr(config, "RAW_DIR", synthetic_root)
    monkeypatch.setattr(config, "GTSRB_DIR", g)
    monkeypatch.setattr(config, "TRAIN_IMAGES_DIR", g / "Final_Training" / "Images")
    monkeypatch.setattr(config, "TEST_IMAGES_DIR", g / "Final_Test" / "Images")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    for name, sub in (("RESULTS_DIR", "results"), ("METRICS_DIR", "results/metrics"), ("FIGURES_DIR", "results/figures"),
                      ("CONFUSION_DIR", "results/cm"), ("COMPARISON_DIR", "results/cmp"), ("LOGS_DIR", "results/logs"),
                      ("MODELS_DIR", "models"), ("DOCS_DIR", "docs")):
        monkeypatch.setattr(config, name, tmp_path / sub)
    import src.pipeline as P
    monkeypatch.setattr(P, "M", tmp_path / "results/metrics")
    config.ensure_dirs()
    return config
