"""Create folders, check dependencies and (optionally) download GTSRB.

    python scripts/setup_project.py              # check environment + dataset
    python scripts/setup_project.py --download   # also download the 3 official GTSRB archives
"""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

REQUIRED = ["numpy", "pandas", "matplotlib", "sklearn", "skimage", "cv2", "PIL", "joblib",
            "streamlit", "docx", "nbformat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download + extract GTSRB (needs internet)")
    args = ap.parse_args()
    config.ensure_dirs()
    missing = []
    for m in REQUIRED:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    print("Missing packages:", missing or "none (pip install -r requirements.txt if any)")
    if args.download:
        from src.data_loader import download_gtsrb
        download_gtsrb()
    from src.data_loader import DatasetNotFoundError, extract_local_archives, locate_dataset
    if not args.download:
        done = extract_local_archives()  # manual route: zips already placed in data/raw/
        if done:
            print("Extracted local archives:", done)
    try:
        info = locate_dataset()
        print(f"Dataset OK: {len(info['class_dirs'])} class folders; labelled test set: "
              f"{'found' if info['test_gt'] else 'NOT found (grouped hold-out will be used)'}")
    except DatasetNotFoundError as exc:
        print("DATASET PENDING:", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
