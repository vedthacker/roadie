#!/usr/bin/env python
"""Build RoadEye_Traffic_Sign_ML_Assignment.zip next to the project folder.

Excluded: virtual envs, __pycache__, .git, caches, the raw dataset, cached features (data/processed), duplicate model files.
Large non-final model files (> 25 MB, e.g. an untuned default SVM) are skipped and listed in ZIP_MANIFEST.txt (the pipeline recreates them).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "RoadEye_Traffic_Sign_ML_Assignment.zip"
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
FINAL_MODELS = {"best_svm_model.joblib", "scaler.joblib", "label_mapping.json", "model_metadata.json", "best_parameters.json"}
BIG = 25 * 1024 * 1024
TOP = ("docs", "notebooks", "src", "models", "results", "app", "tests", "scripts", "data")
ROOT_FILES = ["README.md", "requirements.txt", "requirements-lock.txt", ".gitignore", "LICENSE_NOTICE.md", "DATASET_DOWNLOAD.md",
              "EXTERNAL_RESOURCES.md", "PROJECT_SCOPE.md", "PROJECT_STATUS.md"]


def digest(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def collect():
    files, notes, seen = [], [], {}
    cand = [ROOT / n for n in ROOT_FILES]
    for top in TOP:
        cand += sorted((ROOT / top).rglob("*"))
    for p in cand:
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts) or p.suffix == ".pyc":
            continue
        if rel.parts[0] == "data" and rel.name not in ("README.md", ".gitkeep"):
            if rel.parts[1] in ("raw", "processed"):  # dataset + caches are never shipped
                continue
        if p.suffix == ".joblib":
            if p.name not in FINAL_MODELS and p.stat().st_size > BIG:
                notes.append(f"skipped (large, non-final, recreated by the pipeline): {rel}")
                continue
            d = digest(p)
            if d in seen:
                notes.append(f"skipped duplicate: {rel} (identical to {seen[d]})")
                continue
            seen[d] = str(rel)
        files.append(p)
    return files, notes


def main() -> int:
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "generate_report_assets.py"), "--status-only", "--zip-done"])
    files, notes = collect()
    out = ROOT.parent / ZIP_NAME
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, Path("RoadEye") / p.relative_to(ROOT))
        z.writestr("RoadEye/ZIP_MANIFEST.txt", f"files: {len(files)}\n" + "\n".join(notes) + ("\n" if notes else "") +
                   "excluded by design: virtual env, __pycache__, .git, data/raw, data/processed\n")
    print(f"[zip] {out}  ({out.stat().st_size / 1e6:.2f} MB, {len(files)} files)")
    for n in notes:
        print("  -", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
