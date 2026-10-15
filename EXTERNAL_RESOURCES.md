# External resources and academic-integrity record

Inspection date: 2026-10-06. The four repositories were cloned **read-only** into a temporary folder outside this project and read.
**None of their code or model files is part of this project or the ZIP, and no pickle/joblib/.mat file from them was loaded (unpickled).**
All preprocessing, experiments, tuning, evaluation, figures and documents in this project were implemented and run independently.

## Dataset
| Item | Value |
|---|---|
| Name | German Traffic Sign Recognition Benchmark (GTSRB) |
| Owner | Institut fuer Neuroinformatik, Ruhr-Universitaet Bochum |
| Source used | Official archive files `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, `GTSRB_Final_Test_GT.zip` (see `DATASET_DOWNLOAD.md`) |
| Exact version actually downloaded | **Recorded at download time** in `data/raw/download_manifest.json` (URL, timestamp, SHA-256). *Pending - not downloaded in the build environment.* |
| Licence / usage terms | **To be recorded from the source page** (not verifiable from the build environment) - see `DATASET_DOWNLOAD.md` |
| Citation | J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks 32 (2012) 323-332. |

## Reference repositories

### 1. Computer-Vision-Traffic-Sign-Recognition
* URL: <https://github.com/omarbasha19/Computer-Vision-Traffic-Sign-Recognition> (inspected commit `d782756`, 2026-01-24)
* **What was inspected:** `README.md` and `main.ipynb` (GTSDB detection + GTSRB recognition notebook).
* **Used as reference for:** the idea of HOG + SVM on GTSRB and of comparing with a CNN; its HOG settings (64x64 grayscale, 9 orientations, 8x8 cells, 2x2 blocks, L2-Hys, sqrt transform) and a linear `SVC(probability=True)` as design choices to compare against our own settings (32x32, 4x4 cells, no sqrt, RBF/linear grid).
* **Code/model directly reused:** No.
* **Pretrained model used:** No. The repository does **not** contain `svm_model.pkl` (the notebook loads it from disk, but the file is absent), so there was nothing to load.
* **Independently implemented here:** loader, ROI cropping, audit, track-grouped split, preprocessing, HOG pipeline, all models, tuning, evaluation, app.

### 2. traffic-sign-detection
* URL: <https://github.com/ZhouJiaHuan/traffic-sign-detection> (commit `6e96c4e`, 2018-12-04)
* **What was inspected:** `svm_hog_classification/hog_extract.py`, `svm_train.py`, README; the opcode stream of `svm_model.pkl` (via `pickletools`, which executes nothing).
* **Used as reference for:** HOG settings (64x64, 9 bins, 8x8 cells, 2x2 blocks, L2, sqrt) and `SVC(C=10, probability=True)` in a detection pipeline.
* **Code/model directly reused:** No.
* **Pretrained model used:** No. The pickle is a Python-2-era `sklearn.externals.joblib` SVC (RBF) for a **7-class** label space (straight, left, right, stop, nohonk, crosswalk, background) - not the 43 GTSRB classes - so it is neither loadable safely nor comparable.
* **Independently implemented here:** everything listed above.

### 3. traffic-sign-classification
* URL: <https://github.com/davidakhihiero/traffic-sign-classification> (commit `429fa6c`, 2025-05-08; MIT licence)
* **What was inspected:** `README.md`, `DATASET_INSTRUCTIONS.txt`, `scripts/TrainTrafficSignsSVM.m`, repository layout (MATLAB; `.mat` models; PDF report not read in detail).
* **Used as reference for:** model-comparison design (SVM+HOG vs SVM+SIFT vs CNN) and ROI cropping before resizing (40x40 there). Their README reports 92.20 % accuracy for SVM+HOG - **that is their claim, not ours, and was not verified**.
* **Code/model directly reused:** No.
* **Pretrained model used:** No (`.mat` MATLAB models cannot be used in Python; trained on a Kaggle copy of GTSRB with its own split).
* **Independently implemented here:** everything listed above.

### 4. Traffic-sign-recognition
* URL: <https://github.com/nakul3112/Traffic-sign-recognition> (commit `5cda53b`, 2020-01-08)
* **What was inspected:** `README.md`, `Final_TSR_detect.py` (use of `cv2.HOGDescriptor` and an SVM), opcode stream of `dataset.pkl` (a scikit-learn 0.20.3 linear `SVC` with `probability`, string class labels, trained on a different video/template dataset).
* **Used as reference for:** the two-stage (detection -> classification) structure only.
* **Code/model directly reused:** No.
* **Pretrained model used:** No (different dataset/labels; untrusted pickle never loaded).
* **Independently implemented here:** everything listed above.

### Reference-model comparison decision
> Reference models were inspected as external implementations, but direct numerical comparison was not performed because the original
> preprocessing/split/model metadata does not allow a scientifically fair comparison.

`models/reference/reference_inspection.json` records what was inspected (no external model file is stored there); `notebooks/08_reference_model_comparison.ipynb` documents the inspection.
Reference/pretrained models are never called "our trained model"; our independently trained model lives in `models/svm/`.

## Libraries
numpy, pandas, SciPy, scikit-learn (SVC, KNeighborsClassifier, DecisionTreeClassifier, GridSearchCV, StratifiedGroupKFold), scikit-image (`hog`),
Pillow, matplotlib, joblib, Streamlit, python-docx, nbformat/nbconvert/Jupyter, pytest.

## Generative-AI assistance
Claude (Anthropic) was used for coding support, structure and documentation as permitted by the course guidelines. The student remains responsible for
verifying, running and explaining all code and results (guideline section 9). **No experimental results are included in this package that were not produced by executed code**;
the synthetic test fixture (`tests/synthetic.py`) exists only to unit-test the code and its outputs are not reported anywhere.

## Declaration (student to complete)
I have run the pipeline on the real GTSRB data, verified the code and the results, and can explain every part of this project.
Name: ______________  Roll no.: ________  Date: ________
