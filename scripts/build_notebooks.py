"""Generate the 8 notebooks (kept as a script so they can be regenerated / diffed).  python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = '''import sys, pathlib, json, warnings
warnings.filterwarnings("ignore")
ROOT = pathlib.Path.cwd().resolve()
ROOT = ROOT.parent if ROOT.name == "notebooks" else ROOT
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Image, Markdown
from src import config, pipeline as P
np.random.seed(config.SEED)                      # reproducibility
pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
print("RUN MODE:", config.RUN_MODE, "| seed:", config.SEED)'''

SHOW = '''def show(name, width=900):
    """Display a saved figure from results/figures/."""
    display(Image(filename=str(config.FIGURES_DIR / name), width=width))

def metrics(name):
    return json.loads((config.METRICS_DIR / name).read_text())'''


def nb(title, cells, fname):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\nProject: **RoadEye - Traffic Sign Recognition Using HOG Features and SVM** "
                                        "(Machine Learning Innovative Assignment).\n\n> Run cells top to bottom. All numbers are "
                                        "computed when the cell runs; nothing is typed in by hand."),
               nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(SHOW)]
    for kind, text in cells:
        n.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    n.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(n, OUT / fname)


def main():
    OUT.mkdir(exist_ok=True)

    nb("01 - Data Acquisition & Understanding", [
        ("md", "## 1. Locate and verify the dataset\nThe code never assumes GTSRB exists: `locate_dataset()` raises a clear error "
               "pointing to `DATASET_DOWNLOAD.md` if it is missing.\n\n**Why GTSRB?** It is the standard, citable benchmark for "
               "multi-class traffic-sign recognition (43 classes), with official labels, per-image ROI boxes and a separate labelled "
               "test set."),
        ("code", "from src.data_loader import locate_dataset, build_metadata\ninfo = locate_dataset()\n"
                 "print('class folders:', len(info['class_dirs']), '| labelled test set:', info['test_gt'] is not None)"),
        ("md", "## 2. Audit: counts, classes, sizes, corrupt files, labels, duplicates\n`stage_audit()` reads **every** file "
               "(decoder check + MD5) and writes `results/metrics/dataset_audit.json`, `class_distribution.png`, `image_size_distribution.png`."),
        ("code", "audit = P.stage_audit()\nsummary = {k: audit[k] for k in ['run_mode','n_files_total','n_train_pool_files','n_official_test_files',"
                 "'n_unreadable_or_missing','file_formats','n_classes_train','class_count_min','class_count_max','imbalance_ratio_max_over_min',"
                 "'n_tracks_train','n_features_raw_pixels']}\ndisplay(pd.Series(summary))"),
        ("code", "display(pd.Series(audit['label_problems'], name='label checks'))\ndisplay(pd.Series(audit['duplicate_files_in_train'], name='duplicate analysis'))\n"
                 "print('identical files shared by train pool and official test:', audit['train_test_identical_files'])\n"
                 "display(pd.DataFrame({'width': audit['image_width'], 'height': audit['image_height']}))\n"
                 "print(audit['sanity_checks_vs_official_documentation'])"),
        ("md", "## 3. Class distribution and representative images"),
        ("code", "show('class_distribution.png'); show('image_size_distribution.png', 700)"),
        ("md", "**What to look for:** (i) strong class imbalance -> see Stage 5 in notebook 02/03; (ii) very different original image "
               "sizes -> a fixed-size resize is mandatory; (iii) any unreadable files or label mismatches (should be zero)."),
        ("md", "## 4. Dataset facts to document\n* Source: official GTSRB archives (see `DATASET_DOWNLOAD.md`).\n* Citation: Stallkamp et al., *Neural Networks* 32 (2012).\n"
               "* Licence/usage: **verify on the archive page you downloaded from and record it in `LICENSE_NOTICE.md`.**\n"
               "* Target: `ClassId` in {0..42}, one traffic-sign type per image."),
    ], "01_data_acquisition_understanding.ipynb")

    nb("02 - Data Preparation & EDA", [
        ("md", "## 1. Prepare: clean, preprocess, split\n`stage_prepare()`:\n1. drops unreadable files (none expected),\n2. ROI-crops, converts to grayscale, "
               "resizes to 32x32,\n3. **splits by sign track** (`class_track` groups) so near-identical frames of one physical sign never appear in both "
               "train and validation,\n4. keeps the official test set separate and untouched,\n5. extracts HOG features (cached)."),
        ("code", "data = P.stage_prepare()\ninfo = metrics('split_info.json')\n"
                 "display(pd.Series({k: info[k] for k in ['run_mode','test_source','n_train','n_val','n_test','n_samples_after_cleaning','n_features_raw','n_features_hog','seed']}))\n"
                 "print('leakage checks:', info['leakage_checks'])"),
        ("md", "## 2. Class distribution per split (stratification check)"),
        ("code", "cc = pd.DataFrame(info['class_counts']).fillna(0).astype(int)\ncc['train_share'] = (cc['train'] / cc['train'].sum()).round(4)\n"
                 "cc['val_share'] = (cc['val'] / cc['val'].sum()).round(4)\ndisplay(cc.head(43))"),
        ("md", "## 3. Exploratory data analysis\nBrightness distribution (illumination), most similar class pairs (by mean HOG vector) and outliers."),
        ("code", "eda = P.stage_eda(data)\ndisplay(pd.Series(eda['brightness']))\ndisplay(pd.read_csv(config.METRICS_DIR / 'eda_similar_classes.csv'))\n"
                 "display(pd.Series(eda['hog_feature_stats']))\nshow('eda_brightness_contrast.png', 800); show('eda_class_mean_images.png', 900); show('eda_similar_classes.png', 900); show('eda_size_by_class.png', 900); show('sample_images.png', 900)"),
        ("md", "## 4. Observations that drive decisions (fill in from the printed outputs)\n"
               "* Imbalance ratio (max/min class count) -> decides the balancing strategy (`balancing.json`).\n"
               "* Brightness spread / dark images -> HOG with `transform_sqrt=True` and block normalisation is more robust than raw pixels.\n"
               "* Visually similar classes (speed limits, curve warnings) -> expect confusions there in the error analysis."),
        ("code", "bal = P.stage_balancing(data)\nprint(bal['decision'])\ndisplay(pd.DataFrame(bal['experiment_on_validation']) if bal['experiment_on_validation'] else pd.Series(bal['train_distribution']))"),
    ], "02_data_preparation_and_eda.ipynb")

    nb("03 - Preprocessing & HOG Feature Engineering", [
        ("md", "## 1. What preprocessing does to one image\nOriginal (RGB, variable size) -> ROI crop -> grayscale -> 32x32 -> HOG."),
        ("code", "data = P.stage_prepare(save_figures=True)\nfrom src.data_loader import read_image_rgb\nfrom src.preprocessing import preprocess_image\n"
                 "i = data.idx('train')[0]\nrow = data.meta.iloc[i]\nraw_img = read_image_rgb(row['path'])\n"
                 "roi = (row.roi_x1, row.roi_y1, row.roi_x2, row.roi_y2)\nproc = preprocess_image(raw_img, roi)\n"
                 "fig, ax = plt.subplots(1, 2, figsize=(5, 2.6)); ax[0].imshow(raw_img); ax[0].set_title(f'original {raw_img.shape[1]}x{raw_img.shape[0]}')\n"
                 "ax[1].imshow(proc, cmap='gray'); ax[1].set_title('preprocessed 32x32'); [a.axis('off') for a in ax]; plt.show()"),
        ("md", "## 2. HOG: parameters and dimensions\nHOG describes local edge *orientation* statistics, which capture the sign's outline and symbol shape and are "
               "largely invariant to illumination (gamma compression + block L2-Hys normalisation)."),
        ("code", "from src.hog_features import hog_feature_dim\nprint('HOG parameters:', config.HOG_PARAMS)\n"
                 "print('raw-pixel dimension :', data.raw.shape[1])\nprint('HOG dimension       :', data.hog.shape[1], '(analytical:', hog_feature_dim(), ')')\n"
                 "show('hog_visualization.png', 900)"),
        ("md", "## 3. Do the representations separate the classes?"),
        ("code", "show('raw_vs_hog_features.png', 900)"),
        ("md", "## 4. Feature scaling without leakage\nThe `StandardScaler` is **fitted on the training split only**, then applied to validation (and, later, test)."),
        ("code", "from src.preprocessing import fit_scaler\nXtr, Xva = data.X('hog', 'train'), data.X('hog', 'val')\nsc = fit_scaler(Xtr)\n"
                 "print('train mean after scaling (should be ~0):', float(sc.transform(Xtr).mean().round(6)))\n"
                 "print('val   mean after scaling (NOT forced to 0):', float(sc.transform(Xva).mean().round(6)))\n"
                 "np.save(config.PROCESSED_DIR / 'hog_train_unscaled.npy', Xtr)  # processed features saved for reuse"),
        ("md", "**Feature selection decision:** none applied. All 1764 HOG dimensions are kept; dimensionality reduction (PCA) was not required for the "
               "chosen models and any such step would have to be fitted on training data only."),
    ], "03_preprocessing_and_hog.ipynb")

    nb("04 - Baseline: Raw Pixels + SVM", [
        ("md", "## Baseline definition\nPipeline: image -> ROI crop -> grayscale -> resize 32x32 -> flatten (1024) -> StandardScaler (train only) -> RBF SVM (C=1, gamma='scale').\n\n"
               "The baseline answers **RQ1**: does HOG help compared with raw pixels, with *identical* split, classifier settings and metrics?"),
        ("code", "data = P.stage_prepare(save_figures=False)\nbase = P.stage_baseline(data)\n"
                 "display(pd.Series({k: base[k] for k in ['accuracy','precision_macro','recall_macro','f1_macro','balanced_accuracy','train_accuracy','inference_time_seconds']}))"),
        ("code", "display(pd.Series(metrics('baseline_svm.json')))\nshow('baseline_confusion_matrix.png', 700)"),
        ("md", "**Reading the result:** compare `train_accuracy` with validation accuracy (over-fitting gap). Baseline limitations: raw pixels are sensitive to "
               "brightness, contrast and small shifts, and the 1024-d vector carries no explicit shape information. Test-set numbers are produced only once, in notebook 07."),
    ], "04_baseline_raw_pixel_svm.ipynb")

    nb("05 - Model Comparison (HOG + KNN / Decision Tree / SVM)", [
        ("md", "## Fair comparison protocol\nSame train/validation split, same HOG features, same StandardScaler, same metrics. "
               "Winner is chosen **automatically** by macro-F1 on the validation set (`PRIMARY_METRIC`). The test set is not used here."),
        ("code", "data = P.stage_prepare(save_figures=False)\nif not (config.METRICS_DIR / 'baseline_svm.json').exists():\n    P.stage_baseline(data)\ndf = P.stage_compare(data)\n"
                 "cols = ['model','feature_representation','accuracy','precision_macro','recall_macro','f1_macro','training_time_seconds','inference_time_seconds','train_accuracy']\n"
                 "display(df[cols].round(4))"),
        ("code", "rq = json.loads((config.COMPARISON_DIR / 'research_questions.json').read_text())\n"
                 "print('RQ2 winner among untuned HOG classifiers:', rq['RQ2_best_untuned_hog_classifier'])\nprint('RQ1 raw vs HOG SVM:', rq['RQ1_raw_vs_hog_svm'])\nshow('model_performance_comparison.png', 800)"),
        ("md", "## Does class weighting help? (validation only)"),
        ("code", "print(metrics('balancing.json')['decision'])"),
        ("md", "**Interpretation guide:** KNN suffers from the curse of dimensionality (1764-d) and gets no class weights; a single Decision Tree is high-variance and "
               "axis-aligned; the RBF SVM can model smooth non-linear boundaries in the high-dimensional HOG space. Whether this holds is decided by the numbers above."),
    ], "05_model_comparison.ipynb")

    nb("06 - Hyper-parameter Tuning (GridSearchCV)", [
        ("md", "## Leakage-safe grid search\n* Only the **training split** is used.\n* `StandardScaler` is inside the CV `Pipeline` (refit in each fold).\n"
               "* Folds are `StratifiedGroupKFold` over sign tracks.\n* Scoring: macro-F1.\n* To keep it affordable the grid search runs on a stratified, group-aware "
               f"subsample of the training split (`TUNE_MAX_SAMPLES`); the best configuration is then refit on the **full** training split and scored on validation."),
        ("code", "print('SVM grid :', config.SVM_GRID)\nprint('KNN grid :', config.KNN_GRID)\nprint('Tree grid:', config.TREE_GRID)\nprint('CV folds:', config.CV_FOLDS, '| scoring:', config.TUNING_SCORING, '| subsample:', config.TUNE_MAX_SAMPLES)"),
        ("code", "data = P.stage_prepare(save_figures=False)\nfor f, fn in [('baseline_svm.json', P.stage_baseline), ('svm_default.json', P.stage_compare)]:\n"
                 "    if not (config.METRICS_DIR / f).exists():\n        fn(data)\nresults = P.stage_tune(data, tune_others=True, n_jobs=2)"),
        ("code", "rows = []\nfor k, r in results.items():\n    rows.append({'model': k, 'best_params': r['best_params'], 'best_cv_f1': round(r['best_cv_score'], 4),\n"
                 "                 'val_f1_before': round(r['validation_f1_macro_before_tuning'], 4), 'val_f1_after': round(r['validation_f1_macro_after_tuning'], 4),\n"
                 "                 'n_fits': r['n_fits'], 'search_time_s': round(r['search_time_seconds'], 1), 'subsampled': r['subsampled']})\ndisplay(pd.DataFrame(rows))"),
        ("code", "display(pd.DataFrame(results['svm']['cv_results']).sort_values('rank_test_score').head(10))\nshow('grid_search_svm.png', 500)"),
        ("md", "## Saved artefacts\n`models/svm/` now holds `best_svm_model.joblib`, `scaler.joblib`, `best_parameters.json`, `label_mapping.json`, `model_metadata.json`."),
        ("code", "print(sorted(p.name for p in (config.MODELS_DIR / 'svm').iterdir()))\ndisplay(pd.Series(json.loads((config.MODELS_DIR / 'svm' / 'best_parameters.json').read_text())))"),
        ("md", "Final configuration = the SVM selected by GridSearchCV (declared in advance as the project's proposed model). If the tuned KNN or Tree has a higher "
               "validation macro-F1, that is reported honestly in the comparison table."),
    ], "06_hyperparameter_tuning.ipynb")

    nb("07 - Final Evaluation & Error Analysis", [
        ("md", "## One-shot test evaluation\nThe frozen models are scored on the untouched test set **once**. A lock file stores the model hash: re-running this notebook "
               "replays the stored predictions and refuses to evaluate a *changed* model (that would be tuning on the test set)."),
        ("code", "data = P.stage_prepare(save_figures=False)\nres = P.stage_final_eval(data)\n"
                 "display(pd.Series({k: res[k] for k in ['model','test_source','n_test','accuracy','precision_macro','recall_macro','f1_macro','f1_weighted','balanced_accuracy','validation_accuracy','train_accuracy','generalisation_gap_val_minus_test_accuracy']}))"),
        ("code", "display(pd.read_csv(config.METRICS_DIR / 'final_test_all_models.csv').round(4)[['model','accuracy','precision_macro','recall_macro','f1_macro','balanced_accuracy']])\nshow('model_performance_comparison_test.png', 800)"),
        ("md", "## Confusion matrix and per-class performance"),
        ("code", "show('final_confusion_matrix.png', 750); pc = pd.read_csv(config.METRICS_DIR / 'final_test_per_class.csv'); display(pc.sort_values('f1').head(10).round(3)); show('final_per_class_f1.png', 600)"),
        ("md", "## Error analysis"),
        ("code", "tc = pd.read_csv(config.METRICS_DIR / 'final_test_top_confusions.csv'); display(tc)\nshow('misclassified_examples.png', 900)"),
        ("code", "z = np.load(config.METRICS_DIR / 'test_predictions.npz'); yt, yp = z['y_true'], z['final_pred']\nwrong = yt != yp\n"
                 "speed = set(range(0, 9))  # GTSRB speed-limit family (classes 0-8)\nboth_speed = np.mean([(t in speed and p in speed) for t, p in zip(yt[wrong], yp[wrong])]) if wrong.any() else float('nan')\n"
                 "print(f'errors: {wrong.sum()} of {len(yt)}; share of errors that are speed-limit <-> speed-limit confusions: {both_speed:.1%}')\n"
                 "imgs = data.images[data.idx('test')]; br = imgs.reshape(len(imgs), -1).mean(1)\n"
                 "print('mean brightness of misclassified vs correct images:', round(br[wrong].mean(), 1), round(br[~wrong].mean(), 1))"),
        ("md", "Systematic patterns to discuss in the report: confusions inside the speed-limit family (digit shapes differ by a few pixels at 32x32), dark/low-contrast images, "
               "and very small signs. Never retune after looking at these results."),
    ], "07_final_evaluation_and_error_analysis.ipynb")

    nb("08 - Reference Model Comparison (external, inspection only)", [
        ("md", "## Policy\nExternal repositories are **reference material only**. Their models are kept apart (`models/reference/`), are never described as ours, and are compared "
               "numerically only if preprocessing, labels and split make the comparison fair."),
        ("code", "ref = json.loads((config.MODELS_DIR / 'reference' / 'reference_inspection.json').read_text())\nfor r in ref['repositories']:\n"
                 "    print('-', r['name'], '| model file present:', r.get('model_file_present', 'see model_files'), '| licence:', r.get('license_file'))\n    print('   ', r['observation'])"),
        ("code", "found = sorted(p.name for p in (config.MODELS_DIR / 'reference').glob('*.pkl'))\nprint('external pickle files placed in models/reference/:', found or 'none')\n"
                 "print('(Pickles are never loaded here: unpickling untrusted files can execute arbitrary code.)')"),
        ("md", "## Outcome\nThe primary repository does not contain its `svm_model.pkl`; the repository with a `.pkl` (ZhouJiaHuan) uses a 7-class label space for a different dataset with a 64x64 HOG window; "
               "the MATLAB `.mat` models are not loadable in scikit-learn; the fourth model's training metadata is incomplete."),
        ("code", "print(ref['statement'])"),
        ("md", "**Comparison table is intentionally absent** - fabricating one would be academic misconduct."),
    ], "08_reference_model_comparison.ipynb")
    print("notebooks written to", OUT)


if __name__ == "__main__":
    main()
