# Dataset download - GTSRB

> **Status of this package:** the dataset is *not* included, and it could not be downloaded when this project was
> built (the build sandbox blocks the official hosts: benchmark.ini.rub.de, sid.erda.dk, Hugging Face, Kaggle all returned HTTP 403 / no route). Everything below is therefore the exact procedure **you** run once.
> Until then every result in the documents reads "Pending execution".

## What you need
The **German Traffic Sign Recognition Benchmark (GTSRB)** - three official files:

| File | Contains |
|---|---|
| `GTSRB_Final_Training_Images.zip` | 43 class folders `00000`-`00042`, `.ppm` images + one `GT-xxxxx.csv` per class |
| `GTSRB_Final_Test_Images.zip` | the test images (`.ppm`) |
| `GTSRB_Final_Test_GT.zip` | **`GT-final_test.csv`** - the *labelled* test file (the images zip only has an unlabelled csv) |

* Benchmark page: <https://benchmark.ini.rub.de/> (GTSRB - German Traffic Sign Recognition Benchmark)
* Official archive (University of Copenhagen ERDA): <https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html>
  - direct files: `https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/<file name>`

Expected size according to the published benchmark description (the audit **measures** the real numbers and warns on mismatch):
43 classes, 39,209 training images, 12,630 test images.

## Option A - automatic (needs internet access to the host above)
```bash
python scripts/setup_project.py --download
```
It downloads the three zips into `data/raw/`, extracts them, and writes `data/raw/download_manifest.json`
(source URL, retrieval time, SHA-256 of every archive - this is your "exact version used" record).

## Option B - manual
1. Download the three zip files in a browser from the page above.
2. Put them into `data/raw/` (do not rename).
3. Run `python scripts/setup_project.py` - it extracts them, writes `data/raw/download_manifest.json` and checks the layout.
   (Or unzip them yourself into `data/raw/`; the loader finds the folders by searching.)

Resulting layout (folder names may differ slightly; the loader searches for them):
```
data/raw/GTSRB/Final_Training/Images/00000 ... 00042/
data/raw/GTSRB/Final_Test/Images/*.ppm
data/raw/GT-final_test.csv
```
> A Kaggle copy (`Train/`, `Train.csv`, PNG files) has a different layout and is **not** supported - the project must use the official files.

## Licence / usage terms - fill this in
The licence could not be read by the build environment. Before submission, open the page you downloaded from and write the terms here:

* Licence / terms as shown on the source page: `__________________________`
* Date accessed: `__________`
* Required citation: J. Stallkamp, M. Schlipsing, J. Salmen, C. Igel, *Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition*, Neural Networks 32 (2012) 323-332.

## Check it worked
```bash
python scripts/setup_project.py          # prints "[dataset ready]" with the three located paths
python scripts/run_pipeline.py --max-per-class 100 --max-test 2000   # quick DEVELOPMENT SUBSET run first (labelled as such everywhere)
python scripts/run_pipeline.py                # the real, full experiment
```
If the dataset is missing, the code stops with a clear `DatasetNotFoundError`; it never invents data.
