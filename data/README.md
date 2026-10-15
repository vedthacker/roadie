# data/

| Folder | Content |
|---|---|
| `raw/` | The GTSRB archives and their extracted folders (you put them here; see `../DATASET_DOWNLOAD.md`). Never committed. |
| `processed/` | Created by the pipeline: audit tables, prepared indexes, `splits.csv`, cached preprocessed images (`images_*.npz`) and HOG features (`hog_*.npy`). Safe to delete - it is rebuilt. |

Expected layout after extraction (search-based, exact folder names may differ slightly):

```
data/raw/GTSRB/Final_Training/Images/00000 ... 00042/   (*.ppm + GT-000xx.csv)
data/raw/GTSRB/Final_Test/Images/*.ppm
data/raw/GT-final_test.csv         (labelled test file from GTSRB_Final_Test_GT.zip)
```

The loader (`src/data_loader.py`) finds the pieces by searching `data/raw/`; if something is missing it stops with a message
and never fabricates data.
