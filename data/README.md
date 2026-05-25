# Data — ISIC 2018, Task 1 (Lesion Boundary Segmentation)

The dataset is **not** committed (everything in `data/` except this file is gitignored). Download it locally with the Phase 2 script:

```bash
python scripts/download_data.py
```

> **Size & disk:** the training-images zip is **~11 GB** (2594 full-resolution
> dermoscopy images); the ground-truth masks are ~27 MB. Keep ~25 GB free during
> setup — the zip is deleted after extraction. The script is idempotent (re-runs
> skip completed work); pass `--force` to redownload. It uses only the Python
> standard library, so it runs without the training dependencies installed.

## Expected layout

The script should produce (under `data/isic2018/`, i.e. `config.RAW_DATA_DIR`):

```
data/isic2018/
├── images/        # dermoscopic input images  (ISIC2018_Task1-2_Training_Input, *.jpg)
└── masks/         # binary ground-truth masks  (ISIC2018_Task1_Training_GroundTruth, *_segmentation.png)
```

Train/val/test splitting is done in code (`src/data/dataset.py`) from this single
training set, using `config.SEED` / `config.VAL_SPLIT` / `config.TEST_SPLIT`.

## Source & manual download

- ISIC 2018 data portal: https://challenge.isic-archive.com/data/#2018
- Task 1 files: **Training Input** (images) + **Training GroundTruth** (segmentation masks).

## License & citation

ISIC data is released under a **non-commercial (CC BY-NC)** license — research /
educational use only. Cite the ISIC 2018 dataset papers (Codella et al. 2018;
Tschandl et al. 2018 / HAM10000) when reporting results. Add the BibTeX here once
finalized.
