# derm-xai-audit — "Is the model looking at the lesion?"

A skin-lesion classifier can score well while relying on **shortcuts**: rulers, ink marks, hair,
dark image borders or the surrounding skin, rather than the lesion itself. This project measures
that for a ResNet-18 trained on HAM10000. The expert lesion segmentation masks serve as ground
truth for *where the model should look*.

## Research questions

1. How much of each explanation's attribution (Grad-CAM, Integrated Gradients, Occlusion) falls
   **inside the lesion mask**, compared with the random baseline (the mask's area fraction)?
2. Does the model still predict above chance when the lesion is removed (**background-only** input)?
   How much does it lose when the background is removed (**lesion-only** input)?
3. Do the explanation methods pass **sanity checks** (Adebayo et al., 2018), i.e. do the maps change
   when the model's weights are randomised?
4. Are images with visible **artifacts** (hand-tagged) more likely to draw attribution outside the lesion?

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Setup, data download / verification, lesion-level split, CPU benchmark | done (see `results/data/`, `results/benchmark/`) |
| 1 | Classifier (staged transfer learning on CPU) | not yet run |
| 2 | Explanations + mask-based localisation metrics | not yet run |
| 3 | Lesion-only / background-only shortcut tests | not yet run |
| 4 | Sanity checks (cascading randomisation) | not yet run |
| 5 | Artifact tagging + comparison | not yet run |
| 6 | Report | not yet run |

**Results: not yet run.** The results sections are generated from `results/*.json` by
`python -m dermxai report`. No number appears here unless it came from an actual run.

## Data

**HAM10000** ("Human Against Machine with 10000 training images") contains 10,015 dermoscopic
images in 7 diagnostic classes, downloaded from the Harvard Dataverse (doi:10.7910/DVN/DBW86T,
dataset version 4.0). The lesion masks (`HAM10000_segmentations_lesion_tschandl.zip`) were made by a
single dermatologist: masks predicted by a segmentation network (Tschandl et al., 2019) were
verified, corrected or redrawn by hand (Tschandl et al., 2020).

Licence: **CC BY-NC 4.0** (non-commercial use only). The data is **not** redistributed in this
repository. Only derived split assignments (`data/splits.csv`) are committed.

`results/data/cache_samples.png` shows a few dataset images (one per class) with their lesion-mask
outlines, reproduced from HAM10000 under CC BY-NC 4.0 (resized, outline added).

The dataset contains several images of the same lesion. All splits are therefore made at the
**lesion** level (`lesion_id`) so no lesion appears in two splits. See `docs/LEARNING.md`.

## Hardware

The project is designed for a **CPU-only laptop** (development machine: Intel i5-8265U, 4 cores,
24 GB RAM). It uses at most *physical cores − 1* threads and below-normal process priority.
`scripts/benchmark_cpu.py` measures the cost of every expensive step before it is run.

## How to reproduce (Windows PowerShell)

```powershell
# 1. Environment (project-local venv; CPU-only PyTorch)
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # includes the CPU-only PyTorch index
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe -m pytest

# 2. Data (about 2.65 GB download; the first command only lists files and sizes)
.venv\Scripts\python.exe -m dermxai download                 # lists files and sizes
.venv\Scripts\python.exe -m dermxai download --yes --extract # or the manual steps below
.venv\Scripts\python.exe -m dermxai verify-data
.venv\Scripts\python.exe -m dermxai preprocess --limit 20   # timing run
.venv\Scripts\python.exe -m dermxai preprocess
.venv\Scripts\python.exe -m dermxai split

# 3. CPU cost estimate for all later phases
.venv\Scripts\python.exe scripts\benchmark_cpu.py
```

The downloader sends a descriptive client name: Dataverse rejects the anonymous default
`python-requests` client with HTTP 403. Downloads resume after interruptions, and every file is
checked against the MD5 checksum Dataverse publishes.

**Manual download (fallback).** If the automatic download fails, download the files in a browser:

1. Open https://doi.org/10.7910/DVN/DBW86T and download these four files **one at a time**.
   Selecting several and clicking "Download" produces a single `dataverse_files.zip` instead.
   - `HAM10000_images_part_1.zip`
   - `HAM10000_images_part_2.zip`
   - `HAM10000_segmentations_lesion_tschandl.zip`
   - `HAM10000_metadata`: choose **Original File Format** (CSV), not the tab-delimited version
2. Move them into `data/raw/HAM10000/zips/` and make sure the metadata file is named `HAM10000_metadata.csv`.
3. Run `.venv\Scripts\python.exe -m dermxai download --extract-only`. This checks every file against the
   MD5 checksums published by Dataverse (saved in `data/raw/HAM10000/dataverse_listing.json`), then unzips.

## Citations

- Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source
  dermatoscopic images of common pigmented skin lesions.* Scientific Data 5, 180161 (2018).
  https://doi.org/10.1038/sdata.2018.161
- Tschandl, P. *The HAM10000 dataset* [data set]. Harvard Dataverse, V4.
  https://doi.org/10.7910/DVN/DBW86T
- Tschandl, P., Rinner, C., Apalla, Z., et al. *Human–computer collaboration for skin cancer
  recognition.* Nature Medicine (2020). https://doi.org/10.1038/s41591-020-0942-0 (source of the lesion masks)
- Tschandl, P., Sinz, C. & Kittler, H. *Domain-specific classification-pretrained fully convolutional
  network encoders for skin lesion segmentation.* Computers in Biology and Medicine (2019).
  https://doi.org/10.1016/j.compbiomed.2018.11.010 (network used to initialise the masks)

Method citations (Grad-CAM, Integrated Gradients, Occlusion, sanity checks) are added in the phases
that use them.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY-NC 4.0, owned by its authors, not included here.
