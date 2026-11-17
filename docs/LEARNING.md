# LEARNING.md: methods explained for interview preparation

Each entry has: the idea in plain English, the key equation, where it lives in the code, and
questions an interviewer might ask, with answers. Entries are added phase by phase.

---

## 1. Lesion-level (grouped) splitting

**Idea.** HAM10000 has 10,015 images but fewer distinct lesions, because some lesions were
photographed more than once (different zoom, angle or day). Two photos of the same lesion look
almost identical. If one is in training and the other in test, the model can "recognise" the
test image instead of diagnosing it, and the test score becomes optimistic. This is **data
leakage**. The fix is to split **lesions**, not images: every image inherits the split of its lesion.

**Key equation (the invariant the code enforces).** Let `L(i)` be the lesion of image `i` and
`S(i)` its split. Then

  for all images i, j: L(i) = L(j) ⇒ S(i) = S(j).

Equivalently, the sets of `lesion_id`s in train, val and test are pairwise disjoint.

**Where in the code.**
- `src/dermxai/data/split.py`: `make_lesion_split` builds one row per lesion, splits those rows,
  and maps the split back to images. `assert_no_leakage` checks the invariant and is called on every split.
- `tests/test_split.py`: `test_no_lesion_in_two_splits` checks it on a toy dataframe, and
  `test_leakage_detector_catches_a_leak` checks that the detector itself works.

**What the real data looks like** (from `results/data/verify_report.json`): 10,015 images of 7,470
lesions. 5,514 lesions have one image, and the rest have 2 to 6. So a random image split would put
near-duplicates on both sides for a large share of the data. Our split has 5,228 / 1,121 / 1,121
lesions (70.4 / 14.6 / 14.9% of the images; `results/data/split_summary.json`).

**Interview questions.**
1. *Why not just split images randomly? It's simpler.*
   Images of the same lesion are near-duplicates. A random image split puts some of them on both
   sides, so the test set partly measures memorisation. Grouped splitting gives an honest estimate
   of performance on **new lesions**, which is what matters clinically.
2. *Is lesion-level splitting enough to remove all leakage?*
   No. It removes the most obvious kind. Other dependencies remain: the same patient can have
   several different lesions (HAM10000 has no patient id), and images from the same clinic share
   camera and lighting. So the test set is still "in-distribution", and I state that as a limitation.
3. *Why split into three sets?*
   Validation is used for every decision: early stopping, hyperparameters, thresholds. Test is
   touched once at the end. If I tuned on test, its score would be optimistic for the same reason
   leakage is: information from the test set would have shaped the model.

---

## 2. Stratification (by diagnosis)

**Idea.** The classes are very imbalanced (`nv`, melanocytic naevi, is about two-thirds of the
images). A purely random split could, by bad luck, leave a rare class like `df` (dermatofibroma)
almost absent from validation or test. **Stratified** splitting draws each class separately, so
every split has roughly the same class proportions as the whole dataset.

**Key equation.** For every class c and split s:

  P(dx = c | split = s) ≈ P(dx = c).

Here it is applied to **lesions**: each lesion has exactly one diagnosis (checked by `verify-data`).
Because lesions have different numbers of images, the image-level proportions can deviate
slightly. `python -m dermxai split` prints the actual image counts per class and split.

**Where in the code.** `make_lesion_split` in `src/dermxai/data/split.py` calls scikit-learn's
`train_test_split(..., stratify=lesions["dx"])` twice: first test vs rest, then val vs train.
The val fraction is rescaled (`val_frac / (1 - test_frac)`) so the final fractions match the config.

**Interview questions.**
1. *Why stratify by lesion and not by image?*
   The unit being split is the lesion, so the unit being balanced must be the lesion too. Doing
   both exactly at once is a constrained problem. Stratifying lesions is simple, exact for lesions
   and close for images.
2. *What does the fixed seed buy you?*
   Reproducibility. Anyone running `split` gets the same `data/splits.csv`, and the committed file
   lets them check this. The CLI also refuses to overwrite an existing, different split without
   `--force`, because a changed split would silently invalidate trained models.

---

## 3. Preprocessing once: resizing images and masks together

**Idea.** Decoding a 600×450 JPEG and resizing it costs more CPU time than a small network
forward pass, so doing it every epoch would dominate training on a laptop. We do it **once** and
cache the results. The image and its mask must get **exactly the same geometry**, otherwise every
"is the attribution inside the lesion?" number would be wrong.

**Key equations.** Scale factor `r = 256 / min(H, W)`, new size `(round(rH), round(rW))`, so
600×450 becomes 341×256 and the aspect ratio is kept. Nothing is cropped at this stage, so image
borders (vignettes, rulers) survive for the shortcut analysis.

- Images use **area interpolation** (`cv2.INTER_AREA`): each output pixel is the average of the
  input pixels it covers. That is the correct anti-aliasing filter when shrinking.
- Masks are first binarised (`> 127` gives 255), then resized with **nearest-neighbour**
  interpolation, so every pixel stays exactly 0 or 255. Bilinear interpolation would create grey
  values on the lesion border, and those would need yet another threshold.

**Where in the code.** `src/dermxai/data/preprocess.py` (`resized_shape`, `resize_pair`,
`write_png_atomic`); tests in `tests/test_preprocess.py`. One test draws a known rectangle into a
toy mask and checks that it lands in the right place after resizing (IoU > 0.97) and that the
lesion's area fraction is unchanged.

**Interview questions.**
1. *Why PNG and not JPEG for the cache?*
   PNG is lossless. JPEG would add new compression artifacts, and artifacts are exactly what this
   project studies. A uint8 `.npy` would also be lossless, but takes about 2× the disk space.
2. *Why write to a temporary file and rename it?*
   `os.replace` is atomic: a file is either complete or absent. If the laptop sleeps or I press
   Ctrl+C mid-write, a re-run skips finished files and redoes the rest, with no corrupted images.
3. *How do you know the image and mask really line up after resizing?*
   Two ways. A unit test checks the geometry on a synthetic rectangle. And
   `scripts/show_cache_samples.py` draws the mask outline on real cached images
   (`results/data/cache_samples.png`), so I can check by eye.
4. *Why not resize straight to 224×224?*
   That would either distort the aspect ratio (squash) or crop away the image border. Keeping a
   256-px short side lets training use random 224-px crops as augmentation, while evaluation uses
   the full frame, so no part of the image is hidden from the explanation methods.

---

## 4. Data integrity: checksums and a verification report

**Idea.** Before any experiment, prove that the data on disk is exactly the published dataset and
is internally consistent. Otherwise a corrupted download or a missing mask could silently change
every result.

**What is checked.**
- **MD5 checksums.** Dataverse publishes an MD5 hash for every file. After downloading, we compute
  the MD5 of our copy and compare. A single flipped bit gives a completely different hash. (MD5 is
  not secure against deliberate tampering, but it reliably detects accidental corruption.)
- **Zip contents.** Only real `.jpg`/`.png` files are extracted, by base name only. The mask zip
  turned out to contain a 233-byte `._ISIC_0025504_segmentation.png`, a macOS "AppleDouble"
  metadata file that Macs add when zipping. It is not a mask, and the extractor now skips such files.
  Without the count check (10,016 vs 10,015) this would have gone unnoticed.
- **`verify-data`.** 10,015 metadata rows, unique image ids, one diagnosis per lesion, a JPEG and a
  mask for every image, mask size = image size (all 600×450), no empty masks, and masks strictly 0/255.

**Where in the code.** `src/dermxai/data/download.py` (`md5sum`, `is_data_member`,
`verify_local_files`), `src/dermxai/data/verify.py`; tests in `tests/test_download.py`.

**Interview questions.**
1. *What did your data checks actually catch?*
   The extra macOS metadata file in the mask archive. A small thing, but it shows why you count
   files against the documented numbers instead of trusting a folder listing.
2. *Why is "every lesion has exactly one diagnosis" worth checking?*
   The split is stratified by lesion diagnosis. If two photos of the same lesion carried different
   labels, the lesion would have no single stratum, and the label noise would also affect evaluation.
