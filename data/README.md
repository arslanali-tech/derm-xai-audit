# data/

Nothing in this folder is committed except this file, `splits.csv` and `artifact_tags.csv`.

```
data/
├── raw/HAM10000/                  # created by `python -m dermxai download --yes --extract`
│   ├── zips/                      # the four Dataverse files (MD5-verified)
│   ├── images/ISIC_XXXXXXX.jpg    # 10,015 dermoscopy images (600x450)
│   ├── masks/ISIC_XXXXXXX_segmentation.png
│   ├── HAM10000_metadata.csv
│   └── dataverse_listing.json     # API response used to find file ids, sizes and checksums
├── cache/                         # created by `python -m dermxai preprocess` (regenerable)
│   ├── images/ISIC_XXXXXXX.png    # short side 256 px, lossless
│   ├── masks/ISIC_XXXXXXX.png     # same geometry, values 0/255
│   └── manifest.json
├── splits.csv                     # image_id, lesion_id, dx, split  (committed)
└── artifact_tags.csv              # Phase 5, hand-made (committed; never overwritten by code)
```

HAM10000 is licensed CC BY-NC 4.0 by its authors (see the main README for citations).
