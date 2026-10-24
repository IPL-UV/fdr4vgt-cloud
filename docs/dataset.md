# Dataset documentation

## What is included in this repository

This repository provides:

- `splits/` -- train/val/test assignments for the 117 pixel scenes and 49 point scenes.
- `src/fdr4vgt_cloud/taco_schema.py` -- canonical column names, PyArrow types and descriptions.
- Pipeline scripts that build the full TACO archive from locally available source data.

The actual images, masks, point labels, and built TACO archives are **not included** in this repository and are **not redistributed** here.

## Schema

The dataset is released as two TACO partitions:

- `fdr4vgt-cloud-scenes`: 117 pixel-annotated scenes.
- `fdr4vgt-cloud-points`: 49 point-annotated scenes at 333 m.

### Level-0 columns

See `src/fdr4vgt_cloud/taco_schema.py` for the canonical column names,
PyArrow types and descriptions.

Key columns:

- `id`: scene identifier.
- `scene:sensor`: `PROBA-V`, `SPOT-VGT1`, `SPOT-VGT2`.
- `scene:platform`: `PROBA-V`, `SPOT-4`, `SPOT-5`.
- `annotation:type`: `pixel` or `point`.
- `split:role`: `finetune_train`, `finetune_val`, `test_pixel`, `heldout_unused`, `test_points`.
- `qa:weak_db_split`: split assignment in the weak pretraining database (`train`, `val`, `test`, `none`).
- `qa:in_original_test89`: whether the scene was in the original 89-scene test table.
- `qa:mask_variant_differs`: true for the SPOT scene with a known mask variant.
- `qa:product_leak`: true if another crop of the same base product appears in a different split.

### Assets (when built locally)

Each pixel scene contains:

- `image.tif`: 4-band TOA reflectance COG, float32, EPSG:4326, nodata 0.
- `mask.tif`: single-band uint8 COG, nodata 255, values {0: clear, 1: cloud, 255: no-data}.

Point scenes contain `points.parquet` instead of `mask.tif`.

## Known caveats

1. The 0.5 threshold is the one reported in the paper. A multi-metric threshold of 0.4848 was computed on the original 89-scene test set but is not recommended for general use.
2. Eight SPOT scenes were added to the test set after the initial split and share product-level overlap with training crops. The `qa:product_leak` column flags these cases.
3. Weak pretraining counts should be verified with `pipeline/43_check_pretrain_split.py` before being reported.

## Licenses

- Code: MIT.
- Splits and metadata in this repository: CC-BY-4.0.
- Proba-V/SPOT-VGT images: original provider terms (Terrascope/VITO, Copernicus data policy).
- Human pixel masks and point validation data: not redistributed here; see original provider terms.
