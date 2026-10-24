# Pipeline modules

| Module | Purpose |
|---|---|
| `10_inventory.py` | Build the scene inventory and assert 117 pixel + 49 point scenes. |
| `11_splits.py` | Recalculate train/val/test splits and export `splits/scenes.csv`. |
| `12_metadata.py` | Extract per-scene metadata from product IDs and SPOT LOG files. |
| `20_cog.py` | Convert source images and masks to Cloud Optimized GeoTIFFs. |
| `21_verify_cog.py` | Validate every COG and compare against the source data. |
| `30_build_taco.py` | Build the two TACO partitions (`scenes` and `points`). |
| `31_fix_tacocat_order.py` | Reorder `.tacocat/level0.parquet` and reopen with `tacoreader`. |
| `32_docs.py` | Generate README and HTML from `COLLECTION.json`. |
| `33_verify_taco.py` | Load the TACO locally and check schema, counts and VSI reads. |
| `40_export_pt2.py` | Export the 24 `.pt2` artifacts from checkpoints. |
| `41_verify_pt2.py` | Compare old vs new `.pt2` outputs and test batch sizes. |
| `42_update_mlm_json.py` | Update the 29 MLM JSON files. |
| `43_check_pretrain_split.py` | Verify how many SPOT scenes entered pretraining. |
| `50_upload_dataset.py` | Dry-run upload of the TACO dataset to Hugging Face. |
| `51_upload_model.py` | Dry-run upload of the model artifacts to Hugging Face. |
| `52_verify_remote.py` | Verify the published dataset and model over the network. |

Run the modules in order. Steps 20-21 and 40-41 require the large source
files under `/data` and are skipped by the public CI.
