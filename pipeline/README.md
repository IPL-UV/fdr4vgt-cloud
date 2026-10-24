# Pipeline modules

| Module | Purpose |
|---|---|
| `01_inventory.py` | Build the scene inventory and assert 117 pixel + 49 point scenes. |
| `02_splits.py` | Recalculate train/val/test splits and export `splits/scenes.csv`. |
| `03_metadata.py` | Extract per-scene metadata from product IDs and SPOT LOG files. |
| `04_cog.py` | Convert source images and masks to Cloud Optimized GeoTIFFs. |
| `05_verify_cog.py` | Validate every COG and compare against the source data. |
| `06_build_taco.py` | Build the two TACO partitions (`scenes` and `points`). |
| `07_fix_tacocat_order.py` | Reorder `.tacocat/level0.parquet` and reopen with `tacoreader`. |
| `08_docs.py` | Generate README and HTML from `COLLECTION.json`. |
| `09_verify_taco.py` | Load the TACO locally and check schema, counts and VSI reads. |
| `10_export_pt2.py` | Export the 24 `.pt2` artifacts from checkpoints. |
| `11_verify_pt2.py` | Compare old vs new `.pt2` outputs and test batch sizes. |
| `12_update_mlm_json.py` | Update the 29 MLM JSON files. |
| `13_check_pretrain_split.py` | Verify how many SPOT scenes entered pretraining. |
| `14_upload_dataset.py` | Dry-run upload of the TACO dataset to Hugging Face. |
| `15_upload_model.py` | Dry-run upload of the model artifacts to Hugging Face. |
| `16_verify_remote.py` | Verify the published dataset and model over the network. |

Run the modules in order. Steps 04-05 and 10-11 require the large source
files under `/data` and are skipped by the public CI.
