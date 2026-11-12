# Splits

- `scenes.csv`: 117 pixel-annotated scenes, with role, sensor, date, centroid, paper and clean split labels, and QA columns.
- `points_scenes.csv`: 49 point-annotated Proba-V scenes.
- `weak_pretrain.csv`: 3,647 scene IDs used for weakly supervised pre-training.
- `split_summary.csv`: counts by role, sensor and year.
- `split_overlap.csv`: scene-ID intersections between roles.
- `geo_separation.csv`: distance from each test scene to the nearest fine-tuning scene.

`role` is one of `finetune_train`, `finetune_val`, `test_pixel`, `test_points`, `heldout_unused`, or `pretrain`.

Run:

```bash
python pipeline/17_check_split_overlap.py
python pipeline/18_split_geo_separation.py
```

No-data in `split_summary.csv` is the fraction of pixels outside the valid area or marked no-data. Clear is 0 and cloud is 1.
