# Hugging Face model repository placeholder

This directory mirrors the contents of the `isp-uv-es/FDR4VGT-CLOUD` model
repository on Hugging Face. The actual `.pt2` weights and `mlm.json` files
are not tracked in git because they are large binary artifacts.

## Expected structure

```
hf_model/
├── single/
│   ├── mlm.json
│   ├── model.pt2
│   ├── load.py
│   └── example_data.safetensors
├── ensemble/
│   ├── mlm.json
│   ├── model.pt2
│   ├── load.py
│   └── example_data.safetensors
└── ablation/
    ├── f_global_*.pt2     (7 files)
    ├── proba_*.pt2        (6 files)
    ├── spot_*.pt2         (6 files)
    └── ensemble_global_*.pt2 (3 files)
```

Total: **24 `.pt2` files** and **29 `.mlm.json` files**.

The JSON files are updated by `pipeline/42_update_mlm_json.py`.
The `.pt2` files are exported by `pipeline/40_export_pt2.py`.

## Model card

The model card (`README.md`) for Hugging Face is generated separately and
should mention:

- DeepLabV3 + pixel-wise MLP architecture.
- Recommended threshold **0.5**.
- TOA reflectance input, unscaled, 4 bands.
- Links to the dataset and this GitHub repository.
