# Model documentation

## Architecture

The operational model is `CombinedNet4`:

- DeepLabV3 encoder with MobileNetV2 backbone (no ImageNet weights).
- Pixel-wise 1x1 MLP (`PixelWiseNet`).
- Residual addition followed by a 1x1 fusion convolution.
- Sigmoid wrapper for exported inference.

This is **DeepLabV3**, not DeepLabV3+.

## Exported artifacts

The model repository holds 24 `.pt2` files and 29 JSON files:

- `single/multisensor_single_1dpwdeeplabv3.pt2`: operational single model.
- `ensemble/multisensor_ensemble.pt2`: six-member ensemble that returns a
  probability and a per-pixel uncertainty.
- `single/f_global_*`, `single/proba_*`, `single/spot_*`: the ablation models.

The operational single model and the ensemble were exported with torch 2.7.0.
The ablation models were exported with torch 2.5.1. `torch.export` refuses to
load a file written by a different torch version, so each group needs its own
environment.

All exports use a dynamic batch size from 1 to 256 and include the sigmoid.
Each `.pt2` is rebuilt from its training checkpoint with
`pipeline/10_export_pt2.py` and `configs/export_manifest.yaml`.

The `load.py` published next to the artifacts is the copy in
`notebooks/fdr4vgt_model/load.py`. It is the module that mlstac imports to run
`compiled_model` and `predict_large`.

## Threshold

The recommended threshold is **0.5**.

A threshold of 0.4848 was obtained by optimising a weighted combination of
F2, recall, kappa, IoU, F1, precision and overall accuracy on the original
89-scene test set. It is documented here for reproducibility but not used as
the default.

## Input

- 4 bands in order: Blue, Red, NIR, SWIR.
- TOA reflectance, unscaled, nominal range 0 to 1.
- No normalisation is applied.

## Compatibility

- Operational single model and ensemble: torch 2.7.0.
- Ablation models: torch 2.5.1.
- The exported graph takes 512 x 512 tiles. Smaller crops are padded to 512 and
  cropped back by the published loader.

## Inference

Use `mlstac==0.4.9` and point it to the JSON file:

```python
import mlstac
url = "https://huggingface.co/isp-uv-es/FDR4VGT-CLOUD/resolve/main/single/multisensor_single_1dpwdeeplabv3.json"
model = mlstac.load(url).download("fdr4vgt_model").compiled_model(device="cpu")
```

To check the published artifacts end to end:

```bash
python pipeline/11_verify_pt2.py
```

## License

The code is MIT. Model weights are released under the same terms as the code.
Third-party training data providers retain their own licenses.
