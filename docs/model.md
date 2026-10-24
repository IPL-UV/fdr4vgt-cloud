# Model documentation

## Architecture

The operational model is `CombinedNet4`:

- DeepLabV3 encoder with MobileNetV2 backbone (no ImageNet weights).
- Pixel-wise 1x1 MLP (`PixelWiseNet`).
- Residual addition followed by a 1x1 fusion convolution.
- Sigmoid wrapper for exported inference.

This is **DeepLabV3**, not DeepLabV3+.

## Exported artifacts

The model repository contains 23 `.pt2` files and 28 JSON files:

- `single/model.pt2`: operational single model.
- `ablation/`: 22 additional models for the ablation study.

All exports use torch 2.7.0, dynamic batch size from 1 to 256, and include the sigmoid.

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

Exports are tested with:

- torch 2.7.0 (build environment)
- torch 2.14.1 CPU (forward compatibility)

## Inference

Use `mlstac>=0.4.9` and point it to the JSON file:

```python
import mlstac
loader = mlstac.load("https://huggingface.co/isp-uv-es/FDR4VGT-CLOUD/resolve/main/single/mlm.json")
model = loader.compiled_model(device="cpu")
```

## License

The code is MIT. Model weights are released under the same terms as the code.
Third-party training data providers retain their own licenses.
