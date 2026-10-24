"""Inference helpers for large GeoTIFF images.

Exports:
  - ``predict_large``: tiled inference with Hann-squared blending.
  - ``load_pt2``: load a ``torch.export`` artifact.
  - ``apply_model``: run a model on an in-memory array.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable

import numpy as np
import rasterio as rio
import torch
import torch.nn as nn

__all__ = ["predict_large", "load_pt2", "apply_model"]


def load_pt2(path: str | pathlib.Path, device: str = "cpu") -> nn.Module:
    """Load a torch.export serialized model and return an executable module."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    exported = torch.export.load(str(path))
    model = exported.module()
    model.to(device)
    model.eval()
    return model


def apply_model(
    model: nn.Module,
    patch: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run ``model`` on a (C, H, W) array and return (1, H, W) probabilities."""
    tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
    return out.squeeze(0).cpu().numpy()


def _hann2(shape: tuple[int, int]) -> np.ndarray:
    """Return a 2-D Hann-squared window of the given shape."""
    h, w = shape
    hw = np.hanning(h).reshape(-1, 1)
    ww = np.hanning(w).reshape(1, -1)
    return (hw * ww) ** 2


def _tile_offsets(
    height: int,
    width: int,
    patch_size: int,
    overlap: int,
) -> Iterable[tuple[int, int]]:
    """Yield (row, col) offsets for a sliding window over the image."""
    step = patch_size - overlap
    rows = list(range(0, height - patch_size + 1, step))
    cols = list(range(0, width - patch_size + 1, step))
    # Ensure the last tile touches the bottom/right edge.
    if not rows or rows[-1] + patch_size < height:
        rows.append(height - patch_size)
    if not cols or cols[-1] + patch_size < width:
        cols.append(width - patch_size)
    for r in rows:
        for c in cols:
            yield r, c


def predict_large(
    image_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    model: nn.Module,
    patch_size: int = 512,
    overlap: int = 128,
    device: str = "cpu",
    dtype: np.dtype = np.float32,
) -> pathlib.Path:
    """Predict a full image using overlapping tiles and Hann-squared blending.

    Parameters
    ----------
    image_path:
        Path to the input GeoTIFF (4 bands, EPSG:4326).
    output_path:
        Path where the single-band probability raster will be written.
    model:
        Model that accepts tensors of shape ``(B, 4, patch_size, patch_size)``.
    patch_size:
        Tile size in pixels. Must match the model training input size.
    overlap:
        Overlap between neighbouring tiles in pixels.
    device:
        ``"cpu"`` or ``"cuda"`` device string.
    dtype:
        Output raster dtype.

    Returns
    -------
    Path to the written output raster.
    """
    image_path = pathlib.Path(image_path)
    output_path = pathlib.Path(output_path)

    with rio.open(image_path) as src:
        meta = src.profile.copy()
        height, width = src.height, src.width
        count = src.count
        if count != 4:
            raise ValueError(f"Expected 4 input bands, got {count}")

    # Output metadata: single-band probability
    out_meta = meta.copy()
    out_meta.update(
        {
            "count": 1,
            "dtype": dtype.name,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "nodata": None,
        }
    )

    numerator = np.zeros((1, height, width), dtype=np.float64)
    denominator = np.zeros((1, height, width), dtype=np.float64)

    window = _hann2((patch_size, patch_size))

    model.to(device)
    model.eval()

    with rio.open(image_path) as src:
        for row, col in _tile_offsets(height, width, patch_size, overlap):
            patch = src.read(window=rio.windows.Window(col, row, patch_size, patch_size))
            probs = apply_model(model, patch, device=device)
            if probs.ndim == 2:
                probs = probs[np.newaxis, ...]

            numerator[:, row : row + patch_size, col : col + patch_size] += probs * window
            denominator[:, row : row + patch_size, col : col + patch_size] += window

    # Avoid divide by zero in edge cases.
    out = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    ).astype(dtype)

    with rio.open(output_path, "w", **out_meta) as dst:
        dst.write(out)

    return output_path
