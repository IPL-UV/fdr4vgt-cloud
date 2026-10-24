#!/usr/bin/env python
"""Quick start figure for the FDR4VGT-CLOUD release.

Runs the operational single model on a 512 x 512 crop of a held out
SPOT-VGT scene and draws three panels: the reflectance composite, the
cloud probability and the binary mask at threshold 0.5.

The model and the example scene are read from the local release
folders when they exist. Otherwise the script falls back to the
published files on Hugging Face.

Run from the repository root:

    python docs/images/quickstart_prediction.py

Environment variables:

    FDR4VGT_DATA        folder with the scene subfolders (rgb.tif,
                        human_mask.tif)
    FDR4VGT_MODEL_JSON  mlstac JSON of the single model
"""

from __future__ import annotations

import os
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE / "quickstart_prediction.png"

# Held out test scene of the FDR4VGT-CLOUD test split.
SCENE = "V2KRNP____20121111F149_V003"
SCENE_DATE = "11 November 2012"
CROP_ROW = 4096
CROP_COL = 2048
CROP_SIZE = 512
THRESHOLD = 0.5

DATA_ROOT = pathlib.Path(
    os.environ.get(
        "FDR4VGT_DATA",
        "/data/users/julio/experiments/cloud-proba-spot/release/data/cog",
    )
)
MODEL_JSON = pathlib.Path(
    os.environ.get(
        "FDR4VGT_MODEL_JSON",
        "/data/users/julio/experiments/cloud-proba-spot/release/hf_update/"
        "single/multisensor_single_1dpwdeeplabv3.json",
    )
)
HF_MODEL_JSON = (
    "https://huggingface.co/isp-uv-es/FDR4VGT-CLOUD/resolve/main/"
    "single/multisensor_single_1dpwdeeplabv3.json"
)

# Bands in the GeoTIFF are Blue, Red, NIR, SWIR. The composite is B4-B3-B2,
# that is SWIR, NIR and Red, printed as red, green and blue.
COMPOSITE_BANDS = (3, 2, 1)
COMPOSITE_LABEL = "B4-B3-B2"
STRETCH_PERCENTILES = (2, 98)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.edgecolor": "#9a9a9a",
        "axes.linewidth": 0.6,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

PROB_CMAP = LinearSegmentedColormap.from_list("cloud", ["#eef1f4", "#08306b"])
MASK_CMAP = ListedColormap(["#eef1f4", "#08306b"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_model():
    """Load the single model with mlstac and return the loader and the net."""
    import mlstac

    source = MODEL_JSON if MODEL_JSON.exists() else HF_MODEL_JSON
    loader = mlstac.load(str(source))
    device = "cuda" if _cuda_available() else "cpu"
    net = loader.compiled_model(device=device)
    return loader, net, device


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def read_crop() -> np.ndarray:
    """Read the four band crop used for the figure.

    Uses a local scene when available; otherwise falls back to the sample
    image hosted on Hugging Face.
    """
    path = DATA_ROOT / SCENE / "rgb.tif"
    if path.exists():
        window = rasterio.windows.Window(CROP_COL, CROP_ROW, CROP_SIZE, CROP_SIZE)
        with rasterio.open(path) as src:
            return src.read(window=window)

    from huggingface_hub import hf_hub_download

    sample = hf_hub_download(
        repo_id="isp-uv-es/FDR4VGT-CLOUD",
        filename="ensemble/rgb.tif",
    )
    window = rasterio.windows.Window(0, 0, CROP_SIZE, CROP_SIZE)
    with rasterio.open(sample) as src:
        return src.read(window=window)


def _srgb(values: np.ndarray) -> np.ndarray:
    """Encode linear values for display, lifting the shadows."""
    low = values <= 0.0031308
    return np.where(low, 12.92 * values, 1.055 * np.power(values, 1 / 2.4) - 0.055)


def stretch(image: np.ndarray) -> np.ndarray:
    """Percentile stretch per band over valid pixels, then sRGB encoding."""
    out = np.zeros(image.shape, dtype=np.float32)
    for band in range(image.shape[0]):
        values = image[band][image[band] > 0]
        low, high = np.percentile(values, STRETCH_PERCENTILES)
        out[band] = np.clip((image[band] - low) / max(high - low, 1e-6), 0, 1)
    return _srgb(out).astype(np.float32)


def composite(image: np.ndarray) -> np.ndarray:
    rgb = stretch(image[list(COMPOSITE_BANDS)])
    return np.moveaxis(rgb, 0, -1)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def draw(image: np.ndarray, prob: np.ndarray, output: pathlib.Path) -> None:
    rgb = composite(image)
    mask = (prob > THRESHOLD).astype(np.uint8)

    fig = plt.figure(figsize=(11.2, 4.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0])

    ax_rgb = fig.add_subplot(grid[0, 0])
    ax_prob = fig.add_subplot(grid[0, 1])
    ax_mask = fig.add_subplot(grid[0, 2])

    ax_rgb.imshow(rgb, interpolation="nearest")
    ax_rgb.set_title(f"(a) Reflectance composite, {COMPOSITE_LABEL}", pad=6)

    im = ax_prob.imshow(prob, cmap=PROB_CMAP, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_prob.set_title("(b) Cloud probability", pad=6)

    ax_mask.imshow(mask, cmap=MASK_CMAP, vmin=0, vmax=1, interpolation="nearest")
    ax_mask.set_title(f"(c) Cloud mask at {THRESHOLD:g}", pad=6)

    for ax in (ax_rgb, ax_prob, ax_mask):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Cloud detection with the FDR4VGT-CLOUD single model", fontsize=12.5)
    fig.supxlabel(
        f"SPOT-VGT scene of {SCENE_DATE}, {CROP_SIZE} x {CROP_SIZE} crop, "
        f"probability and mask from mlstac 0.4.9. Dark blue marks cloud.",
        fontsize=8,
        color="#4d4d4d",
    )

    # Three equal square panels with a wide gutter. The colorbar sits under
    # (b) so the gutters between panels stay the same width.
    fig.set_layout_engine("constrained", w_pad=0.32, h_pad=0.06)
    fig.get_layout_engine().set(rect=(0.012, 0.040, 0.988, 0.972))
    fig.draw_without_rendering()

    # Centre the block of panels so the left and right margins match, then
    # freeze the result: the colorbar is placed from these positions.
    margin_left = ax_rgb.get_position().x0
    margin_right = 1.0 - ax_mask.get_position().x1
    shift = 0.5 * (margin_right - margin_left)
    for ax in (ax_rgb, ax_prob, ax_mask):
        box = ax.get_position()
        ax.set_position([box.x0 + shift, box.y0, box.width, box.height])
    fig.set_layout_engine("none")

    cax = ax_prob.inset_axes([0.20, -0.062, 0.60, 0.030])
    cax.set_in_layout(False)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(
        labelsize=7.5,
        length=2,
        width=0.6,
        color="#9a9a9a",
        labelcolor="#4d4d4d",
        pad=2,
    )
    cbar.outline.set_edgecolor("#9a9a9a")
    cbar.outline.set_linewidth(0.6)

    fig.savefig(output, dpi=150, pil_kwargs={"optimize": True})
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    loader, net, device = load_model()
    image = read_crop()
    prob = loader.predict_large(
        image=image,
        model=net,
        device=device,
        batch_size=4,
        num_workers=8,
        nodata=0.0,
    )
    prob = np.squeeze(prob).astype(np.float32)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    draw(image, prob, OUTPUT)

    cover = float((prob > THRESHOLD).mean())
    print(f"scene {SCENE} crop {CROP_ROW},{CROP_COL} {CROP_SIZE} px on {device}")
    print(f"cloud cover at {THRESHOLD:g}: {cover:.3f}")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
