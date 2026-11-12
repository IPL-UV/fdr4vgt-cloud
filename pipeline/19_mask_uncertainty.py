"""Recompute F2, IoU and kappa against four versions of the reference mask.

The reference masks are drawn by one annotator, so the cloud boundary carries a
one-pixel uncertainty. This script takes the probability map of a model on the
test scenes that have a dense pixel mask and rescores it against:

- the original mask
- the cloud class eroded by one pixel (conservative reference)
- the cloud class dilated by one pixel (permissive reference)
- the original mask ignoring a one-pixel band around the cloud edge

Nothing in the image or in the prediction changes, only the reference. The
output table is the basis of Comments 3.2 and 3.12 of the response letter: the
four variants for eight architectures and the pre-trained model, and the four
variants for the three repeated training runs of the operational model.

Inference is batched over tiles and one process runs per device, so the 32
scenes finish in a couple of minutes on one GPU.

Example, two shards on two GPUs:

    python pipeline/19_mask_uncertainty.py --shard 0 --shards 2 --device cuda:0 \
        --model 1dpwdeeplabv3 --ckpt weights/multisensor.ckpt \
        --tables ~/tables --via splits/scenes.csv --out out/shard0.csv

The checkpoints used for the letter are the internal training runs behind the
published weights; any PyTorch checkpoint whose state dict has the same keys
works, and so does a run trained with `src/fdr4vgt_cloud/training/finetune.py`.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fdr4vgt_cloud.inference import _hann2, _tile_offsets  # noqa: E402
from fdr4vgt_cloud.model import SigmoidWrap, get_model  # noqa: E402
from fdr4vgt_cloud.training.config import MODEL_CONFIGS  # noqa: E402

PATCH = 512
OVERLAP = 128
STRUCTURE = ndimage.generate_binary_structure(2, 1)


def load_model(device: str, name: str, ckpt: str):
    """Load a checkpoint written by the fine-tuning script (key prefix `model.`)."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    state = {k.removeprefix("model."): v for k, v in state.items()}
    net = get_model(name)
    net.load_state_dict(state, strict=True)
    return SigmoidWrap(net).eval().to(device)


def predict(image: np.ndarray, model, device: str, batch_size: int) -> np.ndarray:
    """Tiled inference with Hann-squared blending, several tiles per forward."""
    _, height, width = image.shape
    numerator = np.zeros((height, width), dtype=np.float64)
    denominator = np.zeros((height, width), dtype=np.float64)
    window = _hann2((PATCH, PATCH))

    tiles = list(_tile_offsets(height, width, PATCH, OVERLAP))
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        batch = np.stack(
            [
                image[:, row : row + PATCH, col : col + PATCH].astype(np.float32)
                for row, col in chunk
            ]
        )
        tensor = torch.from_numpy(batch).to(device, non_blocking=True)
        with torch.no_grad():
            probs = model(tensor).squeeze(1).float().cpu().numpy()
        for (row, col), prob in zip(chunk, probs, strict=True):
            numerator[row : row + PATCH, col : col + PATCH] += prob * window
            denominator[row : row + PATCH, col : col + PATCH] += window

    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )


def scores(pred: np.ndarray, truth_cloud: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    out = (pred[valid] >= 0.5).astype(np.int8)
    truth = truth_cloud[valid].astype(np.int8)
    tp = int(((truth == 1) & (out == 1)).sum())
    fp = int(((truth == 0) & (out == 1)).sum())
    fn = int(((truth == 1) & (out == 0)).sum())
    tn = int(((truth == 0) & (out == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    total = tp + tn + fp + fn
    po = (tp + tn) / total if total else 0.0
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total**2) if total else 0.0
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    return f2, iou, kappa


def variants(mask: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the four reference variants as (cloud flags, valid flags).

    The nodata class (255) is never scored and is restored after dilation, so the
    morphology cannot turn the border of the scene into cloud.
    """
    nodata = mask == 255
    cloud = mask == 1
    eroded = ndimage.binary_erosion(cloud, STRUCTURE, border_value=0)
    dilated = ndimage.binary_dilation(cloud, STRUCTURE, border_value=0) & ~nodata
    ring = ndimage.binary_dilation(cloud, STRUCTURE) ^ eroded
    clean = ~nodata & ~ring
    return {
        "original": (cloud, ~nodata),
        "erode1": (eroded, ~nodata),
        "dilate1": (dilated, ~nodata),
        "no_ring": (cloud, clean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="1dpwdeeplabv3", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--ckpt", required=True, help="PyTorch checkpoint of the model")
    parser.add_argument("--label", default="", help="Name written in the output table")
    parser.add_argument("--tables", required=True, help="Directory with the metrics table")
    parser.add_argument("--via", required=True, help="Scene split table (splits/scenes.csv)")
    parser.add_argument("--out", default="")
    parser.add_argument("--limit", type=int, default=0, help="Debug: stop after N scenes")
    args = parser.parse_args()

    table = pd.read_csv(
        Path(args.tables) / "metrics_unified_all_sensors_human_mask_gt.csv", low_memory=False
    )
    table = table[(table.model == "f_global_1dpwdeeplabv3") & (table.ground_truth == "human_mask")]
    table["scene_id"] = table.rgb_path.str.split("/").str[-2]
    splits = pd.read_csv(args.via).rename(columns={"id": "scene_id"})
    test_ids = set(splits.loc[splits.role == "test_pixel", "scene_id"])
    table = table[table.scene_id.isin(test_ids)].drop_duplicates("scene_id")
    table = table[table.rgb_path.map(lambda p: (Path(p).parent / "human_mask.tif").exists())]
    table = table.sort_values(["sensor", "scene_id"]).reset_index(drop=True)
    table = table.iloc[args.shard :: args.shards]
    if args.limit:
        table = table.head(args.limit)
    print(f"shard {args.shard}: {len(table)} scenes on {args.device}", flush=True)

    model = load_model(args.device, args.model, args.ckpt)
    rows = []
    for row in table.itertuples():
        start = time.time()
        with rio.open(row.rgb_path) as src:
            image = src.read()
        with rio.open(Path(row.rgb_path).parent / "human_mask.tif") as src:
            mask = src.read(1)
        pred = predict(image, model, args.device, args.batch)
        for name, (cloud, valid) in variants(mask).items():
            f2, iou, kappa = scores(pred, cloud, valid)
            rows.append(
                {
                    "scene_id": row.scene_id,
                    "model": args.label or args.model,
                    "sensor": row.sensor,
                    "variant": name,
                    "F2": round(f2, 4),
                    "IoU": round(iou, 4),
                    "Kappa": round(kappa, 4),
                    "valid_px": int(valid.sum()),
                }
            )
        print(f"{row.scene_id} {row.sensor} {time.time() - start:.1f}s", flush=True)

    out = Path(args.out) if args.out else Path(f"mask_uncertainty_shard{args.shard}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"written {out}", flush=True)


if __name__ == "__main__":
    main()
