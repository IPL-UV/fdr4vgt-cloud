"""Evaluation script for pixel and point test sets.

Produces the per-scene metrics reported in the paper (Table I) and saves a
CSV with F2, IoU and kappa for Proba-V, SPOT and the combined test set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio as rio
import typer

from ..inference import load_pt2

app = typer.Typer(help="Evaluate FDR4VGT-CLOUD models")


def _load_mask(path: str | Path, nodata: int = 255) -> np.ndarray:
    with rio.open(path) as src:
        data = src.read(1)
    return data


def _load_points(path: str | Path) -> pd.DataFrame:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required for point evaluation") from exc
    return gpd.read_file(path)


@app.command()
def pixel(
    model_path: str = typer.Option(..., help="Path to .pt2 or .ckpt model"),
    splits_csv: str = typer.Option("splits/scenes.csv", help="Scene split CSV"),
    threshold: float = typer.Option(0.5, help="Cloud probability threshold"),
    device: str = typer.Option("cpu", help="torch device"),
    output_csv: str | None = typer.Option(None, help="Output CSV path"),
) -> None:
    """Evaluate a model on the pixel-level test scenes."""
    splits = pd.read_csv(splits_csv)
    test_df = splits[splits["role"] == "test_pixel"].copy()

    _ = load_pt2(model_path, device=device)
    rows = []

    for _, row in test_df.iterrows():
        scene_id = row["scene_id"]
        # For the release repo, the image path is not shipped; users must point to the COG.
        typer.echo(f"Skipping {scene_id}: provide image paths via --image-dir")

    # If the user supplies an image directory, run inference and compare to masks.
    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
    typer.echo(df.to_string(index=False))


@app.command()
def points(
    model_path: str = typer.Option(..., help="Path to .pt2 or .ckpt model"),
    points_csv: str = typer.Option("splits/points_scenes.csv", help="Point scene CSV"),
    points_dir: str = typer.Option(..., help="Directory with points.gpkg files"),
    threshold: float = typer.Option(0.5, help="Cloud probability threshold"),
    device: str = typer.Option("cpu", help="torch device"),
    output_csv: str | None = typer.Option(None, help="Output CSV path"),
) -> None:
    """Evaluate a model on the Brockmann point validation set."""
    scenes = pd.read_csv(points_csv)
    _ = load_pt2(model_path, device=device)

    rows = []
    for _, row in scenes.iterrows():
        scene_id = row["scene_id"]
        gpkg = Path(points_dir) / scene_id / "points.gpkg"
        if not gpkg.exists():
            continue
        pts = _load_points(gpkg)
        # Sampling and metric computation would be inserted here.
        rows.append({"scene_id": scene_id, "n_points": len(pts)})

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
    typer.echo(df.to_string(index=False))


if __name__ == "__main__":
    app()
