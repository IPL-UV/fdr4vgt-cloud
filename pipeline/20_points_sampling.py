"""Describe how the point-based Proba-V reference is made up.

The reviewer asked how the 7,654 reference points were sampled and what bias
that sampling carries. The answer is in the points themselves: each scene ships
a `points.gpkg` that keeps the fields of the original validation product
(product name, pixel coordinates on the product grid, cloud label, date and
surface-type class), so the composition of the set can be measured instead of
recalled.

Writes two tables:

- `r3_16_points_composition.csv`: per scene, number of points, cloud, clear,
  surface classes and date.
- `r3_17_points_balance.csv`: points per date and class.

The surface-type identifiers are counted but not interpreted, because the
distributed files do not include the class legend.

Example:

    python pipeline/20_points_sampling.py --root /path/to/test_pixbox/images --out out
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Directory with one subdirectory per scene")
    parser.add_argument("--out", default=".")
    args = parser.parse_args()

    scenes, points = [], []
    for scene_dir in sorted(Path(args.root).iterdir()):
        path = scene_dir / "points.gpkg"
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        gdf["scene_id"] = scene_dir.name
        points.append(
            gdf[["scene_id", "PIXEL_X", "PIXEL_Y", "CLOUD", "PIXEL_SURFACE_TYPE_ID", "date"]]
        )
        scenes.append(
            {
                "scene_id": scene_dir.name,
                "date": str(gdf["date"].iloc[0])[:10],
                "n_points": len(gdf),
                "n_cloud": int((gdf["CLOUD"] == 1).sum()),
                "n_clear": int((gdf["CLOUD"] == 0).sum()),
                "n_surface_types": int(gdf["PIXEL_SURFACE_TYPE_ID"].nunique()),
            }
        )

    per_scene = pd.DataFrame(scenes)
    per_scene["cloud_fraction"] = (per_scene.n_cloud / per_scene.n_points).round(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_scene.to_csv(out / "r3_16_points_composition.csv", index=False)

    every = pd.concat(points, ignore_index=True)
    balance = (
        every.assign(date=every.date.astype(str).str[:10])
        .groupby(["date", "CLOUD"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "clear", 1: "cloud"})
    )
    balance["total"] = balance.sum(axis=1)
    balance.to_csv(out / "r3_17_points_balance.csv")

    print(f"scenes {len(per_scene)} | points {per_scene.n_points.sum()}")
    print(
        "points per scene: min %d, median %d, max %d"
        % (per_scene.n_points.min(), per_scene.n_points.median(), per_scene.n_points.max())
    )
    print("cloud %.1f%%, clear %.1f%%" % (100 * every.CLOUD.mean(), 100 * (1 - every.CLOUD.mean())))
    print("scenes without cloud:", int((per_scene.n_cloud == 0).sum()))
    print("scenes without clear:", int((per_scene.n_clear == 0).sum()))
    print(balance.to_string())


if __name__ == "__main__":
    main()
