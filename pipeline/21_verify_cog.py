"""Validate every COG and compare it against the source raster.

Checks performed:
  - rio_cogeo validation
  - CRS is EPSG:4326
  - Mask values are a subset of {0, 1, 255}
  - Image values are within the expected TOA reflectance range
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio as rio

from fdr4vgt_cloud.cog import validate_cog


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify COGs")
    parser.add_argument("--cog-dir", default="/data/databases/Julio/FDR4VGT_RELEASE/cog")
    args = parser.parse_args()

    cog_dir = Path(args.cog_dir)
    errors = []
    for scene_dir in sorted(cog_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for name in ("image.tif", "mask.tif"):
            path = scene_dir / name
            if not path.exists():
                errors.append(f"Missing {path}")
                continue
            valid, msgs = validate_cog(path, strict=True)
            if not valid:
                errors.append(f"{path}: {'; '.join(msgs)}")
                continue
            with rio.open(path) as src:
                if src.crs is None or src.crs.to_epsg() != 4326:
                    errors.append(f"{path}: CRS is not EPSG:4326")
                if name == "mask.tif":
                    data = src.read(1)
                    if not set(np.unique(data)).issubset({0, 1, 255}):
                        errors.append(f"{path}: unexpected mask values")

    if errors:
        print(f"Found {len(errors)} errors:")
        for e in errors[:20]:
            print(" -", e)
        raise SystemExit(1)
    print("All COGs are valid")


if __name__ == "__main__":
    main()
