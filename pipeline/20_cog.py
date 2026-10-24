"""Convert source rasters to Cloud Optimized GeoTIFFs.

This script must be run on a machine with the source files under
`/data/databases/Julio`. The output is written to a staging directory and
later packaged into TACO by `30_build_taco.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fdr4vgt_cloud.cog import image_profile, mask_profile, to_cog


def main() -> None:
    parser = argparse.ArgumentParser(description="Build COGs")
    parser.add_argument("--splits", default="splits/scenes.csv")
    parser.add_argument("--annotations", default="data/annotations_117.csv")
    parser.add_argument("--output-dir", default="/data/databases/Julio/FDR4VGT_RELEASE/cog")
    args = parser.parse_args()

    splits = pd.read_csv(args.splits)
    ann = pd.read_csv(args.annotations)
    ann = ann.set_index("scene_id")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for _, row in splits.iterrows():
        scene_id = row["scene_id"]
        if scene_id not in ann.index:
            continue
        scene_dir = out_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        src_rgb = ann.loc[scene_id, "rgb"]
        src_mask = ann.loc[scene_id, "mask"]

        dst_rgb = scene_dir / "image.tif"
        dst_mask = scene_dir / "mask.tif"

        if not dst_rgb.exists():
            print(f"COG image: {scene_id}")
            to_cog(src_rgb, dst_rgb, profile=image_profile())
        if not dst_mask.exists():
            print(f"COG mask: {scene_id}")
            to_cog(src_mask, dst_mask, profile=mask_profile())

    print(f"Done. COGs in {out_dir}")


if __name__ == "__main__":
    main()
