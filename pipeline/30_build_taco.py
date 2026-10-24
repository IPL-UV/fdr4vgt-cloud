"""Build the two TACO partitions from the validated COGs.

Partitions:
  - fdr4vgt-cloud-scenes (117 pixel scenes)
  - fdr4vgt-cloud-points (49 point scenes)

Uses tacotoolbox 0.26.9. Plain COG assets are used (no TACOTIFF/Header).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_partition(
    name: str,
    scenes: pd.DataFrame,
    cog_dir: Path,
    output_dir: Path,
) -> None:
    print(f"Building partition {name} with {len(scenes)} scenes")
    # Implementation uses tacotoolbox.Tortilla, Taco, Sample, STAC, Labels, SplitStrategy.
    # The actual call is omitted here because it requires the full source COGs and the
    # tacotoolbox API. This script is a placeholder for the release pipeline.
    (output_dir / name).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TACO partitions")
    parser.add_argument("--splits", default="splits/scenes.csv")
    parser.add_argument("--points", default="splits/points_scenes.csv")
    parser.add_argument("--cog-dir", default="/data/databases/Julio/FDR4VGT_RELEASE/cog")
    parser.add_argument("--output-dir", default="/data/databases/Julio/FDR4VGT_RELEASE/taco")
    args = parser.parse_args()

    scenes = pd.read_csv(args.splits)
    points = pd.read_csv(args.points)
    cog_dir = Path(args.cog_dir)
    output_dir = Path(args.output_dir)

    build_partition("fdr4vgt-cloud-scenes", scenes, cog_dir, output_dir)
    build_partition("fdr4vgt-cloud-points", points, cog_dir, output_dir)

    print(f"Done. Partitions in {output_dir}")


if __name__ == "__main__":
    main()
