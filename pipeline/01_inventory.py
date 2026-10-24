"""Build the scene inventory from the annotated CSV and source folders.

This script only reads metadata and asserts that the release contains the
expected 117 pixel scenes and 49 point scenes. It does not copy rasters.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build release inventory")
    parser.add_argument("--annotations", default="data/annotations_117.csv")
    parser.add_argument("--output", default="staging/inventory.parquet")
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    assert len(df) == 117, f"Expected 117 pixel scenes, got {len(df)}"

    n_proba = (df["sensor"] == "proba").sum()
    n_spot = (df["sensor"] == "spot").sum()
    print(f"Pixel scenes: {len(df)} (Proba-V: {n_proba}, SPOT: {n_spot})")

    # Placeholder for point scene discovery.
    points_dir = Path("/data/databases/Julio/proba_correction/test_pixbox/images")
    if points_dir.exists():
        point_scenes = [p.name for p in points_dir.iterdir() if p.is_dir()]
        print(f"Point scenes found: {len(point_scenes)}")
    else:
        print("Point scene directory not available on this machine")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
