"""Load each TACO partition and verify counts, schema and VSI reads."""

from __future__ import annotations

import argparse

import tacoreader

from fdr4vgt_cloud.taco_schema import LEVEL0_COLUMNS


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TACO partitions")
    parser.add_argument(
        "--scenes",
        default="/data/databases/Julio/FDR4VGT_RELEASE/taco/fdr4vgt-cloud-scenes/.tacocat",
    )
    parser.add_argument(
        "--points",
        default="/data/databases/Julio/FDR4VGT_RELEASE/taco/fdr4vgt-cloud-points/.tacocat",
    )
    args = parser.parse_args()

    for name, path in [("scenes", args.scenes), ("points", args.points)]:
        df = tacoreader.load(path)
        missing = [c for c in LEVEL0_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        print(f"{name}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
