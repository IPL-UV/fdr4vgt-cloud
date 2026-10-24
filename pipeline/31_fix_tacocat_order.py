"""Reorder ``.tacocat/level0.parquet`` to match ``taco_schema.py``.

After reordering, reopen the catalog with ``tacoreader`` to ensure nothing
is broken.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from fdr4vgt_cloud.taco_schema import LEVEL0_COLUMNS


def reorder_tacocat(tacocat_dir: Path) -> None:
    parquet_path = tacocat_dir / "level0.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    table = pq.read_table(parquet_path)
    columns = [c for c in LEVEL0_COLUMNS if c in table.column_names]
    columns += [c for c in table.column_names if c not in LEVEL0_COLUMNS]
    reordered = table.select(columns)
    pq.write_table(reordered, parquet_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix .tacocat column order")
    parser.add_argument(
        "--tacocat",
        default="/data/databases/Julio/FDR4VGT_RELEASE/taco/fdr4vgt-cloud-scenes/.tacocat",
    )
    args = parser.parse_args()

    reorder_tacocat(Path(args.tacocat))
    print(f"Reordered {args.tacocat}/level0.parquet")


if __name__ == "__main__":
    main()
