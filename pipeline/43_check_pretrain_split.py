"""Reproduce the weak pretraining split and report how many SPOT scenes enter.

This check uses the old tacoreader 0.5.6 logic from
``TRAINPROBAV/train_global_cloud_mask/dataloader.py``: it concatenates the
Proba-V and SPOT indices and then calls ``train_test_split`` twice. If the
concatenation is not reset, ``TOTAL_DB.iloc[train_idx]`` may silently drop
SPOT rows.

Output: ``staging/pretrain_split_check.md``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Check pretraining split")
    parser.add_argument(
        "--probav-index",
        default="/data/databases/spot_proba_cloud/CESAR_PROBAV_VEGETATION/tacos/index.parquet",
    )
    parser.add_argument(
        "--spot-index",
        default="/data/databases/spot_proba_cloud/CESAR_SPOT_VEGETATION/tacos/index.parquet",
    )
    parser.add_argument("--output", default="staging/pretrain_split_check.md")
    args = parser.parse_args()

    probav = (
        pd.read_parquet(args.probav_index) if Path(args.probav_index).exists() else pd.DataFrame()
    )
    spot = pd.read_parquet(args.spot_index) if Path(args.spot_index).exists() else pd.DataFrame()

    if probav.empty and spot.empty:
        print("Source indices not available; writing placeholder report")
        text = "# Pretraining split check\n\nSource indices not found on this machine.\n"
    else:
        total = pd.concat([probav, spot], axis=0)
        train_idx, val_idx = train_test_split(total.index, test_size=0.1, random_state=42)
        train_idx, test_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

        def sensor_counts(idx):
            s = total.loc[idx, "sensor"] if "sensor" in total.columns else pd.Series(dtype="object")
            return s.value_counts().to_dict()

        text = f"""# Pretraining split check

Total rows: {len(total)}
Proba-V rows: {len(probav)}
SPOT rows: {len(spot)}

Train sensor counts: {sensor_counts(train_idx)}
Val sensor counts: {sensor_counts(val_idx)}
Test sensor counts: {sensor_counts(test_idx)}
"""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
