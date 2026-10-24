"""Extract per-scene metadata from product IDs and SPOT LOG files.

This script documents the source of every metadata field and avoids
invented constants. It writes ``staging/scene_metadata.parquet``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from pathlib import Path

import pandas as pd


def parse_proba_id(scene_id: str) -> dict:
    """Parse a Proba-V L2A product ID (Collection 1)."""
    pattern = r"PROBAV_L2A_(\d{8})_(\d{6})_(\d)_(\dKM|333M)_V101(?:_(\d+))?"
    m = re.match(pattern, scene_id)
    if not m:
        return {}
    date_str, time_str, camera, resolution, crop = m.groups()
    date = dt.datetime.strptime(date_str, "%Y%m%d").replace(
        hour=int(time_str[:2]),
        minute=int(time_str[2:4]),
        second=int(time_str[4:6]),
        tzinfo=dt.UTC,
    )
    return {
        "scene:platform": "PROBA-V",
        "scene:collection": "V101",
        "scene:camera": int(camera),
        "scene:nominal_resolution_m": 1000 if resolution == "1KM" else 333,
        "stac:time_start": int(date.timestamp() * 1e6),
        "meta:time_source": "product_id",
    }


def parse_spot_id(scene_id: str) -> dict:
    """Parse a SPOT-VGT product ID (V003)."""
    pattern = r"V(\d)KRNP____(\d{8})F(\d{3})_V003"
    m = re.match(pattern, scene_id)
    if not m:
        return {}
    sensor_num, date_str, orbit = m.groups()
    date = dt.datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=dt.UTC)
    platform = "SPOT-4" if sensor_num == "1" else "SPOT-5"
    return {
        "scene:platform": platform,
        "scene:collection": "V003",
        "scene:camera": None,
        "scene:nominal_resolution_m": 1000,
        "stac:time_start": int(date.timestamp() * 1e6),
        "scene:orbit_number": int(orbit),
        "meta:time_source": "product_id_date_only",
    }


def parse_id(scene_id: str, sensor: str) -> dict:
    if sensor == "proba":
        return parse_proba_id(scene_id)
    if sensor == "spot":
        return parse_spot_id(scene_id)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build scene metadata table")
    parser.add_argument("--splits", default="splits/scenes.csv")
    parser.add_argument("--output", default="staging/scene_metadata.parquet")
    args = parser.parse_args()

    splits = pd.read_csv(args.splits)
    records = []
    for _, row in splits.iterrows():
        meta = {"scene_id": row["scene_id"], "sensor": row["sensor"]}
        meta.update(parse_id(row["scene_id"], row["sensor"]))
        records.append(meta)

    df = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output)
    print(f"Wrote {args.output} with {len(df)} rows")


if __name__ == "__main__":
    main()
