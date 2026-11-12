#!/usr/bin/env python
"""Report the distance from each test scene to the nearest fine-tuning scene."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_KM = 6371.0


def haversine(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in km between one point and an array of points."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def main() -> None:
    scenes = pd.read_csv(ROOT / "splits/scenes.csv")
    train = scenes[scenes.role.isin(["finetune_train", "finetune_val"])]
    test = scenes[scenes.role == "test_pixel"]

    rows = []
    for sensor in sorted(test.sensor.unique()):
        tr = train[train.sensor == sensor]
        te = test[test.sensor == sensor]
        if tr.empty or te.empty:
            continue
        dist = [
            haversine(row.centroid_lon, row.centroid_lat, tr.centroid_lon, tr.centroid_lat).min()
            for row in te.itertuples()
        ]
        dist = np.asarray(dist)
        rows.append(
            {
                "sensor": sensor,
                "n_train": len(tr),
                "n_test": len(te),
                "min_km": round(float(dist.min())),
                "median_km": round(float(np.median(dist))),
                "max_km": round(float(dist.max())),
            }
        )

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "splits/geo_separation.csv", index=False)


if __name__ == "__main__":
    main()
