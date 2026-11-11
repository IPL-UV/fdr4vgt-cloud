#!/usr/bin/env python
"""Check scene-ID overlap between the released split roles."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    scenes = pd.read_csv(ROOT / "splits/scenes.csv")
    points = pd.read_csv(ROOT / "splits/points_scenes.csv")
    weak = pd.read_csv(ROOT / "splits/weak_pretrain.csv")

    sets = {
        "pretrain": set(weak.scene_id),
        "finetune": set(scenes.loc[scenes.role == "finetune_train", "scene_id"]),
        "val": set(scenes.loc[scenes.role == "finetune_val", "scene_id"]),
        "test_pixel": set(scenes.loc[scenes.role == "test_pixel", "scene_id"]),
        "test_points": set(points.scene_id),
        "heldout_unused": set(scenes.loc[scenes.role == "heldout_unused", "scene_id"]),
    }

    rows = []
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            inter = sorted(sets[left] & sets[right])
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "n_left": len(sets[left]),
                    "n_right": len(sets[right]),
                    "n_overlap": len(inter),
                    "overlap_ids": ";".join(inter),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "splits/split_overlap.csv", index=False)
    print(out[["left", "right", "n_overlap"]].to_string(index=False))


if __name__ == "__main__":
    main()
