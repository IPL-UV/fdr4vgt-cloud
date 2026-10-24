"""Recalculate splits for the 117 pixel scenes and 49 point scenes.

Outputs:
  - splits/scenes.csv
  - splits/points_scenes.csv

The pixel split follows the published protocol:
  - 78 fine-tuning scenes -> 66 train / 12 val
  - 24 Proba-V held-out scenes -> test_pixel
  - 8 SPOT test scenes -> test_pixel
  - 7 SPOT held-out scenes -> heldout_unused
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def assign_roles(annotations: pd.DataFrame) -> pd.Series:
    roles = []
    for _, row in annotations.iterrows():
        subset = row["subset"]
        sensor = row["sensor"]
        if subset == "fine_tune":
            sp = row["split_paper"]
            roles.append("finetune_val" if sp == "val" else "finetune_train")
        elif subset == "held_out":
            roles.append("heldout_unused" if sensor == "spot" else "test_pixel")
        elif subset == "test":
            roles.append("test_pixel")
        else:
            roles.append("unknown")
    return pd.Series(roles, index=annotations.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build release splits")
    parser.add_argument("--annotations", default="data/annotations_117.csv")
    parser.add_argument(
        "--e2-splits", default="/data/users/julio/experiments/cloud-proba-spot/e2/splits"
    )
    parser.add_argument("--output-dir", default="splits")
    args = parser.parse_args()

    ann = pd.read_csv(args.annotations)
    ann["role"] = assign_roles(ann)

    e2_dir = Path(args.e2_splits)
    finetune = pd.read_csv(e2_dir / "finetune_clean.csv")
    test89 = pd.read_csv(e2_dir / "test_clean_89.csv")
    weak = pd.read_csv(e2_dir / "weak_clean.csv")

    clean_map = {r["scene_id"]: r["split"] for _, r in finetune.iterrows()}
    for sid in test89["scene_id"]:
        clean_map[sid] = "test"
    weak_map = {r["scene_id"]: r["split"] for _, r in weak.iterrows()}

    out = pd.DataFrame(
        {
            "scene_id": ann["scene_id"],
            "sensor": ann["sensor"],
            "date": ann["date"],
            "role": ann["role"],
            "split_paper": ann["split_paper"],
            "split_clean": [clean_map.get(sid, "none") for sid in ann["scene_id"]],
            "qa:weak_db_split": [weak_map.get(sid, "none") for sid in ann["scene_id"]],
            "qa:in_original_test89": [sid in set(test89["scene_id"]) for sid in ann["scene_id"]],
            "qa:mask_variant_differs": ["F164" in sid for sid in ann["scene_id"]],
            "qa:source_image_md5": ann["rgb_md5"],
            "qa:source_mask_md5": ann["mask_md5"],
        }
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.output_dir) / "scenes.csv", index=False)
    print(out["role"].value_counts())

    points = test89[test89["sensor"] == "proba_pixbox"][["scene_id", "sensor", "rgb"]].copy()
    points["role"] = "test_points"
    points.to_csv(Path(args.output_dir) / "points_scenes.csv", index=False)
    print(f"Point scenes: {len(points)}")


if __name__ == "__main__":
    main()
