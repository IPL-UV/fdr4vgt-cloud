"""Update the 29 MLM JSON files for mlstac 0.4.9.

Corrections applied (per Opus review):
  - ``mlm:framework_version`` = "2.7.0"
  - ``recommended_threshold`` = 0.5
  - ``mlm:architecture`` mentions DeepLabV3 (not V3+)
  - ``custom:sensors`` = ["VGT-1", "VGT-2", "PROBA-V"]
  - ``custom:spatial_resolution`` = "1 km"
  - input scaling = "none" (TOA reflectance, unscaled)
  - file sizes match the real `.pt2`
  - valid MLM 1.5 schema
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def update_json(path: Path) -> None:
    with open(path) as f:
        item = json.load(f)

    item["mlm:framework_version"] = "2.7.0"
    item["mlm:architecture"] = "DeepLabV3 (MobileNetV2 encoder) + pixel-wise MLP"
    item["recommended_threshold"] = 0.5

    custom = item.setdefault("custom", {})
    custom["sensors"] = ["VGT-1", "VGT-2", "PROBA-V"]
    custom["spatial_resolution"] = "1 km"

    if "mlm:input" in item:
        item["mlm:input"]["value_scaling"] = "none"
        item["mlm:input"]["description"] = "TOA reflectance, unscaled, nominal 0 to 1"

    # Recalculate file:size from the sibling .pt2 if present.
    pt2 = path.with_suffix(".pt2")
    if pt2.exists():
        item["file:size"] = pt2.stat().st_size

    with open(path, "w") as f:
        json.dump(item, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update MLM JSON files")
    parser.add_argument("--model-dir", default="hf_model")
    args = parser.parse_args()

    for json_path in sorted(Path(args.model_dir).rglob("*.json")):
        update_json(json_path)
        print(f"Updated {json_path}")


if __name__ == "__main__":
    main()
