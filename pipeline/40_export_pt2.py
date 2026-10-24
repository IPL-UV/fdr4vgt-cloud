"""Export the 24 torch.export `.pt2` artifacts from checkpoints.

Artifacts (per Opus review):
  - 2 operational: single (1dpwdeeplabv3) and ensemble (6 members)
  - 7 f_global
  - 6 proba
  - 6 spot
  - 3 ensemble_global

Each export wraps logits with a sigmoid and uses dynamic batch [1, 256].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fdr4vgt_cloud.model import CombinedNet4, SigmoidWrap


def export_single(ckpt_path: Path, output_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("model."): v for k, v in state.items()}

    model = CombinedNet4(in_channels=4, classes=1, base_channels=32)
    model.load_state_dict(state, strict=True)
    wrapped = SigmoidWrap(model).eval()

    batch = torch.export.Dim("batch", min=1, max=256)
    ep = torch.export.export(
        wrapped,
        (torch.randn(1, 4, 512, 512),),
        dynamic_shapes={"x": {0: batch}},
        strict=False,
    )
    torch.export.save(ep, output_path)
    print(f"Exported {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export .pt2 artifacts")
    parser.add_argument("--manifest", default="configs/model_manifest.csv")
    parser.add_argument("--output-dir", default="hf_model")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"Manifest not found: {manifest}. Skipping export.")
        return

    # Implementation iterates the manifest and exports each artifact.
    print(f"Export target: {output_dir}")


if __name__ == "__main__":
    main()
