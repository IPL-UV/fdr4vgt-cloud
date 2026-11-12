"""Export the `.pt2` artifacts that are published on Hugging Face.

The artifacts are built from the training checkpoints with the same code that
was used for the paper, so the published numbers can be reproduced. Every
export wraps the logits with a sigmoid and declares a dynamic batch dimension
from 1 to 256.

Usage:

    python pipeline/10_export_pt2.py --manifest configs/export_manifest.yaml

Paths in the manifest may use environment variables, e.g. ``${TRAINPROBAV}``.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import yaml
from torch.export import Dim

from fdr4vgt_cloud.model import SigmoidWrap, get_model

BATCH = Dim("batch", min=1, max=256)
EXAMPLE = (2, 4, 512, 512)


class EnsembleProbStd(torch.nn.Module):
    """Ensemble that returns the probability of one member and its spread.

    The probability comes from the first member, which is the operational
    model. The uncertainty is the standard deviation across all members,
    normalised by the maximum possible value so that it stays in 0 to 1.
    """

    def __init__(self, models: list[torch.nn.Module], best: int = 0) -> None:
        super().__init__()
        self.models = torch.nn.ModuleList(
            [torch.nn.Sequential(model, torch.nn.Sigmoid()) for model in models]
        )
        self.best = best
        count = len(models)
        self.std_max = math.sqrt(0.25 * count / (count - 1)) if count > 1 else 1.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([model(x) for model in self.models], dim=0).squeeze(2)
        prob = stacked[self.best]
        if len(self.models) > 1:
            spread = torch.clamp(stacked.std(dim=0) / self.std_max, 0.0, 1.0)
        else:
            spread = torch.zeros_like(prob)
        return prob.unsqueeze(1), spread.unsqueeze(1)


def load_checkpoint(path: Path, name: str) -> torch.nn.Module:
    """Build one architecture and load its Lightning checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    state = {key.removeprefix("model."): value for key, value in state.items()}
    model = get_model(name)
    model.load_state_dict(state, strict=True)
    return model.eval()


def export(model: torch.nn.Module, output: Path) -> None:
    """Export one inference graph with a dynamic batch dimension."""
    output.parent.mkdir(parents=True, exist_ok=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    program = torch.export.export(
        model,
        (torch.randn(*EXAMPLE),),
        dynamic_shapes={"x": {0: BATCH}},
        strict=False,
    )
    torch.export.save(program, output)
    print(f"exported {output} ({output.stat().st_size / 1e6:.1f} MB)")


def build(item: dict) -> torch.nn.Module:
    """Turn one manifest entry into the module that gets exported."""
    checkpoints = [Path(os.path.expandvars(p)) for p in item["checkpoints"]]
    names = item["models"]
    if item["kind"] == "single":
        return SigmoidWrap(load_checkpoint(checkpoints[0], names[0]))
    models = [load_checkpoint(path, name) for path, name in zip(checkpoints, names, strict=True)]
    return EnsembleProbStd(models, best=item.get("best", 0)).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export .pt2 artifacts")
    parser.add_argument("--manifest", default="configs/export_manifest.yaml")
    parser.add_argument("--only", default="", help="Export a single artifact by name")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text())
    for item in manifest["artifacts"]:
        if args.only and item["name"] != args.only:
            continue
        print(f"building {item['name']}")
        export(build(item), Path(item["output"]))


if __name__ == "__main__":
    main()
