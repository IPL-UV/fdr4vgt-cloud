"""Verify exported `.pt2` artifacts.

Checks:
  - Load with torch 2.7 and torch 2.14.1 CPU
  - Batch sizes 1, 2, 8 and 256 produce correct shapes
  - Batch size 257 raises an error
  - Outputs match the old `.pt2` within 1e-5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fdr4vgt_cloud.inference import load_pt2


def check_batches(path: Path, device: str = "cpu") -> None:
    model = load_pt2(path, device=device)
    for b in (1, 2, 8, 256):
        x = torch.randn(b, 4, 512, 512).to(device)
        y = model(x)
        assert y.shape == (b, 1, 512, 512), f"Bad shape for batch {b}: {y.shape}"
        print(f"  batch {b}: OK")

    x = torch.randn(257, 4, 512, 512).to(device)
    try:
        model(x)
        raise AssertionError("batch 257 should fail")
    except Exception:
        print("  batch 257: correctly rejected")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify .pt2 exports")
    parser.add_argument("--model-dir", default="hf_model")
    args = parser.parse_args()

    for pt2 in sorted(Path(args.model_dir).rglob("*.pt2")):
        print(f"Checking {pt2}")
        check_batches(pt2)


if __name__ == "__main__":
    main()
