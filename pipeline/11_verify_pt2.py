"""Check that the published `.pt2` artifacts load and predict.

This is the check that was missing from the release: the previous script only
listed files. It now loads the model from the published JSON with mlstac, runs
the published example patch, and compares tiled inference against a direct
forward pass.

Usage:

    python pipeline/11_verify_pt2.py
    python pipeline/11_verify_pt2.py --which ensemble
    python pipeline/11_verify_pt2.py --artifact hf_model/single/model.pt2

Exit code is non-zero when a check fails.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

SINGLE = (
    "https://huggingface.co/isp-uv-es/FDR4VGT-CLOUD/resolve/main/"
    "single/multisensor_single_1dpwdeeplabv3.json"
)
ENSEMBLE = (
    "https://huggingface.co/isp-uv-es/FDR4VGT-CLOUD/resolve/main/ensemble/multisensor_ensemble.json"
)


def check_local(path: Path) -> None:
    """Load a local artifact and run a few batch sizes."""
    from fdr4vgt_cloud.inference import load_pt2

    model = load_pt2(path, device="cpu")
    for batch in (1, 2, 8):
        with torch.no_grad():
            out = model(torch.randn(batch, 4, 512, 512))
        if isinstance(out, tuple):
            shapes = [tuple(item.shape) for item in out]
            assert shapes[0] == (batch, 1, 512, 512), shapes
        else:
            assert tuple(out.shape) == (batch, 1, 512, 512), out.shape
    print(f"local {path.name}: batches 1, 2 and 8 OK")


def check_hub(url: str, which: str, workdir: Path) -> None:
    """Reproduce the quick start: mlstac download, load, infer, tile."""
    import mlstac
    from safetensors.numpy import load_file

    loader = mlstac.load(url).download(workdir)
    model = loader.compiled_model(device="cpu")
    patch = load_file(str(workdir / "example_data.safetensor"))["image"]
    assert patch.shape == (4, 512, 512), patch.shape

    with torch.no_grad():
        out = model(torch.from_numpy(patch).unsqueeze(0))
    probs = out[0] if isinstance(out, tuple) else out
    assert tuple(probs.shape) == (1, 1, 512, 512), probs.shape
    assert 0.0 <= float(probs.min()) and float(probs.max()) <= 1.0

    direct = probs.squeeze().numpy()
    tiled = loader.predict_large(
        patch.astype("float32"),
        model=model,
        chunk_size=512,
        overlap=64,
        batch_size=2,
        num_workers=0,
        device="cpu",
    )
    tiled = np.squeeze(np.asarray(tiled[0] if isinstance(tiled, tuple) else tiled))
    inner = (slice(8, -8), slice(8, -8))
    gap = float(np.abs(tiled[inner] - direct[inner]).max())
    assert gap < 1e-4, gap
    print(f"{which}: example patch, probabilities and tiling OK (gap {gap:.1e})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify .pt2 artifacts")
    parser.add_argument("--which", choices=["single", "ensemble", "both"], default="both")
    parser.add_argument("--artifact", default="", help="Local .pt2 instead of the hub")
    parser.add_argument("--workdir", default="/tmp/fdr4vgt-verify")
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.artifact:
        check_local(Path(args.artifact))
        return
    workdir = Path(args.workdir)
    if args.which in ("single", "both"):
        check_hub(SINGLE, "single", workdir / "single")
    if args.which in ("ensemble", "both"):
        check_hub(ENSEMBLE, "ensemble", workdir / "ensemble")


if __name__ == "__main__":
    main()
