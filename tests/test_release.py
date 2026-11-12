"""Checks that do not need the training data or the network."""

import torch

from fdr4vgt_cloud.inference import load_pt2
from fdr4vgt_cloud.model import SigmoidWrap, get_model


def test_model_zoo_builds_every_architecture():
    names = [
        "1dpwdeeplabv3",
        "1dpwseg",
        "1dpwunet",
        "1dpwunetpp",
        "deeplabv3",
        "unet",
        "unetpp",
        "segformer",
    ]
    for name in names:
        assert isinstance(get_model(name), torch.nn.Module), name


def test_pt2_roundtrip(tmp_path):
    """The published loader must read back a fresh export and predict."""
    model = SigmoidWrap(get_model("1dpwdeeplabv3")).eval()
    # A batch of 2 in the example keeps the batch dimension dynamic; with 1
    # torch.export folds it into a constant.
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        model,
        (torch.randn(2, 4, 128, 128),),
        dynamic_shapes={"x": {0: batch}},
        strict=False,
    )
    path = tmp_path / "roundtrip.pt2"
    torch.export.save(program, path)

    loaded = load_pt2(path)
    with torch.no_grad():
        out = loaded(torch.randn(2, 4, 128, 128))

    assert tuple(out.shape) == (2, 1, 128, 128)
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0
