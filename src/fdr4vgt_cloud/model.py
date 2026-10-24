"""Neural network definitions for FDR4VGT-CLOUD.

All models use a 4-band input (Blue, Red, NIR, SWIR) and output a single
cloud probability logit. The operational model is ``CombinedNet4``: a
DeepLabV3 encoder (MobileNetV2, no pretrained weights) plus a pixel-wise
1x1 MLP and a final 1x1 fusion. For export, logits are wrapped with a
sigmoid (``SigmoidWrap``) so the serialized artifact emits probabilities.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.nn import functional

__all__ = [
    "DeepLabV3Branch",
    "PixelWiseNet",
    "CombinedNet4",
    "SigmoidWrap",
    "get_model",
]


class DeepLabV3Branch(nn.Module):
    """DeepLabV3 segmentation branch with MobileNetV2 encoder."""

    def __init__(self, in_channels: int = 4, classes: int = 1) -> None:
        super().__init__()
        self.deeplabv3 = smp.DeepLabV3(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deeplabv3(x)


class PixelWiseNet(nn.Module):
    """1x1 convolutional MLP that learns a per-pixel cloud score."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(base_channels)
        self.conv3 = nn.Conv2d(base_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = functional.relu(self.bn1(self.conv1(x)))
        x = functional.relu(self.bn2(self.conv2(x)))
        return self.conv3(x)


class CombinedNet4(nn.Module):
    """Operational architecture: DeepLabV3 + pixel-wise MLP + fusion.

    The paper uses the shorthand ``PW-DL3+``. The encoder is DeepLabV3
    (not DeepLabV3+) from ``segmentation-models-pytorch``.
    """

    def __init__(
        self,
        in_channels: int = 4,
        classes: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.seg_branch = DeepLabV3Branch(in_channels=in_channels, classes=classes)
        self.pixel_branch = PixelWiseNet(
            in_channels=in_channels,
            out_channels=classes,
            base_channels=base_channels,
        )
        self.fusion_conv = nn.Conv2d(classes, classes, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seg_out = self.seg_branch(x)
        pixel_out = self.pixel_branch(x)
        fused = seg_out + pixel_out
        return self.fusion_conv(fused)


class SigmoidWrap(nn.Module):
    """Wrap a logit model so the exported artifact emits probabilities."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def get_model(name: str = "1dpwdeeplabv3", **kwargs) -> nn.Module:
    """Factory for the model zoo used in the ablation study.

    Supported names:
      - ``1dpwdeeplabv3`` (operational)
      - ``1dpwunet``, ``1dpwunetpp``, ``1dpwseg``
      - ``deeplabv3``, ``unet``, ``unetpp``, ``segformer``
    """
    name = name.lower()
    classes = kwargs.get("classes", 1)
    in_channels = kwargs.get("in_channels", 4)

    single = {
        "deeplabv3": DeepLabV3Branch,
        "unet": lambda **kw: smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=kw.get("in_channels", 4),
            classes=kw.get("classes", 1),
        ),
        "unetpp": lambda **kw: smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=kw.get("in_channels", 4),
            classes=kw.get("classes", 1),
        ),
        "segformer": lambda **kw: smp.Segformer(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=kw.get("in_channels", 4),
            classes=kw.get("classes", 1),
        ),
    }

    if name == "1dpwdeeplabv3":
        return CombinedNet4(in_channels=in_channels, classes=classes)
    if name in single:
        return single[name](in_channels=in_channels, classes=classes)
    raise ValueError(f"Unknown model name: {name}")
