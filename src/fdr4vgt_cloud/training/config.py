"""Shared configuration for training, fine-tuning and evaluation.

No credentials or tokens are stored here. Paths are read from environment
variables or passed explicitly so the code can run on different machines.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Config", "SEED", "PATCH_SIZE", "MODEL_CONFIGS"]


SEED = 42
PATCH_SIZE = 512


# Operational hyperparameters used for the published models.
MODEL_CONFIGS = {
    "1dpwdeeplabv3": {
        "architecture": "CombinedNet4",
        "encoder": "mobilenet_v2",
        "in_channels": 4,
        "classes": 1,
        "base_channels": 32,
        "description": "DeepLabV3 + pixel-wise MLP (operational model)",
    },
    "1dpwunet": {
        "architecture": "CombinedNet2",
        "encoder": "mobilenet_v2",
        "in_channels": 4,
        "classes": 1,
        "base_channels": 32,
        "description": "UNet + pixel-wise MLP",
    },
    "1dpwunetpp": {
        "architecture": "CombinedNet3",
        "encoder": "mobilenet_v2",
        "in_channels": 4,
        "classes": 1,
        "base_channels": 32,
        "description": "UNet++ + pixel-wise MLP",
    },
    "1dpwseg": {
        "architecture": "CombinedNet",
        "encoder": "mobilenet_v2",
        "in_channels": 4,
        "classes": 1,
        "base_channels": 32,
        "description": "Segformer + pixel-wise MLP",
    },
    "deeplabv3": {"architecture": "DeepLabV3", "encoder": "mobilenet_v2"},
    "unet": {"architecture": "UNet", "encoder": "mobilenet_v2"},
    "unetpp": {"architecture": "UNetPlusPlus", "encoder": "mobilenet_v2"},
    "segformer": {"architecture": "Segformer", "encoder": "mobilenet_v2"},
}


@dataclass
class Config:
    """Training configuration object."""

    # Data
    probav_tacos: list[str] | None = None
    spot_tacos: list[str] | None = None
    finetune_tortilla: str | None = None
    splits_csv: str | None = None

    # Training
    model_name: str = "1dpwdeeplabv3"
    batch_size: int = 32
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 100
    scheduler_step: int = 100
    scheduler_gamma: float = 0.1
    precision: str = "16-mixed"

    # Reproducibility
    seed: int = SEED

    # Outputs
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> Config:
        """Build a config from environment variables."""
        probav = os.getenv("FDR_PROBAV_TACOS")
        spot = os.getenv("FDR_SPOT_TACOS")
        finetune = os.getenv("FDR_FINETUNE_TORTILLA")
        return cls(
            probav_tacos=probav.split(",") if probav else None,
            spot_tacos=spot.split(",") if spot else None,
            finetune_tortilla=finetune,
        )

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
