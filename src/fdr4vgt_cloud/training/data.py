"""Data loading for the weak pretraining stage.

Expects TACO ``.tortilla`` files containing Proba-V and SPOT TOA images and
cloud masks. The split is recalculated from the combined index with
``train_test_split(..., random_state=42)`` to match the published procedure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import tacoreader
import torch
from sklearn.model_selection import train_test_split

__all__ = ["CloudDataModule", "CloudDataset"]


def _prepare_sample(
    input_path: str | Path,
    cloudmask1: str | Path,
    cloudmask2: str | Path,
    patch_size: int = 512,
    fixed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one training patch from image and mask files."""
    with rio.open(input_path) as src:
        width, height = src.width, src.height
        if fixed:
            centroid = (
                (src.bounds.left + src.bounds.right) / 2,
                (src.bounds.top + src.bounds.bottom) / 2,
            )
            x = int(centroid[0] - patch_size / 2)
            x = max(0, min(x, width - patch_size))
            y = int(centroid[1] - patch_size / 2)
            y = max(0, min(y, height - patch_size))
        else:
            x = np.random.randint(0, max(width - patch_size, 1) + 1)
            y = np.random.randint(0, max(height - patch_size, 1) + 1)

        window = rio.windows.Window.from_slices(
            (y, min(y + patch_size, height)),
            (x, min(x + patch_size, width)),
        )
        data = src.read(window=window)

    with rio.open(cloudmask1) as src:
        target1 = src.read(window=window) * 1.0
        invalid_mask = ~(target1 == 0) * 1.0

    with rio.open(cloudmask2) as src:
        final_mask = ((src.read(window=window) - 1) > 0.5) * 1

    final_mask = invalid_mask * final_mask
    input_data = (invalid_mask * data).astype(np.float32)

    input_data = torch.from_numpy(input_data).float()
    final_mask = torch.from_numpy(final_mask).float()

    # Resize to patch_size if the tile was smaller than the requested patch.
    if input_data.shape[1] != patch_size or input_data.shape[2] != patch_size:
        input_data = torch.nn.functional.interpolate(
            input_data.unsqueeze(0),
            size=(patch_size, patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        final_mask = torch.nn.functional.interpolate(
            final_mask.unsqueeze(0),
            size=(patch_size, patch_size),
            mode="nearest",
        ).squeeze(0)

    return input_data, final_mask


class CloudDataset(torch.utils.data.Dataset):
    """Dataset that samples patches from a TACO dataframe."""

    def __init__(
        self,
        dataframe: tacoreader.TortillaDataFrame,
        patch_size: int = 512,
        fixed: bool = False,
    ) -> None:
        self.dataframe = dataframe
        self.patch_size = patch_size
        self.fixed = fixed

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = self.dataframe.read(idx)
        image = dataset.read(0)
        cloudmask1 = dataset.read(1)
        cloudmask2 = dataset.read(2)
        return _prepare_sample(
            input_path=image,
            cloudmask1=cloudmask1,
            cloudmask2=cloudmask2,
            patch_size=self.patch_size,
            fixed=self.fixed,
        )


class CloudDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for the weak pretraining split."""

    def __init__(
        self,
        probav_files: list[str] | None = None,
        spot_files: list[str] | None = None,
        batch_size: int = 16,
        patch_size: int = 512,
        test_size: float = 0.1,
        random_state: int = 42,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.probav_files = probav_files or []
        self.spot_files = spot_files or []
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.test_size = test_size
        self.random_state = random_state
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        probav_db = tacoreader.load(self.probav_files) if self.probav_files else None
        spot_db = tacoreader.load(self.spot_files) if self.spot_files else None

        frames = [f for f in (probav_db, spot_db) if f is not None]
        if not frames:
            raise ValueError("At least one TACO file must be provided")

        total_db = pd.concat(frames, axis=0)
        train_idx, val_idx = train_test_split(
            total_db.index,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        train_idx, test_idx = train_test_split(
            train_idx,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        self.train_df = tacoreader.TortillaDataFrame(
            total_db.iloc[train_idx].reset_index(drop=True)
        )
        self.val_df = tacoreader.TortillaDataFrame(total_db.iloc[val_idx].reset_index(drop=True))
        self.test_df = tacoreader.TortillaDataFrame(total_db.iloc[test_idx].reset_index(drop=True))

        if stage == "fit" or stage is None:
            self.train_dataset = CloudDataset(self.train_df, self.patch_size)
            self.val_dataset = CloudDataset(self.val_df, self.patch_size, fixed=True)
        if stage == "test" or stage is None:
            self.test_dataset = CloudDataset(self.test_df, self.patch_size, fixed=True)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def build_split_indices(
    probav_files: list[str] | None,
    spot_files: list[str] | None,
    test_size: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a table with the pretraining split assignment per scene.

    Useful for reporting ``qa:weak_db_split`` in the released dataset.
    """
    probav_db = tacoreader.load(probav_files) if probav_files else None
    spot_db = tacoreader.load(spot_files) if spot_files else None
    frames = [f for f in (probav_db, spot_db) if f is not None]
    total_db = pd.concat(frames, axis=0)

    train_idx, val_idx = train_test_split(
        total_db.index,
        test_size=test_size,
        random_state=random_state,
    )
    train_idx, test_idx = train_test_split(
        train_idx,
        test_size=test_size,
        random_state=random_state,
    )

    split = pd.Series(index=total_db.index, dtype="object")
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split.to_frame(name="split")
