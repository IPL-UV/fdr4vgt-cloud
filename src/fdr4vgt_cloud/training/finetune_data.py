"""Data loading for the fine-tuning stage on human-annotated scenes.

The 78 annotated scenes are split into 66 train / 12 val with
``train_test_split(test_size=0.15, random_state=42)``. The 8 held-out test
scenes and 7 unused scenes are loaded separately for evaluation.
"""

from __future__ import annotations

import einops
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rasterio as rio
import torch
from sklearn.model_selection import train_test_split

__all__ = ["FinetuneDataModule", "load_image_chunks"]


def load_image_chunks(
    dataframe: pd.DataFrame,
    patch_size: int = 512,
    image_col: str = "rgb",
    mask_col: str = "mask",
) -> tuple[np.ndarray, np.ndarray]:
    """Load all annotated scenes and tile them into 512x512 chunks."""
    x_container, y_container = [], []
    for _, row in dataframe.iterrows():
        with rio.open(row[image_col]) as src, rio.open(row[mask_col]) as src2:
            x = src.read().astype(np.float32)
            xmask = (x.mean(axis=0) != 0) * 1
            y = (src2.read() * 1.0 - 1) * xmask
            x = x * xmask

        h, w = x.shape[1], x.shape[2]
        h_pad = (h // patch_size + 1) * patch_size - h
        w_pad = (w // patch_size + 1) * patch_size - w

        x_new = np.pad(x, ((0, 0), (0, h_pad), (0, 0)), mode="edge")
        x_new = np.pad(x_new, ((0, 0), (0, 0), (0, w_pad)), mode="edge")
        y_new = np.pad(y, ((0, 0), (0, h_pad), (0, 0)), mode="edge")
        y_new = np.pad(y_new, ((0, 0), (0, 0), (0, w_pad)), mode="edge")

        x_chunks = einops.rearrange(
            x_new,
            "c (h1 h2) (w1 w2) -> (h1 w1) c h2 w2",
            h2=patch_size,
            w2=patch_size,
        )
        y_chunks = einops.rearrange(
            y_new,
            "c (h1 h2) (w1 w2) -> (h1 w1) c h2 w2",
            h2=patch_size,
            w2=patch_size,
        )
        x_container.append(x_chunks)
        y_container.append(y_chunks)

    return np.concatenate(x_container, axis=0), np.concatenate(y_container, axis=0)


class _ArrayDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()


class FinetuneDataModule(pl.LightningDataModule):
    """Lightning data module for the human-annotated fine-tuning split."""

    def __init__(
        self,
        splits_csv: str,
        patch_size: int = 512,
        batch_size: int = 16,
        val_size: float = 0.15,
        random_state: int = 42,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.splits_csv = splits_csv
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.random_state = random_state
        self.num_workers = num_workers

    def setup(self, stage: str | None = None) -> None:
        df = pd.read_csv(self.splits_csv)
        train_val = df[df["role"] == "finetune_train"].copy()
        # Add the validation scenes from the E2 clean split if they are not already train.
        clean_val = df[df["split_clean"] == "val"].copy()
        train_ids = set(train_val["scene_id"])
        extra_val = clean_val[~clean_val["scene_id"].isin(train_ids)].copy()

        train_df = train_val
        if len(extra_val):
            val_df = pd.concat([df[df["role"] == "finetune_val"], extra_val], ignore_index=True)
        else:
            val_df = df[df["role"] == "finetune_val"].copy()

        # Fallback: if the CSV does not contain explicit val labels, split train.
        if val_df.empty:
            train_df, val_df = train_test_split(
                train_df,
                test_size=self.val_size,
                random_state=self.random_state,
            )

        if stage == "fit" or stage is None:
            x_train, y_train = load_image_chunks(train_df, self.patch_size)
            x_val, y_val = load_image_chunks(val_df, self.patch_size)
            self.train_dataset = _ArrayDataset(x_train, y_train)
            self.val_dataset = _ArrayDataset(x_val, y_val)

        if stage == "test" or stage is None:
            test_df = df[df["role"] == "test_pixel"].copy()
            x_test, y_test = load_image_chunks(test_df, self.patch_size)
            self.test_dataset = _ArrayDataset(x_test, y_test)

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
