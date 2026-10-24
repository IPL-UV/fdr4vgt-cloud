"""Weak pretraining script for the global cloud mask.

Example:
    python -m fdr4vgt_cloud.training.pretrain \
        --model 1dpwdeeplabv3 \
        --batch-size 32 \
        --max-epochs 100 \
        --probav-tacos "probav/*.tortilla" \
        --spot-tacos "spot/*.tortilla"
"""

from __future__ import annotations

import datetime

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from ..model import get_model
from .config import Config
from .data import CloudDataModule

app = typer.Typer(help="Weak pretraining for FDR4VGT-CLOUD")


class LitModel(pl.LightningModule):
    """Lightning wrapper for the segmentation models."""

    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.get("scheduler_step", 100),
            gamma=self.hparams.get("scheduler_gamma", 0.1),
        )
        return [optimizer], [scheduler]


@app.command()
def main(
    model: str = typer.Option("1dpwdeeplabv3", help="Model name from MODEL_CONFIGS"),
    probav_tacos: str | None = typer.Option(None, help="Comma-separated TACO files for Proba-V"),
    spot_tacos: str | None = typer.Option(None, help="Comma-separated TACO files for SPOT"),
    batch_size: int = typer.Option(32, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate"),
    max_epochs: int = typer.Option(100, help="Maximum epochs"),
    patience: int = typer.Option(100, help="Early stopping patience"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint output directory"),
    gpus: int | None = typer.Option(None, help="GPU device index; None = CPU"),
    seed: int = typer.Option(42, help="Random seed"),
    offline: bool = typer.Option(False, help="Disable external loggers if True"),
) -> None:
    """Run the weak pretraining stage."""
    pl.seed_everything(seed)
    cfg = Config(
        model_name=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        probav_tacos=probav_tacos.split(",") if probav_tacos else None,
        spot_tacos=spot_tacos.split(",") if spot_tacos else None,
    )
    cfg.ensure_dirs()

    data_module = CloudDataModule(
        probav_files=cfg.probav_tacos,
        spot_files=cfg.spot_tacos,
        batch_size=cfg.batch_size,
        random_state=cfg.seed,
    )

    dl_model = get_model(cfg.model_name, classes=1, in_channels=4)
    lit_model = LitModel(
        model=dl_model,
        learning_rate=cfg.learning_rate,
    )
    lit_model.hparams.scheduler_step = cfg.scheduler_step
    lit_model.hparams.scheduler_gamma = cfg.scheduler_gamma

    uuid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.checkpoint_dir,
        filename=f"{model}__{uuid}" + "__{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        verbose=True,
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    trainer_kwargs: dict = {
        "max_epochs": cfg.max_epochs,
        "callbacks": callbacks,
        "logger": not offline,
    }
    if gpus is not None:
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = [gpus]
    else:
        trainer_kwargs["accelerator"] = "cpu"

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(lit_model, data_module)
    trainer.test(lit_model, data_module, ckpt_path="best")


if __name__ == "__main__":
    app()
