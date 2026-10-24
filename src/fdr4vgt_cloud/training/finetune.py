"""Fine-tuning script starting from a pretrained checkpoint.

Example:
    python -m fdr4vgt_cloud.training.finetune \
        --model 1dpwdeeplabv3 \
        --pretrained-ckpt checkpoints/1dpwdeeplabv3__...ckpt \
        --splits-csv splits/scenes.csv \
        --batch-size 16 \
        --max-epochs 30
"""

from __future__ import annotations

import datetime

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from ..model import get_model
from .config import Config
from .finetune_data import FinetuneDataModule
from .pretrain import LitModel

app = typer.Typer(help="Fine-tune FDR4VGT-CLOUD on human annotations")


@app.command()
def main(
    model: str = typer.Option("1dpwdeeplabv3", help="Model name"),
    pretrained_ckpt: str | None = typer.Option(None, help="Path to pretrained checkpoint"),
    splits_csv: str = typer.Option("splits/scenes.csv", help="CSV with split:role column"),
    batch_size: int = typer.Option(16, help="Batch size"),
    learning_rate: float = typer.Option(1e-3, help="Adam learning rate"),
    max_epochs: int = typer.Option(30, help="Maximum epochs"),
    patience: int = typer.Option(10, help="Early stopping patience"),
    checkpoint_dir: str = typer.Option("checkpoints", help="Checkpoint output directory"),
    gpus: int | None = typer.Option(None, help="GPU device index; None = CPU"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:
    """Fine-tune a model on the 78 annotated scenes."""
    pl.seed_everything(seed)
    cfg = Config(
        model_name=model,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
        splits_csv=splits_csv,
    )
    cfg.ensure_dirs()

    data_module = FinetuneDataModule(
        splits_csv=cfg.splits_csv,
        batch_size=cfg.batch_size,
        random_state=cfg.seed,
    )

    dl_model = get_model(cfg.model_name, classes=1, in_channels=4)
    if pretrained_ckpt:
        ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        # Remove lightning "model." prefix if present.
        state = {k.removeprefix("model."): v for k, v in state.items()}
        missing, unexpected = dl_model.load_state_dict(state, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

    lit_model = LitModel(dl_model, learning_rate=cfg.learning_rate)
    lit_model.hparams.scheduler_step = 100
    lit_model.hparams.scheduler_gamma = 0.1

    uuid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=cfg.checkpoint_dir,
        filename=f"ft_{model}__{uuid}" + "__{epoch:03d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        verbose=True,
        mode="min",
    )
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer_kwargs: dict = {
        "max_epochs": cfg.max_epochs,
        "callbacks": callbacks,
        "logger": False,
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
