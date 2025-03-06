import os
from typing import Any

import numpy as np
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from pytorch_msssim import MS_SSIM as MultiScaleSSIM

import config
from utilities.augmentations import inverse_normalize
from utilities.helpers import (  # Stage identifiers; Scheduler
    STAGE_TEST,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    linear_warmup_cosine_decay,
)
from utilities.io import save_image

METRIC_MSSSIM = "ssim"
METRIC_L1LOSS = "L1"


class Reconstruction(LightningModule):
    checkpoint_monitor = "val/loss"
    checkpoint_mode = "min"

    identifier = "reconstruction"

    def __init__(self, underlying_model: str, underlying_model_args: dict):
        super().__init__()

        # Save model arguments
        self.save_hyperparameters("underlying_model", "underlying_model_args")

        # Imported here, because it is incomplete in the global scope due to circular dependency
        from . import get_model

        # Instantiate underlying model
        self.underlying_model = get_model(underlying_model, underlying_model_args)

        # Metrics
        self.criterion_ssim = MultiScaleSSIM(data_range=1.0, size_average=True, channel=3)
        self.criterion_loss = nn.L1Loss()

    @staticmethod
    def convert_numpy(x):
        x = np.array(x).transpose(1, 2, 0)
        return (x * 255).astype(np.uint8)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.underlying_model.parameters(), **config.OPTIMIZER_ARGS)
        scheduler = linear_warmup_cosine_decay(optimizer, config.EPOCHS, **config.SCHEDULER_ARGS)
        return [optimizer], [
            {
                "name": "learning_rate",
                "scheduler": scheduler,
                "monitor": self.checkpoint_monitor,
                "interval": "epoch",
            }
        ]

    def forward(self, x):
        return self.underlying_model(x)

    # PER STEP

    def training_step(self, batch, batchIndex):
        return self._common_step(STAGE_TRAIN, batch, batchIndex)

    def validation_step(self, batch, batchIndex):
        return self._common_step(STAGE_VALIDATION, batch, batchIndex)

    def test_step(self, batch, batchIndex):
        return self._common_step(STAGE_TEST, batch, batchIndex)

    # COMMON

    def _common_step(self, stage: str, batch: Any, batchIndex: int):
        x, y = batch

        # Compute RGB prediction
        z = self.underlying_model(x)

        # Output in-progress images for the first batch, every 10 epochs, from the main process
        if self.trainer.is_global_zero and (
                (self.trainer.current_epoch % 10 == 0 and batchIndex == 0) or (stage == STAGE_TEST)
        ):
            self._output_images(stage, batchIndex, x, y, z)

        # Compute ssim and loss
        ssim = self.criterion_ssim(z, y)
        loss = self.criterion_loss(z, y)

        # Compute combo loss (equal weighting)
        combo_loss = loss + (1.0 - ssim)

        # Log measures
        self.log(f"{stage}/ssim", ssim, sync_dist=True)
        self.log(f"{stage}/L1", loss, sync_dist=True)
        self.log(f"{stage}/loss", combo_loss, sync_dist=True)

        # Return combo loss for optimization
        return combo_loss

    def _output_images(self, stage: str, batchIndex: int, x, y, z):

        if self.trainer.sanity_checking:
            stage = "sanity"

        # Get relevant output directory
        directory = os.path.join(
            config.CHECKPOINT_DIRECTORY,
            config.SESSION_NAME,
            config.SESSION_VERSION,
            "images",
            stage,
        )

        # Ensure directory is available
        os.makedirs(directory, exist_ok=True)

        # Push tensors to CPU, undoing the normalization so we can see them in jpg form
        x = inverse_normalize(x.detach().cpu(), **config.AUGMENTATION_ARGS["normalization"])
        y = inverse_normalize(y.detach().cpu(), **config.AUGMENTATION_ARGS["normalization"])
        z = inverse_normalize(z.detach().cpu(), **config.AUGMENTATION_ARGS["normalization"])

        # For the first 16 images in the batch
        for i, xi, yi, zi in zip(range(0, 16), x, y, z):
            prefix = f"b{batchIndex:03d}+{i:03d}_e{self.trainer.current_epoch:03d}"

            save_image(os.path.join(directory, f"{prefix}_lbl.jpg"), self.convert_numpy(yi))
            save_image(os.path.join(directory, f"{prefix}_img.jpg"), self.convert_numpy(xi))
            save_image(os.path.join(directory, f"{prefix}_out.jpg"), self.convert_numpy(zi))
