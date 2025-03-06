import os
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pytorch_lightning import LightningModule
from pytorch_toolbelt.losses import MULTICLASS_MODE, JaccardLoss
from pytorch_toolbelt.zoo import resnet34_unet64_s4
from torchmetrics.classification.jaccard import JaccardIndex
from torchvision.utils import draw_segmentation_masks

import config
from utilities.augmentations import inverse_normalize
from utilities.helpers import (
    STAGE_TEST,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    add_stage_metrics,
    compute_stage_metric,
    get_stage_metric,
    linear_warmup_cosine_decay,
    print_if_rank_zero,
    tensor_pad,
    tensor_unpad,
)
from utilities.io import resolve_checkpoint_path, save_image

from .reconstruction import Reconstruction


def overwrite_state_dict(source: LightningModule, target: LightningModule):
    """
    Overwrites the target module state with the source module. Mismatched parameters are reported
    and removed from the copy.
    """

    # Get the state dictionaries of each model
    source_state = source.state_dict()
    target_state = target.state_dict()

    # Discover mismatched state (layer name or shape conflict).
    # We expect that the last layer will likely mismatch when loading the reconstruction
    # checkpoint onto resunet, as reconstruction outputs color.
    mismatch = []
    for k, v in source_state.items():
        if k not in target_state or target_state[k].shape != v.shape:
            mismatch.append(k)

    # If there were any mismatched keys...
    if len(mismatch) > 0:

        # Remove mismatched keys
        for k in mismatch:
            source_state.pop(k)

        # Report to user
        print_if_rank_zero(
            f"Found {len(mismatch)} missing or mismatched parameters. "
            "The model will still have original values for these parameters."
        )
        print_if_rank_zero(f" -> {mismatch}")

    print_if_rank_zero("Overwriting target model state!")
    target.load_state_dict(source_state, strict=False)


METRIC_IOU_PER_CLASS = "iou_per_class"
METRIC_IOU_AVERAGE = "iou_average"


class ResUNet34(LightningModule):
    checkpoint_monitor = "val/iou_avg"
    checkpoint_mode = "max"

    identifier = "resunet34"

    def __init__(
            self, pretrain: Union[str, bool], num_classes: int, ignore_class: Optional[int] = None
    ):
        super().__init__()

        # Save model arguments
        self.save_hyperparameters("pretrain", "num_classes", "ignore_class")

        self.ignore_class = ignore_class
        self.num_classes = num_classes
        self.pretrain = pretrain

        # Determine what pre-training initialization to use
        # - No pre-training (random initialization)
        # - Library pre-trained (encoder imagenet checkpoint)
        # - Custom pre-trained (path to checkpoint)

        if self.pretrain is False or self.pretrain is None:

            # We do not want the library provided checkpoint.
            self.pretrain = False

        elif self.pretrain is True or self.pretrain == "imagenet":

            # We do want to the library provided checkpoint.
            self.pretrain = True

        else:

            # We want to load the user provided checkpoint.
            resolved_checkpoint = resolve_checkpoint_path(self.pretrain)
            if resolved_checkpoint is None:
                raise Exception(f"Unable to resolve checkpoint path for '{self.pretrain}'.")
            self.pretrain = resolved_checkpoint

        # Instantiate the underlying model
        self.underlying_model = resnet34_unet64_s4(
            input_channels=3,
            num_classes=self.num_classes,
            dropout=0.2,
            pretrained=(self.pretrain is True),
        )

        # (Optional) Overwrite with user checkpoint
        if type(self.pretrain) == str:

            # Load checkpoint
            checkpoint = Reconstruction.load_from_checkpoint(self.pretrain)

            # ...
            if hasattr(checkpoint, "underlying_model"):

                # The pre-training edition of this model
                underlying_model = checkpoint.underlying_model

                if type(underlying_model) == type(self):
                    # Replace the matching state from the underlying model onto ourself
                    overwrite_state_dict(underlying_model, self)
                else:
                    raise Exception("Unable to copy checkpoint state, model type mismatch.")

            else:
                raise Exception("Unable to load pre-training checkpoint, no 'underlying_model'.")

        # Get class sequence, skipping the ignore class (if provided)
        class_sequence = list(range(0, self.num_classes))
        if self.ignore_class is not None:
            class_sequence = [c for c in class_sequence if c != self.ignore_class]

        # Create Jaccard Loss
        self.criterion = JaccardLoss(MULTICLASS_MODE, np.array(class_sequence), smooth=1e-3)

        # Create logging metrics
        self.iou_per_class = {}
        self.iou_overall = {}

        # Add the overall average iou metrics
        iou_average = JaccardIndex(self.num_classes, average="macro", ignore_index=ignore_class)
        add_stage_metrics(self, METRIC_IOU_AVERAGE, iou_average)

        # Add the per-class iou metrics
        iou_per_class = JaccardIndex(self.num_classes, average=None, ignore_index=ignore_class)
        add_stage_metrics(self, METRIC_IOU_PER_CLASS, iou_per_class)

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
        x, pad = tensor_pad(x, 64)
        return tensor_unpad(self.underlying_model(x), pad)

    # PER STEP

    def training_step(self, batch, batchIndex):
        return self._common_step(STAGE_TRAIN, batch, batchIndex)

    def validation_step(self, batch, batchIndex):
        return self._common_step(STAGE_VALIDATION, batch, batchIndex)

    def test_step(self, batch, batchIndex):
        return self._common_step(STAGE_TEST, batch, batchIndex)

    # PER EPOCH

    def training_epoch_end(self, _):
        self._common_epoch_end(STAGE_TRAIN)

    def validation_epoch_end(self, _):
        self._common_epoch_end(STAGE_VALIDATION)

    def test_epoch_end(self, _):
        self._common_epoch_end(STAGE_TEST)

    def _common_step(self, stage: str, batch, batchIndex: int):
        x, y = batch

        # Forward
        z = self.forward(x)

        # Output in-progress and testing images
        if self.trainer.is_global_zero and (
                (stage is STAGE_TEST) or (batchIndex == 0 and self.trainer.current_epoch % 10 == 0)
        ):
            self._output_images(stage, batchIndex, x, y, z)

        # Compute loss
        loss = self.criterion(z, y)

        # Update metrics
        get_stage_metric(self, stage, METRIC_IOU_PER_CLASS)(z, y)
        get_stage_metric(self, stage, METRIC_IOU_AVERAGE)(z, y)

        # Record loss
        self._common_log_metrics(stage, {"loss": loss})

        # Return loss for optimization
        return loss

    def _common_epoch_end(self, stage: str) -> Dict[str, Union[torchmetrics.Metric, torch.Tensor]]:
        metrics = {}

        # Record per-class metrics
        iou_class_values = compute_stage_metric(self, stage, METRIC_IOU_PER_CLASS)
        for i, v in enumerate(iou_class_values):
            metrics[f"iou_class_{i}"] = v

        # Record per-class std.dev
        metrics["iou_std_dev"] = torch.std(iou_class_values)

        # Record average iou
        iou_average = compute_stage_metric(self, stage, METRIC_IOU_AVERAGE)
        metrics["iou_avg"] = iou_average

        # Record metrics
        self._common_log_metrics(stage, metrics)

    def _common_log_metrics(self, stage: str, metrics: dict):
        """Actually submits metrics to pytorch lightning."""

        # Record metrics
        for k, v in metrics.items():
            self.log(f"{stage}/{k}", v, sync_dist=True)

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

        def convert_one_hot(tensor: torch.Tensor):
            tensor = nn.functional.one_hot(tensor, self.num_classes)
            tensor = tensor.transpose_(3, 1)  # BxWxHxC -> BxCxWxH
            tensor = tensor.transpose_(2, 3)  # BxCxWxH -> BxCxHxW
            return tensor.type(torch.bool)

        def convert_numpy(x):
            return np.array(x).transpose(1, 2, 0)

        # Push tensors to CPU, converting into a form useful for generating images
        x = inverse_normalize(x.detach().cpu(), **config.AUGMENTATION_ARGS["normalization"])
        y = convert_one_hot(y.detach().cpu())
        z = convert_one_hot(torch.argmax(z.detach(), dim=1).cpu())

        # Convert input input image to uint8 (0 - 255)
        x = (x * 255).type(torch.uint8)

        # Get the relevant datamodule
        datamodule = self.trainer.datamodule

        # Get segmentation palette
        colors = None  # torchvision automatic palette
        if hasattr(datamodule, "get_class_color"):
            colors = [(*datamodule.get_class_color(c),) for c in range(self.num_classes)]

        # For the first 16 images in the batch
        for i, xi, yi, zi in zip(range(0, 16), x, y, z):
            prefix = f"b{batchIndex:03d}+{i:03d}_e{self.trainer.current_epoch:03d}"

            # Generate images
            img = convert_numpy(xi)
            lbl = convert_numpy(draw_segmentation_masks(xi, yi, colors=colors))
            out = convert_numpy(draw_segmentation_masks(xi, zi, colors=colors))

            # Save images
            save_image(os.path.join(directory, f"{prefix}_img.jpg"), img)
            save_image(os.path.join(directory, f"{prefix}_lbl.jpg"), lbl)
            save_image(os.path.join(directory, f"{prefix}_out.jpg"), out)
