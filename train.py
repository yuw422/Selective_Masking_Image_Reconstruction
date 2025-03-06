import logging
import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torchinfo import summary

import config
from datasets import get_datamodule
from models import get_model
from utilities.helpers import print_if_rank_zero
from utilities.io import resolve_checkpoint_path, save_yaml

# Reduce verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def main():
    # Set the random seed on everything, to make things more replicable
    seed_everything(config.RANDOM_SEED, workers=True)

    # Prepare Data Module
    datamodule = get_datamodule(config.DATASET_NAME, config.DATASET_ARGS)

    # Prepare Model Module
    model = get_model(config.MODEL_NAME, config.MODEL_ARGS)
    summary(model, (2, 3, 256, 256), col_names=("input_size", "output_size", "num_params"), verbose=1,
            device=f"cuda:{config.GPU}" if torch.cuda.is_available() else "cpu")


    # Attempt to resolve "last checkpoint"
    checkpoint_path = resolve_checkpoint_path(
        os.path.join(config.SESSION_NAME, config.SESSION_VERSION, "checkpoints", "last.ckpt")
    )

    if checkpoint_path:
        print_if_rank_zero(f"Resuming Checkpoint: {checkpoint_path}")
        # TODO: Should probably load the config from the checkpoint dir too

    # Construct the PL trainer for training
    trainer = Trainer(
        max_epochs=config.EPOCHS,
        # Do not print model summary
        enable_model_summary=False,
        # GPU Selection
        auto_select_gpus=True,
        devices=[config.GPU],
        accelerator="gpu",
        benchmark=True,
        # Multi-GPU Comminucation
        strategy=DDPStrategy(find_unused_parameters=False),
        sync_batchnorm=True,
        # Logging and checkpoints
        callbacks=[
            LearningRateMonitor(),
            ModelCheckpoint(
                monitor=model.checkpoint_monitor,
                mode=model.checkpoint_mode,
                save_last=True,  # checkpoint resume
                save_top_k=1,  # testing
            ),
        ],
        log_every_n_steps=5,
        logger=TensorBoardLogger(
            save_dir=config.CHECKPOINT_DIRECTORY,
            default_hp_metric=False,
            version=config.SESSION_VERSION,
            name=config.SESSION_NAME,
        ),
    )

    # Save config to disk
    save_yaml(
        os.path.join(
            config.CHECKPOINT_DIRECTORY,
            config.SESSION_NAME,
            config.SESSION_VERSION,
            "configuration.yml",
        ),
        config.CONFIG_DATA,
    )

    # Actually train the model
    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    print_if_rank_zero("TRAINING COMPLETE")

    # Halt the use of multi-gpu, we will restart with only one for testing
    # See: https://github.com/Lightning-AI/lightning/issues/8375#issuecomment-878739663
    torch.distributed.destroy_process_group()
    if trainer.is_global_zero:

        # Get best checkpoint
        checkpoint_path = trainer.checkpoint_callback.best_model_path
        if not checkpoint_path:
            raise Exception("Unable to test, unable to determine checkpoint path.")

        # Construct the PL trainer for testing
        trainer = Trainer(
            max_epochs=0,
            # Do not print model summary
            enable_model_summary=False,
            # GPU Selection
            auto_select_gpus=True,
            devices=[config.GPU],  # Only need one for testing
            accelerator="gpu",
            benchmark=True,
            # Configure logger
            log_every_n_steps=5,
            logger=TensorBoardLogger(
                save_dir=config.CHECKPOINT_DIRECTORY,
                default_hp_metric=False,
                version=config.SESSION_VERSION,
                name=config.SESSION_NAME,
            ),
        )

        # Test the model (in single gpu mode)
        trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_path)

        print("TESTING COMPLETE")


if __name__ == "__main__":
    main()
