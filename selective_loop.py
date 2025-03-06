import logging
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import config
from datasets.nassar_unlabelled import NassarSelective
from datasets.pascal_voc_unlabelled import PascalSelective
from datasets.cityscapes_unlabelled import CitySelective
from datasets.sugarbeet import SugarSelective
from augmentations.reconstruction import RecRandomMask
from models import get_model
from utilities.helpers import print_if_rank_zero
from utilities.io import resolve_checkpoint_path, save_yaml

# Reduce verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

seed_everything(config.RANDOM_SEED, workers=True)


def init_trainer(idx):
    # Prepare Model Module
    model = get_model(config.MODEL_NAME, config.MODEL_ARGS)
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
            version=f"{config.SESSION_VERSION}/step{idx}",
            name=config.SESSION_NAME,
        ),
    )
    return model, trainer


def train_step(trainer, model, datamodule, ckpt_path):
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
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    print_if_rank_zero("TRAINING COMPLETE")

    # Halt the use of multi-gpu, we will restart with only one for testing
    # See: https://github.com/Lightning-AI/lightning/issues/8375#issuecomment-878739663
    torch.distributed.destroy_process_group()


def get_checkpoint(trainer, idx):
    if trainer.is_global_zero:

        # Get best checkpoint
        # checkpoint_path = trainer.checkpoint_callback.best_model_path
        checkpoint_path = resolve_checkpoint_path(
            os.path.join(config.SESSION_NAME, config.SESSION_VERSION, f"step{idx}", "checkpoints", "last.ckpt")
        )
        print(os.path.join(config.SESSION_NAME, config.SESSION_VERSION, f"step{idx}", "checkpoints", "last.ckpt"))
        print(f'Checking ckpt path: {checkpoint_path}')
        if not checkpoint_path:
            raise Exception("Unable to test, unable to determine checkpoint path.")
    return checkpoint_path


def init_random_train(selective_d):
    model, trainer = init_trainer(0)
    print(os.path.join(config.SESSION_NAME, config.SESSION_VERSION, "step0", "checkpoints", "last.ckpt"))
    ckpt_path = resolve_checkpoint_path(
        os.path.join(config.SESSION_NAME, config.SESSION_VERSION, "step0", "checkpoints", "last.ckpt")
    )
    init_train_part = selective_d.create_partition_dataset(0)
    train_step(trainer, model, init_train_part, ckpt_path)
    return model, trainer


def selective_step_train(selective_d, idx):
    model, trainer = init_trainer(idx)
    ckpt_path = None
    checkpoint_path = get_checkpoint(trainer, idx - 1)
    print(f"Load model from selective step {idx - 1} to train: {checkpoint_path}")
    model = model.load_from_checkpoint(checkpoint_path)

    selective_dmodule = selective_d.create_partition_selective_dataset(idx, selective_d.aug_selective)
    train_step(trainer, model, selective_dmodule, ckpt_path)
    return model, trainer


def selective_masking(model, d, trainer, num_sample, idx):
    print(f"Start selective masking step {idx}")
    selective_part = d.create_partition_dataset(idx, d.aug_infer)
    selective_data = selective_part.get_aug_dataset('test')

    checkpoint_path = get_checkpoint(trainer, idx - 1)
    test_model = model.load_from_checkpoint(checkpoint_path)
    test_model.eval()
    for i in range(len(selective_data)):
        sample_loss = []
        for j in range(num_sample):
            sample = selective_data.get_item_with_path(i)
            # x,y [3, h, w], need [b, 3, h, w]
            x, y = sample["data"]
            # print(y.shape, x.shape)
            y_pred = test_model(torch.unsqueeze(x, 0))
            # print(y.shape, y_pred.shape)
            tile_locs = RecRandomMask.get_patch_loc(y.shape[1], y.shape[2], config.SP_COUNT)
            loss_map = test_model._get_tile_loss_map(y_pred, torch.unsqueeze(y, 0), tile_locs)
            sample_loss.append(np.array(loss_map))
        sample_loss = np.array(sample_loss)
        ave_sample_loss = np.mean(sample_loss, axis=0)
        # print(sample_loss.shape, ave_sample_loss.shape)
        # print(x.shape, y.shape)
        # input("CHECK X Y SHAPE")
        masked_img = test_model._mask_image(y, ave_sample_loss, tile_locs)
        d.save_masked_image(sample["path"], masked_img)

        d.save_vis(sample["path"], test_model._inverse_nomalize(y), 'gt')
        d.save_vis(sample["path"], test_model._inverse_nomalize(x), 'random_mask')
        d.save_vis(sample["path"], masked_img, 'select_mask')

        if i % int(len(selective_data) / 20) == 0:
            print(f"Selective Step {idx}: {int(i / len(selective_data) * 100)}%")
        # input(f"Done Selective Masking Sample {i}")
    print(f"Selective Step {idx} Complete")


def main():
    num_sample = 5
    num_partition = 10
    if "city" in config.SESSION_NAME:
        d = CitySelective(config.DATASET_ARGS['batch_size'], num_partition)
    elif "nassar" in config.SESSION_NAME:
        d = NassarSelective(config.DATASET_ARGS['batch_size'], num_partition)
    elif "pascal" in config.SESSION_NAME:
        d = PascalSelective(config.DATASET_ARGS['batch_size'], num_partition)
    elif "sugar" in config.SESSION_NAME:
        d = SugarSelective(config.DATASET_ARGS['batch_size'], num_partition)

    # train on random masking part0
    model, trainer = init_random_train(d)

    for part_idx in range(1, num_partition):
        # selective masking for next part
        selective_masking(model, d, trainer, num_sample, part_idx)
        # train on
        selective_step_train(d, part_idx)
        # input(f"STOP CHECK STEP {part_idx}")


if __name__ == "__main__":
    main()
