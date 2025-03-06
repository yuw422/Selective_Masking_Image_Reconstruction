import logging

import numpy as np
import torch.utils.data as data
from pytorch_lightning import seed_everything

import config
from datasets import get_datamodule
from utilities.io import load_image

# Reduce verbosity
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def main():
    # Set the random seed on everything, to make things more replicable
    seed_everything(config.RANDOM_SEED, workers=True)

    # Prepare Data Module
    datamodule = get_datamodule(config.DATASET_NAME, config.DATASET_ARGS)

    # ...
    datamodule.prepare_data()
    datamodule.setup()

    # Assumes the specific implementation of datamodules in this repo
    train = datamodule._datasets["train"]
    test = datamodule._datasets["test"]
    val = datamodule._datasets["val"]

    color_sum = np.zeros((3,))
    color_sum_squared = np.zeros((3,))
    pixel_count = 0

    for sample in list(data.ConcatDataset([train, test, val])):
        # Load image
        image = load_image(sample)

        # Sum colors
        color_sum += np.sum(image, axis=(0, 1)) / 255.0
        color_sum_squared += np.sum(image ** 2, axis=(0, 1)) / 255.0

        # Count pixels
        pixel_count += image.shape[0] * image.shape[1]

    # Compute mean and std.dev
    color_mean = color_sum / pixel_count
    color_std = np.sqrt(color_sum_squared / pixel_count - color_mean ** 2)

    # Report mean and std.dev
    print("color_mean", color_mean)
    print("color_std", color_std)


if __name__ == "__main__":
    main()
