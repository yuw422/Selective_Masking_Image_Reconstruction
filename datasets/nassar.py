import os
from typing import List, Optional, Tuple

import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import config
from augmentations import get_augmentation
from utilities.augmentations import AugmentedDataset
from utilities.helpers import (
    STAGE_TEST,
    STAGE_TRAIN,
    STAGE_VALIDATION,
    get_sufficient_num_workers,
)
from utilities.io import load_image

CLASS_NAMES = {
    0: 'background',
    1: 'crop',
    2: 'weed',
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 128, 0),
    2: (128, 0, 0),
}


class Nassar(LightningDataModule):
    identifier = "nassar"

    def __init__(self, batch_size: int):
        super().__init__()

        self.batch_size = batch_size

        def get_samples(subset):
            dir_nassar = 'nassar/tiled_dataset/train'

            directory = os.path.join(config.DATASET_DIRECTORY, dir_nassar)
            data_list_fn = directory + f"/data_list/{subset}.txt"
            print(f"Loading Nassar, file list {data_list_fn}")
            count = 0
            data_list = []
            with open(data_list_fn) as f:
                for line in f:
                    data_list.append(line[:-1])
                    count += 1
                print(f"Loaded {subset} file, Count: {len(data_list)}")
            if subset != "test":
                data_list = data_list[:int(len(data_list)/5)]
            print(f"Dataset Len {len(data_list)}")
            return [(directory + f"/data/{x}.png", directory + f"/mask/{x}.png") for x in data_list]

        self._datasets = {
            STAGE_TRAIN: get_samples("train"),
            STAGE_VALIDATION: get_samples("val"),
            STAGE_TEST: get_samples("test"),
        }

        # Instantiate desired augmentation class
        self.augmentations = get_augmentation(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)

    @property
    def train_samples(self) -> List[Tuple[str, str]]:
        return self._datasets[STAGE_TRAIN]

    @property
    def val_samples(self) -> List[Tuple[str, str]]:
        return self._datasets[STAGE_VALIDATION]

    @property
    def test_samples(self) -> List[Tuple[str, str]]:
        return self._datasets[STAGE_TEST]

    def get_dataset(self, stage: str) -> Optional[Dataset]:
        return Nassar._Dataset(self._datasets[stage])

    def train_dataloader(self):
        return self._get_dataloader(STAGE_TRAIN)

    def val_dataloader(self):
        return self._get_dataloader(STAGE_VALIDATION)

    def test_dataloader(self):
        return self._get_dataloader(STAGE_TEST)

    def get_class_color(self, i: int) -> np.ndarray:
        return CLASS_COLORS[i]

    def get_class_name(self, i: int) -> str:
        return CLASS_NAMES[i]

    def _get_dataloader(self, stage: str):
        return DataLoader(
            AugmentedDataset(self, self.get_dataset(stage), self.augmentations, stage),
            batch_size=(1 if stage == STAGE_TEST else self.batch_size),
            persistent_workers=True,
            num_workers=get_sufficient_num_workers(),
            shuffle=(stage == STAGE_TRAIN),
            pin_memory=True,
            drop_last=True,
        )

    class _Dataset(Dataset):
        """"""

        def __init__(self, samples: List[Tuple[str, str]]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            # Get image and label paths
            image_path, label_path = self.samples[idx]

            # Read the image and label
            image = load_image(image_path)

            # Load color label
            raw_label = load_image(label_path)

            return image, raw_label
