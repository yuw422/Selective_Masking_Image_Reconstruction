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


class SugarBeetsSeg(LightningDataModule):
    identifier = "sugarbeets_seg"

    def __init__(self, batch_size: int):
        super().__init__()

        self.batch_size = batch_size

        def get_samples(subset):
            dir_sugarbeets = 'sugarbeets/ijrr_sugarbeets_2016_annotations'  # server location
            directory = os.path.join(config.DATASET_DIRECTORY, dir_sugarbeets)

            data_list_fn = directory + "/sugarbeets_weeds.txt"
            print(f"Loading Sugarbeets Segmentation, file list {data_list_fn}")
            count = 0
            data_list = []
            with open(data_list_fn) as f:
                for line in f:
                    data_list.append(directory+SugarBeetsSeg.get_label_image_path(line[:-1]))
                    count += 1
                if subset == 'train':
                    data_list = data_list[:int(0.7 * count)]  # use first 80% to train and rest as val
                elif subset == 'val':
                    data_list = data_list[int(0.7 * count):int(0.8 * count)]
                elif subset == 'test':
                    data_list = data_list[int(0.8 * count):]
                print(f"Loaded dataset file, Count: {len(data_list)}")
            return [(SugarBeetsSeg.get_rgb_path(x), x) for x in data_list]

        self._datasets = {
            STAGE_TRAIN: get_samples("train"),
            STAGE_VALIDATION: get_samples("val"),
            STAGE_TEST: get_samples("test"),
        }

        # Instantiate desired augmentation class
        self.augmentations = get_augmentation(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)

    @staticmethod
    def get_label_image_path(mask_path):
        # split_address = mask_path.split("ijrr_sugarbeets_2016_annotations")
        return mask_path

    @staticmethod
    def get_rgb_path(mask_path):
        split_address = mask_path.split("annotations/dlp/iMapCleaned")
        return split_address[0] + "images/rgb" + split_address[1]

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
        return SugarBeetsSeg._Dataset(self._datasets[stage])

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

    @staticmethod
    def convert2trainid(raw_mask):
        # Suger_beets_2016_bonn/dataset/visualizeRandomSample.py
        mask = np.zeros(raw_mask.shape)
        mask[raw_mask == 1] = 0
        mask[raw_mask == 97] = 0
        # crop
        mask[raw_mask == 10000] = 1
        mask[raw_mask == 10001] = 1
        mask[raw_mask == 10002] = 1
        # weed
        mask[raw_mask > 10000] = 2
        return mask

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

            return image, SugarBeetsSeg.convert2trainid(raw_label)
