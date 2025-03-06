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
    19: 'other',
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
}

CLASS_COLORS = {
    19: (0, 0, 0),
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
}

ID_MAP = {
    0: (19, (0, 0, 0)),
    1: (19, (0, 0, 0)),
    2: (19, (0, 0, 0)),
    3: (19, (0, 0, 0)),
    4: (19, (0, 0, 0)),
    5: (19, (111, 74, 0)),
    6: (19, (81, 0, 81)),
    7: (0, (128, 64, 128)),
    8: (1, (244, 35, 232)),
    9: (19, (250, 170, 160)),
    10: (19, (230, 150, 140)),
    11: (2, (70, 70, 70)),
    12: (3, (102, 102, 156)),
    13: (4, (190, 153, 153)),
    14: (19, (180, 165, 180)),
    15: (19, (150, 100, 100)),
    16: (19, (150, 120, 90)),
    17: (5, (153, 153, 153)),
    18: (19, (153, 153, 153)),
    19: (6, (250, 170, 30)),
    20: (7, (220, 220, 0)),
    21: (8, (107, 142, 35)),
    22: (9, (152, 251, 152)),
    23: (10, (70, 130, 180)),
    24: (11, (220, 20, 60)),
    25: (12, (255, 0, 0)),
    26: (13, (0, 0, 142)),
    27: (14, (0, 0, 70)),
    28: (15, (0, 60, 100)),
    29: (19, (0, 0, 90)),
    30: (19, (0, 0, 110)),
    31: (16, (0, 80, 100)),
    32: (17, (0, 0, 230)),
    33: (18, (119, 11, 32)),
    -1: (19, (0, 0, 142)),
}


class Cityscapes(LightningDataModule):
    identifier = "cityscapes"

    def __init__(self, batch_size: int):
        super().__init__()

        self.batch_size = batch_size

        def get_samples(subset):
            # Get the directory of the specific year
            directory = os.path.join(config.DATASET_DIRECTORY, 'cityscapes')
            image_dir = "leftImg8bit_trainvaltest/leftImg8bit/"
            label_dir = "gtFine_trainvaltest/gtFine/"
            image_name = "leftImg8bit.png"
            label_name = "gtFine_labelIds.png"

            subset_folder = 'val' if subset == 'test' else 'train'
            subset_image_dir = os.path.join(directory, image_dir, subset_folder)
            subset_label_dir = os.path.join(directory, label_dir, subset_folder)

            data_list_fn = subset_image_dir + f"/{subset_folder}.txt"
            print(f"Loading {subset} set, file list {data_list_fn}")
            count = 0
            data_list = []
            with open(data_list_fn) as f:
                for line in f:
                    data_list.append(line[:-1])
                    count += 1
                if subset == 'train':
                    data_list = data_list[:int(0.8 * count)]  # use first 80% to train and rest as val
                elif subset == 'val':
                    data_list = data_list[int(0.8 * count):]
                elif subset == 'test':
                    data_list = data_list
                print(f"Loaded dataset file, Count: {len(data_list)}")
            return [(subset_image_dir + "/" + x + "_" + image_name, subset_label_dir + "/" + x + "_" + label_name) for x
                    in data_list]

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
        return Cityscapes._Dataset(self._datasets[stage])

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
    def id2class(id):
        return ID_MAP[id][0]

    @staticmethod
    def convert2trainid(raw_mask):
        mask = np.zeros(raw_mask.shape)
        # print(np.unique(raw_mask, return_counts=True))
        for i in range(-1, 34):
            mask[raw_mask == i] = Cityscapes.id2class(i)
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

            return image, Cityscapes.convert2trainid(raw_label)
