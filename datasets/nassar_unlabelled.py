import os
from copy import deepcopy
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image

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


class NassarUnlabelled(LightningDataModule):
    identifier = "nassar_unlabel"

    def __init__(self, batch_size: int, data_list=None, aug=None):
        super().__init__()

        self.batch_size = batch_size

        def get_samples(subset, data_list):
            dir_nassar = 'nassar/ortho_tiles/'
            directory = os.path.join(config.DATASET_DIRECTORY, dir_nassar)
            if data_list is None:
                data_list_fn = directory + "/datalist.txt"
                print(f"Loading Nassar, file list {data_list_fn}")
                count = 0
                data_list = []
                with open(data_list_fn) as f:
                    for line in f:
                        data_list.append(line[:-1])
                        count += 1
            else:
                count = len(data_list)
                print(f"Loading Nassar from predetemined data list")
            if subset == 'train':
                data_list = data_list[:int(0.9 * count)]
            elif subset == 'val':
                data_list = data_list[int(0.9 * count):]
            elif subset == 'test':
                data_list = data_list
            print(f"Dataset Len {len(data_list)}")
            return [directory + f"/{x}" for x in data_list]

        self._datasets = {
            STAGE_TRAIN: get_samples("train", data_list),
            STAGE_VALIDATION: get_samples("val", data_list),
            STAGE_TEST: get_samples("test", data_list),
        }

        # Instantiate desired augmentation class
        # print(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)
        if aug is None:
            self.augmentations = get_augmentation(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)
        else:
            self.augmentations = get_augmentation(aug['names'], aug['args'])

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
        return NassarUnlabelled._Dataset(self._datasets[stage])

    def train_dataloader(self):
        return self._get_dataloader(STAGE_TRAIN)

    def val_dataloader(self):
        return self._get_dataloader(STAGE_VALIDATION)

    def test_dataloader(self):
        return self._get_dataloader(STAGE_TEST)

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

    def get_aug_dataset(self, stage: str):
        return AugmentedDataset(self, self.get_dataset(stage), self.augmentations, stage)

    class _Dataset(Dataset):
        """"""

        def __init__(self, samples: List[Tuple[str, str]]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            # Get image paths
            image_path = self.samples[idx]

            # Read the image
            image = load_image(image_path)

            return image, None


class NassarSelectiveMask(NassarUnlabelled):
    identifier = "nassar_selective"

    def __init__(self, batch_size: int, data_list=None, aug=None):
        super().__init__(batch_size, data_list, aug)

        self.batch_size = batch_size

        def get_samples(subset, data_list):
            dir_nassar = 'nassar/ortho_tiles/'
            directory = os.path.join(config.DATASET_DIRECTORY, dir_nassar)
            dir_nassar_selective_mask = 'nassar/selective_mask/'
            mask_directory = os.path.join(config.DATASET_DIRECTORY, dir_nassar_selective_mask)

            count = len(data_list)
            print(f"Loading Nassar from predetemined data list")
            if subset == 'train':
                data_list = data_list[:int(0.9 * count)]
            elif subset == 'val':
                data_list = data_list[int(0.9 * count):]
            elif subset == 'test':
                data_list = data_list
            print(f"Dataset Len {len(data_list)}")
            return [(mask_directory + f"/{x}", directory + f"/{x}") for x in data_list]

        self._datasets = {
            STAGE_TRAIN: get_samples("train", data_list),
            STAGE_VALIDATION: get_samples("val", data_list),
            STAGE_TEST: get_samples("test", data_list),
        }

        # Instantiate desired augmentation class
        # print(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)
        if aug is None:
            self.augmentations = get_augmentation(config.AUGMENTATION_NAME, config.AUGMENTATION_ARGS)
        else:
            self.augmentations = get_augmentation(aug['names'], aug['args'])

    def get_dataset(self, stage: str) -> Optional[Dataset]:
        return NassarSelectiveMask._Dataset(self._datasets[stage])

    def _get_dataloader(self, stage: str):
        return DataLoader(
            AugmentedDataset(self, self.get_dataset(stage), self.augmentations, stage, norm_label=True),
            batch_size=(1 if stage == STAGE_TEST else self.batch_size),
            persistent_workers=True,
            num_workers=get_sufficient_num_workers(),
            shuffle=(stage == STAGE_TRAIN),
            pin_memory=True,
            drop_last=True,
        )

    def get_aug_dataset(self, stage: str):
        return AugmentedDataset(self, self.get_dataset(stage), self.augmentations, stage, norm_label=True)

    class _Dataset(Dataset):
        """"""

        def __init__(self, samples: List[Tuple[str, str]]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            # Get image paths
            selective_mask_image_path, image_path = self.samples[idx]

            # Read the image
            smimg = load_image(selective_mask_image_path)
            image = load_image(image_path)

            return smimg, image


class NassarSelective(object):
    """
    create a list of partitions of nassar unlabelled datamodules

    """

    def __init__(self, batch_size: int, num_partitions: int):
        super().__init__()

        self.batch_size = batch_size
        dir_nassar = 'nassar/ortho_tiles/'
        dir_nassar_selective_mask = 'nassar/selective_mask/'
        dir_nassar_vis = 'nassar/vis/'
        self.selective_dir = os.path.join(config.DATASET_DIRECTORY, dir_nassar_selective_mask)
        self.vis_dir = os.path.join(config.DATASET_DIRECTORY, dir_nassar_vis)
        Path(self.selective_dir).mkdir(exist_ok=True)
        Path(self.vis_dir).mkdir(exist_ok=True)
        directory = os.path.join(config.DATASET_DIRECTORY, dir_nassar)
        data_list_fn = directory + "/datalist.txt"
        print(f"Loading Nassar, file list {data_list_fn}")
        self.data_list = []
        with open(data_list_fn) as f:
            for line in f:
                self.data_list.append(line[:-1])
        # [directory + f"/{x}" for x in data_list]
        self.partition_list = self.create_partitions_list(num_partitions)
        self.aug_train = {"names": config.AUGMENTATION_NAME, "args": config.AUGMENTATION_ARGS}
        # {'augmentation_steps': ['HorizontalFlip', 'ColorJitter'],
        # 'normalization': {'mean': [0.45286129, 0.43170348, 0.39989259],
        # 'std': [0.44426655, 0.46648413, 0.48871749]}, 'resize': 256}
        # use
        self.aug_infer = deepcopy(self.aug_train)
        self.aug_infer["args"]["augmentation_steps"] = []
        self.aug_selective = deepcopy(self.aug_train)
        self.aug_selective["names"] = "custom"
        print(f"AUG TRAIN {self.aug_train}")
        print(f"AUG INFER {self.aug_infer}")
        print(f"AUG SELEC {self.aug_selective}")

    def create_partitions_list(self, num_partitions):
        return [self.data_list[i::num_partitions] for i in range(num_partitions)]

    def create_partition_dataset(self, idx, aug=None):
        return NassarUnlabelled(self.batch_size, self.partition_list[idx], aug=aug)

    def create_partition_selective_dataset(self, idx, aug=None):
        if isinstance(idx, int):
            print(f"Creating Selective Partition idx: {idx}")
            return NassarSelectiveMask(self.batch_size, self.partition_list[idx], aug=aug)
        elif isinstance(idx, list):
            part_list = []
            for i in idx:
                part_list += self.partition_list[i]
            print(f"Creating Selective Partition from list: {idx}, len {len(part_list)}")
            return NassarSelectiveMask(self.batch_size, part_list, aug=aug)

    def save_masked_image(self, input_image_path, masked_image):
        p_split = input_image_path.split("ortho_tiles")
        # print(input_image_path)
        save_path = "selective_mask".join(p_split)
        # print(f"Save masked image at {save_path}")
        im = Image.fromarray(masked_image)
        im.save(save_path)

    def save_vis(self, input_image_path, image, image_type):
        p_split = input_image_path.split("ortho_tiles")
        p_split[1] = p_split[1].split(".png")[0] + f"_{image_type}.png"
        # print(input_image_path)
        save_path = "vis".join(p_split)
        # print(f"Save {image_type} vis at {save_path}")
        # print(image.shape)
        # input("CHECK PATH")
        im = Image.fromarray(image)
        im.save(save_path)
