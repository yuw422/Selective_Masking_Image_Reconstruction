import glob
import os
import albumentations as A
from copy import deepcopy
from pathlib import Path
from PIL import Image
from typing import List, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

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


class PascalVocUnlabelled(LightningDataModule):
    identifier = "pascal_unlabel"

    def __init__(self, batch_size: int, data_list=None, aug=None):
        super().__init__()

        self.batch_size = batch_size

        # Full path to VOC directory
        voc_directory = os.path.join(config.DATASET_DIRECTORY, "pascal/VOCtrainvaltest/VOCdevkit")

        def get_samples(era, path):
            # Get the directory of the specific year
            directory = os.path.join(voc_directory, era)

            # Load sample names in each group
            with open(os.path.join(directory, "ImageSets/Segmentation", path), "r") as f:
                names = f.read().splitlines()

            def get_label_pair(name):
                img = os.path.join(directory, "JPEGImages", name + ".jpg")
                img_exist = os.path.exists(img)

                lbl = os.path.join(directory, "SegmentationClass", name + ".png")
                lbl_exist = os.path.exists(lbl)

                # print(f"path: '{path}'", f"'{name}'", img_exist, lbl_exist)
                return (img, lbl) if (img_exist and lbl_exist) else None

            # Map samples to image pairs
            return list(filter(None, [get_label_pair(name) for name in names]))

        if data_list is None:
            # Get all known segmentation samples
            train_samples = get_samples("VOC2012", "train.txt")
            test_samples = get_samples("VOC2007", "test.txt")
            val_samples = get_samples("VOC2012", "val.txt")

            # We only want images without a segmentation label
            def is_labelled_image(path):
                name = os.path.splitext(os.path.basename(path))[0]
                return (name in train_samples) or (name in val_samples) or (name in test_samples)

            # Finds all pascal voc images without a segmentation annotation
            unlabelled_samples = [
                img_path
                for img_path in glob.glob(
                    os.path.join(voc_directory, "VOC2012/JPEGImages/**/*.jpg"),
                    recursive=True,
                )
                # no reason to exclude labelled image from pretraining
                # if not is_labelled_image(img_path)
            ]
        else:
            print(f"Loading Pascal VOC from predetemined data list")
            unlabelled_samples = data_list

        # Count how many samples we have
        num_samples = len(unlabelled_samples)

        # Compute the size of each split (70/15/15)
        # test_size = int(num_samples * 0.15)
        # val_size = int(num_samples * 0.15)
        # train_size = num_samples - val_size - test_size
        #
        # # Split samples into a training and validation splits
        # train_samples, val_samples, test_samples = random_split(
        #     lengths=[train_size, val_size, test_size],
        #     dataset=unlabelled_samples,
        #     # We forcefully set a generator here with our random seed to ensure
        #     # a deterministic split of the unlabelled images.
        #     generator=torch.Generator().manual_seed(config.RANDOM_SEED),
        # )
        train_samples = unlabelled_samples[:int(0.9 * num_samples)]
        # we don't really care about pretraining accuracy here
        val_samples = unlabelled_samples[int(0.9 * num_samples):]
        test_samples = unlabelled_samples

        self._datasets = {
            STAGE_TRAIN: train_samples,
            STAGE_VALIDATION: val_samples,
            STAGE_TEST: test_samples,
        }

        # Instantiate desired augmentation class
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
        return PascalVocUnlabelled._Dataset(self._datasets[stage])

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

        def __init__(self, samples: List[str]):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            # Get image path
            image_path = self.samples[idx]

            # Read the image
            image = load_image(image_path)

            return image, None


class PascalSelectiveMask(PascalVocUnlabelled):
    identifier = "pascal_selective"

    def __init__(self, batch_size: int, data_list=None, aug=None):
        super().__init__(batch_size, data_list, aug)

        self.batch_size = batch_size

        def get_samples(subset, data_list):
            count = len(data_list)
            print(f"Loading Pascal VOC from predetemined data list")
            if subset == 'train':
                data_list = data_list[:int(0.9 * count)]
            elif subset == 'val':
                data_list = data_list[int(0.9 * count):]
            elif subset == 'test':
                data_list = data_list
            print(f"Dataset Len {len(data_list)}")
            return [("selective_mask".join(x.split("JPEGImages")), x) for x in data_list]

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
        return PascalSelectiveMask._Dataset(self._datasets[stage])

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
            self.resize = config.AUGMENTATION_ARGS['resize']

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            # Get image paths
            selective_mask_image_path, image_path = self.samples[idx]

            # Read the image
            smimg = load_image(selective_mask_image_path)
            trans = A.Compose(
                [
                    # Not made square, because we test 1 images at a time
                    A.SmallestMaxSize(max_size=self.resize),
                ])
            image = trans(image=load_image(image_path))['image']

            return smimg, image


class PascalSelective(object):
    """
    create a list of partitions of nassar unlabelled datamodules

    """

    def __init__(self, batch_size: int, num_partitions: int):
        super().__init__()

        self.batch_size = batch_size
        voc_directory = os.path.join(config.DATASET_DIRECTORY, "pascal/VOCtrainvaltest/VOCdevkit")
        dir_voc_selective_mask = 'VOC2012/selective_mask/'
        dir_voc_vis = 'VOC2012/vis/'
        self.selective_dir = os.path.join(voc_directory, dir_voc_selective_mask)
        self.vis_dir = os.path.join(voc_directory, dir_voc_vis)
        Path(self.selective_dir).mkdir(exist_ok=True)
        Path(self.vis_dir).mkdir(exist_ok=True)

        print("Loading Pascal VOC")
        self.data_list = [
            img_path
            for img_path in glob.glob(
                os.path.join(voc_directory, "VOC2012/JPEGImages/**/*.jpg"),
                recursive=True,
            )]
        print(f"Dataset Total Len {len(self.data_list)}")
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
        return PascalVocUnlabelled(self.batch_size, self.partition_list[idx], aug=aug)

    def create_partition_selective_dataset(self, idx, aug=None):
        if isinstance(idx, int):
            print(f"Creating Selective Partition idx: {idx}")
            return PascalSelectiveMask(self.batch_size, self.partition_list[idx], aug=aug)
        elif isinstance(idx, list):
            part_list = []
            for i in idx:
                part_list += self.partition_list[i]
            print(f"Creating Selective Partition from list: {idx}, len {len(part_list)}")
            return PascalSelectiveMask(self.batch_size, part_list, aug=aug)

    def save_masked_image(self, input_image_path, masked_image):
        p_split = input_image_path.split("JPEGImages")
        # print(input_image_path)
        save_path = "selective_mask".join(p_split)
        # print(f"Save masked image at {save_path}")
        im = Image.fromarray(masked_image)
        im.save(save_path)

    def save_vis(self, input_image_path, image, image_type):
        p_split = input_image_path.split("JPEGImages")
        p_split[1] = p_split[1].split(".png")[0] + f"_{image_type}.png"
        # print(input_image_path)
        save_path = "vis".join(p_split)
        # print(f"Save {image_type} vis at {save_path}")
        # print(image.shape)
        # input("CHECK PATH")
        im = Image.fromarray(image)
        im.save(save_path)
