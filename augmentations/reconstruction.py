import random
from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np

from utilities.augmentations import AugmentationBase, apply_augmentations

from .custom import CustomAugmentations


class RandomRecMasking(AugmentationBase):
    """
    Implements Image Reconstruction w/ Coarse Cutouts with additional user
    specified augmentation steps.
    """

    identifier = "random_rec_mask"

    def __init__(
            self,
            normalization: Dict[str, Tuple[float, float, float]],
            resize: int,
            augmentation_steps: List[Union[str, Dict]],
    ):
        super().__init__(normalization)

        self.resize = resize

        # Compute mean in 0-255 range
        mean = np.array(normalization["mean"]) * 255

        # ...
        self._base_augmentations = CustomAugmentations(
            normalization,
            resize,
            augmentation_steps,
        )

        # ...
        self._augmentations = A.Compose(
            [
                RecRandomMask(512,
                              fill_color=mean,
                              ratio=0.5,
                              p=1),
            ]
        )

    def apply_train(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        image, label = self._base_augmentations.apply_train(image, label)[0]
        # first one is (aug_image, none), second one is (image, none)
        return [
            apply_augmentations(self._augmentations, image, label),
            (image, label),
        ]

    def apply_val(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        image, label = self._base_augmentations.apply_val(image, label)[0]
        return [
            apply_augmentations(self._augmentations, image, label),
            (image, label),
        ]

    def apply_test(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        image, label = self._base_augmentations.apply_test(image, label)[0]
        return [
            apply_augmentations(self._augmentations, image, label),
            (image, label),
        ]


class CoarseCutout(A.ImageOnlyTransform):
    """
    Replicates the standard "CoarseDropout" augmentation available in Albumentations, but allows
    specifying the cutouts to be per-channel. The number of holes is also based on a threshold of
    approximate total pixels masked out.
    """

    def __init__(
            self,
            min_size=2,
            max_size=16,
            per_channel=False,
            ratio=0.5,
            fill_color: np.ndarray = np.array([0, 0, 0]),
            always_apply: bool = False,
            p: float = 0.5,
    ):
        super().__init__(always_apply, p)

        self.per_channel = per_channel

        self.min_height = min_size
        self.min_width = min_size

        self.max_height = max_size
        self.max_width = max_size

        self.ratio = ratio
        self.fill_color = fill_color

    def apply(self, img: np.ndarray, **params) -> np.ndarray:

        img = np.copy(img)
        height, width = img.shape[:2]

        # The total number of pixels
        total_pixels = width * height * (3 if self.per_channel else 1)
        count = 0

        # Will cut regions until the desired ratio is acheived
        while (count / total_pixels) < self.ratio:

            # Generate hole size
            hole_height = random.randint(self.min_height, self.max_height)
            hole_width = random.randint(self.min_width, self.max_width)

            # Generate hole position
            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width

            # Apply cutout
            if self.per_channel:
                # Slice out a rect from a random channel
                c = random.randint(0, 2)
                img[y1:y2, x1:x2, c] = self.fill_color[c]
            else:
                # Slice out a rect from the image
                img[y1:y2, x1:x2] = self.fill_color

            # Count these pixels removed. Technically invalid since any overlapping
            # rect will result in those pixels being counted twice, but its fast.
            count += hole_width * hole_height

        return img


class RecRandomMask(A.ImageOnlyTransform):
    def __init__(self,
                 num_patch,
                 fill_color: np.ndarray = np.array([0, 0, 0]),
                 ratio=0.5,
                 always_apply: bool = False,
                 p: float = 0.5, ):
        super().__init__(always_apply, p)
        self.num_patch = num_patch
        self.percentage = ratio
        self.fill_color = fill_color

    @staticmethod
    def get_patch_loc(h, w, num_patch):
        hpixel = h / np.sqrt(num_patch)
        wpixel = w / np.sqrt(num_patch)
        hloc = np.arange(0, h, np.ceil(hpixel))
        wloc = np.arange(0, w, np.ceil(wpixel))
        # if last patch is smaller than 1/2 patch_size, merge with previous patch
        if h - hloc[-1] < hpixel / 2:
            hloc[-1] = h
        else:
            hloc = np.append(hloc, [h])

        if w - wloc[-1] < wpixel / 2:
            wloc[-1] = w
        else:
            wloc = np.append(wloc, [w])
        hloc = hloc.astype(np.int32)
        wloc = wloc.astype(np.int32)
        # print(hloc)
        # print(wloc)
        locs = []
        for i in range(len(hloc) - 1):
            for j in range(len(wloc) - 1):
                locs.append((hloc[i], hloc[i + 1], wloc[j], wloc[j + 1]))
        return locs

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        img = np.copy(img)
        h, w = img.shape[:2]
        locs = self.get_patch_loc(h, w, self.num_patch)
        mask_idx = random.sample(range(len(locs)), int(len(locs) * self.percentage))
        # print(len(mask_idx))
        for i in mask_idx:
            img[locs[i][0]:locs[i][1], locs[i][2]:locs[i][3], :] = self.fill_color
        return img
