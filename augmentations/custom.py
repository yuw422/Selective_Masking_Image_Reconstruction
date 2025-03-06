from typing import Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np

from utilities.augmentations import (
    AugmentationBase,
    apply_augmentations,
    get_augmentation,
)


class CustomAugmentations(AugmentationBase):
    """
    Implements a custom augmentation set (random resized crop + user defined list)
    """

    identifier = "custom"

    def __init__(
            self,
            normalization: Dict[str, Tuple[float, float, float]],
            resize: int,
            augmentation_steps: List[Union[str, Dict]],
    ):
        super().__init__(normalization)

        self.resize = resize

        # Augmenations for the train stage
        self._train = A.Compose(
            [
                A.RandomResizedCrop(
                    width=self.resize,
                    height=self.resize,
                    scale=(0.25, 1.0),
                ),
                # Populate with user selected augmentations
                *[get_augmentation(x) for x in augmentation_steps],
            ]
        )

        # Augmenations for the validation stage
        self._val = A.Compose(
            [
                # Made square to allow batching
                A.SmallestMaxSize(max_size=self.resize),
                A.CenterCrop(width=self.resize, height=self.resize),
            ]
        )

        # Augmenations for the test stage
        self._test = A.Compose(
            [
                # Not made square, because we test 1 images at a time
                A.SmallestMaxSize(max_size=self.resize),
            ]
        )

    def apply_train(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        return [apply_augmentations(self._train, image, label)]

    def apply_val(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        return [apply_augmentations(self._val, image, label)]

    def apply_test(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        return [apply_augmentations(self._test, image, label)]
