import abc
import inspect
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch.utils.data.dataset import Dataset

from utilities.helpers import STAGE_TEST, STAGE_TRAIN, STAGE_VALIDATION


def get_augmentation(arg: Union[str, Any]):
    """
    Looks up and creates an albumentations object by name (case insensitive).
    """

    # Constructs a map to all albumentation transforms
    lookup = {
        name.lower(): clazz  # snake case would be better, but oh well
        for name, clazz in inspect.getmembers(
            inspect.getmodule(A),
            lambda m: inspect.isclass(m) and issubclass(m, A.BasicTransform),
        )
    }

    # Either basic name for defaults or dictionary
    (name, kwargs) = (arg, None) if isinstance(arg, str) else (arg.pop("name"), arg)

    # Look up augmentation
    clazz = lookup.get(name.lower(), None)
    if not clazz:
        raise LookupError(f"Unable to find augmentation named '{name}'.")

    return clazz(**(kwargs or {}))


def apply_augmentations(
        augmentation: Callable[..., dict],
        image: np.ndarray,
        label: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Process an image or image/label pair with an Albumentations augmentation callable.
    """

    if label is not None:
        result = augmentation(image=image, mask=label)
        return (result["image"], result["mask"])
    else:
        result = augmentation(image=image)
        return (result["image"], None)


class AugmentationBase(abc.ABC):
    """
    Base class for augmentations.
    """

    def __init__(self, normalization: Dict[str, Tuple[float, float, float]]) -> None:
        super().__init__()

        self.normalization = normalization

    @abc.abstractmethod
    def apply_train(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        pass

    @abc.abstractmethod
    def apply_val(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        pass

    @abc.abstractmethod
    def apply_test(
            self, image: np.ndarray, label: Optional[np.ndarray]
    ) -> List[Tuple[np.ndarray, Optional[np.ndarray]]]:
        pass


class AugmentedDataset(Dataset):
    """
    Wraps a dataset with augmentation base.
    """

    def __init__(
            self,
            datamodule: LightningDataModule,
            dataset: Dataset,
            augmentations: AugmentationBase,
            stage: str,
            norm_label=False
    ):
        self.augmentations = augmentations
        self.datamodule = datamodule
        self.dataset = dataset
        self.stage = stage

        self._normalize = A.Compose(
            [
                A.Normalize(**self.augmentations.normalization),
                ToTensorV2(transpose_mask=True),
            ]
        )
        self.norm_label = norm_label

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        results = []

        # Select appropriate augmentation
        if self.stage == STAGE_TRAIN:
            apply = self.augmentations.apply_train
        elif self.stage == STAGE_VALIDATION:
            apply = self.augmentations.apply_val
        elif self.stage == STAGE_TEST:
            apply = self.augmentations.apply_test
        else:
            raise Exception("Invalid Stage")

        # Apply augmentations to the relevant data point
        for result in apply(*self.dataset[idx % len(self.dataset)]):

            # Ensure we have (img, lbl) pair even if lbl is None
            if not isinstance(result, tuple):
                result = (result, None)

            # Normalize and convert to tensor
            results.append(self._convert_tensor(*result))

        # Don't return as a list for a singular item
        if len(results) == 1:
            return results[0]

        return results

    def get_item_with_path(self, idx):
        return {"data": self.__getitem__(idx), "path": self.dataset.samples[idx]}

    def _convert_tensor(self, image, label):
        # print("COV TENSOR")
        original_label = deepcopy(label)
        image, label = apply_augmentations(self._normalize, image, label)
        if label is not None:
            if self.norm_label:
                # print("NORM LABEL")
                label, _ = apply_augmentations(self._normalize, original_label, None)
                return (image.type(torch.float32), label.type(torch.float32))
            else:
                # print("INT LABEL")
                return (image.type(torch.float32), label.type(torch.int64))
        else:
            # print("NONE LABEL")
            return image.type(torch.float32)


def inverse_normalize(
        x: torch.Tensor,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
):
    """
    Reverses the normalization transform performed on a tensor.
    """

    # Clone the tensor so we can return the inverse without affecting the original
    x = torch.clone(x)

    # https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/16
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    x.mul_(std).add_(mean)
    return x
