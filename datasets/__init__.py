import inspect
import sys
from typing import Any, Dict

from pytorch_lightning import LightningDataModule

from utilities.validation import strip_and_warn_kwargs, validate_datamodule_type

from .pascal_voc_segmentation import PascalVocSegmentation
from .pascal_voc_unlabelled import PascalVocUnlabelled
from .pascal_voc_superpixel import PascalVocUnlabelledSuperpixel
from .imagenet import ImageNet
from .imagenet_superpixel import ImageNetSuperpixel
from .cityscapes import Cityscapes
from .cityscapes_unlabelled import CityscapesUnlabelled
from .sugarbeet import SugarBeets
from .sugarbeet_seg import SugarBeetsSeg
from .nassar import Nassar
from .nassar_unlabelled import NassarUnlabelled

__all__ = [
    "PascalVocSegmentation",
    "PascalVocUnlabelled",
    "PascalVocUnlabelledSuperpixel",
    "ImageNet",
    "ImageNetSuperpixel",
    "Cityscapes",
    "CityscapesUnlabelled",
    "SugarBeets",
    "SugarBeetsSeg",
    "Nassar",
    "NassarUnlabelled"
]


def get_all_datamodules() -> Dict[str, Any]:
    """
    Gets the table all known model types.
    """

    module = sys.modules[__name__]

    def is_module(type_):
        return (
                inspect.isclass(type_)
                and issubclass(type_, LightningDataModule)
                and (type_ != LightningDataModule)
        )

    lookup = {}

    for member_name, _ in inspect.getmembers(module, is_module):
        # Get module member
        member = getattr(module, member_name)
        # Store the member (if member has an identifier)
        if hasattr(member, "identifier"):
            lookup[member.identifier] = member

    return lookup


def get_datamodule(name: str, params: dict) -> LightningDataModule:
    """
    Instantiates the specified data module by name.
    """

    module_table = get_all_datamodules()
    if name not in module_table:
        options = ", ".join([f"'{x}'" for x in module_table.keys()])
        raise KeyError(f"Unable to find model '{name}'. Options are {options}.")

    # Get model, validate it has the necessary properties for training
    module_type = module_table[name]
    validate_datamodule_type(module_type)

    # Instantiate the datamodule
    strip_and_warn_kwargs(module_type, params)
    return module_type(**params)
