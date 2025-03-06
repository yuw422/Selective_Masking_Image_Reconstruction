import inspect
import sys
from typing import Any, Dict

from utilities.augmentations import AugmentationBase
from utilities.validation import strip_and_warn_kwargs, validate_augmentation_type

from .custom import CustomAugmentations
from .reconstruction import RandomRecMasking

__all__ = [
    "CustomAugmentations",
    "RandomRecMasking",
]


def get_all_augmentations() -> Dict[str, Any]:
    """
    Gets the table all known augmentation class types.
    """

    module = sys.modules[__name__]

    def is_module(type_):
        return (
                inspect.isclass(type_)
                and issubclass(type_, AugmentationBase)
                and (type_ != AugmentationBase)
        )

    lookup = {}

    for member_name, _ in inspect.getmembers(module, is_module):
        # Get module member
        member = getattr(module, member_name)
        # Store the member (if member has an identifier)
        if hasattr(member, "identifier"):
            lookup[member.identifier] = member

    return lookup


def get_augmentation(name: str, params: dict) -> AugmentationBase:
    """
    Instantiates the specified augmentation class by name.
    """

    module_table = get_all_augmentations()
    if name not in module_table:
        options = ", ".join([f"'{x}'" for x in module_table.keys()])
        raise KeyError(f"Unable to find augmentation '{name}'. Options are {options}.")

    # Get model, validate it has the necessary properties for training
    module_type = module_table[name]
    validate_augmentation_type(module_type)

    # Instantiate the augmentation class
    strip_and_warn_kwargs(module_type, params)
    return module_type(**params)
