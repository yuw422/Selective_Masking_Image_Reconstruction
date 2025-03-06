import inspect
import sys
from typing import Any, Dict

from pytorch_lightning import LightningModule

from utilities.validation import strip_and_warn_kwargs, validate_module_type

from .reconstruction import Reconstruction
from .resunet34 import ResUNet34
from .reconstruction_selective import ReconstructionSelective

__all__ = [
    "Reconstruction",
    "ReconstructionSelective",
    "ResUNet34",
]


def get_all_models() -> Dict[str, Any]:
    """
    Gets the table all known model types.
    """

    module = sys.modules[__name__]

    def is_module(type_):
        return (
                inspect.isclass(type_)
                and issubclass(type_, LightningModule)
                and (type_ != LightningModule)
        )

    lookup = {}

    for member_name, _ in inspect.getmembers(module, is_module):
        # Get module member
        member = getattr(module, member_name)
        # Store the member (if member has an identifier)
        if hasattr(member, "identifier"):
            lookup[member.identifier] = member

    return lookup


def get_model(name: str, params: dict) -> LightningModule:
    """
    Instantiates the specified model by name.
    """

    module_table = get_all_models()
    if name not in module_table:
        options = ", ".join([f"'{x}'" for x in module_table.keys()])
        raise KeyError(f"Unable to find model '{name}'. Options are {options}.")

    # Get model, validate it has the necessary properties for training
    module_type = module_table[name]
    validate_module_type(module_type)

    # Instantiate the model
    strip_and_warn_kwargs(module_type, params)
    return module_type(**params)
