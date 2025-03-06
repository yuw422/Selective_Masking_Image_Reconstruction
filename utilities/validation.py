import inspect


def validate_datamodule_type(module):
    """
    Validates the datamodule type has all the attributes necessary to work with the training script.
    """

    pass  # Nothing special to validate


def validate_augmentation_type(module):
    """
    Validates the augmentation type has all the attributes necessary to work with the training script.
    """

    pass  # Nothing special to validate


def validate_module_type(module):
    """
    Validates the module type has all the attributes necessary to work with the training script.
    """

    if not hasattr(module, "checkpoint_monitor"):
        raise RuntimeError("Modules have a 'checkpoint_monitor' class property (ex, \"loss\").")

    if not hasattr(module, "checkpoint_mode"):
        raise RuntimeError("Modules have a 'checkpoint_mode' class property (ex, \"min\").")


def strip_and_warn_kwargs(_class: type, params: dict):
    """
    Strips any parameters not suitable for the specified class constructor,
    warning the user about each.

    Parameters
    ----------
    _class : type
        Some class to check the parameters.
    params : dict
        The keyword arguments to pass to the class constructor.
    """

    # Remove unusable keyword arguments and warn the user
    signature = inspect.signature(_class.__init__)
    invalid_kwargs = [k for k in params if k not in signature.parameters]
    for k in invalid_kwargs:
        print(
            f"[WARN] Constructor for {_class.__qualname__} does not contain '{k}', "
            "but it was specified!"
        )
        params.pop(k)
