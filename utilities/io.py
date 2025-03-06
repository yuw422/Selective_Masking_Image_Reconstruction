import os
from typing import Any, Optional

import numpy as np
import yaml
from PIL import Image

import config


def load_yaml(path: str):
    """
    Loads the specified YAML document.

    Parameters
    ----------
    path : str
        Some path to a YAML document.

    Returns
    -------
    dict
        The dict representation of the YAML document.
    """

    if os.path.isfile(path):
        with open(path, "r") as file:
            return yaml.safe_load(file)
    else:
        return None


def save_yaml(path: str, data: Any):
    """
    Writes the specified dict to disk as a YAML document.

    Parameters
    ----------
    path : str
        Some path to write the document to disk.
    data : dict
        Some dict to write to disk.
    """

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    with open(path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)


def get_image_size(path: str):
    """
    Loads the specified image (without decoding content) and returns size (H,W).
    """

    with Image.open(path) as im:
        return (im.size[1], im.size[0])


def load_image(path: str, dtype: Optional[np.dtype] = None, convert: Optional[str] = None):
    """
    Loads the specified Image and returns as a numpy array.
    """

    im = Image.open(path)
    if convert is not None:
        im = im.convert(convert)  # type: ignore

    return np.array(im, dtype)


def save_image(path: str, data: np.ndarray):
    """
    Writes the specified (H,W,C) formatted numpy array as an image to disk.
    """

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    im = Image.fromarray(data)
    im.save(path, quality=95)


def expand_path(path: str, relative: bool = False):
    """
    Resolves symbols and variables in the path and returns either the absolute
    or relative path.

    Parameters
    ----------
    path : str
        Some path to a file.
    relative : bool, optional
        If the computed path should be returned relative to the current directory, by default False

    Returns
    -------
    str
        The computed path.
    """

    # Resolve path symbols and variables (ex, '~/project/blah.txt')
    path = os.path.expandvars(path)
    path = os.path.expanduser(path)

    # Get the full complete path (ex, '/home/user/project/blah.txt')
    path = os.path.realpath(path)

    if relative:
        # Computes the relative path (ex, 'blah.txt')
        path = os.path.relpath(path)

    return path


def resolve_checkpoint_path(hint: str) -> Optional[str]:
    """
    Attempts to resolve the checkpoint specified by the hint.

    The hint can be:
    - A full path to the desired .ckpt file.
    - A relative path to config.CHECKPOINT_DIRECTORY.
    - A short form of "name/version" resolved in config.CHECKPOINT_DIRECTORY.
    """

    # Attempt to resolve if the path directly specifies a checkpoint
    if hint.endswith(".ckpt"):

        # Full path to a checkpoint file
        if os.path.exists(hint):
            return hint

        # Resolve relative to the checkpoint directory
        checkpoint_path = os.path.join(config.CHECKPOINT_DIRECTORY, hint)
        if os.path.exists(checkpoint_path):
            return checkpoint_path

    # Attempt to resolve short form "name/version" into full checkpoint path
    checkpoint_directory = os.path.join(config.CHECKPOINT_DIRECTORY, hint, "checkpoints")
    if os.path.exists(checkpoint_directory):

        # Get checkpoint file
        checkpoints = set(os.listdir(checkpoint_directory))
        checkpoints.remove("last.ckpt")  # we wan't the top-1 checkpoint

        # Get the first checkpoint (should be )
        if len(checkpoints) == 1:
            checkpoint = list(checkpoints)[0]
            return os.path.join(checkpoint_directory, checkpoint)
        else:
            raise Exception(
                "Unable to resolve appropriate checkpoint! "
                f"More than one top-k checkpoints exist in '{checkpoint_directory}'"
            )

    # Unable to resolve checkpoint
    return None
