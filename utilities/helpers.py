import math
import os
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric

STAGE_TRAIN = "train"
STAGE_VALIDATION = "val"
STAGE_TEST = "test"


def add_stage_metrics(obj, attribute: str, metric: Metric):
    """
    Adds `STAGE_TRAIN`, `STAGE_VALIDATION` and `STAGE_TEST` copies of the specified metric.
    """

    for stage in [STAGE_TRAIN, STAGE_VALIDATION, STAGE_TEST]:
        setattr(obj, f"{stage}_{attribute}", metric)  # ex: train_iou
        metric = deepcopy(metric)  # each stage has unique state


def get_stage_metric(obj, stage: str, name: str) -> Metric:
    """
    Gets the stage relevant metric object for the specified object.
    """
    return getattr(obj, f"{stage}_{name}")  # ex: train_iou


def compute_stage_metric(obj, stage: str, attribute: str, *args, **kwargs):
    """
    Computes the aggregate result of the stage relevant metric and resets it.
    """

    metric = get_stage_metric(obj, stage, attribute)
    result = metric.compute(*args, **kwargs)
    metric.reset()
    return result


def tensor_pad(x, stride):
    # https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch

    h, w = x.shape[-2:]

    # ...
    mod_h = h % stride
    mod_w = w % stride

    # ...
    new_w = (w + stride - mod_w) if mod_w > 0 else (w)
    new_h = (h + stride - mod_h) if mod_h > 0 else (h)

    # ...
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = torch.nn.functional.pad(x, pads, "reflect")

    return out, pads


def tensor_unpad(x, pad):
    # https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch

    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]: -pad[3], :]

    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]: -pad[1]]

    return x


def is_rank_zero():
    """
    Determines if this process is the "main" process to help do things only once
    instead of doing once per process in multi-gpu (ie, ddp) configurations.
    """

    # Get Ranks (ie, which GPU/Process/Node index is currently executing w/ DDP)
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    GLOBAL_RANK = int(os.getenv("GLOBAL_RANK", "0"))

    return GLOBAL_RANK == 0 and LOCAL_RANK == 0


def print_if_rank_zero(*args, **kwargs):
    """
    Prints only if on the main process (rank zero).
    """

    if is_rank_zero():
        print(*args, **kwargs)


def get_sufficient_num_workers():
    """
    Compute an okay number of workers processes to use.
    """

    return max(1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count()))


def linear_warmup_cosine_decay(
        optimizer: Optimizer,
        max_epochs: int,
        warmup_epochs: int,
        min_multiplier: float,
):
    # https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b
    # https://arxiv.org/abs/2103.12682 metions it
    # Cosine Decay LR

    PI = 3.14159265

    def to_min(x):
        m = min_multiplier
        return m + (x * (1.0 - m))

    def lr_func(e):
        if e < warmup_epochs:
            # Ramp up (0.0 to 1.0)
            return to_min(e / warmup_epochs)
        else:
            # Ramp down with cosine (1.0 to 0.0) until max epochs
            time = (e - warmup_epochs) / (max_epochs - warmup_epochs)
            return to_min((math.cos(time * PI) + 1.0) / 2.0)

    return LambdaLR(optimizer, lr_func)
