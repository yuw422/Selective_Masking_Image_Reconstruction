import numpy as np
import torch
from typing import Any
from utilities.helpers import STAGE_TEST
from pytorch_msssim import SSIM
from augmentations.reconstruction import RecRandomMask
from models.reconstruction import Reconstruction
import config
from utilities.augmentations import inverse_normalize


class ReconstructionSelective(Reconstruction):
    identifier = "reconstruction_selective"

    def __init__(self, underlying_model: str, underlying_model_args: dict):
        super().__init__(underlying_model, underlying_model_args)
        self.criterion_ssim = SSIM(data_range=1.0, size_average=True, channel=3, win_size=3)
        self.mean = np.array(config.AUGMENTATION_ARGS["normalization"]["mean"]) * 255

    def _get_tile_loss_map(self, z, y, tile_locs):
        """
        calculate combo loss for tiles, return list of loss in same order of tile list
        """
        loss_map = []
        for l in tile_locs:
            # img[locs[i][0]:locs[i][1], locs[i][2]:locs[i][3], :]
            # torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 256, 256])
            z_tile = z[:, :, l[0]:l[1], l[2]:l[3]]
            y_tile = y[:, :, l[0]:l[1], l[2]:l[3]]
            # print(l)
            # print(z_tile.shape, y_tile.shape)
            ssim = self.criterion_ssim(z_tile, y_tile)
            loss = self.criterion_loss(z_tile, y_tile)
            # Compute combo loss (equal weighting)
            combo_loss = loss + (1.0 - ssim)
            loss_map.append(combo_loss.item())
            # print(f"SSIM {ssim}, L1 {loss}, Loss {combo_loss}")
            # input("STOP CHECK")
        return loss_map

    def _mask_image(self, x, loss_map, tile_locs):
        # loss_map numpy array of (sample,)
        # tile_locs list of (sample)
        # x is tensor, have shape [3, h, w]
        img = self.convert_numpy(inverse_normalize(x, **config.AUGMENTATION_ARGS["normalization"]))
        mask_list = np.argpartition(loss_map, -int(loss_map.shape[0] * 0.5))[-int(loss_map.shape[0] * 0.5):]
        # print(len(mask_list))
        # input()
        for m_idx in mask_list:
            img[tile_locs[m_idx][0]:tile_locs[m_idx][1], tile_locs[m_idx][2]:tile_locs[m_idx][3], :] = self.mean
        return img

    def _inverse_nomalize(self, x):
        # x is tensor, have shape [c, h, w]
        img = self.convert_numpy(inverse_normalize(x, **config.AUGMENTATION_ARGS["normalization"]))
        # print(f"inverse norm in shape {x.shape}")
        return img
