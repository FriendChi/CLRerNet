import torch.nn.functional as F
from torch import nn

import typing as t

import torch
import torch.nn as nn
from einops import rearrange


import math
import torch
import torch.nn as nn

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SegDecoder(nn.Module):
    """
    Segmentation decoder head for auxiliary loss.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/seg_decoder.py
    """

    def __init__(
        self,
        image_height,
        image_width,
        num_classes,
        prior_feat_channels=64,
        refine_layers=3,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(0.1)
        self.conv = nn.Conv2d(prior_feat_channels * refine_layers, num_classes, 1)
        self.image_height = image_height
        self.image_width = image_width
        self.att = eca_layer(prior_feat_channels * refine_layers)

    def forward(self, x):
        x = self.dropout(x)
        x = self.att(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=[self.image_height, self.image_width],
            mode="bilinear",
            align_corners=False,
        )
        return x
