import torch.nn.functional as F
from torch import nn

import typing as t

import torch
import torch.nn as nn
from einops import rearrange


import math
import torch
import torch.nn as nn

class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        # AvgPool（全局平均池化）：
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        # StdPool（全局标准池化）
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        # CFC（全连接层）
        z = self.cfc(u)  # (b, c, 1)
        # BN（归一化）
        z = self.bn(z)
        # Sigmoid
        g = torch.sigmoid(z)


        g = g.view(b, c, 1, 1)
        return x * g.expand_as(x)



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
        self.mlka = SRMLayer(prior_feat_channels * refine_layers)

    def forward(self, x):
        x = self.dropout(x)
        x = self.mlka(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=[self.image_height, self.image_width],
            mode="bilinear",
            align_corners=False,
        )
        return x
