import torch.nn.functional as F
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSmoothAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(DynamicSmoothAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.mlp = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, 1),
            nn.Sigmoid()  # 将缩放因子限制在 [0, 1] 范围内
        )

    def forward(self, x):
        # 全局平均池化，得到每个通道的平均值
        mu = self.avg_pool(x).squeeze(-1).squeeze(-1)  # [B, C]
        
        # 动态生成缩放因子
        alpha = self.mlp(mu).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 缩小小于平均值的元素
        mask = (x < mu.unsqueeze(-1).unsqueeze(-1)).float()  # 小于平均值的元素掩码
        x_scaled = x * (1 - mask) + x * mask * alpha  # 缩小小于平均值的元素
        
        # 归一化 x_scaled 得到注意力权重
        attention = F.softmax(x_scaled.view(x.size(0), x.size(1), -1), dim=-1)  # [B, C, H*W]
        attention = attention.view_as(x)  # [B, C, H, W]
        
        # 应用注意力权重
        return x * attention  # [B, C, H, W]

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
        self.mlka = DynamicSmoothAttention(prior_feat_channels * refine_layers)

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
