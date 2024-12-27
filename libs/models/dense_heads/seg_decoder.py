import torch.nn.functional as F
from torch import nn
import torch

class SegDecoder(nn.Module):
    """
    Segmentation decoder head for auxiliary loss.
    Adapted from:
    https://github.com/Turoad/CLRNet/blob/main/clrnet/models/utils/seg_decoder.py
    """

    def __init__(
        self,
        image_height,  # 输入图像的高度
        image_width,   # 输入图像的宽度
        num_classes,   # 类别数  4 lanes + 1 background
        prior_feat_channels=64,  # 特征通道数，默认为 64
        refine_layers=3,         # 细化层数，默认为 3
    ):
        super().__init__()
        # self.dropout = nn.Dropout2d(0.1)  # 空间维度上的 Dropout 操作，概率为 0.1
        self.conv = nn.Conv2d(
            prior_feat_channels * refine_layers,  # 输入通道数 = 特征通道数 × 细化层数
            num_classes,  # 输出通道数 = 类别数
            1  # 卷积核大小为 1 × 1
        )

        self.conv1 = nn.Conv2d(
        in_channels=prior_feat_channels*2,
        out_channels=prior_feat_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
        )
        self.conv2 = nn.Conv2d(
        in_channels=prior_feat_channels*2,
        out_channels=prior_feat_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
        )
        self.image_height = image_height  # 保存输入图像高度
        self.image_width = image_width    # 保存输入图像宽度

    def forward(self, batch_features):
        batch_features[1] = torch.cat((F.interpolate(batch_features[0], size=batch_features[1].shape[2:], mode='nearest'),batch_features[1]),dim=1)
        batch_features[1] = self.conv1(batch_features[1])
        batch_features[2] = torch.cat((F.interpolate(batch_features[1], size=batch_features[2].shape[2:], mode='nearest'),batch_features[2]),dim=1)
        batch_features[2] = self.conv2(batch_features[2])        
        
        x = self.conv(batch_features[2])     # 通过 1 × 1 卷积生成类别概率图
        x = F.interpolate(
            x,
            size=[self.image_height, self.image_width],  # 上采样到输入图像的大小
            mode="bilinear",  # 双线性插值
            align_corners=False,  # 不对齐角点
        )
        return x  # 返回上采样后的结果

