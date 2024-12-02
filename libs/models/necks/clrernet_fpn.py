"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/necks/fpn.py
"""


import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS
import torch

@NECKS.register_module
class CLRerNetFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        """
        Feature pyramid network for CLRerNet.
        Args:
            in_channels (List[int]): Channel number list. 输入的特征图通道数列表。
            out_channels (int): Number of output feature map channels. 输出特征图的通道数。
            num_outs (int): Number of output feature map levels. 输出特征图的层数。
        """
        super(CLRerNetFPN, self).__init__()  # 调用父类的初始化方法，初始化 nn.Module
        assert isinstance(in_channels, list)  # 确保 in_channels 是一个列表
        self.in_channels = in_channels  # 保存输入通道列表
        self.out_channels = out_channels  # 保存输出通道数
        self.num_ins = len(in_channels)  # 输入的特征图数量（等于 in_channels 的长度）
        self.num_outs = num_outs  # 输出特征图的层数

        self.backbone_end_level = self.num_ins  # 设置骨干网络的结束层（即输入的层数）
        self.start_level = 0  # 设置起始层，通常为 0
        self.lateral_convs = nn.ModuleList()  # 用于存储 lateral 卷积层的列表
        self.fpn_convs = nn.ModuleList()  # 用于存储 FPN 卷积层的列表

        # 初始化 lateral 卷积和 FPN 卷积层
        for i in range(self.start_level, self.backbone_end_level):
            #横向卷积层,1*1卷积用于保持通道统一
            l_conv = ConvModule(
                in_channels[i],  # 输入通道数
                out_channels,  # 输出通道数
                1,  # 卷积核大小为 1
                conv_cfg=None,  # 卷积配置（未指定）
                norm_cfg=None,  # 归一化配置（未指定）
                act_cfg=None,  # 激活函数配置（未指定）
                inplace=False,
            )
            # FPN 卷积层，处理后的通道和尺寸都不变
            fpn_conv = ConvModule(
                out_channels*2,  # 输入通道数为输出通道数
                out_channels,  # 输出通道数
                3,  # 卷积核大小为 3
                padding=1,  # padding 为 1
                conv_cfg=None,  # 卷积配置（未指定）
                norm_cfg=None,  # 归一化配置（未指定）
                act_cfg=None,  # 激活函数配置（未指定）
                inplace=False,
            )

            self.lateral_convs.append(l_conv)  # 将 lateral 卷积层添加到列表中
            self.fpn_convs.append(fpn_conv)  # 将 FPN 卷积层添加到列表中

    def forward(self, inputs):
        """
        Args:
            inputs (List[torch.Tensor]): Input feature maps.
              Example of shapes:
                ([1, 64, 80, 200], [1, 128, 40, 100], [1, 256, 20, 50], [1, 512, 10, 25]).
        Returns:
            outputs (Tuple[torch.Tensor]): Output feature maps.
              The number of feature map levels and channels correspond to
               `num_outs` and `out_channels` respectively.
              Example of shapes:
                ([1, 64, 40, 100], [1, 64, 20, 50], [1, 64, 10, 25]).
        """
        if type(inputs) == tuple:  # 如果输入是 tuple 类型，将其转换为 list 类型
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)  # 确保输入的特征图数量不小于 in_channels 的长度

        if len(inputs) > len(self.in_channels):  # 如果输入的特征图数量大于 in_channels 的长度
            for _ in range(len(inputs) - len(self.in_channels)):  # 删除多余的输入特征图
                del inputs[0]

        # 构建 lateral 卷积层的输出
        laterals = [
            lateral_conv(inputs[i + self.start_level])  # 通过 lateral 卷积对每个输入特征图进行处理
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # 构建自顶向下的路径
        used_backbone_levels = len(laterals)  # 使用的骨干网络层数，即 lateral 卷积层的数量
        for i in range(used_backbone_levels - 1, 0, -1):  # 从最后一层往前遍历
            prev_shape = laterals[i - 1].shape[2:]  # 获取上一层的空间维度（不包括 batch 和通道）
            laterals[i - 1] = torch.cat([laterals[i - 1], F.interpolate(laterals[i], size=prev_shape, mode='nearest'  )], dim=1) 

        # 对每一层的特征图进行 FPN 卷积处理
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        
        # 返回每一层的输出特征图
        return tuple(outs)

