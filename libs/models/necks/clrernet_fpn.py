"""
Adapted from:
https://github.com/Turoad/CLRNet/blob/main/clrnet/models/necks/fpn.py
"""


import torch.nn as nn
import torch.nn.functional as F
import torch
from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS

class SFM(nn.Module):
    def __init__(self, in_channels):
        """
        原版 SFM 模块，增加特征尺寸对齐功能。
        Args:
            in_channels: 输入特征图的通道数（假设每层输入的通道数相同）。
        """
        super(SFM, self).__init__()
        # 通道加权系数生成
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)  # 通道降维并生成权重
        self.sigmoid = nn.Sigmoid()  # 激活函数，确保权重范围为 [0, 1]

    def forward(self, *features):
        """
        Args:
            *features: 可变数量的输入特征图（来自不同层）。
        Returns:
            融合后的特征图。
        """
        # 获取基准特征的尺寸（以第一个特征为基准）
        base_size = features[0].shape[2:]  # (H, W)
        
        aligned_features = []  # 存储对齐后的特征
        for feature in features:
            current_size = feature.shape[2:]  # 当前特征的尺寸
            if current_size < base_size:  # 如果尺寸小于基准尺寸
                # 上采样到基准尺寸
                feature = F.interpolate(feature, size=base_size, mode='nearest')
            elif current_size > base_size:  # 如果尺寸大于基准尺寸
                # 下采样到基准尺寸
                feature = F.adaptive_max_pool2d(feature, output_size=base_size)
            aligned_features.append(feature)
        
        # 初始化融合特征
        fused_feature = 0
        for feature in aligned_features:
            # 计算通道权重
            attention = self.avg_pool(feature)  # 全局池化 (B, C, 1, 1)
            attention = self.sigmoid(self.fc(attention))  # (B, 1, 1, 1)
            
            # 权重加权后相加
            fused_feature += feature * attention
        
        return fused_feature
 



@NECKS.register_module
class CLRerNetFPN(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        """
        Feature pyramid network for CLRerNet.
        Args:
            in_channels (List[int]): Channel number list.
            out_channels (int): Number of output feature map channels.
            num_outs (int): Number of output feature map levels.
        """
        super(CLRerNetFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.backbone_end_level = self.num_ins
        self.start_level = 0
        self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()

        self.sfm1 = SFM(out_channels)
        self.sfm2 = SFM(out_channels)
        self.sfm3 = SFM(out_channels)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                inplace=False,
            )
            # fpn_conv = ConvModule(
            #     out_channels,
            #     out_channels,
            #     3,
            #     padding=1,
            #     conv_cfg=None,
            #     norm_cfg=None,
            #     act_cfg=None,
            #     inplace=False,
            # )

            self.lateral_convs.append(l_conv)
            # self.fpn_convs.append(fpn_conv)

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
        if type(inputs) == tuple:
            inputs = list(inputs)

        assert len(inputs) >= len(self.in_channels)  # 4 > 3

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        outs = [l.clone() for l in laterals]

        outs[1] = self.sfm1(laterals[1], laterals[0],laterals[2])
        outs[2] = self.sfm2(laterals[2], outs[1])
        outs[0] = self.sfm3(laterals[0], outs[1])

        # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     prev_shape = laterals[i - 1].shape[2:]
        #     laterals[i - 1] += F.interpolate(
        #         laterals[i], size=prev_shape, mode='nearest'
        #     )

        # outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        return tuple(outs)
