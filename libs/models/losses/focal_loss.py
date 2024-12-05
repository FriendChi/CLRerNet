# Adapted from: https://github.com/lucastabelini/LaneATT/blob/main/lib/focal_loss.py

# pylint: disable-all
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from torch import Tensor
# Source: https://github.com/kornia/kornia/blob/f4f70fefb63287f72bc80cd96df9c061b1cb60dd/kornia/losses/focal.py


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    """
    将整数标签 tensor 转换为 one-hot 编码 tensor。
    """
    if not torch.is_tensor(labels):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(type(labels))
        )  # 检查输入是否为张量
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(labels.dtype)
        )  # 检查标签数据类型是否为 int64
    if num_classes < 1:
        raise ValueError(
            "The number of classes must be bigger than one."
            " Got: {}".format(num_classes)
        )  # 确保类数大于1

    shape = labels.shape  # 获取标签的形状
    one_hot = torch.zeros(
        shape[0], num_classes, *shape[1:], device=device, dtype=dtype
    )  # 创建一个全零的 one-hot 编码张量
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps  # 填充 one-hot 编码，避免数值问题添加 `eps`



def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = "none",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    计算 Focal Loss，用于解决类别不平衡问题。
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(input)))
    if not len(input.shape) >= 2:
        raise ValueError(
            "Invalid input shape, we expect BxCx*. Got: {}".format(input.shape)
        )  # 确保输入的维度 >= 2
    if input.size(0) != target.size(0):
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(
                input.size(0), target.size(0)
            )
        )  # 确保输入和目标的 batch size 一致
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(
            "Expected target size {}, got {}".format(input.size()[2:], target.size())
        )  # 确保输入和目标形状匹配
    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".format(
                input.device, target.device
            )
        )  # 确保输入和目标在相同设备上

    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps  # 对输入进行 softmax 转换
    target_one_hot: torch.Tensor = one_hot(
        target, num_classes=input.shape[1], device=input.device, dtype=input.dtype
    )  # 将目标标签转换为 one-hot 编码

    weight = torch.pow(-input_soft + 1.0, gamma)  # 计算焦点因子 (1 - p_t)^gamma
    focal = -alpha * weight * torch.log(input_soft)  # 计算 Focal Loss 的每个元素
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)  # 按类别加权计算损失

    if reduction == "none":
        loss = loss_tmp  # 不进行损失归约
    elif reduction == "mean":
        loss = torch.mean(loss_tmp)  # 求平均损失
    elif reduction == "sum":
        loss = torch.sum(loss_tmp)  # 求损失总和
    else:
        raise NotImplementedError("Invalid reduction mode: {}".format(reduction))  # 不支持的归约模式
    return loss

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: Tensor = None,
                 pos_weight: Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot
        return

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise 
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

@LOSSES.register_module
class KorniaFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float,  # Focal Loss 的 alpha 参数，用于控制类间权重
        gamma: float = 2.0,  # Focal Loss 的 gamma 参数，控制难易样本的权重
        loss_weight: float = 1.0,  # 损失权重
        reduction: str = "none",  # 损失归约方式：'none'、'mean' 或 'sum'
    ) -> None:
        super(KorniaFocalLoss, self).__init__()
        # self.alpha: float = alpha
        # self.gamma: float = gamma
        # self.reduction: str = reduction
        # self.eps: float = 1e-6  # 防止数值稳定性问题的微小常数
        self.loss_weight = loss_weight  # 最终损失的加权

        self.pll = Poly1FocalLoss(num_classes=5, epsilon=1.0, alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(  # 前向传播方法
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Focal Loss。
        """
        return (
            self.pll(input,target)
            * self.loss_weight
        )  # 调用 focal_loss 函数并乘以损失权重
