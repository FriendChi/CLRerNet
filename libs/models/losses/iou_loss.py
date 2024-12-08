import torch
from mmdet.models.builder import LOSSES


@LOSSES.register_module
class CLRNetIoULoss(torch.nn.Module):
    def __init__(self, loss_weight=1.0, lane_width=15 / 800):
        """
        行 IoU 损失函数，用于车道线的预测任务。
        Args:
            loss_weight (float): 损失权重，用于调节 IoU 损失在整体损失中的权重。
            lane_width (float): 虚拟车道线的半宽（相对坐标）。
        """
        super(CLRNetIoULoss, self).__init__()
        self.loss_weight = loss_weight  # 损失权重
        self.lane_width = lane_width  # 车道半宽


    def calc_iou(self, pred, target, pred_width, target_width):
        """
        计算预测与目标之间的行 IoU 值。
        Args:
            pred: 预测车道线，形状为 (Nl, Nr)，相对坐标。
            target: 目标车道线，形状为 (Nl, Nr)，相对坐标。
            pred_width: 预测线条的虚拟宽度，形状 (Nl, Nr)。
            target_width: 目标线条的虚拟宽度，形状 (Nl, Nr)。
        Returns:
            torch.Tensor: 计算出的 IoU 值，形状为 (N)。
        """
        px1 = pred - pred_width  # 预测线的左边界
        px2 = pred + pred_width  # 预测线的右边界
        tx1 = target - target_width  # 目标线的左边界
        tx2 = target + target_width  # 目标线的右边界

        invalid_mask = target  # 标记无效目标点
        # 去除无效点（坐标不在合法范围内）
        invalid_masks = (invalid_mask < 0) | (invalid_mask >= 1.0)
        ovr = torch.min(px2, tx2) - torch.max(px1, tx1)  # 计算交集宽度
        union = torch.max(px2, tx2) - torch.min(px1, tx1)  # 计算并集宽度


        ovr[invalid_masks] = 0.0  # 无效点的交集设置为0
        union[invalid_masks] = 0.0  # 无效点的并集设置为0
        iou = ovr.sum(dim=-1) / (union.sum(dim=-1) + 1e-9)  # 计算 IoU 值
        return iou


    def forward(self, pred, target):
        """
        计算 IoU 损失。
        Args:
            pred: 预测车道线，形状为 (Nl, Nr)。
            target: 目标车道线，形状为 (Nl, Nr)。
        Returns:
            torch.Tensor: 损失值。
        """
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"  # 确保预测和目标形状一致
        width = torch.ones_like(target) * self.lane_width  # 为预测和目标创建统一的虚拟宽度
        iou = self.calc_iou(pred, target, width, width)  # 计算 IoU 值
        return (1 - iou).mean() * self.loss_weight  # 返回 IoU 损失（1 - IoU 平均值）



@LOSSES.register_module
class LaneIoULoss(CLRNetIoULoss):
    def __init__(self, loss_weight=1.0, lane_width=7.5 / 800, img_h=320, img_w=1640):
        """
        LaneIoU 损失函数的实现。
        Args:
            loss_weight (float): 损失权重。
            lane_width (float): 虚拟车道线半宽。
            img_h (int): 图像高度。
            img_w (int): 图像宽度。
        """
        super(LaneIoULoss, self).__init__(loss_weight, lane_width)  # 初始化基类
        self.max_dx = 1e4  # 限制横向偏移的最大值
        self.img_h = img_h  # 图像高度
        self.img_w = img_w  # 图像宽度


    def _calc_lane_width(self, pred, target):
        """
        动态计算车道线的虚拟宽度。
        Args:
            pred: 预测车道线，形状为 (Nl, Nr)。
            target: 目标车道线，形状为 (Nl, Nr)。
        Returns:
            torch.Tensor: 预测和目标线条的虚拟宽度。
        """
        n_strips = pred.shape[1] - 1  # 车道线的网格数
        dy = self.img_h / n_strips * 2  # 网格高度（两格之间）
        
        # 计算预测线宽度
        _pred = pred.clone().detach()  # 分离预测
        pred_dx = (_pred[:, 2:] - _pred[:, :-2]) * self.img_w  # 水平方向差异
        pred_width = self.lane_width * torch.sqrt(pred_dx.pow(2) + dy**2) / dy  # 根据斜边计算宽度 越倾斜，宽度越大
        pred_width = torch.cat([pred_width[:, 0:1], pred_width, pred_width[:, -1:]], dim=1)  # 填补首尾

        # 计算目标线宽度
        target_dx = (target[:, 2:] - target[:, :-2]) * self.img_w  # 水平方向差异
        target_dx[torch.abs(target_dx) > self.max_dx] = 0  # 剔除异常差异
        target_width = self.lane_width * torch.sqrt(target_dx.pow(2) + dy**2) / dy  # 根据斜边计算宽度 
        target_width = torch.cat([target_width[:, 0:1], target_width, target_width[:, -1:]], dim=1)  # 填补首尾

        return pred_width, target_width

    def distance_loss_weight(pred,weight_type="linear",alpha=1):
        # 生成权重（基于采样点的索引）
        N, Nr = pred.shape
        weights = torch.arange(Nr, device=pred.device).float()  # 采样点索引
        
        # 根据不同类型计算权重
        if weight_type == 'linear':
            weights = weights / Nr  # 线性权重
        elif weight_type == 'exponential':
            weights = torch.exp(-alpha * weights)  # 指数加权 较大的 alpha 会加大远处的权重
        elif weight_type == 'polynomial':
            weights = (1 - weights / Nr) ** alpha  # 多项式加权 较大的 alpha 会加大远处的权重
        elif weight_type == 'cosine':
            weights = torch.cos(torch.pi * weights / Nr)  # 余弦加权
        elif weight_type == 'logarithmic':
            weights = torch.log(1 + weights)  # 对数加权
        else:
            raise ValueError("Invalid weight_type. Choose from 'linear', 'exponential', 'polynomial'.")
        
        # 权重归一化
        weights = weights / weights.sum()
        return weights

    def forward(self, pred, target):
        """
        计算动态宽度的 IoU 损失。
        Args:
            pred: 预测车道线，形状为 (Nl, Nr)。
            target: 目标车道线，形状为 (Nl, Nr)。
        Returns:
            torch.Tensor: 损失值。
        """
        assert (
            pred.shape == target.shape
        ), "prediction and target must have the same shape!"  # 确保预测和目标形状一致
        pred_width, target_width = self._calc_lane_width(pred, target)  # 动态计算宽度
        iou = self.calc_iou(pred, target, pred_width, target_width)  # 计算 IoU 值
        distance_loss_weight = self.distance_loss_weight(pred,weight_type="linear")
        return (1 - iou).mean() * self.loss_weight*distance_loss_weight  # 返回 IoU 损失

