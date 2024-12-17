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


    def calc_iou(self, pred, target, pred_width, target_width,smooth_type='log',alpha=1.0):
        """
        计算带平滑距离惩罚的 DIoU 损失，同时移除无效点的影响。
        Args:
            pred: 预测车道线，形状为 (Nl, Nr)。
            target: 目标车道线，形状为 (Nl, Nr)。
            pred_width: 预测线条的虚拟宽度，形状 (Nl, Nr)。
            target_width: 目标线条的虚拟宽度，形状 (Nl, Nr)。
            alpha: 平滑系数。
            smooth_type: 平滑方式，可选 'log', 'sqrt' 或其他。
        Returns:
            torch.Tensor: 计算出的 DIoU 损失。
        """
        # 计算边界框
        px1 = pred - pred_width
        px2 = pred + pred_width
        tx1 = target - target_width
        tx2 = target + target_width

        # 标记无效点
        invalid_mask = (target < 0) | (target >= 1.0)

        # IoU 计算
        ovr = torch.clamp(torch.min(px2, tx2) - torch.max(px1, tx1), min=0.0)
        union = torch.clamp(torch.max(px2, tx2) - torch.min(px1, tx1), min=1e-9)
        iou = ovr.sum(dim=-1) / union.sum(dim=-1)
        iou[invalid_mask.all(dim=-1)] = 0.0  # 移除无效行的 IoU 影响

        # 中心点距离
        center_pred = (px1 + px2) / 2
        center_target = (tx1 + tx2) / 2
        center_distance = ((center_pred - center_target).pow(2).sum(dim=-1))
        center_distance[invalid_mask.all(dim=-1)] = 0.0  # 无效行设置为 0

        # 包围框距离
        enclosing_left = torch.min(px1, tx1)
        enclosing_right = torch.max(px2, tx2)
        enclosing_distance = ((enclosing_right - enclosing_left).pow(2).sum(dim=-1) + 1e-9)
        enclosing_distance[invalid_mask.all(dim=-1)] = 1.0  # 避免除以 0

        # 平滑距离
        if smooth_type == 'log':
            smooth_distance = torch.log(1 + (center_distance / enclosing_distance))
        elif smooth_type == 'sqrt':
            smooth_distance = torch.sqrt(center_distance / enclosing_distance + 1e-6)
        else:
            raise ValueError("Unsupported smooth type")

        # DIoU 计算
        diou = iou - alpha * smooth_distance
        diou[invalid_mask.all(dim=-1)] = 0.0  # 无效行的 DIoU 设置为 0

        return diou


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
        return (1 - iou).mean() * self.loss_weight  # 返回 IoU 损失

