# modified based on:
# https://github.com/open-mmlab/mmdetection/blob/v2.28.0/mmdet/apis/inference.py
# Copyright (c) OpenMMLab. All rights reserved.

import cv2
import torch
from mmcv.parallel import collate, scatter

from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp


def inference_one_image(model, img_path):
    """对单张图像进行推理。
    参数:
        model (nn.Module): 加载好的检测模型。
        img_path (str): 图像文件路径。
    返回:
        img (np.ndarray): 图像数据，形状为(宽, 高, 通道)。
        preds (List[np.ndarray]): 检测到的车道线。
    """
    # 读取图像文件
    img = cv2.imread(img_path)
    ori_shape = img.shape  # 获取原始图像的形状

    # 构建输入数据字典
    data = dict(
        filename=img_path,  # 文件名
        sub_img_name=None,  # 子图像名称（无）
        img=img,  # 图像数据
        gt_points=[],  # 真实标注点（无）
        id_classes=[],  # 类别ID（无）
        id_instances=[],  # 实例ID（无）
        img_shape=ori_shape,  # 图像形状
        ori_shape=ori_shape,  # 原始图像形状
    )

    # 获取模型配置
    cfg = model.cfg
    # 设置模型头部的测试配置，关闭车道线模式
    model.bbox_head.test_cfg.as_lanes = False
    # 获取模型所在的设备（GPU或CPU）
    device = next(model.parameters()).device

    # 构建测试流水线
    test_pipeline = Compose(cfg.data.test.pipeline)
    # 对输入数据进行预处理
    data = test_pipeline(data)
    # 数据打包成批次格式
    data = collate([data], samples_per_gpu=1)

    # 提取图像元信息和图像数据
    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]

    # 如果模型在GPU上，则将数据分散到指定设备
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    # 推理模型
    with torch.no_grad():  # 关闭梯度计算
        results = model(return_loss=False, rescale=True, **data)

    # 获取车道线检测结果
    lanes = results[0]['result']['lanes']
    # 处理车道线预测结果
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])

    return img, preds


def get_prediction(lanes, ori_h, ori_w):
    """从检测到的车道线中提取预测结果。
    参数:
        lanes: 检测到的车道线。
        ori_h: 原始图像的高度。
        ori_w: 原始图像的宽度。
    返回:
        preds: 处理后的车道线预测结果。
    """
    preds = []
    for lane in lanes:
        # 将车道线数据转换为NumPy数组
        lane = lane.cpu().numpy()
        xs = lane[:, 0]  # 车道线的x坐标
        ys = lane[:, 1]  # 车道线的y坐标

        # 过滤掉无效的x坐标（不在[0, 1)范围内的值）
        valid_mask = (xs >= 0) & (xs < 1)
        xs = xs * ori_w  # 将x坐标映射回原始图像宽度
        lane_xs = xs[valid_mask]  # 过滤后的x坐标
        lane_ys = ys[valid_mask] * ori_h  # 将y坐标映射回原始图像高度

        # 将坐标反转（从底部到顶部排序）
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]

        # 将车道线点表示为(x, y)对
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]

        # 对车道线点进行插值处理，生成更平滑的预测结果
        interp_pred = interp(pred, n=5)
        preds.append(interp_pred)

    return preds
