import copy
import numpy as np
import cv2

GT_COLOR = (255, 0, 0)
PRED_HIT_COLOR = (0, 255, 0)
PRED_MISS_COLOR = (0, 0, 255)


def draw_lane(lane, img=None, img_shape=None, width=30, color=(255, 255, 255)):
    """
    在图像上绘制车道线。
    参数:
        lane (np.ndarray): 单条车道线的N个(x, y)坐标，形状为(N, 2)。
        img (np.ndarray): 源图像。
        img_shape (tuple): 当img为None时使用的空白图像形状。
        width (int): 车道线的宽度（粗细）。
        color (tuple): 车道线的颜色（BGR格式）。
    返回:
        img (np.ndarray): 绘制了车道线的输出图像。
    """
    # 如果没有提供图像，则创建一个空白图像
    if img is None:
        img = np.zeros(img_shape, dtype=np.uint8)
    # 将车道线坐标转换为整数类型
    lane = lane.astype(np.int32)
    # 遍历车道线上的相邻点对，并绘制线段
    for p1, p2 in zip(lane[:-1], lane[1:]):
        cv2.line(img, tuple(p1), tuple(p2), color, thickness=width)
    return img


def visualize_lanes(
    src,
    preds,
    annos=list(),
    pred_ious=None,
    iou_thr=0.5,
    concat_src=False,
    save_path=None,
):
    """
    可视化预测结果和真实标签中的车道线标记。
    参数:
        src (np.ndarray): 源图像。
        preds (List[np.ndarray]): 车道线预测结果。
        annos (List[np.ndarray]): 真实标签中的车道线标注。
        pred_ious (List[np.ndarray]): 预测与真实标签的IoU值。
        iou_thr (float): 预测与真实标签匹配的IoU阈值。
        concat_src (bool): 是否将原始图像与绘制结果垂直拼接。
        save_path (str): 输出图像文件路径。
    返回:
        dst (np.ndarray): 输出图像。
    """
    # 创建源图像的深拷贝以避免修改原图
    dst = copy.deepcopy(src)
    
    # 绘制真实标签中的车道线（使用GT_COLOR颜色）
    for anno in annos:
        dst = draw_lane(anno, dst, dst.shape, width=4, color=GT_COLOR)
    
    # 判断预测结果是否达到IoU阈值
    if pred_ious is None:
        hits = [True for _ in range(len(preds))]  # 如果没有提供IoU值，默认全部命中
    else:
        hits = [iou > iou_thr for iou in pred_ious]  # 根据IoU值判断是否命中
    
    # 绘制预测结果中的车道线（根据命中情况选择颜色）
    for pred, hit in zip(preds, hits):
        color = PRED_HIT_COLOR if hit else PRED_MISS_COLOR  # 命中或未命中的颜色
        dst = draw_lane(pred, dst, dst.shape, width=4, color=color)
    
    # 如果需要，将原始图像与绘制结果垂直拼接
    if concat_src:
        dst = np.concatenate((src, dst), axis=0)
    
    # 如果提供了保存路径，则将结果图像保存到文件
    if save_path:
        cv2.imwrite(save_path, dst)
    
    return dst
