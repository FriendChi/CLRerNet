# modified based on:
# https://github.com/open-mmlab/mmdetection/blob/v2.28.0/mmdet/apis/inference.py
# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import cv2
import torch
from mmcv.parallel import collate, scatter

from libs.datasets.pipelines import Compose
from libs.datasets.metrics.culane_metric import interp

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import matplotlib.pyplot as plt
# python demo/image_demo.py demo/demo.jpg  configs/clrernet/culane/clrernet_culane_dla34_ema.py work_dirs47/epoch_50.pth --out-file=result.png
def inference_one_image(model, img_path):
    """Inference on an image with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img_path (str): Image path.
    Returns:
        img (np.ndarray): Image data with shape (width, height, channel).
        preds (List[np.ndarray]): Detected lanes.
    """
    img = cv2.imread(img_path)
    ori_shape = img.shape
    data = dict(
        filename=img_path,
        sub_img_name=None,
        img=img,
        gt_points=[],
        id_classes=[],
        id_instances=[],
        img_shape=ori_shape,
        ori_shape=ori_shape,
    )

    cfg = model.cfg
    model.bbox_head.test_cfg.as_lanes = False
    device = next(model.parameters()).device  # model device

    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    data['img_metas'] = data['img_metas'].data[0]
    data['img'] = data['img'].data[0]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    # [多个形状不一样的tensor]
    lanes = results[0]['result']['lanes']
    # print(lanes)
    # print('lanes.shape:',lanes)
    preds = get_prediction(lanes, ori_shape[0], ori_shape[1])

    # Grad CAM ======================================================
    class MMLaneWrapper(torch.nn.Module):
        def __init__(self, model,img_meta = data['img_metas']):
            super().__init__()
            self.model = model
            self.img_meta = img_meta

        def forward(self, x):
            # with torch.no_grad():
            model.eval()
            d = {'img_metas':self.img_meta,'img':x}
            results = model(return_loss=False, rescale=True, **d)
            # x 就是 input_tensor，也就是 data['img']
            return results[0]['result']['lanes'][0]

    
    m = MMLaneWrapper(model)

    class LanePointsTarget:
        def __init__(self):
            """
            lane_points: Tensor of shape [n, 2], representing points on a lane.
            """


        def __call__(self, model_output):
            # 假设 model_output 是一个列表，每个元素对应一条车道线的输出
            # 我们可以根据需要选择适当的处理方式
            # 例如，返回第一个车道线的所有点的和
            return sum([model_output[i].sum() for i in range(len(model_output))])


    # cam = GradCAM(model=m, target_layers=[m.model.backbone.model.level5])
    cam = GradCAM(model=m, target_layers=[m.model.neck.fpn_convs[0]])
    
    targets = [LanePointsTarget()]     
    # targets = None

    grayscale_cam = cam(input_tensor=data['img'], targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(data['img'][0].detach().cpu().permute(1,2,0).numpy().astype(dtype=np.float32),
                                      grayscale_cam, use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('/work/result/cam_result.png', dpi=300, bbox_inches='tight', pad_inches=0)


    return img, preds


def get_prediction(lanes, ori_h, ori_w):
    """
    根据输入的车道线数据（lanes），生成预测结果。
    
    参数：
    - lanes: 包含多个车道线的列表，每个车道线是一个 Tensor。
    - ori_h: 原始图像的高度。
    - ori_w: 原始图像的宽度。
    
    返回：
    - preds: 处理后的车道线预测结果列表，每个车道线是一组插值点。
    """
    preds = []  # 用于存储处理后的车道线预测结果

    for lane in lanes:
        # print(lane.shape)
        # 将 Tensor 转换为 NumPy 数组，并确保其位于 CPU 上
        lane = lane.cpu().numpy()

        # 提取车道线的 x 和 y 坐标
        xs = lane[:, 0]  # 车道线的 x 坐标
        ys = lane[:, 1]  # 车道线的 y 坐标

        # 筛选出有效的 x 坐标（范围在 [0, 1) 内）
        valid_mask = (xs >= 0) & (xs < 1)

        # 将 x 坐标从归一化范围 [0, 1) 映射到原始图像宽度
        xs = xs * ori_w

        # 筛选出有效的 x 和 y 坐标
        lane_xs = xs[valid_mask]  # 有效的 x 坐标
        lane_ys = ys[valid_mask] * ori_h  # 将 y 坐标映射到原始图像高度

        # 将坐标反转顺序（从大到小排列）
        lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]

        # 将 x 和 y 坐标组合成点的列表
        pred = [(x, y) for x, y in zip(lane_xs, lane_ys)]

        # 对点进行插值处理，生成更平滑的车道线
        interp_pred = interp(pred, n=5)  # 插值函数，n 表示插值点的数量

        # 将插值后的车道线加入结果列表
        preds.append(interp_pred)

    return preds  # 返回所有车道线的预测结果

'''

CLRerNet(
  (backbone): DLANet(
    (model): DLA(
      (base_layer): Sequential(
        (0): Conv2d(3, 16, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (level0): Sequential(
        (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (level1): Sequential(
        (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (level2): Tree(
        (tree1): BasicBlock(
          (conv1): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (tree2): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (root): Root(
          (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (project): Sequential(
          (0): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (level3): Tree(
        (tree1): Tree(
          (tree1): BasicBlock(
            (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (tree2): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (root): Root(
            (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (project): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (tree2): Tree(
          (tree1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (tree2): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (root): Root(
            (conv): Conv2d(448, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (level4): Tree(
        (tree1): Tree(
          (tree1): BasicBlock(
            (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (tree2): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (root): Root(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (project): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (tree2): Tree(
          (tree1): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (tree2): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
          (root): Root(
            (conv): Conv2d(896, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (level5): Tree(
        (tree1): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (tree2): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (root): Root(
          (conv): Conv2d(1280, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
        )
        (downsample): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (project): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
    )
  )
  (neck): CLRerNetFPN(
    (lateral_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ConvModule(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ConvModule(
        (conv): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (fpn_convs): ModuleList(
      (0): ConvModule(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ConvModule(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ConvModule(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
    )
  )
  (bbox_head): CLRerHead(
    (anchor_generator): CLRerNetAnchorGenerator(
      (prior_embeddings): Embedding(192, 3)
    )
    (attention): ROIGather(
      (attention): AnchorVecFeatureMapAttention(
        (resize): FeatureResize()
        (f_key): ConvModule(
          (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
        (f_query): Sequential(
          (0): Conv1d(192, 192, kernel_size=(1,), stride=(1,), groups=192)
          (1): ReLU()
        )
        (f_value): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (W): Conv1d(192, 192, kernel_size=(1,), stride=(1,), groups=192)
      )
      (convs): ModuleList(
        (0): ConvModule(
          (conv): Conv2d(64, 48, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
        (1): ConvModule(
          (conv): Conv2d(64, 48, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
        (2): ConvModule(
          (conv): Conv2d(64, 48, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
          (bn): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
      )
      (catconv): ModuleList(
        (0): ConvModule(
          (conv): Conv2d(48, 64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
        (1): ConvModule(
          (conv): Conv2d(96, 64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
        (2): ConvModule(
          (conv): Conv2d(144, 64, kernel_size=(9, 1), stride=(1, 1), padding=(4, 0), bias=False)
          (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (activate): ReLU(inplace=True)
        )
      )
      (fc): Linear(in_features=2304, out_features=64, bias=True)
      (fc_norm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
    )
    (loss_cls): KorniaFocalLoss()
    (loss_bbox): SmoothL1Loss()
    (loss_seg): CLRNetSegLoss(
      (criterion): NLLLoss()
    )
    (loss_iou): LaneIoULoss()
    (reg_modules): ModuleList(
      (0): Linear(in_features=64, out_features=64, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU(inplace=True)
    )
    (cls_modules): ModuleList(
      (0): Linear(in_features=64, out_features=64, bias=True)
      (1): ReLU(inplace=True)
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): ReLU(inplace=True)
    )
    (reg_layers): Linear(in_features=64, out_features=76, bias=True)
    (cls_layers): Linear(in_features=64, out_features=2, bias=True)
    (seg_decoder): SegDecoder(
      (dropout): Dropout2d(p=0.1, inplace=False)
      (conv): Conv2d(192, 5, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)

'''
