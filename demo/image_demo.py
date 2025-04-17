# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import init_detector

from libs.api.inference import inference_one_image
from libs.utils.visualizer import visualize_lanes


def parse_args():
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser = ArgumentParser()
    # 添加必需的参数：图片文件路径
    parser.add_argument('img', help='图像文件')
    # 添加必需的参数：配置文件路径
    parser.add_argument('config', help='配置文件')
    # 添加必需的参数：模型检查点文件路径
    parser.add_argument('checkpoint', help='检查点文件')
    # 添加可选参数：输出文件路径，默认为'result.png'
    parser.add_argument('--out-file', default='result.png', help='输出文件路径')
    # 添加可选参数：推理设备，默认为'cuda:0'（GPU）
    parser.add_argument('--device', default='cuda:0', help='推理使用的设备')
    # 添加可选参数：边界框分数阈值，默认为0.3
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='边界框分数阈值'
    )
    # 解析命令行参数并返回
    args = parser.parse_args()
    return args


def main(args):
    # 根据配置文件和检查点文件构建模型，并指定使用设备
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # 对单张图像进行推理，返回原始图像和预测结果
    src, preds = inference_one_image(model, args.img)
    # 可视化车道线检测结果，并保存到指定路径
    dst = visualize_lanes(src, preds, save_path=args.out_file)


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 调用主函数执行程序
    main(args)
    args = parse_args()
    main(args)
