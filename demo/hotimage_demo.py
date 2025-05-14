# based on https://github.com/open-mmlab/mmdetection/blob/v2.28.0/demo/image_demo.py
# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet.apis import init_detector

from libs.api.inference import inference_one_hotimage
from libs.utils.visualizer import visualize_lanes
import os

def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    # parser.add_argument('--out-file', default='result.png', help='Path to output file')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold'
    )
    args = parser.parse_args()
    return args

# python demo/hotimage_demo.py configs/clrernet/culane/clrernet_culane_dla34_ema.py work_dirs47/epoch_50.pth
# python demo/hotimage_demo.py configs/clrernet/culane/clrernet_culane_dla34.py /work/work_dirs45/latest.pth
def main(args):
    base_dir = '/work/dataset/culane/'
    txt_path = '/work/dataset/culane/list/test_split/test6_curve.txt'
    # txt_path = '/work/dataset/culane/list/test_split/test8_night.txt'
    out_dir = '/work/result/'
    # 获取文件名部分（不包括路径）
    filename = os.path.basename(txt_path)

    # 去掉'.txt'后缀并根据'_'分割字符串
    parts = filename.rstrip('.txt').split('_')

    # 提取目标字符串，这里假设目标字符串是最后一个'_'后的部分
    target_str = parts[-1]
    with open(txt_path, 'r') as file:
        path_list = [line.strip() for line in file]
    model = init_detector(args.config, args.checkpoint, device=args.device)
    start = 0
    end = 100
    for index, path in enumerate(path_list[start:end], start=start):
        out_file_path = out_dir +target_str+'_'+str(index)+'.png'
        img_path = base_dir+path
        args.img = img_path
        args.out_file_path = out_file_path
        
        # test a single image
        src, preds = inference_one_hotimage(model, args)
        print(out_file_path,args.img)


if __name__ == '__main__':
    args = parse_args()
    main(args)
