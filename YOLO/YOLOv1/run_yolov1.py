# !/usr/bin/env python
# -*- coding=utf-8 -*-
"""
Demo for yolov1
"""

import os
import cv2
import argparse
from YOLO.YOLOv1.yolo_v1 import Yolo

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'YOLOv1 Object Detection"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--input', type=str, default='../datasets/car.jpg', help='File path of image to detect', required = False)
    parser.add_argument('--output', type=str, default='detected_image.jpg', help='File path of output image', required = False)
    parser.add_argument('--verbose', type=str, default=True, help='Verbose', required = False)

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    try:
        assert os.path.exists(args.input)
    except:
        print('There is no such file%s'%args.input)
        return None

    return args


"""main"""
def run_yolov1():

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    yolo_net = Yolo("./weights/YOLO_small.ckpt", args.verbose)
    yolo_net.detect_from_file(args.input, detected_image_file=args.output)

if __name__ == '__main__':
    run_yolov1()
