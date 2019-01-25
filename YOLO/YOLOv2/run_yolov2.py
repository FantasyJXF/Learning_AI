"""
Demo for yolov2
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
import argparse

from YOLO.YOLOv2.model import darknet
from YOLO.YOLOv2.detect_ops import decode
from YOLO.YOLOv2.utils import preprocess_image, postprocess, draw_detection
from YOLO.YOLOv2.config import anchors, class_names

"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of 'YOLOv2 Object Detection"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--input', type=str, default='../datasets/cat.jpg', help='File path of image to detect', required = True)
    parser.add_argument('--output', type=str, default='result.jpg', help='File path of output image', required = True)

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
def run_yolov2():

    # parse arguments
    args = parse_args()
    if args is None:
        exit()

    input_size = (416, 416)
    image_file = args.input
    image = cv2.imread(image_file)
    image_shape = image.shape[:2]
    image_cp = preprocess_image(image, input_size)
    """
    image = Image.open(image_file)
    image_cp = image.resize(input_size, Image.BICUBIC)
    image_cp = np.array(image_cp, dtype=np.float32)/255.0
    image_cp = np.expand_dims(image_cp, 0)
    #print(image_cp)
    """

    images = tf.placeholder(tf.float32, [1, input_size[0], input_size[1], 3])
    detection_feat = darknet(images)
    feat_sizes = input_size[0] // 32, input_size[1] // 32
    detection_results = decode(detection_feat, feat_sizes, len(class_names), anchors)

    # download URL : https://pan.baidu.com/s/1mrM95_wz6LTvIOZBHDxsWQ
    checkpoint_path = "./checkpoint_dir/yolo2_coco.ckpt"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        bboxes, obj_probs, class_probs = sess.run(detection_results, feed_dict={images: image_cp})

    bboxes, scores, class_inds = postprocess(bboxes, obj_probs, class_probs,
                                            image_shape=image_shape)
    img_detection = draw_detection(image, bboxes, scores, class_inds, class_names)
    #cv2.imwrite(args.output, img_detection)
    cv2.imshow("detection results", img_detection)

    cv2.waitKey(0)

if __name__ == '__main__':
    run_yolov2()