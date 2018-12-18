# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from skimage import filters,io
import matplotlib.pyplot as plt

filename = "Char_Index.txt"
INPUT_DIR = "Char_Image"
OUTPUT_DIR = "Char_Image_Binary"

def PreProcessingToCorrect():
    input1_ = []
    input2_ = []
    with open("Char_Index_Err.txt") as file_idx_err:
        while True:
            # 整行读取数据
            lines = file_idx_err.readline()
            if not lines:
                break
            # 将整行数据分割处理，这里分隔符为空格
            tmp1, tmp2 = [i for i in lines.split()]
            input1_.append(tmp1)
            input2_.append(tmp2)

        input1_ = np.array(input1_[1:])
        input2_ = np.array(input2_[1:])
        IndexFileName_Err = input2_

    for k in range(0, len(IndexFileName_Err)):
        path_e = os.path.join(OUTPUT_DIR, IndexFileName_Err[k])
        img_e = io.imread(path_e)
        a, b = img_e.shape
        dst = np.zeros((a,b), dtype=np.uint8)
        for i in range(0, a):
            for j in range(0, b):
                dst[i, j] = 255 - img_e[i, j]
        io.imsave(path_e, dst)
        # io.imshow(img_e)