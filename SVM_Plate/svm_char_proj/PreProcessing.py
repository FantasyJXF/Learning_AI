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

def PreProcessing():
    input1 = []
    input2 = []
    input3 = []
    with open(filename, 'r') as file_to_read:
        while True:
            # 整行读取数据
            lines = file_to_read.readline()
            if not lines:
                break
                pass
            # 将整行数据分割处理，这里分隔符为空格
            tmp1, tmp2, tmp3 = [i for i in lines.split()]
            input1.append(tmp1)
            input2.append(tmp2)
            input3.append(tmp3)
        input1 = np.array(input1[1:])
        input2 = np.array(input2[1:])
        input3 = np.array(input3[1:])
    indexFileName = input3
    input_y = input2

    for k in range(0, len(indexFileName)):
        path = os.path.join(INPUT_DIR, indexFileName[k])
        image = io.imread(path, as_gray=True)
        # image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        thresh = filters.threshold_otsu(image)
        # 乘以1.0是因为将True,False转换为浮点数
        dst = (image > thresh) * 1
        a, b = dst.shape
        # # 绘图显示
        # plt.figure("thresh", figsize=(8,8))
        #
        # plt.subplot(121)
        # plt.title("Original")
        # plt.imshow(image, plt.cm.gray)
        #
        # plt.subplot(122)
        # plt.title("binary")
        # # 0表示黑,1表示白
        # plt.imshow(dst, plt.cm.gray)
        # plt.show()

        # 四个角的二值化数据大于2,白底
        if (dst[0, 0] + dst[0, b - 1] + dst[a - 1, 0] + dst[a - 1, b - 1] >= 2):
            for i in range(0, a):
                for j in range(0, b):
                    # 将每一个像素的值反向,转换成黑底白字
                    dst[i, j] = 1 - dst[i, j]
        path_out = os.path.join(OUTPUT_DIR, str(k + 1) + ".jpg")
        dst = dst.astype(np.uint8) * 255
        io.imsave(path_out, dst)
        # io.imshow(dst)

if __name__ == '__main__':
    PreProcessing()