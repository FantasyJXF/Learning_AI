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

input1 = []
input2 = []
input3 = []
#Efield = []
with open(filename, 'r') as  file_to_read:
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

for k in range(0, len(indexFileName)):
    path = os.path.join(INPUT_DIR, indexFileName[k])
    image = io.imread(path, as_grey=True)
    #image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    thresh = filters.threshold_otsu(image)
    dst = (image > thresh) * 1.0
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
    # plt.imshow(dst, plt.cm.gray)
    # plt.show()

    if (dst[0,0] + dst[0, b-1] + dst[a-1, 0] + dst[a-1,b-1] >= 2):
        for i in range(0,a):
            for j in range(0,b):
                dst[i,j] = 1 - dst[i,j]
    path_out = os.path.join(OUTPUT_DIR, str(k+1)+".jpg")
    io.imsave(path_out, dst)

