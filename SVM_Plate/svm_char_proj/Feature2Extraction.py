# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
from skimage import filters,io
#import matplotlib.pyplot as plt

OUTPUT_DIR = "Char_Image_Binary"
square8 = 8

def Feature2Extraction():
    with open("feature2.txt", 'w+') as fd:
        for k in range(0, 1000):
            path = os.path.join(OUTPUT_DIR, str(k + 1) + '.jpg')
            image = io.imread(path)
            thresh = filters.threshold_otsu(image)
            dst = (image > thresh) * 1
            a, b = dst.shape
            bh = math.ceil(a / square8)
            bw = math.ceil(b / square8)
            size_ = bh * bw
            # 定义特征向量
            C = np.zeros(size_, dtype=np.int)

            for i in range(0, a):
                x = math.floor(i / 8.0)
                for j in range(0, b):
                    if dst[i, j] == 1:
                        y = math.floor(j / 8.0)
                        C[bw * x + y] += 1

            fd.write(str(k + 1) + '\t')
            for i in range(0, size_ - 1):
                fd.write(str(C[i]) + ',')
            if k < 999:
                fd.write(str(C[size_ - 1]) + '\n')
            else:
                fd.write(str(C[size_ - 1]))

if __name__ == '__main__':
    Feature2Extraction()