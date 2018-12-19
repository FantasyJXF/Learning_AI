# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
from skimage import filters,io
#import matplotlib.pyplot as plt

OUTPUT_DIR = "../outputs/Char_Image_Binary"

def Feature5Extraction():
    with open("../outputs/feature5.txt", 'w+') as fd:
        for k in range(0, 1000):
            path = os.path.join(OUTPUT_DIR, str(k + 1) + '.jpg')
            image = io.imread(path)
            thresh = filters.threshold_otsu(image)
            dst = (image > thresh) * 1
            a, b = dst.shape
            size_ = a + b
            # 定义特征向量
            C = np.zeros(size_, dtype=np.int)

            for i in range(a):
                for j in range(b - 1):
                    if dst[i, j] != dst[i, j + 1] and dst[i, j + 1] == 1:
                        C[i] += 1

            for j in range(b):
                for i in range(a - 1):
                    if dst[i, j] != dst[i + 1, j] and dst[i + 1, j] == 1:
                        C[a + j] += 1

            fd.write(str(k + 1) + '\t')
            for i in range(0, size_ - 1):
                fd.write(str(C[i]) + ',')
            if k < 999:
                fd.write(str(C[size_ - 1]) + '\n')
            else:
                fd.write(str(C[size_ - 1]))

if __name__ == '__main__':
    Feature5Extraction()