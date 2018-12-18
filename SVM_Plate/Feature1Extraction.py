# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import numpy as np
from skimage import filters,io
import matplotlib.pyplot as plt

OUTPUT_DIR = "Char_Image_Binary"

def Feature1Extraction():
    with open("feature1.txt", 'w+') as fd:
        for k in range(0, 1000):
            path = os.path.join(OUTPUT_DIR, str(k + 1) + '.jpg')
            image = io.imread(path)
            thresh = filters.threshold_otsu(image)
            dst = (image > thresh) * 1
            a, b = dst.shape
            C = np.zeros(a + b, dtype=np.int)

            # 每一行
            for i in range(0, a):
                for j in range(0, b):
                    if (dst[i, j] == 1):
                        C[i] += 1
            # 每一列
            for j in range(0, b):
                for i in range(0, a):
                    if (dst[i, j] == 1):
                        C[a + j] += 1

            fd.write(str(k + 1) + '\n')
            for i in range(0, a + b - 1):
                fd.write(str(C[i]) + ',')
            if k < 999:
                fd.write(str(C[a + b - 1]) + '\n')
            else:
                fd.write(str(C[a + b - 1]))
