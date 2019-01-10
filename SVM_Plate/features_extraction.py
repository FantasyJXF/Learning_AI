# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os

import sys

import math

import numpy as np

from skimage import io,filters

import matplotlib.pyplot as plt

def features_extraction(path):
    image = io.imread(path)
    thresh = filters.threshold_otsu(image)
    dst = (image > thresh) * 1
    a, b = dst.shape
    
    '''
    Feature1
    '''
    C1 = np.zeros(a + b, dtype=np.int)
    # 每一行
    for i in range(0, a):
        for j in range(0, b):
            if (dst[i, j] == 1):
                C1[i] += 1
    # 每一列
    for j in range(0, b):
        for i in range(0, a):
            if (dst[i, j] == 1):
                C1[a + j] += 1

        
    '''
    Feature2
    '''
    bh2 = math.ceil(a / 8)
    bw2 = math.ceil(b / 8)
    size2 = bh2 * bw2
    # 定义特征向量
    C2 = np.zeros(size2, dtype=np.int)
    for i in range(0, a):
        x = math.floor(i / 8.0)
        for j in range(0, b):
            if dst[i, j] == 1:
                y = math.floor(j / 8.0)
                C2[bw2 * x + y] += 1

        
    '''
    Feature3
    '''    
    bh3 = math.ceil(a / 4)
    bw3 = math.ceil(b / 4)
    size3 = bh3 * bw3
    # 定义特征向量
    C3 = np.zeros(size3, dtype=np.int)
    for i in range(0, a):
        x = math.floor(i / 8.0)
        for j in range(0, b):
            if dst[i, j] == 1:
                y = math.floor(j / 8.0)
                C3[bw3 * x + y] += 1
 
        
    '''
    Feature4
    '''
    bh4 = math.ceil(a / 6)
    bw4 = math.ceil(b / 6)
    size4 = bh4 * bw4
    # 定义特征向量
    C4 = np.zeros(size4, dtype=np.int)
    for i in range(0, a):
        x = math.floor(i / 8.0)
        for j in range(0, b):
            if dst[i, j] == 1:
                y = math.floor(j / 8.0)
                C4[bw4 * x + y] += 1

        
    '''
    Feature5
    '''
    size5 = a + b
    # 定义特征向量
    C5 = np.zeros(size5, dtype=np.int)
    for i in range(a):
        for j in range(b - 1):
            if dst[i, j] != dst[i, j + 1] and dst[i, j + 1] == 1:
                C5[i] += 1
    for j in range(b):
        for i in range(a - 1):
            if dst[i, j] != dst[i + 1, j] and dst[i + 1, j] == 1:
                C5[a + j] += 1

    features = np.concatenate((C1, C2, C3, C4, C5)).reshape(1, -1)
    return features
