# !/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.svm import SVC

filename = "Char_Index.txt"

def SVM_Training():
    X = features_collection()
    y_ = labels_extraction(filename)
    pass


def features_collection():
    feature1 = feature_extraction("feature1.txt")
    feature2 = feature_extraction("feature2.txt")
    feature3 = feature_extraction("feature3.txt")
    feature4 = feature_extraction("feature4.txt")
    feature5 = feature_extraction("feature5.txt")
    features = np.append(
                np.append(
                    np.append(
                        np.append(feature1,
                            feature2, axis=1),
                            feature3, axis=1),
                            feature4, axis=1),
                            feature5, axis=1)
    return features

def labels_extraction(file_name):
    labels = []
    with open(file_name, 'r') as fp:
        while True:
            # 整行读取数据
            line = fp.readline()
            if not line:
                break
            # 将整行数据分割处理，这里分隔符为空格
            tmp1, tmp2, tmp3 = [i for i in line.split()]
            labels.append(tmp2)
    return np.array(labels[1:])

def feature_extraction(feature_file):
    feature = []
    with open(feature_file, 'r') as fp:
        for line in fp:
            linearr = line.strip().split('\t')
            feature.append(linearr[1].split(','))
    return np.array(feature)

if __name__ == "__main__":
    SVM_Training()