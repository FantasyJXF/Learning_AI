# !/usr/bin/env python
# -*- coding: utf-8 -*-

#import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split

filename = "../datasets/Char_Index.txt"

def SVM_Training():
    X = features_collection()
    y_ = labels_extraction(filename)

    # 分开训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y_, test_size=0.3, random_state=0)

    # 设置模型1
    svc = SVC(kernel='poly', degree=3, gamma=1, coef0=0)
    # 训练模型
    clf = svc.fit(X_train, y_train)
    # 计算测试集精度
    score = clf.score(X_test, y_test)
    print('多项式核模型精度为%s' % score)

    # 设置模型2
    svc2 = SVC(kernel='rbf', degree=2, gamma=1.7)
    # 训练模型
    clf2 = svc2.fit(X_train, y_train)
    # 计算测试集精度
    score2 = clf2.score(X_test, y_test)
    print('RBF核模型精度为%s' % score2)

def features_collection():
    feature1 = feature_extraction("../outputs/feature1.txt")
    feature2 = feature_extraction("../outputs/feature2.txt")
    feature3 = feature_extraction("../outputs/feature3.txt")
    feature4 = feature_extraction("../outputs/feature4.txt")
    feature5 = feature_extraction("../outputs/feature5.txt")
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