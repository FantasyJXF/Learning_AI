# !/usr/bin/env python
# -*- coding: utf-8 -*-

from SVM_Plate.svm_char_proj import PreProcessing, PreProcessingToCorrect, \
                      Feature1Extraction, Feature2Extraction,\
                      Feature3Extraction, Feature4Extraction,\
                      Feature5Extraction

def main(argv=None):
    PreProcessing.PreProcessing()  # 图像预处理,把要处理的图像转成二值图
    PreProcessingToCorrect.PreProcessingToCorrect()  # 对没有正确转化的图像进行手工校正
    Feature1Extraction.Feature1Extraction()    # 提取特征1,为每一行和每一列的白点数
    Feature2Extraction.Feature2Extraction()    # 提取特征2, 为区域密度,区域大小为8*8
    Feature3Extraction.Feature3Extraction()    # 提取特征3, 为区域密度,区域大小为4*4
    Feature4Extraction.Feature4Extraction()    # 提取特征4, 为区域密度,区域大小为6*6
    Feature5Extraction.Feature5Extraction()    # 提取特征5, 为每一行和每一列的线段数目
    # Feature6Extraction.Feature6Extraction()    # 提取特征6, 为字符左右上下与边界的距离

if __name__ == "__main__":
    main()