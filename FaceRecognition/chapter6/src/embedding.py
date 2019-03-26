# -*- coding: utf-8 -*-
'''
代码的逻辑就是

1. 先导入模型参数
2. 然后导入两张图片，分别获取其经过模型后得到的128维特征向量
3. 最后计算两个向量的欧氏距离

代码中有几个参数：

- image_size：图片长宽尺寸，这里要求输入的图片是长宽相等的，但是不要求两张人脸图大小一致，这里设置的尺寸是代码中会将人脸图读取后重新拉伸压缩成这个大小，这个尺寸最好比200大，太小了会运行失败
- modeldir：预训练好的模型路径
- image_name1：第一张人脸图的图片名
- image_name2：第二张人脸图的图片名
'''

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet

image_size = 200 #don't need equal to real image size, but this value should not small than this
modeldir = '/home/jxf/Learning_AI/FaceRecognition/models/20170512-110547/20170512-110547.pb' #change to your model dir
image_name1 = '/home/jxf/Learning_AI/FaceRecognition/datasets/casia_mtcnnpy_182/0000105/006_0.png' #change to your image name
image_name2 = '/home/jxf/Learning_AI/FaceRecognition/datasets/casia_mtcnnpy_182/0000107/004_0.png' #change to your image name

print('建立facenet embedding模型')
tf.Graph().as_default()
sess = tf.Session()

facenet.load_model(modeldir)
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

print('facenet embedding模型建立完毕')

scaled_reshape = []

image1 = scipy.misc.imread(image_name1, mode='RGB')
image1 = cv2.resize(image1, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image1 = facenet.prewhiten(image1)
scaled_reshape.append(image1.reshape(-1,image_size,image_size,3))
emb_array1 = np.zeros((1, embedding_size))
emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]

image2 = scipy.misc.imread(image_name2, mode='RGB')
image2 = cv2.resize(image2, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
image2 = facenet.prewhiten(image2)
scaled_reshape.append(image2.reshape(-1,image_size,image_size,3))
emb_array2 = np.zeros((1, embedding_size))
emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]

dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))
print("128维特征向量的欧氏距离：%f "%dist)