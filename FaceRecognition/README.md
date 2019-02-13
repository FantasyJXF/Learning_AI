# Face Recognition

人脸识别主要分为人脸验证(1:1)和人脸识别(1:N).

关键点在于损失函数的定义,通过深度神经网络将人脸图像映射到为一个特征向量(embedding),训练的目的是使得相同人脸特征向量欧式距离近,不同人脸欧式距离远.

* 三元组损失 Triplet Loss

* 中心损失 Center Loss


## 参考

* Papers
  * MTCNN_paper
  * FaceNet - A Unified Embedding for Face Recognition and Clustering
  * A Discriminative Feature Learning Approach for Deep Face Recognition