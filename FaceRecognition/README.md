# Face Recognition

人脸识别主要分为人脸验证(1:1)和人脸识别(1:N).

关键点在于损失函数的定义,通过深度神经网络将人脸图像映射到为一个特征向量(embedding),训练的目的是使得相同人脸特征向量欧式距离近,不同人脸欧式距离远.

* 三元组损失 Triplet Loss

* 中心损失 Center Loss

* ArcLoss(STOA)

## 参考

### Papers

* 人脸检测
  * MTCNN - Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks
  * WIDER FACE - A Face Detection Benchmark
  * Landmarks - Deep learning face attributes in the wild
* 人脸识别
  * FaceNet - A Unified Embedding for Face Recognition and Clustering
  * Sphereface - Deep hypersphere embedding for face recognition
  * CosFace - Large Margin Cosine Loss for Deep Face Recognition
  * AM-Softmax - Additive margin softmax for face verification
  * Center Loss - A Discriminative Feature Learning Approach for Deep Face Recognition
  * ArcFace - Additive Angular Margin Loss for Deep Face Recognition