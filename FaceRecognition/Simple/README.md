# 使用TensorFlow训练人脸识别模型

## 目录介绍
> * `lwf_dataset`: LWF带标签的人脸数据集
> * `my_faces`: 我的照片
> * `other_faces`: 从LWF数据集中提取出的人脸图像
> * `tmp`: 保存模型的训练结果
> * `get_my_faces.py`: 通过摄像头获取本人脸的照片
> * `is_my_face.py`: 判断是否是本人的脸
> * `set_other_people.py`: 从lwf_dataset中提取出人脸
> * `train_faces.py`: 人脸训练过程

**说明：**

* 程序中使用的是`dlib`来识别人脸部分
* LFW数据集网站：http://vis-www.cs.umass.edu/lfw/
* 使用CNN网络模型
* 模型识别过程：运行`is_my_face.py`程序，让摄像头拍到本人即可