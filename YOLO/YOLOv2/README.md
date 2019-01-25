# YOLO v2

> Refer to
> * \<YOLO9000:Better,Faster,Stronger\>
> * [小白将的知乎](https://zhuanlan.zhihu.com/p/35325884)
> * [KOD-chen的GitHub](https://github.com/KOD-Chen/YOLOv2-Tensorflow)

## 目录说明

* `config.py` ---- Anchors & Class names
* `detect_ops.py` ---- 从模型输出提取bbox回归、物体分类等信息
* `loss.py`  ---- 计算YOLOv2的损失函数
* `model.py` ---- YOLOv2的网络模型,DARKNET-19
* `run_yolov2.py` ---- YOLOv2演示程序
* `utils.py` ---- 功能函数，包含预处理输入图片，筛选边界框NMS，绘制筛选后的边界框

> 执行检测的指令为：
> ```
> python run_yolov2.py --input file_path --output file_path
> ```