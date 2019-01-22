# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_infrence.py和mnist_train.py中定义的常量和函数
import mnist_lenet_inference
import mnist_lenet_train

DATA_SRC = "../../datasets/mnist"

# 每10秒加载一次最新的模型,并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出placeholder
        x = tf.placeholder(tf.float32, [
                None,
                mnist_lenet_inference.IMAGE_SIZE,
                mnist_lenet_inference.IMAGE_SIZE,
                mnist_lenet_inference.NUM_CHANNELS],
            name='x-input')
        y_ = tf.placeholder(tf.float32, 
                [None, mnist_lenet_inference.OUTPUT_NODE], 
            name='y-input')

        xs, ys = mnist.validation.images, mnist.validation.labels
        reshaped_xs = np.reshape(xs, (-1, 
                                      mnist_lenet_inference.IMAGE_SIZE,
                                      mnist_lenet_inference.IMAGE_SIZE,
                                      mnist_lenet_inference.NUM_CHANNELS))
        validate_feed = {x: reshaped_xs, y_: ys}

        # 测试时不关注正则化损失的值
        y = mnist_lenet_inference.inference(x, False, None)

        # 使用前向传播的结果计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_lenet_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中的正确率的变化
        while True:
            with tf.Session() as sess:
                # 通过checkpoint文件自动找到目录张最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(mnist_lenet_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.all_model_checkpoint_paths:
                    for path in ckpt.all_model_checkpoint_paths:
                        # 加载模型
                        saver.restore(sess, path)
                        # 通过文件名得到模型保存时迭代的轮数
                        global_step = path.split('/')[-1].split('-')[-1]
                        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                        print("After %s train steps, validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint found")
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets(DATA_SRC, one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()

