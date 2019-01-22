# -*- coding: utf-8 -*-
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from TensorFlow_Practices.tf_mnist import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99

DATA_SRC = "../../datasets/mnist"
MODEL_SAVE_PATH = "../models"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 定义输入输出placeholder
    with tf.name_scope("input"):
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    #  直接使用mnist_inference.py中定义的传播过程
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("moving_average"):
        # 定义损失函数,学习率,滑动平均操作以及训练过程
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        # 将EMA应用到所有可训练的参数
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.name_scope("loss_function"):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses"))

    with tf.name_scope("train_step"):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # 初始化TensroFlow持久化类
        saver = tf.train.Saver()

        # 初始化TensroBoard
        writer = tf.summary.FileWriter("tensorboard_log", tf.get_default_graph())

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # 在训练的过程中不在测试模型在验证数据上的表现,验证和测试的过程将会有一个独立的程序来完成
            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)

                # 每1000轮保存一次模型
                if i % 1000 == 0:
                    # 配置运行时需要记录的信息
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    # 运行时记录运行信息的proto
                    run_metadata = tf.RunMetadata()
                    # 将配置信息和记录运行信息的proto传入运行的过程,从而记录运行时的每一个节点的时间,空间和开销
                    _, loss_value, step = sess.run([train_op, loss, global_step],
                                                   feed_dict={x: xs, y_: ys},
                                                   options=run_options,
                                                   run_metadata=run_metadata)
                    # 将节点在运行时的信息写入日志文件
                    writer.add_run_metadata(run_metadata, tag=("tag%d" % i), global_step=i)
                    saver.save(
                        sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                        global_step=global_step)
                    # 输出当前的训练情况
                    print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                else:
                    _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
        writer.close()

def main(argv=None):
    mnist = input_data.read_data_sets(DATA_SRC, one_hot=True)
    train(mnist)


if __name__ == "__main__":
    tf.app.run()


