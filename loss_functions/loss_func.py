# coding: utf-8
#
# I have seen a few different mean squared error loss functions in various posts for regression models in Tensorflow:
# https://stackoverflow.com/questions/41338509/tensorflow-mean-squared-error-loss-function

# loss = tf.reduce_sum(tf.pow(prediction - Y,2))/(n_instances)
# loss = tf.reduce_mean(tf.squared_difference(prediction, Y))
# loss = tf.nn.l2_loss(prediction - Y)
#
# What are the differences between these?


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def test1():
    shape_obj = (5, 5)
    shape_obj = (100, 6, 12)  # 三维的数据   [12个[6个[100个]]]
    Y1 = tf.random_normal(shape=shape_obj)  # 正态分布的随机数填充
    Y2 = tf.random_normal(shape=shape_obj)
    print Y1  # Tensor("random_normal:0", shape=(100, 6, 12), dtype=float32)
    print Y2  # Tensor("random_normal_1:0", shape=(100, 6, 12), dtype=float32)

    print reduce(lambda x, y: x * y, shape_obj)  # 100 * 12 * 6 = 7200个元素
    loss1 = tf.reduce_sum(tf.pow(Y1 - Y2, 2)) / (reduce(lambda x, y: x * y, shape_obj))  # 均方误差
    loss2 = tf.reduce_mean(tf.squared_difference(Y1, Y2))  # (Y1-Y2)*(Y1-Y2)
    loss3 = tf.nn.l2_loss(Y1 - Y2)  # output = sum(t ** 2) / 2

    with tf.Session() as sess:
        print sess.run(Y1)
        print sess.run(Y2)
        print sess.run([loss1, loss2, loss3])  # [1.9610037, 1.9610037, 7059.6133]
        # from result, we can see
        # tf.pow(Y1 - Y2, 2) == tf.squared_difference(Y1, Y2)
        # tf.reduce_sum()/number_of_element == reduce_mean


def test2():  # 通过例子2可以看到 虽然两个方法代表一个东西，但是通过填充0和1使计算误差扩大，最后的结果差距也是很大的
    shape_obj = (5000, 5000, 10)
    Y1 = tf.zeros(shape=shape_obj)
    Y2 = tf.ones(shape=shape_obj)

    loss1 = tf.reduce_sum(tf.pow(Y1 - Y2, 2)) / (reduce(lambda x, y: x * y, shape_obj))
    loss2 = tf.reduce_mean(tf.squared_difference(Y1, Y2))

    with tf.Session() as sess:
        # print sess.run(Y1)
        # print sess.run(Y2)
        print sess.run([loss1, loss2])  # [1.0, 0.26843545]


if __name__ == "__main__":
    # test1()
    test2()
