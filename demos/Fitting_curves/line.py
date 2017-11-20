# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math


def fitting_line():
    # 构建点x和标签y
    num_points = 1000
    vectors_set = []
    for i in xrange(num_points):
        x1 = np.random.normal(0.0, 0.55)  # 高斯分布
        y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
        vectors_set.append([x1, y1])
    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]

    # 画出原始图
    plt.plot(x_data, y_data, 'ro', label='Original data')
    plt.legend()
    plt.show()

    # 第一步，根据数据，选择一个模型。这里正太分布的数据，选择一条直线做拟合
    # 构建y=w*x+b，并用梯度下降法优化w和b，并输出最终的w与b的值
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # shape(1,), [-1, 1)
    b = tf.Variable(tf.zeros([1]))  # [0]
    # init = tf.global_variables_initializer()  # Returns an Op that initializes global variables.
    # sess = tf.Session()
    # sess.run(init)
    # print(sess.run(W))
    # print(sess.run(b))
    y = W * x_data + b  # 模型

    # 用损失函数确定误差
    loss = tf.reduce_mean(tf.square(y - y_data))  # square 平方  这个loss是均方误差
    # 等价于 tf.reduce_sum(tf.pow(y - y_data, 2)) / 1000 (1000个点)  这个就是均方误差的方程式了

    global_step = tf.Variable(0, trainable=False)
    initial_learning_rate = 0.1  # 初始学习率  下面是tf提供随着训练自动慢慢减小learning_rate的方法

    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=100000,
                                               decay_rate=0.96,
                                               staircase=True)

    # 设定梯度下降的学习率
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # 0.5 is learning_rate  可以调整看结果差距
    # 降低损失
    train = optimizer.minimize(loss,
                               global_step=global_step)  # minimize方法会调用compute_gradients()和apply_gradients()方法 也就是计算梯度 和 应用梯度

    learning_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in xrange(51):  # 基于上次训练结果继续训练  增大后可以看到拟合程度变高
        sess.run(train)  # 训练
    print(
        step, sess.run(W),
        sess.run(b))  # (50, array([ 0.08307622], dtype=float32), array([ 0.29846427], dtype=float32))

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.legend()
    plt.show()


def fitting_others():
    x = np.random.rand(100).astype(np.float32)
    y = 3 * x * x + 1
    plt.plot(x, y, 'ro')
    plt.legend()
    plt.show()
    w1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    w2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    w3 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y_ = w1 * x + w2 * x * x + w3 * x * x * x + b
    loss = tf.reduce_mean(tf.square(y - y_))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)
    for step in xrange(100):
        sess.run(train)
    # print step, sess.run(w1), sess.run(w2), sess.run(w3), sess.run(b)

    plt.plot(x, y, 'ro')
    x = np.linspace(0, 1, 100)
    y__ = sess.run(w1) * x + sess.run(w2) * x * x + sess.run(w3) * x * x * x + sess.run(b)
    plt.plot(x, y__, 'b')
    plt.legend()
    plt.show()


def computeYbyX(x):
    noise = np.random.normal(-100, 100, x.shape)  # 正态分布产生噪点
    # normal(
    #   loc:float 此概率分布的均值 （对应着整个分布的中心centre）
    #   scale：float 此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
    #   size：int or tuple of ints 输出的shape，默认为None，只输出一个值
    # )
    return 400 * np.sin(x)  # + 2 * x * x + noise  # 用它模拟要生成的曲线


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 从截断的正态分布里输出随机值  标准差0.1
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def two_layers():  # 用到2层网络
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    hidden_dim = 400

    xTrain = np.linspace(-20, 20, hidden_dim + 1).reshape([1, -1])  # 在[-20, 20]区间内取值 401个点， reshape变成 1行 n列的数组
    # noise = np.random.normal(-0.2, 0.2, xTrain.shape)
    yTrain = computeYbyX(xTrain)  # 得到对应X的Y

    # save train image
    # plt.clf()
    # plt.plot(xTrain[0], yTrain[0], 'ro', label='train data')
    # plt.legend()
    # plt.savefig('curve_data.png', dpi=200)

    x = tf.placeholder(tf.float32, [1, hidden_dim + 1])  # 创建一个[1, 401] 的 x， 使用前需要以 feed_dict={x: rand_array} 形式填充数据

    w = weight_variable([hidden_dim, 1])
    b = bias_variable([hidden_dim, 1])

    w2 = weight_variable([1, hidden_dim])
    b2 = bias_variable([1])

    w3 = weight_variable([hidden_dim + 1, hidden_dim + 1])  # 401*401
    b3 = bias_variable([1, hidden_dim + 1])  # 1*401
    # y = tf.matmul(x, w3) + b3 # 单层网络

    # 第一层网络，tf.matmul(w, x) + b
    # w是400*1的矩阵，x是1*401的矩阵，填充xTrain训练数据，b是400*1的增量  这一层就是常见的线性模型y=wx+b
    # sigmoid是下一层网络的激活函数，sigmoid(y)
    hidden = tf.nn.sigmoid(tf.matmul(w, x) + b)  # 400*401
    # hidden = tf.matmul(w, x) + b; # 不用sigmoid函数 结果是直线

    # 第二层网络  两层的好处是对w,b的初值设定没那么严格
    # y = tf.matmul(w2, hidden) + b2  # 1*400 × 400*401 + 1*1 -> 1*401

    y = tf.matmul(x, w3) + b3  # 这个单层模型形状比较接近，但怎么改得拟合？  (1*401) * (401*401)

    loss = tf.reduce_mean(tf.square(y - yTrain))  # 均方误差
    step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
    optimizer = tf.train.AdamOptimizer(rate)
    # optimizer = tf.train.GradientDescentOptimizer(rate) # 梯度下降 画不出曲线？
    train = optimizer.minimize(loss, global_step=step)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    for time in range(0, 10001):
        train.run({x: xTrain}, sess)
        # if time % 1000 == 0:
        #     plt.clf()
        #     plt.plot(xTrain[0], yTrain[0], 'ro', label='train data')
        #     plt.plot(xTrain[0], y.eval({x: xTrain}, sess)[0], label='fit line')
        #     plt.legend()
        #     plt.savefig('curve_fitting_' + str(int(time / 1000)) + '.png', dpi=200)
    print "w3:", sess.run(w3)
    print "b3:", sess.run(b3)
    xTest = np.linspace(-20, 20, hidden_dim + 1).reshape([1, -1])
    yTest = computeYbyX(xTest)
    plt.clf()
    plt.plot(xTest[0], yTest[0], 'mo', label='test data')
    plt.plot(xTest[0], y.eval({x: xTest}, sess)[0], label=u'fit line')
    plt.legend()
    plt.savefig('curve_fitting_test.png', dpi=200)
    plt.show()


def classify():
    vectors_set = []
    for i in xrange(200):
        theta = np.random.random_sample() * 2 * np.pi
        r = np.random.uniform(0, 1)
        circle_1_x = math.sin(theta) * (r ** 0.5) + 1  # np.random.uniform(0, 2)
        circle_1_y = math.cos(theta) * (r ** 0.5) + 1  # np.random.uniform(0, 2)
        vectors_set.append([circle_1_x, circle_1_y])
    for i in xrange(200):
        theta = np.random.random_sample() * 2 * np.pi
        r = np.random.uniform(0, 1)
        circle_1_x = math.sin(theta) * (r ** 0.5) + 3
        circle_1_y = math.cos(theta) * (r ** 0.5) + 3
        vectors_set.append([circle_1_x, circle_1_y])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    plt.plot(x_data, y_data, 'ro', label='Original data')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # fitting_line()
    # fitting_others()
    two_layers()
    # classify()

    # sess = tf.Session()
    # x = tf.constant([[[1., 1.], [2., 2.]], [[3., 3.], [4., 4.]]])
    # print(sess.run(x))
    # print(sess.run(tf.reduce_mean(x)))
    # print(sess.run(tf.reduce_mean(x, 0)))
    # print(sess.run(tf.reduce_mean(x, 1)))
    # print(sess.run(tf.reduce_mean(x, 2)))
