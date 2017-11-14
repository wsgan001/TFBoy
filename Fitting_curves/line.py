# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


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

    # 构建y=w*x+b，并用梯度下降法优化w和b，并输出最终的w与b的值
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # shape(1,), [-1, 1)
    b = tf.Variable(tf.zeros([1]))  # [0]
    # init = tf.global_variables_initializer()  # Returns an Op that initializes global variables.
    # sess = tf.Session()
    # sess.run(init)
    # print(sess.run(W))
    # print(sess.run(b))

    y = W * x_data + b  # line
    loss = tf.reduce_mean(tf.square(y - y_data))  # square 平方  loss是均方误差？

    global_step = tf.Variable(0, trainable=False)
    initial_learning_rate = 0.5  # 初始学习率  下面是tf提供随着训练自动慢慢减小learning_rate的方法
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=10,
                                               decay_rate=0.9)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)  # 0.5 is learning_rate  可以调整看结果差距
    train = optimizer.minimize(loss)  # minimize方法会调用compute_gradients()和apply_gradients()方法 也就是计算梯度 和 应用梯度
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for step in xrange(8):  # 训练8次  增大后可以看到拟合程度变高
        sess.run(train)
    print(step, sess.run(W), sess.run(b))

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
    noise = np.random.normal(-100, 100, x.shape)
    return 400 * np.sin(x) + 2 * x * x + noise


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def two_layers():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    xTrain = np.linspace(-20, 20, 401).reshape([1, -1])
    noise = np.random.normal(-0.2, 0.2, xTrain.shape)
    yTrain = computeYbyX(xTrain)

    # save train image
    plt.clf()
    plt.plot(xTrain[0], yTrain[0], 'ro', label='train data')
    plt.legend()
    plt.savefig('curve_data.png', dpi=200)

    x = tf.placeholder(tf.float32, [1, 401])

    hiddenDim = 400

    w = weight_variable([hiddenDim, 1])
    b = bias_variable([hiddenDim, 1])

    w2 = weight_variable([1, hiddenDim])
    b2 = bias_variable([1])

    w3 = weight_variable([401, 401])
    b3 = bias_variable([1, 401])

    hidden = tf.nn.sigmoid(tf.matmul(w, x) + b)
    y = tf.matmul(w2, hidden) + b2

    loss = tf.reduce_mean(tf.square(y - yTrain))
    step = tf.Variable(0, trainable=False)
    rate = tf.train.exponential_decay(0.15, step, 1, 0.9999)
    optimizer = tf.train.AdamOptimizer(rate)
    train = optimizer.minimize(loss, global_step=step)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for time in range(0, 10001):
        train.run({x: xTrain}, sess)
        if time % 1000 == 0:
            plt.clf()
            plt.plot(xTrain[0], yTrain[0], 'ro', label='train data')
            plt.plot(xTrain[0], y.eval({x: xTrain}, sess)[0], label='fit line')
            plt.legend()
            plt.savefig('curve_fitting_' + str(int(time / 1000)) + '.png', dpi=200)

    xTest = np.linspace(-40, 40, 401).reshape([1, -1])
    yTest = computeYbyX(xTest)
    plt.clf()
    plt.plot(xTest[0], yTest[0], 'mo', label='test data')
    plt.plot(xTest[0], y.eval({x: xTest}, sess)[0], label=u'fit line')
    plt.legend()
    plt.savefig('curve_fitting_test.png', dpi=200)
    plt.show()


if __name__ == "__main__":
    fitting_line()
    fitting_others()
    two_layers()

    # sess = tf.Session()
    # x = tf.constant([[[1., 1.], [2., 2.]], [[3., 3.], [4., 4.]]])
    # print(sess.run(x))
    # print(sess.run(tf.reduce_mean(x)))
    # print(sess.run(tf.reduce_mean(x, 0)))
    # print(sess.run(tf.reduce_mean(x, 1)))
    # print(sess.run(tf.reduce_mean(x, 2)))
