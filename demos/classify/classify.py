# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 生成螺旋形的线形不可分数据点
np.random.seed(0)
N = 100  # 每个类的数据个数
D = 2  # 输入维度
K = 3  # 类的个数
X = np.zeros((N * K, D))  # (300,2)
y = np.zeros(N * K, dtype='uint8')  # (300)  相当于label
for j in xrange(K):
    # print j
    ix = range(N * j, N * (j + 1))
    # print ix
    r = np.linspace(0.0, 1, N)  # radius [100 * 1]
    # print r
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta  [100 * 1]
    # print t
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    # >>> np.c_[np.array([1, 2, 3]), np.array([4, 5, 6])]
    # array([[1, 4],
    #        [2, 5],
    #        [3, 6]])
    # >>> np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
    # array([[1, 2, 3, 0, 0, 4, 5, 6]])

    # print X,X[:, 0], X[:, 1]  输出X, X第一列，X第二列
    y[ix] = j
    # print y
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)  # 散点图
plt.xlim([-1, 1])
plt.ylim([-1, 1])
# plt.legend()
# plt.show()

num_label = 3
num_data = N * num_label
labels = (np.arange(num_label) == y[:, None]).astype(np.float32)  # y=[a,b,c,d] y[:, None]=[[a],[b],[c],[d]]
# print np.arange(num_label)  # [0 1 2]
# print np.arange(num_label) == y[:, None]  # [[ True False False]...[False, True, False]...[False, False, True]]
# print labels.shape  # (300, 3)
# print X.shape  # (300, 2)

hidden_size_1 = 50
hidden_size_2 = 50

beta = 0.001  # L2 正则化系数
learning_rate = 0.1  # 学习速率

graph = tf.Graph()
with graph.as_default():
    x = tf.constant(X.astype(np.float32))
    tf_labels = tf.constant(labels)

    # 隐藏层1
    hidden_layer_weights_1 = tf.Variable(tf.truncated_normal([D, hidden_size_1], stddev=math.sqrt(
        2.0 / num_data)))  # [2*50]  stddev=math.sqrt(2.0 / num_data))截断正太分布的标准差
    hidden_layer_bias_1 = tf.Variable(tf.zeros([hidden_size_1]))  # [1*50]

    # 隐藏层2
    hidden_layer_weights_2 = tf.Variable(
        tf.truncated_normal([hidden_size_1, hidden_size_2], stddev=math.sqrt(2.0 / hidden_size_1)))  # [50 * 50]
    hidden_layer_bias_2 = tf.Variable(tf.zeros([hidden_size_2]))  # [1 * 50]

    # 输出层
    out_weights = tf.Variable(
        tf.truncated_normal([hidden_size_2, num_label], stddev=math.sqrt(2.0 / hidden_size_2)))  # [50 * 3]
    out_bias = tf.Variable(tf.zeros([num_label]))  # [1 * 3]

    z1 = tf.matmul(x, hidden_layer_weights_1) + hidden_layer_bias_1  # [300*2] * [2*50] + [1*50] = [300*50]
    h1 = tf.nn.relu(z1)  # 使用max(features, 0) 修正   relu:Rectified Linear Unit线性整流函数

    z2 = tf.matmul(h1, hidden_layer_weights_2) + hidden_layer_bias_2  # [300*50] * [50*50] = [300*50]
    h2 = tf.nn.relu(z2)

    logits = tf.matmul(h2, out_weights) + out_bias  # [300*50] * [50*3] + [1*3] = [300*3]

    # L2正则化  tf.nn.l2_loss output = sum(t ** 2) / 2
    regularization = tf.nn.l2_loss(hidden_layer_weights_1) + tf.nn.l2_loss(hidden_layer_weights_2) + tf.nn.l2_loss(
        out_weights)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits) + beta * regularization)  # [300*1]
    # softmax_cross_entropy_with_logits() Computes softmax cross entropy between logits and labels.
    # Measures the probability error in discrete classification tasks in which the classes are mutually exclusive
    # (each entry is in exactly one class).
    # For example, each CIFAR-10 image is labeled with one and only one label:
    # an image can be a dog or a truck, but not both.

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_prediction = tf.nn.softmax(logits)  # [300*3]

    weights = [hidden_layer_weights_1, hidden_layer_bias_1, hidden_layer_weights_2, hidden_layer_bias_2, out_weights,
               out_bias]

num_steps = 5000


def accuracy(predictions, labels):
    # print predictions [300*3]
    # print labels  [300*3]
    # print np.argmax(predictions, 1) # 取一个维度的最大值， 返回的是索引
    # 比如array([[0, 1, 2],[3, 4, 5]]) 在0方向返回array([1, 1, 1])，在1方向上返回array([2, 2])
    # 这里[300*3]处理后是[300*1]
    # print np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))  # 比较两个数组，每一个量相同为True  sum取所有True的数量
    # print predictions.shape, predictions.shape[0], predictions.shape[1]  # (300, 3), 300, 3
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def relu(x):
    return np.maximum(0, x)


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])

        if (step % 1000 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, labels))
    # print session.run(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=logits))

    w1, b1, w2, b2, w3, b3 = weights
    # 显示分类器
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # print x_min, x_max
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # print y_min, y_max
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # print xx.shape  # (184, 189)

    # meshgrid 例子
    # nx, ny = (3, 2)
    # x = np.linspace(0, 1, nx)
    # y = np.linspace(0, 1, ny)
    # xv, yv = np.meshgrid(x, y)
    # xv
    # array([[ 0. ,  0.5,  1. ],
    #        [ 0. ,  0.5,  1. ]])
    # yv
    # array([[ 0.,  0.,  0.],
    #        [ 1.,  1.,  1.]])
    # xv, yv = np.meshgrid(x, y, sparse=True)  # make sparse output arrays
    # xv
    # array([[ 0. ,  0.5,  1. ]])
    # yv
    # array([[ 0.],
    #        [ 1.]])

    Z = np.dot(relu(np.dot(relu(np.dot(np.c_[xx.ravel(), yy.ravel()], w1.eval()) + b1.eval()), w2.eval()) + b2.eval()),
               w3.eval()) + b3.eval()
    # xx.ravel() 在1维上[1*N]排序
    # np.dot 点积
    # print Z.shape  # (34776, 3)
    Z = np.argmax(Z, axis=1)
    # print Z.shape  # (34776,)
    Z = Z.reshape(xx.shape)
    # print Z.shape  # (184, 189)
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    # plt.contourf 等高线区域填充颜色  plt.contour 等高线
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.show()
