# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

tf.set_random_seed(1)  # 随机数种子，保证后面循环都用同一组随机数
np.random.seed(1)

u = 0
sig = math.sqrt(0.2)
BATCH_SIZE = 64
num_points = 100
# vectors_set = []
# for i in xrange(num_points):
#     x1 = np.linspace(u - 10 * sig, u + 10 * sig, 100)
#     y1 = np.exp(-(x1 - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
#     vectors_set.append([x1, y1])
# x_data = [v[0] for v in vectors_set]
# y_data = [v[1] for v in vectors_set]

x_data = np.linspace(-20, 20, num_points)
y_data = []
ori_data = []
for i in xrange(num_points):
    y = x_data[i] * x_data[i] / 4
    y_data.append(y)
    ori_data.append()

PAINT_POINTS = np.vstack([np.linspace(-1, 1, num_points) for _ in range(BATCH_SIZE)])

# 画出原始图
plt.plot(x_data, y_data, 'r-', label='Original data')
plt.legend()
plt.show()

# D net
real_data = tf.placeholder(tf.float32, [None, num_points])
D_in_real = tf.layers.dense(real_data, 128, tf.nn.relu(), name='real')
D_prob_real = tf.layers.dense(D_in_real, 1, tf.nn.sigmoid(), name='out')

D_in_fake = tf.layers.dense(G_out, 128, tf.nn.relu(), name='fake', reuse=True)
D_prob_fake = tf.layers.dense(D_in_fake, 1, tf.nn.sigmoid(), name='out', reuse=True)

# G net
G_in = tf.placeholder(tf.float32, [None, 10])
G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu())
G_out = tf.layers.dense(G_l1, num_points)

# loss
D_loss = -tf.reduce_mean(tf.log(D_prob_real) + tf.log(1 - D_prob_fake))
G_loss = tf.reduce_mean(tf.log(1 - D_prob_fake))

D_train = tf.train.AdamOptimizer(0.02).minimize(D_loss,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                           scope='Discriminator'))
G_train = tf.train.AdamOptimizer(0.02).minimize(G_loss,
                                                var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                           scope='Generator'))


with tf.Session as sess:
    sess.run(tf.global_variables_initializer())

    plt.ion()
    for step in range(5000):
        real_line =



if __name__ == "__main__":
    print draw_line()
