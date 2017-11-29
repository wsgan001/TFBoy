# coding: utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf.set_random_seed(1)  # 随机数种子，保证后面循环都用同一组随机数
# np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001  # learning rate for generator
LR_D = 0.0001  # learning rate for discriminator
N_IDEAS = 5  # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])  # (64, 15)


# print PAINT_POINTS
# print np.linspace(-1, 1, ART_COMPONENTS).shape  # (15)
# print PAINT_POINTS.shape  # (64, 15)
# 因为设置了random seed，所以这里linespace 随机值都一样，15个点分了64组

# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')


# plt.legend(loc='upper right')
# plt.show()
# test = np.array([[[1], [2], [3]], [[4], [5], [6]]])
# print test.shape  # (2,3,1)
# print test[:, np.newaxis].shape  # (2,1,3,1)

# print np.power(PAINT_POINTS, 2).shape
# print np.power(PAINT_POINTS, 3).shape

def artist_works():  # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]  # [1,2) 取值， shape (64)
    # print np.random.uniform(1, 2, size=BATCH_SIZE).shape
    # print a  # (64, 1)
    # print np.power(PAINT_POINTS, 2).shape  # (64, 15)
    paintings = a * np.power(PAINT_POINTS, 3) + (a - 1)  # (64,15)
    return paintings


with tf.variable_scope('Generator'):
    # with tf.name_scope('G_in'):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])  # random ideas (could from normal distribution) (N, 5)
    # with tf.name_scope('G_l1'):
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)  # 全连接层， G_in 输入tensor， 128该层神经单元节点数量， relu激活函数
    # with tf.name_scope('G_out'):
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)  # making a painting from these random ideas

with tf.variable_scope('Discriminator'):
    # with tf.name_scope('real_art'):
    real_art = tf.placeholder(tf.float32, [None, ART_COMPONENTS],
                              name='real_in')  # receive art work from the famous artist
    # with tf.name_scope('D_l0'):
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
    # with tf.name_scope('prob_artist0'):
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid,
                                   name='out')  # probability that the art work is made by artist
    # reuse layers for generator
    # with tf.name_scope('D_l1'):
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)  # receive art work from a newbie like G
    # with tf.name_scope('prob_artist1'):
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out',
                                   reuse=True)  # probability that the art work is made by artist

    # with tf.name_scope('D_loss'):
D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1))
# tf.summary.scalar('D_loss', D_loss)
# with tf.name_scope('G_loss'):
G_loss = tf.reduce_mean(tf.log(1 - prob_artist1))
# tf.summary.scalar('G_loss', G_loss)

# with tf.name_scope('D_train'):
train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
# with tf.name_scope('G_train'):
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

with tf.Session() as sess:
    # sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    # plt.ion()  # something about continuous plotting
    for step in range(5000):
        artist_paintings = np.power(PAINT_POINTS, 2)  # artist_works()  # real painting from artist
        # print artist_paintings
        G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)  # (64, 5)
        G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],  # train and get results
                                        {G_in: G_ideas, real_art: artist_paintings})[:3]
        # print sess.run([G_out, prob_artist0, D_loss, train_D, train_G],  # train and get results
        #                                 {G_in: G_ideas, real_art: artist_paintings})


        if step % 50 == 0:  # plotting
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting', )
            plt.plot(PAINT_POINTS[0], np.power(PAINT_POINTS[0], 3), c='#74BCFF', lw=3, label='upper bound')
            # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.01)
            # rs = sess.run(merged, feed_dict={G_in: G_ideas, real_art: artist_paintings})
            # writer.add_summary(rs, step)

plt.ioff()
plt.show()
