# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.0001
BATCH_SIZE = 64
num_points = 50
num_ideas = 15
PAINT_POINTS = np.vstack([np.linspace(-1, 1, num_points) for _ in range(BATCH_SIZE)])


def create_real():
    return np.power(PAINT_POINTS, 3)


# G net
with tf.variable_scope('Generator'):
    with tf.name_scope('G_in'):
        G_in = tf.placeholder(tf.float32, [None, num_ideas])
    with tf.name_scope('G_l1'):
        G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    with tf.name_scope('G_out'):
        G_out = tf.layers.dense(G_l1, num_points)

# D net
with tf.variable_scope('Discriminator'):
    real_data = tf.placeholder(tf.float32, [None, num_points], name='real_in')
    with tf.name_scope('D_in_real'):
        D_in_real = tf.layers.dense(real_data, 128, tf.nn.relu, name='l')
    with tf.name_scope('D_prob_real'):
        D_prob_real = tf.layers.dense(D_in_real, 1, tf.nn.sigmoid, name='out')
    with tf.name_scope('D_in_fake'):
        D_in_fake = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)
    with tf.name_scope('D_prob_fake'):
        D_prob_fake = tf.layers.dense(D_in_fake, 1, tf.nn.sigmoid, name='out', reuse=True)

# loss
with tf.name_scope('D_loss'):
    D_loss = tf.reduce_mean(-tf.log(D_prob_real) - tf.log(1 - D_prob_fake))
    tf.summary.scalar('D_loss', D_loss)
with tf.name_scope('G_loss'):
    G_loss = tf.reduce_mean(tf.log(1 - D_prob_fake))
    tf.summary.scalar('G_loss', G_loss)

with tf.name_scope('D_train'):
    D_train = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
with tf.name_scope('G_train'):
    G_train = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/", sess.graph)
    plt.ion()
    for step in range(10000):
        real_line = create_real()
        G_idea = np.random.randn(BATCH_SIZE, num_ideas)
        G_paintings, prob_real, prob_fake, d_loss, g_loss = sess.run([G_out, D_prob_real, D_prob_fake, D_loss, G_loss, D_train, G_train],  # train and get results
                                        {G_in: G_idea, real_data: real_line})[:5]

        if step % 500 == 0:  # plotting
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated', )
            plt.plot(PAINT_POINTS[0], np.power(PAINT_POINTS[0], 3), c='r', lw=3, label='Real')
            plt.ylim((-1.5, 1.5))
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.01)
            rs = sess.run(merged, feed_dict={G_in: G_idea, real_data: real_line})
            writer.add_summary(rs, step)
            print "Prob real = %.2f, fake = %.2f, d_loss = %.2f, g_loss = %.2f" % (prob_real.mean(), prob_fake.mean(), d_loss, g_loss)
plt.ioff()
plt.show()
