# coding: utf-8
import tensorflow as tf
# auto download MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # None 代表任意长度
W = tf.Variable(tf.zeros([784, 10]))  # weight
b = tf.Variable(tf.zeros([10]))  # bias
y = tf.nn.softmax(tf.matmul(x, W) + b)  # [None*784] * [784*10]

# create a new placeholder to save correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

# 计算交叉熵，具体见交叉熵计算公式  tf.log(y) 计算y中所有元素的log值  reduction_indices=[1] 横向压扁，0是纵向压扁
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 上面的代码虽然按照公式实现，但是数字不稳定，更稳定的方法是使用softmax_cross_entropy_with_logits接口
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
