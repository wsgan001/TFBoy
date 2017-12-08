# coding: utf-8
import tensorflow as tf

# input tensor shape [batch,         in_height,    in_width,    in_channels]
# filter shape       [filter_height, filter_width, in_channels, out_channels]


# assume input img
# 1 2 3
# 4 5 6
# 7 8 9
input_img = tf.constant([[[[1], [2], [3]],
                          [[4], [5], [6]],
                          [[7], [8], [9]]]], tf.float32, [1, 3, 3, 1])
# filter: 2
# output:
# 2  4  6
# 8  10 12
# 14 16 18
conv_filter1 = tf.constant([[[[2]]]], tf.float32, [1, 1, 1, 1])
op1 = tf.nn.conv2d(input_img, conv_filter1, strides=[1, 1, 1, 1], padding='VALID')

# add channel of img
input_img2 = tf.constant([[[[1, 1, 1, 1, 1],
                            [2, 2, 2, 2, 2],
                            [3, 3, 3, 3, 3]],
                           [[4, 4, 4, 4, 4],
                            [5, 5, 5, 5, 5],
                            [6, 6, 6, 6, 6]],
                           [[7, 7, 7, 7, 7],
                            [8, 8, 8, 8, 8],
                            [9, 9, 9, 9, 9]]]], tf.float32, [1, 3, 3, 5])
# 如果把图像设置5通道,卷积核输入5通道输出1通道
conv_filter2 = tf.constant([[[[2]]]], tf.float32, [1, 1, 5, 1])
op2 = tf.nn.conv2d(input_img2, conv_filter2, strides=[1, 1, 1, 1], padding='VALID')
# result
# 10 20 30
# 40 50 60
# 70 80 90


# 卷积核输入5通道输出5通道
conv_filter3 = tf.constant([[[[2]]]], tf.float32, [1, 1, 5, 5])
op3 = tf.nn.conv2d(input_img2, conv_filter3, strides=[1, 1, 1, 1], padding='VALID')
# 结果数值依然是5通道的叠加
#[[[[ 10.  10.  10.  10.  10.]
#   [ 20.  20.  20.  20.  20.]
#   [ 30.  30.  30.  30.  30.]]
#
#  [[ 40.  40.  40.  40.  40.]
#   [ 50.  50.  50.  50.  50.]
#   [ 60.  60.  60.  60.  60.]]
#
#  [[ 70.  70.  70.  70.  70.]
#   [ 80.  80.  80.  80.  80.]
#   [ 90.  90.  90.  90.  90.]]]]


# filter:
# 2 4
# 3 1
# output:
# 27 37
# 57 67
conv_filter4 = tf.constant([[[[2]], [[4]]], [[[3]], [[1]]]], tf.float32, [2, 2, 1, 1])
op4 = tf.nn.conv2d(input_img, conv_filter4, strides=[1, 1, 1, 1], padding='VALID')


# 把padding设置为same
# output
# 27 37 24
# 57 67 39
# 46 52 18
conv_filter5 = tf.constant([[[[2]], [[4]]], [[[3]], [[1]]]], tf.float32, [2, 2, 1, 1])
op5 = tf.nn.conv2d(input_img, conv_filter5, strides=[1, 1, 1, 1], padding='SAME')
# 这里的padding是卷积核可以移动到图像边缘之外,具体算法如下
# 关于padding的计算:http://www.jianshu.com/p/05c4f1621c7e
# input WxW, filter FxF, Stride S, output new_w,new_h
# new_h = W / S = 3 / 1 = 3
# 高度需要添加像素 lack = (new_h - 1) x S + F - W = (3-1)x1+2-3=1
# 在input顶部添加像素数: top = lack / 2 = 1/2 = 0 整型
# 在input底部添加像素数: bottom = lack - top = 1
# 上面是垂直方向.水平方向一样方法
# 所以最后的input变换成
# 1  2  3  0
# 4  5  6  0
# 7  8  9  0
# 0  0  0  0
# 用2x2的卷积核做卷积,结果就是3x3的



# conv的步长改为2 对应conv_filter1结果为
# 2  6
# 14 18
conv_filter6 = tf.constant([[[[2]]]], tf.float32, [1, 1, 1, 1])
op6 = tf.nn.conv2d(input_img, conv_filter6, strides=[1, 2, 2, 1], padding='VALID')
# 对于2D的,stride通常取[1, stride, stride, 1]
# data_format参数 NHWC 或 NCHW     [batch, height, width, channels]  对应stride的[1,2,2,1]
# 我们把stride改一下
op7 = tf.nn.conv2d(input_img, conv_filter6, strides=[1, 2, 1, 1], padding='SAME')
# 结果为
# 2  4  6
# 14 16 18


if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(op7)
        # print sess.run(op1)
        # print sess.run(op2)
        # print sess.run(op3)
