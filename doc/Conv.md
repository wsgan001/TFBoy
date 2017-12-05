
### [深度学习]Charpter 9:卷积网络
`卷积网络`convolutional network,也叫做`卷积神经网络`convolutional neural network CNN
专门用来处理类似**网格结构**数据的神经网络.
比如
* 时间序列,轴上的一维网格
* 图像数据,二维像素网格
我们把至少在网络中一层中使用卷积运算来替代一般的矩阵乘法运算的神经网络 称为 卷积网络



#### 卷积 convolution
CNN中用到的卷积和其他领域的定义并不完全一致

> 关于数学概念,参考这里[如何通俗易懂地解释卷积？](https://www.zhihu.com/question/22298352)

卷积公式
$$s(t) = \int x(a)w(t-a)da$$

卷积通常用星号表示
$$s(t)=(x*w)(t)$$


在卷积网络的术语中,卷积的第一个参数(函数x)通常叫做**输入(input)**,第二个参数(函数w)叫做**核函数(kernel function)**.输出有时被称作**特征映射(feature map)**

> 在模型检测的文章里经常会提到**feature map**,所以这个要记住

上面求积分是考虑连续的情况,离散的卷积如下:
$$
s(t)=(x*w)(t)=\sum_{a=-\infty}^\infty x(a)w(t-a)
$$

在机器学习中,输入通常是多维数组(Tensor),而核通常是由学习算法优化得到的多维数组的参数.