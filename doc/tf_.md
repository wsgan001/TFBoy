http://www.cnblogs.com/lienhua34/p/5998375.html

Tensorflow是一个基于图的计算系统，其主要应用于机器学习。
从Tensorflow名字的字面意思可以拆分成两部分来理解：Tensor+flow。
* Tensor：中文名可以称为“张量”，其本质就是任意维度的数组。一个向量就是一个1维的Tensor，一个矩阵就是2维的Tensor。
* Flow：指的就是图计算中的数据流。

当我们想要使用Tensorflow做什么事情的时候，一般需要三个操作步骤：
1. 创建Tensor；
2. 添加Operations（Operations输入Tensor，然后输出另一个Tensor）；
3. 执行计算（也就是运行一个可计算的图）。

Tensorflow有个图的概念，Operations会添加到图中，作为图的节点。在添加某Operation的时候，不会立即执行该Operation。Tensorflow会等待所有Operation添加完毕，然后Tensorflow会优化该计算图，以便决定如何执行计算。

下面我们通过两个向量相加的简单例子来看一下Tensorflow的基本用法。
```python
import tensorflow as tf
with tf.Session():
  input1 = tf.constant([1.0 1.0 1.0 1.0])
  input2 = tf.constant([2.0 2.0 2.0 2.0])
  output = tf.add(input1, input2)
  result = output.eval()
  print result
```

Tensorflow的计算必须要在一个Session的上下文中。Session会包含一个计算图，而这个图你添加的Tensors和Operations。当然，你在添加Tensor和Operation的时候，它们都不会立即进行计算，而是等到最后需要计算Session的结果的时候。当Tensorflow之后了计算图中的所有Tensor和Operation之后，其会知道如何去优化和执行图的计算。

两个tf.constant() 语句向计算图中创建了两个Tensor。调用tf.constant()的动作大致可以说为，创建两个指定维度的Tensor，以及两个constant操作符用于初始化相对应的Tensor（不会立即执行）。

tf.add()语句向计算图中添加了一个add操作，当不会立即执行，这时候add操作的结果还无法获取。此时，计算图大致如下所示
![](https://github.com/lienhua34/notes/raw/master/tensorflow/asserts/addvec.jpg?_=5998375)

当我们最后调用output.eval()时，会触发Tensorflow执行计算图，从而获取output计算结点的结果。

Variable的使用
我们上面的例子使用的Tensor是常量（constant），而在我们实际的机器学习任务中，我们往往需要变量（variable）来记录一下可变的状态（例如神经网络节点的权重参数等）。下面我们来看一个简单的variable例子。

```python
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
  # Set up two variables, total and weights, that we'll change repeatedly.
  total = tf.Variable(tf.zeros([1, 2]))
  weights = tf.Variable(tf.random_uniform([1,2]))

  # Initialize the variables we defined above.
  tf.initialize_all_variables().run()

  # This only adds the operators to the graph right now. The assignment
  # and addition operations are not performed yet.
  update_weights = tf.assign(weights, tf.random_uniform([1, 2], -1.0, 1.0))
  update_total = tf.assign(total, tf.add(total, weights))

  for _ in range(5):
    # Actually run the operation graph, so randomly generate weights and then
    # add them into the total. Order does matter here. We need to update
    # the weights before updating the total.
    sess.run(update_weights)
    sess.run(update_total)

    print weights.eval(), total.eval()
```

上面的代码就是创建了两个变量total和weights（都是1维的tensor），total所有元素初始化为0，而weights的元素则用-1到1之间的随机数进行初始化。然后在某个迭代中，使用-1到1之间的随机数来更新变量weights的元素，然后添加到变量total中。

在调用tf.Variable()的时候，只是定了变量以及变量的初始化操作（实际上并未执行）。所有变量都需要在开始执行图计算之前进行初始化。调用tf.initialize_all_variables().run()来对所有变量进行初始化。

在for循环中，

sess.run(update_weights)

触发执行更新weights变量的计算。

sess.run(update_total)

则处理了将变量total和变量weights进行相加，并将结果赋值到变量total。


Tensorflow是基于图（Graph）的计算系统。而图的节点则是由操作（Operation）来构成的，而图的各个节点之间则是由张量（Tensor）作为边来连接在一起的。所以Tensorflow的计算过程就是一个Tensor流图。Tensorflow的图则是必须在一个Session中来计算。

Session
Session提供了Operation执行和Tensor求值的环境。如下面所示，

```python
import tensorflow as tf

# Build a graph.
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b

# Launch the graph in a session.
sess = tf.Session()

# Evaluate the tensor 'c'.
print sess.run(c)
sess.close()

# result: [3., 8.]
```

一个Session可能会拥有一些资源，例如Variable或者Queue。当我们不再需要该session的时候，需要将这些资源进行释放。有两种方式，

调用session.close()方法；
使用with tf.Session()创建上下文（Context）来执行，当上下文退出时自动释放。

```python
import tensorflow as tf

# Build a graph.
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b

with tf.Session() as sess:
    print sess.run(c)
```

Session类的构造函数如下所示：

tf.Session.__init__(target='', graph=None, config=None)

如果在创建Session时没有指定Graph，则该Session会加载默认Graph。如果在一个进程中创建了多个Graph，则需要创建不同的Session来加载每个Graph，而每个Graph则可以加载在多个Session中进行计算。

执行Operation或者求值Tensor有两种方式：

调用Session.run()方法： 该方法的定义如下所示，参数fetches便是一个或者多个Operation或者Tensor。

tf.Session.run(fetches, feed_dict=None)

调用Operation.run()或则Tensor.eval()方法： 这两个方法都接收参数session，用于指定在哪个session中计算。但该参数是可选的，默认为None，此时表示在进程默认session中计算。

那如何设置一个Session为默认的Session呢？有两种方式：

1. 在with语句中定义的Session，在该上下文中便成为默认session；上面的例子可以修改成：

```python
import tensorflow as tf

# Build a graph.
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b

with tf.Session():
   print c.eval()
```

2. 在with语句中调用Session.as_default()方法。 上面的例子可以修改成：

```python
import tensorflow as tf

# Build a graph.
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
sess = tf.Session()
with sess.as_default():
    print c.eval()
sess.close()
```


Graph
Tensorflow中使用tf.Graph类表示可计算的图。图是由操作Operation和张量Tensor来构成，其中Operation表示图的节点（即计算单元），而Tensor则表示图的边（即Operation之间流动的数据单元）。

tf.Graph.__init__()

创建一个新的空Graph

在Tensorflow中，始终存在一个默认的Graph。如果要将Operation添加到默认Graph中，只需要调用定义Operation的函数（例如tf.add()）。如果我们需要定义多个Graph，则需要在with语句中调用Graph.as_default()方法将某个graph设置成默认Graph，于是with语句块中调用的Operation或Tensor将会添加到该Graph中。

例如，

```python
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    c1 = tf.constant([1.0])
with tf.Graph().as_default() as g2:
    c2 = tf.constant([2.0])

with tf.Session(graph=g1) as sess1:
    print sess1.run(c1)
with tf.Session(graph=g2) as sess2:
    print sess2.run(c2)

# result:
# [ 1.0 ]
# [ 2.0 ]
```

如果将上面例子的sess1.run(c1)和sess2.run(c2)中的c1和c2交换一下位置，运行会报错。因为sess1加载的g1中没有c2这个Tensor，同样地，sess2加载的g2中也没有c1这个Tensor。

Operation
一个Operation就是Tensorflow Graph中的一个计算节点。其接收零个或者多个Tensor对象作为输入，然后产生零个或者多个Tensor对象作为输出。Operation对象的创建是通过直接调用Python operation方法（例如tf.matmul()）或者Graph.create_op()。

例如c = tf.matmul(a, b)表示创建了一个类型为MatMul的Operation，该Operation接收Tensor a和Tensor b作为输入，而产生Tensor c作为输出。

当一个Graph加载到一个Session中，则可以调用Session.run(op)来执行op，或者调用op.run()来执行（op.run()是tf.get_default_session().run()的缩写）。

Tensor
Tensor表示的是Operation的输出结果。不过，Tensor只是一个符号句柄，其并没有保存Operation输出结果的值。通过调用Session.run(tensor)或者tensor.eval()方可获取该Tensor的值。



关于Tensorflow的图计算过程
我们通过下面的代码来看一下Tensorflow的图计算过程：

```python
import tensorflow as tf
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)
d = tf.constant(4)
add1 = tf.add(a, b)
mul1 = tf.mul(b, c)
add2 = tf.add(c, d)
output = tf.add(add1, mul1)
with tf.Session() as sess:
    print sess.run(output)

# result: 9
```

上面的代码构成的Graph如下图所示，
![](https://github.com/lienhua34/notes/raw/master/tensorflow/asserts/graph_compute_flow.jpg?_=5998853)

当Session加载Graph的时候，Graph里面的计算节点都不会被触发执行。当运行sess.run(output)的时候，会沿着指定的Tensor output来进图路径往回触发相对应的节点进行计算（图中红色线表示的那部分）。当我们需要output的值时，触发Operation tf.add(add1, mul1)被执行，而该节点则需要Tensor add1和Tensor mul1的值，则往回触发Operation tf.add(a, b)和Operation tf.mul(b, c)。以此类推。

所以在计算Graph时，并不一定是Graph中的所有节点都被计算了，而是指定的计算节点或者该节点的输出结果被需要时。